"""El campo del grafo DENTRO del lazo: M zonas acopladas solo por el campo.

QUE FALTABA
-----------
`field.py` estudia el campo AISLADO: colocacion, umbral de flutter, propagacion
de un impulso. Pero el lazo cerrado usaba un campo escalar de un nodo, asi que
la pieza distintiva de la propuesta -- el acoplamiento por el laplaciano del
grafo de subsistemas -- no estaba nunca acoplada al MPC. Aqui si.

MONTAJE
-------
M zonas identicas (tramos de un lazo solar, modulos de un invernadero, etapas de
una linea de proceso). Cada zona tiene su PROPIA planta, su PROPIO MPC y su
PROPIO filtro de seguridad. NO hay ningun acoplamiento fisico entre ellas y
tampoco comunicacion explicita: lo unico que las une es el campo homeostatico.

La demanda llega SOLO A LA ZONA 1. Las demas tienen su valor de reposo. La
pregunta es si la decision de compromiso viaja aguas abajo, cuanto se atenua y
con cuanto retraso.

POR QUE LA COMPARACION ONDA/DIFUSION ES JUSTA POR CONSTRUCCION
--------------------------------------------------------------
Las dos ramas comparten la MISMA rigidez K = w0^2 I + c^2 L, luego comparten el
MISMO perfil estacionario: en reposo ambas resuelven K h = K h_d. La diferencia
es exclusivamente DINAMICA -- con que rapidez y con que fase se propaga el
cambio -- asi que no hace falta igualar nada a mano: la ganancia en continua ya
esta igualada por la estructura.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .field import FieldParams, chain_graph, operators
from .mompc import MOMPC, shifted_sequence, solve
from .weights import admissible_set, certified_radius, simplex_grid


# --------------------------------------------------------------------------
#                          EL CAMPO SOBRE EL GRAFO
# --------------------------------------------------------------------------

class GraphField:
    """Campo homeostatico de M nodos. Rama de onda (2o orden, Verlet) o de
    difusion (1er orden, IMEX). Misma K en ambas, luego mismo estacionario.

    FORZAMIENTO: hay DOS semanticas y la diferencia es estructural, no de
    sintonia. La revision adversarial de la arena de transporte la encontro
    midiendo la ganancia en CONTINUA del operador, y esta verificada en A18.

      forcing="target"  (el original)   f = K (h_d - h)
          En equilibrio K(h - h_d) = 0 => h = h_d EXACTAMENTE, luego
          H(0) = K^-1 K = I y la ganancia cruzada en continua es CERO. Cada
          nodo alcanza SU propia consigna y una alerta SOSTENIDA no cruza: solo
          cruzan los transitorios de flanco. Es la semantica correcta si cada
          nodo tiene consigna propia y el grafo solo debe acoplar la DINAMICA.

      forcing="source"  (transporte)    f = w0^2 h_d
          En equilibrio K h = w0^2 h_d => h = w0^2 K^-1 h_d, cuya parte fuera de
          la diagonal NO es cero: una inyeccion sostenida en un nodo SI llega a
          los demas, atenuada con la distancia. Es la semantica necesaria si el
          campo debe TRANSPORTAR informacion desde una fuente.

    Con b = beta = 0 (G = 0) las dos ramas comparten exactamente la misma
    ganancia en continua bajo cualquiera de los dos forzamientos, de modo que el
    nulo onda/difusion sigue siendo justo. Con b != 0 NO: G entra sobre la
    velocidad en la rama de onda y sobre la posicion en la difusiva, asi que el
    estacionario deja de coincidir y el nulo quedaria confundido.
    """

    def __init__(self, M: int, p: FieldParams, branch: str = "wave",
                 a0: float = 0.5, theta: float = 0.5, graph=chain_graph,
                 forcing: str = "target"):
        if branch not in ("wave", "diffusion"):
            raise ValueError(branch)
        if forcing not in ("target", "source"):
            raise ValueError(forcing)
        self.M, self.p, self.branch, self.theta = M, p, branch, theta
        self.forcing = forcing
        L, A = graph(M)
        self.K, self.C, self.G = operators(L, A, p)
        if forcing == "source" and (p.b != 0.0 or p.beta != 0.0):
            raise ValueError("forcing='source' exige b = beta = 0: con G != 0 el "
                             "estacionario de onda y difusion difiere y el nulo "
                             "de mecanismo quedaria confundido")
        self.a0 = a0
        self.h = np.full(M, a0)
        self.v = np.zeros(M)
        # IMEX para la rama difusiva: (I + lam(K+G))^{-1}, incondicionalmente estable
        lam = 1.0 / p.gamma
        self._Minv = np.linalg.inv(np.eye(M) + lam * (self.K + self.G))
        self._lam = lam

    def dc_gain(self) -> np.ndarray:
        """H(0): que fraccion de una inyeccion sostenida en el nodo j llega al i."""
        if self.forcing == "target":
            return np.eye(self.M)
        return self.p.w0 ** 2 * np.linalg.inv(self.K)

    def verlet_stable(self, dt: float = 1.0) -> bool:
        """El Verlet explicito es condicionalmente estable: dt*sqrt(lmax(K)) < 2.
        Con c grande el integrador desborda, y una configuracion desbordada no
        puede entrar en un barrido de sintonia como si fuera un candidato mas."""
        if self.branch != "wave":
            return True
        return bool(dt * np.sqrt(max(np.linalg.eigvalsh(self.K).max(), 0.0)) < 2.0)

    def reset(self) -> None:
        self.h = np.full(self.M, self.a0)
        self.v = np.zeros(self.M)

    def _acc(self, h, v, hd):
        if self.forcing == "target":
            return -(self.C + self.G) @ v - self.K @ (h - hd)
        return -(self.C + self.G) @ v - self.K @ h + self.p.w0 ** 2 * hd

    def step(self, hd: np.ndarray, dt: float = 1.0) -> np.ndarray:
        hd = np.asarray(hd, float).ravel()
        if self.branch == "wave":
            acc = self._acc(self.h, self.v, hd)
            self.h = self.h + dt * self.v + 0.5 * dt * dt * acc
            vh = self.v + 0.5 * dt * acc
            self.v = vh + 0.5 * dt * self._acc(self.h, vh, hd)
        else:
            if self.forcing == "target":
                # gamma h' + (K+G)(h - hd) = 0 -> (I+lam(K+G)) h+ = h + lam(K+G) hd
                self.h = self._Minv @ (self.h + self._lam * (self.K + self.G) @ hd)
            else:
                # gamma h' + (K+G) h = w0^2 hd
                self.h = self._Minv @ (self.h + self._lam * self.p.w0 ** 2 * hd)
            self.v = np.zeros(self.M)
        return self.h.copy()

    def sync(self, node: int, a_applied: float, blocked: bool) -> None:
        if blocked:
            self.v[node] *= self.theta
            self.h[node] = a_applied


class Independent(GraphField):
    """Control nulo: M campos escalares SIN acoplamiento (c = D = 0).

    Aisla exactamente lo que aporta el grafo: misma dinamica local, cero
    transporte. Si el resultado con acoplamiento no bate a este, el grafo no
    esta haciendo nada.
    """

    def __init__(self, M: int, p: FieldParams, **kw):
        p0 = FieldParams(w0=p.w0, zeta=p.zeta, c=0.0, D=0.0, b=0.0, beta=0.0,
                         gamma=p.gamma)
        super().__init__(M, p0, **kw)


# --------------------------------------------------------------------------
#                          EL LAZO DISTRIBUIDO
# --------------------------------------------------------------------------

@dataclass
class DistTrace:
    alpha: np.ndarray        # (T, M) peso aplicado en cada zona
    demand: np.ndarray       # (T,)  demanda limpia, que solo ve la zona 1
    x: np.ndarray            # (T, M, n)
    u: np.ndarray            # (T, M)
    stage: np.ndarray        # (T, M)
    blocked: np.ndarray      # (T, M)


def distributed_loop(mompc: MOMPC, field: GraphField, demand: Callable[[int], float],
                     T: int = 80, seed: int = 0, x0: Sequence[float] = (5.0, 5.0),
                     noise_std: float = 0.10, dist_amp: float = 0.6,
                     dist_period: float = 17.0, a_rest: float = 0.5,
                     grid_points: int = 21, mismatch: Optional[np.ndarray] = None,
                     dist_std: float = 0.0) -> Optional[DistTrace]:
    """M zonas independientes, cada una con su MPC y su filtro; el campo es lo
    unico que las conecta. La demanda entra SOLO en la zona 1."""
    prob = mompc.problem
    plant = prob.plant
    M = field.M
    rng = np.random.default_rng(seed)
    grid = simplex_grid(2, grid_points)
    step = float(np.linalg.norm(grid[1] - grid[0]))
    Areal = plant.A if mismatch is None else plant.A + mismatch

    field.reset()
    X = [np.asarray(x0, float).copy() for _ in range(M)]
    a_prev = [a_rest] * M
    sols = []
    for i in range(M):
        s = solve(mompc, X[i], np.array([1 - a_rest, a_rest]))
        if s is None:
            return None
        sols.append(s)

    AL = np.zeros((T, M)); DEM = np.zeros(T); XS = np.zeros((T, M, plant.n))
    US = np.zeros((T, M)); ST = np.zeros((T, M)); BL = np.zeros((T, M), bool)

    for t in range(T):
        adc = float(np.clip(demand(t), 0.0, 1.0))
        DEM[t] = adc
        # objetivo del campo: la demanda SOLO en el nodo 0, reposo en los demas
        hd = np.full(M, a_rest)
        hd[0] = float(np.clip(adc + rng.normal(0.0, noise_std), 0.0, 1.0))
        h = field.step(hd, 1.0)

        for i in range(M):
            a_req = float(np.clip(h[i], 0.0, 1.0))
            Us = shifted_sequence(mompc, sols[i])
            Js = prob.costs(X[i], Us)
            Ja = float(np.array([1 - a_prev[i], a_prev[i]]) @ Js)
            adm = admissible_set(mompc, X[i], Ja, grid=grid)
            a_vec, blocked = adm.nearest(np.array([1 - a_req, a_req]))
            a_app = float(a_vec[1])
            field.sync(i, a_app, blocked)

            s = solve(mompc, X[i], a_vec)
            if s is None:
                return None
            u = float(s.U[0, 0])
            c0 = prob.objectives[0].stage(X[i], np.array([u]))
            c1 = prob.objectives[1].stage(X[i], np.array([u]))
            # coste NEUTRAL: ponderado por la demanda LIMPIA, la misma para todas
            # las zonas -- es lo que hace comparables a las de aguas abajo
            ST[t, i] = (1.0 - adc) * c0 + adc * c1
            AL[t, i] = a_app; XS[t, i] = X[i]; US[t, i] = u; BL[t, i] = blocked

            wd = np.zeros(plant.n)
            wd[-1] = dist_amp * np.sin(2 * np.pi * t / dist_period)
            wn = dist_std * rng.standard_normal(plant.n) if dist_std > 0 else 0.0
            X[i] = np.clip(Areal @ X[i] + plant.B.ravel() * u + wd + wn,
                           -plant.xmax, plant.xmax)
            a_prev[i] = a_app
            sols[i] = s

    return DistTrace(alpha=AL, demand=DEM, x=XS, u=US, stage=ST, blocked=BL)


# --------------------------------------------------------------------------
#                              METRICAS
# --------------------------------------------------------------------------

def transport_profile(tr: DistTrace, warmup: int = 20) -> dict:
    """Cuanta senal de la demanda llega a cada zona, y con cuanto retraso.

    amplitud_i = desv. tipica de alpha_i, normalizada por la de la zona 1
    retraso_i  = desfase (en muestras) que maximiza la correlacion cruzada con
                 la demanda
    """
    A = tr.alpha[warmup:]
    d = tr.demand[warmup:] - tr.demand[warmup:].mean()
    M = A.shape[1]
    amp = np.array([A[:, i].std() for i in range(M)])
    amp_rel = amp / amp[0] if amp[0] > 1e-12 else np.zeros(M)
    lag = np.zeros(M); corr = np.zeros(M)
    for i in range(M):
        a = A[:, i] - A[:, i].mean()
        if a.std() < 1e-12:
            lag[i] = np.nan; corr[i] = 0.0; continue
        best = (-2.0, 0)
        for L in range(0, 13):
            if L == 0:
                c = np.corrcoef(a, d)[0, 1]
            else:
                c = np.corrcoef(a[L:], d[:-L])[0, 1]
            if c > best[0]:
                best = (c, L)
        corr[i], lag[i] = best[0], best[1]
    return {"amplitud": amp, "amplitud_relativa": amp_rel,
            "retraso": lag, "correlacion": corr,
            "coste_por_zona": tr.stage[warmup:].sum(axis=0),
            "coste_aguas_abajo": float(tr.stage[warmup:, 1:].sum()),
            "bloqueos_por_zona": tr.blocked[warmup:].sum(axis=0)}
