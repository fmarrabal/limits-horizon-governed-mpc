"""Transporte: el unico uso del campo de 2o orden que cinco arenas no han matado.

LA PREGUNTA
-----------
Una perturbacion que se propaga FISICAMENTE aguas abajo por una cadena de
subsistemas da al nodo i informacion sobre el FUTURO del nodo i+1 que el
horizonte local de i+1 no puede obtener a ningun coste. Es informacion de
INSTANTE DE LLEGADA, no de amplitud -- lo cual la salva de la proposicion de
invariancia de escala (A15), que solo mata las demandas fabricadas modulando
amplitudes.

LA OBJECION ESTRUCTURAL QUE HAY QUE RESOLVER ANTES DE CONSTRUIR NADA
---------------------------------------------------------------------
Un MPC nominal EN REPOSO no hace nada, y NINGUN meta-parametro cambia eso:
con x = 0 y sin perturbacion en la prediccion, u* = 0 para todo alpha y para
todo N. Luego un aviso que llega mientras el nodo esta en reposo es
ESTRUCTURALMENTE INUTILIZABLE por el canal del meta-parametro.

Para que el aviso sirva hacen falta TRES cosas a la vez:

  1. EXCITACION DE FONDO permanente, para que el nodo nunca este en reposo y el
     meta-parametro tenga siempre efecto.
  2. UNA NO LINEALIDAD ACTIVA (saturacion del actuador), para que el estado de
     partida importe: llegar al evento con |x| pequeno compra tiempo que el
     actuador saturado ya no puede comprar.
  3. QUE LA ACCION PREVENTIVA CUESTE. Subir alpha (guardar margen) quema
     esfuerzo contra el ruido de fondo; si fuera gratis, el mejor peso CONSTANTE
     ya lo haria siempre y no habria nada que programar en el tiempo.

Con las tres, el optimo varia en el tiempo por el CUANDO, no por el CUANTO. Es
la unica forma conocida de esquivar A15 legitimamente.

EL TEST DE VIABILIDAD, QUE ES LO PRIMERO QUE SE EJECUTA
--------------------------------------------------------
Un gobernador CLARIVIDENTE que conoce el instante de llegada t_k y actua con
antelacion `lead` pasos. Se barre `lead` desde 0 (= reactivo, el limite que
alcanza cualquier detector local) hacia arriba:

  * si el coste es PLANO en `lead`, el aviso anticipado no compra NADA por el
    canal del meta-parametro, y la arena esta MUERTA sea cual sea el mecanismo
    de transporte: no hace falta implementar campo, ni linea de retardo, ni
    nada.
  * si el coste BAJA con `lead` hasta un `lead*` > 0, entonces existe valor en
    la anticipacion y la pregunta de COMO transportarla (onda contra difusion
    contra linea de retardo pura) pasa a ser legitima.

Este modulo implementa el test. Nada mas hasta que el test pase.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .plant import Plant


# --------------------------------------------------------------------------
#                    MPC PONDERADO CON SATURACION, CACHEADO
# --------------------------------------------------------------------------

class WeightedMPC:
    """min sum_{k<N} (1-a) u'Ru + a x'Qx  s.a. |u| <= umax.

    Sin coste terminal ni restricciones de estado: la factibilidad es trivial y
    la unica no linealidad del lazo es la CAJA DE ENTRADA, que es exactamente la
    que queremos que este activa. El QP condensado se precomputa para cada peso
    de la rejilla (G = (1-a) G_R + a G_Q es afin en a, pero OSQP necesita su
    propia factorizacion, asi que se cachea uno por peso).
    """

    def __init__(self, plant: Plant, Q: np.ndarray, R: np.ndarray, N: int,
                 alphas: Sequence[float]):
        from scipy import sparse
        import osqp
        self.plant, self.N = plant, int(N)
        self.Q = 0.5 * (np.atleast_2d(Q) + np.atleast_2d(Q).T)
        self.R = 0.5 * (np.atleast_2d(R) + np.atleast_2d(R).T)
        self.alphas = np.round(np.asarray(alphas, float), 6)
        n, m, A, B = plant.n, plant.m, plant.A, plant.B
        Sx = np.zeros((n * (N + 1), n))
        Su = np.zeros((n * (N + 1), m * N))
        Ak = np.eye(n)
        for i in range(N + 1):
            Sx[i * n:(i + 1) * n] = Ak
            for j in range(i):
                Su[i * n:(i + 1) * n, j * m:(j + 1) * m] = np.linalg.matrix_power(A, i - 1 - j) @ B
            Ak = A @ Ak
        Qb = np.zeros((n * (N + 1), n * (N + 1)))
        for k in range(N):
            Qb[k * n:(k + 1) * n, k * n:(k + 1) * n] = self.Q
        Rb = np.kron(np.eye(N), self.R)
        self._GQ = 2.0 * (Su.T @ Qb @ Su)
        self._GR = 2.0 * Rb
        self._FxQ = 2.0 * (Su.T @ Qb @ Sx)
        self._cache: Dict[float, dict] = {}
        umax = np.tile(plant.umax, N)
        Aid = sparse.identity(m * N, format="csc")
        for a in self.alphas:
            G = a * self._GQ + (1.0 - a) * self._GR
            G = 0.5 * (G + G.T)
            prob = osqp.OSQP()
            prob.setup(P=sparse.triu(sparse.csc_matrix(G), format="csc"),
                       q=np.zeros(m * N), A=Aid, l=-umax, u=umax,
                       verbose=False, eps_abs=1e-9, eps_rel=1e-9,
                       max_iter=20000, polish=False)
            self._cache[float(a)] = dict(prob=prob, Fx=a * self._FxQ)
        self.n_solves = 0

    def snap(self, a: float) -> float:
        """Peso mas cercano de la rejilla (el gobernador propone continuo)."""
        return float(self.alphas[int(np.argmin(np.abs(self.alphas - a)))])

    def u0(self, x: np.ndarray, a: float) -> float:
        c = self._cache[self.snap(a)]
        x = np.asarray(x, float).ravel()
        c["prob"].update(q=c["Fx"] @ x)
        res = c["prob"].solve()
        if res.info.status_val not in (1, 2):
            raise RuntimeError(f"OSQP fallo (a={a}): {res.info.status}")
        self.n_solves += 1
        return float(np.asarray(res.x, float)[0])


# --------------------------------------------------------------------------
#                        LA CADENA Y SU PERTURBACION
# --------------------------------------------------------------------------

def transport_plant(dt: float = 1.0, wp: float = 0.30, zp: float = 0.20,
                    umax: float = 1.0, xmax: float = 1e4) -> Plant:
    """Subsistema de la cadena: 2o orden estable, controlable, con actuador
    de autoridad LIMITADA (es la no linealidad que hace que el margen valga)."""
    from scipy.linalg import expm
    Ac = np.array([[0.0, 1.0], [-wp ** 2, -2 * zp * wp]])
    Bc = np.array([[0.0], [1.0]])
    Mm = np.zeros((3, 3))
    Mm[:2, :2] = Ac
    Mm[:2, 2:] = Bc
    E = expm(Mm * dt)
    return Plant(A=E[:2, :2], B=E[:2, 2:], xmax=np.array([xmax, xmax]),
                 umax=np.array([umax]), name="tramo")


@dataclass(frozen=True)
class Traffic:
    """Perturbacion = ruido de fondo permanente + eventos SOSTENIDOS que viajan.

    El fondo mantiene al nodo fuera del reposo (condicion 1). Los eventos son
    escalones de duracion W, no impulsos: un impulso ya ha ocurrido cuando se
    detecta y no deja nada que anticipar, mientras que un escalon sostenido
    castiga durante W pasos, que es donde el margen previo se cobra.
    """

    T: int = 900
    sigma_bg: float = 0.045      # ruido de fondo (condicion 1)
    amp: float = 0.95            # amplitud del evento
    width: int = 14              # duracion del evento
    gap_lo: int = 70             # separacion minima entre eventos
    gap_hi: int = 110
    seed: int = 0

    def build(self) -> Tuple[np.ndarray, List[int]]:
        rng = np.random.default_rng(self.seed + 5150)
        d = self.sigma_bg * rng.standard_normal(self.T)
        t, starts = 60, []
        while t + self.width + 40 < self.T:
            sgn = 1.0 if rng.random() < 0.5 else -1.0
            d[t:t + self.width] += sgn * self.amp
            starts.append(t)
            t += int(rng.integers(self.gap_lo, self.gap_hi + 1))
        return d, starts


def true_cost(x: np.ndarray, u: float, q: float = 1.0, r: float = 1.0) -> float:
    """Coste de evaluacion NEUTRAL: fijo, ajeno al peso que cada gobernador usa."""
    x = np.asarray(x, float).ravel()
    return float(q * x[0] ** 2 + r * float(u) ** 2)


# --------------------------------------------------------------------------
#                  LAZO CON PROGRAMA DE PESO ARBITRARIO
# --------------------------------------------------------------------------

def run_schedule(mpc: WeightedMPC, plant: Plant, d: np.ndarray,
                 a_seq: np.ndarray, warm: int = 60,
                 q: float = 1.0, r: float = 1.0) -> dict:
    """Ejecuta el lazo con el programa de peso dado y devuelve coste y
    diagnostico de saturacion (que hay que vigilar: sin saturacion activa la
    arena vuelve a caer bajo A15)."""
    T = len(d)
    a_seq = np.asarray(a_seq, float)
    if a_seq.ndim == 0:
        a_seq = np.full(T, float(a_seq))
    x = np.zeros(2)
    tot = 0.0
    nsat = 0
    umax = float(plant.umax[0])
    peak = 0.0
    for t in range(T):
        u = mpc.u0(x, a_seq[t])
        if abs(u) > 0.999 * umax:
            nsat += 1
        if t >= warm:
            tot += true_cost(x, u, q, r)
            peak = max(peak, abs(float(x[0])))
        w = np.zeros(2)
        w[1] = d[t]
        x = plant.A @ x + plant.B.ravel() * u + w
        if not np.all(np.isfinite(x)) or np.max(np.abs(x)) > 1e8:
            return {"coste": np.inf, "sat": 1.0, "pico": np.inf}
    return {"coste": tot, "sat": nsat / T, "pico": peak}


# --------------------------------------------------------------------------
#          LA VERSION CON ALMACENAMIENTO (la que puede usar el aviso)
# --------------------------------------------------------------------------
#
# POR QUE HIZO FALTA. El primer intento fallo por una razon estructural, no de
# sintonia: con una planta asintoticamente estable y regulacion al origen, el
# equilibrio en reposo es x = 0 PARA TODO peso y PARA TODO horizonte. Luego no
# hay nada que pre-posicionar, y un aviso anticipado es inutilizable por el canal
# del meta-parametro por mucho que se adelante.
#
#   PROPOSICION (inutilidad de la anticipacion sin almacenamiento). Si el punto
#   de operacion optimo en reposo NO depende del meta-parametro, entonces el
#   valor optimo del meta-parametro antes de un evento coincide con su valor
#   optimo en reposo, y la antelacion no compra nada.
#
# El remedio es un estado de ALMACENAMIENTO cuyo nivel deseado SI dependa del
# peso: entonces alpha fija el punto de operacion, y anticipar significa
# literalmente CARGAR antes de que llegue la demanda. Es el caso canonico
# (deposito pulmon en una linea de proceso, almacenamiento termico en un campo
# solar, estado de carga en una microrred) y el unico donde el aviso tiene por
# donde entrar.

def storage_plant(dt: float = 1.0, tau_act: float = 3.0, umax: float = 0.6,
                  bmax: float = 1e4) -> Plant:
    """x = [nivel b, caudal f]. El caudal tiene retardo de actuador tau_act, de
    modo que llegar a tiempo exige EMPEZAR ANTES: es lo que hace que la
    antelacion pueda valer algo.

        b+ = b + dt*f  (- dt*d)        f+ = f + (dt/tau)(u - f)
    """
    A = np.array([[1.0, dt], [0.0, 1.0 - dt / tau_act]])
    B = np.array([[0.0], [dt / tau_act]])
    return Plant(A=A, B=B, xmax=np.array([bmax, bmax]),
                 umax=np.array([umax]), name="deposito")


class SetpointMPC:
    """MPC ponderado con CONSIGNAS DISTINTAS por objetivo.

        J_i = sum_k (x_k - xr_i)' Q_i (x_k - xr_i) + u_k' R_i u_k

    y el peso alpha mezcla J_0 y J_1. Como las consignas difieren, alpha fija el
    PUNTO DE OPERACION -- que es justo lo que faltaba: aqui el meta-parametro si
    tiene efecto con el sistema en reposo, y por tanto el aviso anticipado tiene
    por donde entrar.
    """

    def __init__(self, plant: Plant, objs: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]],
                 N: int, alphas: Sequence[float]):
        from scipy import sparse
        import osqp
        if len(objs) != 2:
            raise ValueError("dos objetivos")
        self.plant, self.N = plant, int(N)
        self.alphas = np.round(np.asarray(alphas, float), 6)
        n, m, A, B = plant.n, plant.m, plant.A, plant.B
        Sx = np.zeros((n * (N + 1), n)); Su = np.zeros((n * (N + 1), m * N))
        Ak = np.eye(n)
        for i in range(N + 1):
            Sx[i * n:(i + 1) * n] = Ak
            for j in range(i):
                Su[i * n:(i + 1) * n, j * m:(j + 1) * m] = np.linalg.matrix_power(A, i - 1 - j) @ B
            Ak = A @ Ak
        self._Sx, self._Su = Sx, Su
        Qbs, Rbs, Xrs = [], [], []
        for (Qi, Ri, xri) in objs:
            Qb = np.zeros((n * (N + 1), n * (N + 1)))
            for k in range(N):
                Qb[k * n:(k + 1) * n, k * n:(k + 1) * n] = np.atleast_2d(Qi)
            Qbs.append(Qb)
            Rbs.append(np.kron(np.eye(N), np.atleast_2d(Ri)))
            Xrs.append(np.tile(np.asarray(xri, float).ravel(), N + 1))
        self._cache: Dict[float, dict] = {}
        umaxv = np.tile(plant.umax, N)
        Aid = sparse.identity(m * N, format="csc")
        for a in self.alphas:
            w = (1.0 - float(a), float(a))
            Qb = w[0] * Qbs[0] + w[1] * Qbs[1]
            Rb = w[0] * Rbs[0] + w[1] * Rbs[1]
            QX = w[0] * (Qbs[0] @ Xrs[0]) + w[1] * (Qbs[1] @ Xrs[1])
            G = 2.0 * (Su.T @ Qb @ Su + Rb)
            G = 0.5 * (G + G.T)
            prob = osqp.OSQP()
            prob.setup(P=sparse.triu(sparse.csc_matrix(G), format="csc"),
                       q=np.zeros(m * N), A=Aid, l=-umaxv, u=umaxv,
                       verbose=False, eps_abs=1e-9, eps_rel=1e-9,
                       max_iter=20000, polish=False)
            self._cache[float(a)] = dict(prob=prob, FxQ=2.0 * (Su.T @ Qb @ Sx),
                                         c=-2.0 * (Su.T @ QX))
        self.n_solves = 0

    def snap(self, a: float) -> float:
        return float(self.alphas[int(np.argmin(np.abs(self.alphas - a)))])

    def u0(self, x: np.ndarray, a: float) -> float:
        c = self._cache[self.snap(a)]
        x = np.asarray(x, float).ravel()
        c["prob"].update(q=c["FxQ"] @ x + c["c"])
        res = c["prob"].solve()
        if res.info.status_val not in (1, 2):
            raise RuntimeError(f"OSQP fallo (a={a}): {res.info.status}")
        self.n_solves += 1
        return float(np.asarray(res.x, float)[0])


def storage_cost(x: np.ndarray, u: float, b_econ: float, b_crit: float,
                 q: float = 1.0, r: float = 1.0, pen: float = 60.0) -> float:
    """Coste de evaluacion NEUTRAL, ajeno a todo peso:

        (b - b_econ)^2  +  r u^2  +  pen * max(0, b_crit - b)^2

    Mantener el deposito por encima de b_econ CUESTA (inmovilizado, perdidas),
    y quedarse por debajo de b_crit cuesta MUCHO (rotura de servicio). Esa
    asimetria es la que hace que cargar con antelacion sea una decision y no una
    obviedad.
    """
    b = float(np.asarray(x, float).ravel()[0])
    short = max(0.0, b_crit - b)
    return float(q * (b - b_econ) ** 2 + r * float(u) ** 2 + pen * short ** 2)


def run_storage(mpc: SetpointMPC, plant: Plant, d: np.ndarray, a_seq,
                b_econ: float, b_crit: float, warm: int = 60,
                r: float = 1.0, pen: float = 60.0) -> dict:
    """Lazo del deposito. La perturbacion d es un CONSUMO que descarga el nivel."""
    T = len(d)
    a_seq = np.asarray(a_seq, float)
    if a_seq.ndim == 0:
        a_seq = np.full(T, float(a_seq))
    x = np.array([b_econ, 0.0])
    tot = 0.0; nsat = 0; nshort = 0; bmin = np.inf
    umax = float(plant.umax[0])
    for t in range(T):
        u = mpc.u0(x, a_seq[t])
        if abs(u) > 0.999 * umax:
            nsat += 1
        if t >= warm:
            tot += storage_cost(x, u, b_econ, b_crit, r=r, pen=pen)
            if x[0] < b_crit:
                nshort += 1
            bmin = min(bmin, float(x[0]))
        w = np.array([-d[t], 0.0])
        x = plant.A @ x + plant.B.ravel() * u + w
        if not np.all(np.isfinite(x)) or np.max(np.abs(x)) > 1e8:
            return {"coste": np.inf, "sat": 1.0, "roturas": T, "bmin": -np.inf}
    return {"coste": tot, "sat": nsat / T, "roturas": nshort / max(T - warm, 1),
            "bmin": float(bmin)}


def is_causal(gate, T: int = 400, t0: int = 200, width: int = 16,
              nodes: Sequence[int] = (1, 2, 3), tol: float = 1e-9) -> bool:
    """¿El mecanismo usa solo informacion que ya existe?

    POR QUE ESTA FUNCION EXISTE. En la primera version del experimento de
    transporte el mapa h -> alpha admitia un DESPLAZAMIENTO libre, y la rejilla
    de sintonia incluia valores NEGATIVOS. Un desplazamiento negativo ADELANTA
    la senal: los dos mecanismos ganadores encendian su puerta hasta ocho pasos
    ANTES de que la fuente midiera el evento. Ganaban usando informacion que
    nadie tiene. Ningun barrido de sintonia puede incluir configuraciones asi.

    La prueba es una sonda: un unico pulso que la fuente mide en `t0`. Si la
    puerta de cualquier nodo se mueve antes de `t0`, la configuracion es acausal
    y queda descartada, igual que se descarta una integracion inestable.
    """
    src = np.zeros(T)
    src[t0:t0 + width] = 1.0
    for i in nodes:
        g = np.asarray(gate(src, T, i), float)
        base = g[:t0].min() if t0 > 0 else g[0]
        if np.any(g[:t0] > base + tol):
            return False
    return True


def clairvoyant_schedule(T: int, starts: Sequence[int], width: int,
                         a_lo: float, a_hi: float, lead: int,
                         tail: int = 0) -> np.ndarray:
    """Programa CLARIVIDENTE: peso alto desde `lead` pasos ANTES de cada evento
    hasta `tail` pasos despues de terminar. lead = 0 es el limite reactivo
    ideal (saber del evento en el instante exacto en que empieza), que es lo
    mejor que puede lograr cualquier detector local sin informacion aguas
    arriba. Todo lead > 0 requiere informacion que solo el vecino tiene."""
    a = np.full(T, float(a_lo))
    for s in starts:
        i0 = max(0, s - int(lead))
        i1 = min(T, s + int(width) + int(tail))
        a[i0:i1] = float(a_hi)
    return a
