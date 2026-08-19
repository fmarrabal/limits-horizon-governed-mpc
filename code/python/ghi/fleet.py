"""Caso de aplicacion: M lazos compartiendo UN presupuesto de computo.

POR QUE ESTE CASO Y NO OTRO
---------------------------
De los dos resultados positivos del programa, el del gobierno del horizonte es
el que NO tiene objecion de estado del arte: no depende de ninguna asimetria de
informacion (que la prealimentacion estandar destruiria), sino del COSTE DE
CALCULAR. Y su beneficio se multiplica en el montaje industrial mas comun de
todos: un solo controlador embebido que atiende a MUCHOS lazos.

EL ARGUMENTO, QUE ES DE MULTIPLEXACION ESTADISTICA
---------------------------------------------------
Con horizonte FIJO hay que dimensionar para el peor caso de CADA lazo a la vez:
el presupuesto necesario es M * N_max, aunque los lazos casi nunca esten todos
en su peor momento simultaneamente. Con horizonte GOBERNADO, cada lazo pide
horizonte largo solo cuando lo necesita, y como los eventos no coinciden, la
suma pedida es mucho menor casi siempre. Es la misma razon por la que una
central telefonica no necesita una linea por abonado.

El riesgo esta en la CORRELACION: si la perturbacion barre todos los lazos a la
vez -- un frente de nubes sobre un campo de colectores, una racha sobre un
invernadero -- los picos coinciden y la ganancia de multiplexacion se evapora.
Este modulo mide las dos cosas: la ganancia y su desaparicion con la
correlacion. Reportar solo la primera seria vender el resultado a medias.

QUE PASA CUANDO EL PRESUPUESTO NO LLEGA
----------------------------------------
Un presupuesto duro obliga a REPARTIR. Aqui el reparto es proporcional a la
demanda: los lazos que el gobernador declara dificiles conservan mas horizonte
que los que declara comodos. Ese es exactamente el trabajo para el que sirve un
gobernador -- no elegir el horizonte en abstracto, sino ORDENAR PRIORIDADES bajo
escasez -- y es donde se ve si su lectura de la dificultad vale algo.

NOTA DE HONESTIDAD: esto es una simulacion con parametros realistas de un lazo
termico (constantes de tiempo de minutos, muestreo de 30 s, eventos de nube),
no una campana de medidas sobre planta. Lo que se afirma es un mecanismo y su
magnitud en ese modelo, no un rendimiento medido en campo.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .plant import Plant
from .suboptimality import (ALPHA_CLIP, GovOrder1, GovThreshold, HorizonMPC,
                            alpha_observed)


# --------------------------------------------------------------------------
#                        EL LAZO TERMICO
# --------------------------------------------------------------------------

def thermal_loop(dt: float = 30.0, tau_f: float = 120.0, tau_m: float = 300.0,
                 k_fm: float = 1.2, umax: float = 0.6) -> Plant:
    """Lazo de colector solar simplificado: fluido rapido acoplado a metal lento.

        T_f' = (-T_f + k_fm (T_m - T_f) + u) / tau_f
        T_m' = ( k_fm (T_f - T_m) ) / tau_m

    tau_f ~ 2 min (tiempo de residencia del fluido) y tau_m ~ 5 min (inercia del
    absorbedor) son ordenes de magnitud habituales en tubo absorbedor de campos
    cilindro parabolicos. La entrada u es el caudal normalizado, saturado.

    LO QUE HACE QUE EL HORIZONTE IMPORTE, y sin lo cual este banco estaria vacio:
    lo que se penaliza es la temperatura del ABSORBEDOR (segundo estado), a la
    que el caudal solo llega A TRAVES del fluido. Ese retardo entre accion y
    efecto es lo unico que hace que mirar mas lejos sirva de algo: si se penaliza
    el estado sobre el que la entrada actua directamente, N = 2 ya ve todo el
    efecto y ningun horizonte mayor compra nada (medido: razon N2/N16 = 1.01).
    """
    from scipy.linalg import expm
    Ac = np.array([[-(1.0 + k_fm) / tau_f, k_fm / tau_f],
                   [k_fm / tau_m, -k_fm / tau_m]])
    Bc = np.array([[1.0 / tau_f], [0.0]])
    Mm = np.zeros((3, 3))
    Mm[:2, :2] = Ac
    Mm[:2, 2:] = Bc
    E = expm(Mm * dt)
    return Plant(A=E[:2, :2], B=E[:2, 2:], xmax=np.array([1e4, 1e4]),
                 umax=np.array([umax]), name="lazo-termico")


@dataclass(frozen=True)
class CloudField:
    """Nubes sobre M lazos: una parte COMUN (el frente que barre el campo) y
    una parte INDEPENDIENTE (nubes sueltas sobre cada lazo).

    `rho` es la fraccion de varianza que aporta el frente comun. rho = 0 son
    lazos independientes; rho = 1 es un frente que golpea a todos a la vez. Es
    el unico mando que decide si la multiplexacion funciona.
    """

    M: int = 8
    T: int = 720                 # 6 h a 30 s
    rho: float = 0.0             # correlacion entre lazos
    rate: float = 0.012          # prob. de que empiece un evento por paso y lazo
    depth: float = 0.85          # profundidad de la caida de irradiancia
    dur_lo: int = 8
    dur_hi: int = 26
    sigma_bg: float = 0.012
    seed: int = 0

    def build(self) -> np.ndarray:
        rng = np.random.default_rng(self.seed + 31337)
        D = self.sigma_bg * rng.standard_normal((self.T, self.M))

        def eventos(gen, cols):
            t = 0
            while t < self.T:
                if gen.random() < self.rate:
                    dur = int(gen.integers(self.dur_lo, self.dur_hi + 1))
                    amp = self.depth * (0.5 + 0.5 * gen.random())
                    D[t:min(self.T, t + dur), cols] -= amp
                    t += dur
                else:
                    t += 1

        # los eventos comunes se sortean UNA vez para todos los lazos
        n_com = int(round(self.rho * self.M))
        if self.rho > 0:
            gc = np.random.default_rng(self.seed + 90210)
            eventos(gc, slice(None))
        for i in range(self.M):
            if i < n_com:
                continue                        # ya cubierto por el frente comun
            eventos(np.random.default_rng(self.seed * 1000 + i), i)
        return D


# --------------------------------------------------------------------------
#                    REPARTO DEL PRESUPUESTO
# --------------------------------------------------------------------------

def allocate(requests: Sequence[int], budget: int, N_min: int,
             N_max: int) -> List[int]:
    """Reparte un presupuesto DURO de horizonte entre los lazos.

    Si la suma pedida cabe, se concede tal cual. Si no, se recorta de forma
    proporcional a la demanda por encima del minimo, de modo que quien el
    gobernador declara dificil conserva mas horizonte. Se garantiza N_min a
    todos: un lazo sin horizonte no es un lazo con menos calidad, es un lazo
    sin control.
    """
    req = np.asarray(requests, int)
    M = len(req)
    base = N_min * M
    if budget <= base:
        return [N_min] * M
    if req.sum() <= budget:
        return req.tolist()
    extra = req - N_min
    tot = extra.sum()
    if tot <= 0:
        return [N_min] * M
    share = (budget - base) * extra / tot
    give = np.floor(share).astype(int)
    # el resto se reparte a los que mas fraccion perdieron, para no desperdiciar
    resto = (budget - base) - give.sum()
    if resto > 0:
        orden = np.argsort(-(share - give))
        give[orden[:int(resto)]] += 1
    return np.clip(N_min + give, N_min, N_max).tolist()


# --------------------------------------------------------------------------
#                            EL LAZO DE LA FLOTA
# --------------------------------------------------------------------------

@dataclass
class FleetResult:
    coste: float
    computo_medio: float
    computo_p95: float
    frac_recortado: float           # pasos en que el presupuesto ATO
    N_medio: float
    divergio: bool


def make_gov(kind: str, N_min: int, N_max: int, N0: Optional[int] = None,
             tau: float = 8.0, a_ref: float = 0.3):
    """El gobernador se sintoniza PARA ESTA ARENA, igual que se barre N para los
    horizontes fijos. Heredar la constante de otra arena seria compararlo en
    desventaja: tau y a_ref se ajustan en semillas disjuntas de las de
    evaluacion (ver `run_fleet.py`)."""
    if kind == "fijo":
        return ("fijo", int(N0))
    if kind == "fugas":
        return ("fugas", GovOrder1(N_min, N_max, tau=tau, a_ref=a_ref))
    if kind == "umbral":
        return ("umbral", GovThreshold(N_min, N_max, up=4, dn=1))
    raise ValueError(kind)


def run_fleet(plant: Plant, Q: np.ndarray, R: np.ndarray, D: np.ndarray,
              gov_kind: str, budget: Optional[int], N_min: int = 2,
              N_max: int = 16, N_fixed: int = 16, warm: int = 60,
              reparto_igual: bool = False, tau: float = 8.0,
              a_ref: float = 0.3) -> FleetResult:
    """Ejecuta M lazos con un presupuesto de computo COMPARTIDO por paso.

    El computo se contabiliza como suma de horizontes concedidos. El instrumento
    que mide alpha_N se reutiliza del propio solve del lazo, de modo que el
    gobernador no se cobra aparte lo que ya se ha pagado.
    """
    T, M = D.shape
    mpc = HorizonMPC(plant, Q=Q, R=R, N_min=N_min, N_max=N_max)
    govs = [make_gov(gov_kind, N_min, N_max, N_fixed, tau, a_ref)
            for _ in range(M)]
    fijo = gov_kind == "fijo"
    X = [np.zeros(plant.n) for _ in range(M)]
    peticion = [N_fixed if fijo else N_min] * M
    # solve del INSTRUMENTO ya pagado en el paso anterior: (N, V). Si el
    # horizonte concedido coincide, se reutiliza y no se vuelve a cobrar
    cache = [None] * M
    tot, comp, recortes, Ns = 0.0, [], 0, []
    for t in range(T):
        if budget is None:
            Ng = list(peticion)
        elif reparto_igual:
            # NULO DE REPARTO (corregido el 19-ago-2026). El brazo igualitario
            # debe aislar UNA cosa: el reparto entre lazos. Por eso conserva al
            # gobernador y su adaptacion TEMPORAL -- gasta en cada paso el mismo
            # total que gastaria el reparto proporcional -- y solo cambia COMO
            # se distribuye ese total: a partes iguales, ignorando que lazo esta
            # en apuros. El resto de la division se rota con t para no favorecer
            # sistematicamente a ningun lazo.
            #
            # La version anterior fijaba q = budget // M en todos los pasos, con
            # lo que destruia tambien la adaptacion temporal y colapsaba EXACTO
            # al horizonte fijo N = budget/M (coincidencia digito a digito con
            # la frontera fija): no descomponia nada. Ese caso sigue disponible
            # como gov_kind='fijo' con N=budget//M.
            ref = allocate(peticion, budget, N_min, N_max)
            total = int(sum(ref))
            base, resto = divmod(total - N_min * M, M)
            Ng = [N_min + base] * M
            for k in range(resto):
                Ng[(t + k) % M] += 1
            Ng = [int(min(N_max, max(N_min, g))) for g in Ng]
        else:
            Ng = allocate(peticion, budget, N_min, N_max)
        if budget is not None and sum(peticion) > budget:
            recortes += 1
        paso = 0
        for i in range(M):
            N = Ng[i]
            # CONTABILIDAD: el solve del INSTRUMENTO del paso anterior ya evaluo
            # V_N(x_t) a horizonte N. Si el horizonte concedido no ha cambiado,
            # ese mismo QP sirve de solve de control y no se vuelve a cobrar; si
            # ha cambiado, hay que rehacerlo y se cobra. Asi, con horizonte
            # estable el gobernador cuesta lo mismo que un horizonte fijo.
            if cache[i] is not None and cache[i][0] == N:
                s = cache[i][1]
            else:
                s = mpc.solve(X[i], N)
                paso += N
            u = s.u0
            st = mpc.stage(X[i], u)
            if t >= warm:
                tot += st
            w = np.zeros(plant.n)
            w[0] = D[t, i]
            Xn = plant.A @ X[i] + plant.B.ravel() * u + w
            if fijo:
                cache[i] = None
            else:
                s2 = mpc.solve(Xn, N)          # instrumento: se cobra siempre
                paso += N
                cache[i] = (N, s2)             # y queda disponible para t+1
                a = alpha_observed(s.V, s2.V, st)
                peticion[i] = govs[i][1].update(
                    None if a is None else float(np.clip(a, -ALPHA_CLIP, ALPHA_CLIP)))
            X[i] = Xn
            Ns.append(N)
        comp.append(paso)
        big = np.concatenate(X)
        if not np.all(np.isfinite(big)) or np.max(np.abs(big)) > 1e8:
            return FleetResult(np.inf, float(np.mean(comp)),
                               float(np.percentile(comp, 95)),
                               recortes / max(t + 1, 1), float(np.mean(Ns)), True)
    return FleetResult(tot, float(np.mean(comp)), float(np.percentile(comp, 95)),
                       recortes / T, float(np.mean(Ns)), False)
