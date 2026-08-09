"""alpha_N de Grune-Pannek como senal de interocepcion, y gobernadores de horizonte.

POR QUE ESTE MODULO, Y POR QUE AHORA
------------------------------------
El principio de diseno que dejaron los experimentos de agosto es:

    el campo debe modular LO QUE EL HORIZONTE DEL MPC NO PUEDE VER.

El precio de la energia fallo ese criterio (el horizonte ya lo arbitra: el
oraculo miope salio un 17% peor que un peso constante). El indice de
suboptimalidad alpha_N lo cumple por definicion: mide la discrepancia entre el
paisaje de valor de horizonte finito y la realidad del lazo cerrado, y solo se
revela AL CERRAR EL LAZO. El controlador no puede leerlo de su propio coste en
el momento de resolver.

EL MARCO (Lincoln-Rantzer 2006; Grune-Rantzer 2008, IEEE TAC 53(9); Grune 2009,
SIAM J. Control Optim. 48(2); Grune & Pannek, libro 2a ed., caps. 4 y 6; el
horizonte adaptativo por umbrales esta en el cap. 7 de la 1a ed. / cap. 10 de
la 2a)
-------------------------------------------------------------------------------
MPC SIN ingredientes terminales:  V_N(x) = min_U sum_{k=0}^{N-1} l(x_k, u_k).
La desigualdad de programacion dinamica relajada dice: si

    V_N(x_t) - V_N(x_{t+1})  >=  alpha * l(x_t, u_t)     con alpha > 0 UNIFORME,

entonces la trayectoria converge con cota de rendimiento

    J_infinito(x, mu_N)  <=  V_N(x) / alpha

(estabilidad asintotica en sentido Lyapunov si la desigualdad vale en un
entorno, no solo a lo largo de una trayectoria).

El ALPHA OBSERVADO a posteriori es directamente medible en linea:

    alpha_t = [ V_{N_t}(x_t) - V_{N_t}(x_{t+1}) ] / l(x_t, u_t)

con un paso de retardo (V_{N_t}(x_{t+1}) se conoce en t+1).

POR QUE AQUI NO VALE EL BANCO CON INGREDIENTES TERMINALES
---------------------------------------------------------
Con (P_i, K_f, Omega) correctos -- nuestro banco tiene S_i = 0 exacto -- el
decrecimiento V(x+) <= V(x) - l se cumple EN NOMINAL Y DENTRO DE LA REGION
FACTIBLE, luego alpha_t >= 1 ahi: la senal solo informaria de perturbaciones y
desajuste, no de la suficiencia del horizonte. Sin coste terminal alpha es
informativo YA EN NOMINAL -- ese es el argumento correcto, no que con terminal
nunca informe. Ademas el regimen sin terminal es el del MPC industrial. Por eso
este modulo define su propio MPC, sin terminal y sin restricciones de estado
duras (solo caja de entrada): la factibilidad es trivial para todo N >= 2 y
toda la dificultad queda en el diagnostico de rendimiento.

LA DUALIDAD DE MONOTONIA (la observacion estructural que motiva el experimento)
-------------------------------------------------------------------------------
Sin coste terminal, l >= 0 y factibilidad anidada (trivial con solo caja de
entrada):                             V_{N+1}(x) >= V_N(x)   (truncamiento)
Con coste terminal que es CLF LOCAL en una region terminal invariante y ley
K_f admisible, y para x factible:     V_{N+1}(x) <= V_N(x)   (extension por K_f)

Los dos marcos tienen direcciones seguras OPUESTAS al variar N:

  * marco de secuencia desplazada (terminal): SUBIR N es gratis, bajar exige
    x en X_{N-1}. Es el marco donde la modulacion de N quedo sin teorema.
  * marco alpha_N (sin terminal): con W_t := V_{N_t}(x_t), BAJAR N nunca sube
    W (porque V_{N'} <= V_N si N' <= N), y de la definicion de alpha_t sale la
    IDENTIDAD CONTABLE  W_{t+1} <= W_t - alpha_t l_t  cuando N no sube. Subir
    N da un salto MULTIPLICATIVO, W_{t+1} <= gamma_max (W_t - alpha_t l_t),
    valido solo sobre un conjunto de niveles compacto via controlabilidad de
    coste -- y sin un alpha_bar > 0 UNIFORME no hay contraccion entre saltos.

SOBRE LO QUE ESTO ES Y LO QUE NO ES (correccion tras revision adversarial):
alpha_t es una cantidad MEDIDA a posteriori, no garantizada por nada; puede
ser negativa. La identidad telescopica de arriba es una CONTABILIDAD
verificable, no un certificado a priori: un "certificado" que no puede fallar
no certifica. La cota real J_inf <= V_N/alpha (Lincoln & Rantzer 2006;
Grune & Rantzer 2008) exige alpha UNIFORME; el alpha(N) constructivo a priori
es el de Grune 2009 y este modulo NO lo computa. Lo que este modulo hace es
usar alpha_t como DIAGNOSTICO en linea y como SENAL de interocepcion.
Analogamente, el paralelo con el average dwell-time sobre los aumentos de N es
solo eso, un paralelo: el campo inercial NO limita la tasa de aumentos (con
demanda en escalon 2->16 el Verlet sube 2->5->11->16 en tres pasos, MAS rapido
que el umbral +2 de la literatura); lo que limita es la frecuencia de
REVERSIONES. Ambas correcciones provienen de la revision adversarial y estan
verificadas numericamente.

DECISIONES DE INGENIERIA DOCUMENTADAS
-------------------------------------
* N_min = 2, no 1: con l = x'Qx + Ru^2 y R > 0, el MPC de N = 1 minimiza
  R u^2 en solitario y devuelve u* = 0 SIEMPRE (no controla nada). N = 1 es
  degenerado, no "barato".
* alpha_t solo se evalua si l_t >= L_MIN: cerca del equilibrio es un cociente
  0/0. Sin informacion se toma alpha := 1.0 ("comodo"), que empuja la demanda
  a N_min: planta asentada => computo minimo.
* El alpha que ve el GOBERNADOR se recorta a [-2, 2] (un solo paso de kick
  produce valores absurdamente negativos); el alpha CRUDO se registra aparte
  para las metricas.
* En los pasos con kick exogeno, alpha_t < 0 no es culpa del horizonte sino de
  la perturbacion. Para las metricas de salud del certificado esos pasos se
  separan; para el gobernador se dejan pasar (DEBE reaccionar a ellos).
* Contabilidad de computo: se cargan TODOS los QP resueltos (sum de N por
  solve), instrumento incluido, igual para todos los gobernadores. Cuando el
  horizonte no cambia, el solve del instrumento ES el solve principal y se
  carga una sola vez.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .plant import Plant
from .regulators import match_first_order

L_MIN = 1e-4          # umbral de informatividad de alpha
V_JUMP = 1.0          # salto de V que delata un evento aunque l sea ~0
ALPHA_CLIP = 2.0      # recorte del alpha que ve el gobernador


# --------------------------------------------------------------------------
#                 MPC SIN TERMINAL, PARAMETRICO EN EL HORIZONTE
# --------------------------------------------------------------------------

@dataclass
class SolN:
    U: np.ndarray
    V: float
    u0: float
    N: int


class HorizonMPC:
    """V_N(x) = min sum_{k=0}^{N-1} l(x_k,u_k)  s.a. |u| <= umax, para todo
    N en [N_min, N_max]. Sin coste terminal, sin region terminal, sin
    restricciones de estado: la factibilidad es trivial y alpha_N es la unica
    nocion de certificado en juego.

    Todo (matrices de prediccion, QP condensado, solver OSQP) se precomputa y
    cachea POR HORIZONTE: el lazo solo actualiza el termino lineal.
    """

    def __init__(self, plant: Plant, Q: np.ndarray, R: np.ndarray,
                 N_min: int = 2, N_max: int = 16):
        if N_min < 2:
            raise ValueError("N_min >= 2: con N=1 y R>0 el optimo es u*=0 siempre")
        self.plant, self.N_min, self.N_max = plant, N_min, N_max
        self.Q = 0.5 * (np.atleast_2d(Q) + np.atleast_2d(Q).T)
        self.R = 0.5 * (np.atleast_2d(R) + np.atleast_2d(R).T)
        self.n, self.m = plant.n, plant.m
        self._cache: Dict[int, dict] = {}
        for N in range(N_min, N_max + 1):
            self._cache[N] = self._build(N)
        self.n_solves = 0
        self.compute = 0            # suma de N sobre todos los solves

    def _build(self, N: int) -> dict:
        from scipy import sparse
        import osqp
        n, m = self.n, self.m
        A, B = self.plant.A, self.plant.B
        Sx = np.zeros((n * (N + 1), n))
        Su = np.zeros((n * (N + 1), m * N))
        Ak = np.eye(n)
        for i in range(N + 1):
            Sx[i*n:(i+1)*n] = Ak
            for j in range(i):
                Su[i*n:(i+1)*n, j*m:(j+1)*m] = np.linalg.matrix_power(A, i-1-j) @ B
            Ak = A @ Ak
        # etapas 0..N-1 con Q; el bloque N queda a CERO (sin coste terminal)
        Qb = np.zeros((n * (N + 1), n * (N + 1)))
        for k in range(N):
            Qb[k*n:(k+1)*n, k*n:(k+1)*n] = self.Q
        Rb = np.kron(np.eye(N), self.R)
        G = 2.0 * (Su.T @ Qb @ Su + Rb)
        G = 0.5 * (G + G.T)
        Fx = 2.0 * (Su.T @ Qb @ Sx)          # g = Fx @ x0
        C0 = Sx.T @ Qb @ Sx                   # termino constante: x0' C0 x0
        P = sparse.triu(sparse.csc_matrix(G), format="csc")
        Aid = sparse.identity(m * N, format="csc")
        umax = np.tile(self.plant.umax, N)
        prob = osqp.OSQP()
        prob.setup(P=P, q=np.zeros(m * N), A=Aid, l=-umax, u=umax,
                   verbose=False, eps_abs=1e-9, eps_rel=1e-9,
                   max_iter=20000, polish=False)
        return dict(Sx=Sx, Su=Su, G=G, Fx=Fx, C0=C0, prob=prob, N=N)

    def solve(self, x: np.ndarray, N: int) -> SolN:
        N = int(np.clip(N, self.N_min, self.N_max))
        c = self._cache[N]
        x = np.asarray(x, float).ravel()
        g = c["Fx"] @ x
        c["prob"].update(q=g)
        res = c["prob"].solve()
        if res.info.status_val not in (1, 2):
            raise RuntimeError(f"OSQP fallo con N={N}: {res.info.status}")
        z = np.asarray(res.x, float)
        V = 0.5 * z @ c["G"] @ z + g @ z + x @ c["C0"] @ x
        self.n_solves += 1
        self.compute += N
        return SolN(U=z.reshape(N, self.m).T, V=float(max(V, 0.0)),
                    u0=float(z[0]), N=N)

    def stage(self, x: np.ndarray, u: float) -> float:
        x = np.asarray(x, float).ravel()
        return float(x @ self.Q @ x + self.R[0, 0] * u * u)


def alpha_observed(V_t: float, V_next: float, stage_t: float) -> Optional[float]:
    """alpha_t = [V_N(x_t) - V_N(x_{t+1})] / l(x_t,u_t).

    GATE BILATERAL (arreglo de la revision adversarial). El gate original
    (None si l < L_MIN) neutralizaba exactamente la muestra mas informativa:
    tras un kick sobre planta asentada, el DENOMINADOR es el coste de etapa del
    estado pre-kick (~1e-20), asi que alpha salia None, el mapa lo leia como
    "comodo" y demandaba N_min JUSTO despues del kick -- el paso [2,2,16,2,...]
    de las trazas. Un evento con l ~ 0 pero |dV| grande ES informativo: el
    salto de V delata el kick. Por eso:

        l >= L_MIN           -> cociente normal
        l <  L_MIN, |dV| grande -> cociente con denominador saturado a L_MIN
                                   (magnitud enorme, signo correcto)
        l <  L_MIN, |dV| ~ 0    -> None (asentada de verdad, sin informacion)
    """
    dV = V_t - V_next
    if stage_t >= L_MIN:
        return dV / stage_t
    if abs(dV) >= V_JUMP:
        return dV / L_MIN
    return None


# --------------------------------------------------------------------------
#                       GOBERNADORES DE HORIZONTE
# --------------------------------------------------------------------------

def demand_from_alpha(alpha: Optional[float], N_min: int, N_max: int,
                      a_ref: float = 0.6, a_floor: float = 0.0) -> float:
    """Mapa estatico deficit-de-alpha -> horizonte deseado.

        alpha >= a_ref  (comodo)          ->  N_min
        alpha <= a_floor (certificado roto) ->  N_max
        entre medias, interpolacion lineal.

    alpha = None (sin informacion, planta asentada) se trata como alpha = 1:
    asentada => computo minimo. Este mapa es el analogo EXACTO del alpha_d
    estatico del TFM: los gobernadores de orden 0/1/2 lo comparten y solo
    difieren en la dinamica con que lo siguen.
    """
    a = 1.0 if alpha is None else float(np.clip(alpha, -ALPHA_CLIP, ALPHA_CLIP))
    frac = float(np.clip((a_ref - a) / (a_ref - a_floor), 0.0, 1.0))
    return N_min + (N_max - N_min) * frac


class Governor:
    """Interfaz: update(alpha) -> N entero en [N_min, N_max]."""

    name = "base"

    def __init__(self, N_min: int, N_max: int, N0: Optional[int] = None):
        self.N_min, self.N_max = N_min, N_max
        self.N = int(N0 if N0 is not None else N_max)

    def update(self, alpha: Optional[float]) -> int:
        raise NotImplementedError

    def reset(self) -> None:
        pass


class GovFixed(Governor):
    """Horizonte constante. Con N0 = N_max es la referencia conservadora."""

    def __init__(self, N_min, N_max, N0):
        super().__init__(N_min, N_max, N0)
        self.name = f"fijo-{self.N}"

    def update(self, alpha):
        return self.N


class GovThreshold(Governor):
    """La regla de la literatura de horizonte adaptativo: umbrales con
    histeresis e incrementos suaves (sube rapido, baja despacio).

        alpha < a_lo  ->  N += up
        alpha > a_hi  ->  N -= dn

    Es orden 0 CON histeresis: sin estado continuo, sin inercia. Es el baseline
    que un revisor exigira (Grune & Pannek, cap. 7; la practica estandar).
    """

    name = "umbral"

    def __init__(self, N_min, N_max, a_lo=0.3, a_hi=0.8, up=2, dn=1, N0=None):
        super().__init__(N_min, N_max, N0 if N0 is not None else N_min)
        self.a_lo, self.a_hi, self.up, self.dn = a_lo, a_hi, up, dn
        self._N0 = self.N

    def update(self, alpha):
        a = 1.0 if alpha is None else float(np.clip(alpha, -ALPHA_CLIP, ALPHA_CLIP))
        if a < self.a_lo:
            self.N = min(self.N + self.up, self.N_max)
        elif a > self.a_hi:
            self.N = max(self.N - self.dn, self.N_min)
        return self.N

    def reset(self):
        self.N = self._N0


class GovMap(Governor):
    """Orden 0 puro: N = round(demanda(alpha)). El analogo del alpha_d del TFM,
    sin memoria de ninguna clase."""

    name = "orden-0"

    def __init__(self, N_min, N_max, a_ref=0.6, a_floor=0.0):
        super().__init__(N_min, N_max, N_min)
        self.a_ref, self.a_floor = a_ref, a_floor

    def update(self, alpha):
        nd = demand_from_alpha(alpha, self.N_min, self.N_max, self.a_ref, self.a_floor)
        self.N = int(round(nd))
        return self.N


class GovOrder1(Governor):
    """Filtro de primer orden sobre la demanda. tau se iguala EN DISCRETO a la
    atenuacion en Nyquist del gobernador de segundo orden (misma disciplina que
    en el banco de pesos: sin igualacion la comparacion no dice nada)."""

    name = "orden-1"

    def __init__(self, N_min, N_max, tau, a_ref=0.6, a_floor=0.0):
        super().__init__(N_min, N_max, N_min)
        self.tau, self.a_ref, self.a_floor = float(tau), a_ref, a_floor
        self.x = float(N_min)

    def update(self, alpha):
        nd = demand_from_alpha(alpha, self.N_min, self.N_max, self.a_ref, self.a_floor)
        self.x += (nd - self.x) / self.tau
        self.x = float(np.clip(self.x, self.N_min, self.N_max))
        self.N = int(round(self.x))
        return self.N

    def reset(self):
        self.x = float(self.N_min)
        self.N = self.N_min


class GovOrder2(Governor):
    """El campo homeostatico (un nodo) sobre el horizonte: Verlet de velocidad,

        n'' + 2 zeta w0 n' + w0^2 (n - n_d(alpha)) = 0

    con anti-windup por theta SOLO cuando el recorte a [N_min, N_max] actua
    (leccion del fallo 1 del banco de pesos: sync jamas se dispara sin recorte;
    el redondeo a entero NO es un recorte y no toca el estado)."""

    name = "orden-2"

    def __init__(self, N_min, N_max, w0=0.7, zeta=0.5, theta=0.5,
                 a_ref=0.6, a_floor=0.0):
        super().__init__(N_min, N_max, N_min)
        self.w0, self.zeta, self.theta = float(w0), float(zeta), float(theta)
        self.a_ref, self.a_floor = a_ref, a_floor
        self.x, self.v = float(N_min), 0.0

    def _acc(self, x, v, nd):
        return -2 * self.zeta * self.w0 * v - self.w0 ** 2 * (x - nd)

    def update(self, alpha):
        nd = demand_from_alpha(alpha, self.N_min, self.N_max, self.a_ref, self.a_floor)
        acc = self._acc(self.x, self.v, nd)
        self.x += self.v + 0.5 * acc
        vh = self.v + 0.5 * acc
        self.v = vh + 0.5 * self._acc(self.x, vh, nd)
        lo, hi = float(self.N_min), float(self.N_max)
        if self.x < lo or self.x > hi:          # recorte real -> anti-windup
            self.x = float(np.clip(self.x, lo, hi))
            self.v *= self.theta
        self.N = int(round(self.x))
        return self.N

    def reset(self):
        self.x, self.v = float(self.N_min), 0.0
        self.N = self.N_min


def matched_governors(N_min: int, N_max: int, w0: float = 0.7, zeta: float = 0.5,
                      theta: float = 0.5, a_ref: float = 0.6) -> Dict[str, Callable[[], Governor]]:
    """La bateria estandar, ampliada tras la revision adversarial.

    * orden-1 aparece con DOS igualaciones y hay que reportar ambas: en un lazo
      sin ruido de medida la igualacion en Nyquist no fija nada relevante (solo
      elige un tau largo), y la revision demostro que el resultado del 1er
      orden se INVIERTE al cambiarla (tau=14.7 -> gana; tau=2.9 -> corr -0.1).
      Con ruido, Nyquist recupera su sentido.
        - orden-1-nyq: tau igualado en Nyquist al 2o orden (14.73 con w0=0.7)
        - orden-1-set: tau = 1/(zeta*w0) (igualacion por tiempo de asentamiento)
    * umbral-up4: el umbral de la literatura con subida agresiva up=4. El up=2
      por defecto estaba infra-sintonizado justo donde se media H2 (con up=4
      bate al campo en el escenario D); sin este baseline la comparacion seria
      un hombre de paja.
    * orden-2-z1: el campo criticamente amortiguado (sin timbre), para separar
      "segundo orden" de "esta sintonia".
    """
    tau_nyq, _g = match_first_order(w0, zeta, 1.0)
    tau_set = 1.0 / (zeta * w0)
    return {
        f"fijo-{N_max}": lambda: GovFixed(N_min, N_max, N_max),
        "umbral": lambda: GovThreshold(N_min, N_max),
        "umbral-up4": lambda: GovThreshold(N_min, N_max, up=4),
        "orden-0": lambda: GovMap(N_min, N_max, a_ref),
        "orden-1-nyq": lambda: GovOrder1(N_min, N_max, tau_nyq, a_ref),
        "orden-1-set": lambda: GovOrder1(N_min, N_max, tau_set, a_ref),
        "orden-2": lambda: GovOrder2(N_min, N_max, w0, zeta, theta, a_ref),
        "orden-2-z1": lambda: GovOrder2(N_min, N_max, w0, 1.0, theta, a_ref),
    }


# --------------------------------------------------------------------------
#                            EL LAZO CERRADO
# --------------------------------------------------------------------------

@dataclass
class AlphaTrace:
    N: np.ndarray             # horizonte aplicado
    alpha_raw: np.ndarray     # alpha observado (nan si no informativo)
    x: np.ndarray
    u: np.ndarray
    stage: np.ndarray         # coste de etapa REAL del lazo
    kick_step: np.ndarray     # True en los pasos donde entro un kick
    compute: int              # suma de N sobre TODOS los solves (instrumento incluido)
    n_solves: int
    n_extra: int              # solves extra del instrumento (cambio de N)
    diverged: bool


def make_kicks(T: int, seed: int, size_lo: float, size_hi: float,
               first: int = 30, gap_lo: int = 35, gap_hi: int = 45
               ) -> Dict[int, float]:
    """Kicks de velocidad en instantes cuasi-periodicos, magnitud y signo
    aleatorios. La DIFICULTAD de cada evento es |kick|: es el analogo del K del
    HBP, y la metrica central es corr(dificultad, computo asignado)."""
    rng = np.random.default_rng(seed)
    kicks: Dict[int, float] = {}
    t = first
    while t < T - 20:
        s = rng.uniform(size_lo, size_hi) * (1 if rng.random() < 0.5 else -1)
        kicks[t] = float(s)
        t += int(rng.integers(gap_lo, gap_hi + 1))
    return kicks


def closed_loop(mpc: HorizonMPC, gov: Governor, T: int, kicks: Dict[int, float],
                x0: Sequence[float] = (0.0, 0.0),
                A_real: Optional[np.ndarray] = None,
                x_div: float = 50.0) -> AlphaTrace:
    """El lazo con la contabilidad honesta del instrumento.

    En cada paso t:
      1. instrumento: V_{N_prev}(x_t), para cerrar alpha_{t-1}. Si el gobernador
         mantiene N, este solve ES el principal y se carga una vez.
      2. el gobernador decide N_t a la vista de alpha_{t-1}
      3. solve principal (si N cambio), aplicar u_0, avanzar la planta
    """
    plant = mpc.plant
    A_r = plant.A if A_real is None else np.asarray(A_real, float)
    x = np.asarray(x0, float).copy()
    gov.reset()

    Ns, als, xs, us, sts, kks = [], [], [], [], [], []
    mpc.n_solves = 0
    mpc.compute = 0
    n_extra = 0
    V_prev: Optional[float] = None
    N_prev: Optional[int] = None
    l_prev: Optional[float] = None
    diverged = False

    for t in range(T):
        alpha: Optional[float] = None
        sol_old: Optional[SolN] = None
        if t > 0:
            sol_old = mpc.solve(x, N_prev)                    # instrumento
            alpha = alpha_observed(V_prev, sol_old.V, l_prev)

        N = gov.update(alpha)
        if sol_old is not None and N == N_prev:
            sol = sol_old                                     # reutilizado: 0 extra
        else:
            sol = mpc.solve(x, N)
            n_extra += int(sol_old is not None)

        u = sol.u0
        l_t = mpc.stage(x, u)

        Ns.append(N)
        als.append(np.nan if alpha is None else alpha)
        xs.append(x.copy())
        us.append(u)
        sts.append(l_t)
        kks.append(t in kicks)

        w = np.zeros(plant.n)
        if t in kicks:
            w[-1] = kicks[t]
        x = A_r @ x + plant.B.ravel() * u + w
        if np.max(np.abs(x)) > x_div:
            diverged = True
            break
        V_prev, N_prev, l_prev = sol.V, N, l_t

    return AlphaTrace(N=np.array(Ns), alpha_raw=np.array(als),
                      x=np.array(xs), u=np.array(us), stage=np.array(sts),
                      kick_step=np.array(kks, bool),
                      compute=mpc.compute, n_solves=mpc.n_solves,
                      n_extra=n_extra, diverged=diverged)


# --------------------------------------------------------------------------
#                               METRICAS
# --------------------------------------------------------------------------

def trace_metrics(tr: AlphaTrace, kicks: Dict[int, float], window: int = 12,
                  alpha_min: float = 0.1) -> dict:
    """Las cuatro magnitudes del protocolo, mas el detalle por evento.

    - computo: suma de N sobre todos los solves (proxy lineal del coste QP)
    - coste:   suma de l(x_t, u_t) del lazo real
    - salud del certificado: alpha en pasos SIN kick (en los pasos con kick el
      alpha negativo es culpa de la perturbacion, no del horizonte)
    - asignacion: para cada kick, computo medio en la ventana posterior; la
      correlacion |kick| vs N_medio es el analogo del corr(K, E[n_iter]) del HBP
    """
    T = len(tr.N)
    # Indexado del alpha (corregido tras revision adversarial): als[t] guarda el
    # alpha de la transicion t-1 -> t. El kick de t_k entra al avanzar
    # x_{t_k} -> x_{t_k+1}, luego el UNICO alpha contaminado por el kick es el
    # del indice t_k + 1. El del indice t_k se refiere a la transicion
    # PRE-kick y es limpio.
    contaminated = np.zeros(T, bool)
    for tk in kicks:
        if tk + 1 < T:
            contaminated[tk + 1] = True
    ok = ~contaminated & ~np.isnan(tr.alpha_raw)
    a_clean = tr.alpha_raw[ok]

    events: List[Tuple[float, float, float]] = []   # (|kick|, N_medio, coste_rec)
    for tk, s in kicks.items():
        if tk + 1 >= T:
            continue
        w_end = min(tk + 1 + window, T)
        events.append((abs(s),
                       float(np.mean(tr.N[tk + 1:w_end])),
                       float(np.sum(tr.stage[tk + 1:w_end]))))
    if len(events) >= 3:
        sz = np.array([e[0] for e in events])
        nm = np.array([e[1] for e in events])
        corr = float(np.corrcoef(sz, nm)[0, 1]) if sz.std() > 1e-12 and nm.std() > 1e-12 else np.nan
    else:
        corr = np.nan

    return dict(
        computo=int(tr.compute),
        coste=float(np.sum(tr.stage)),
        N_medio=float(np.mean(tr.N)),
        n_extra=int(tr.n_extra),
        alpha_min_limpio=float(np.min(a_clean)) if a_clean.size else np.nan,
        frac_alpha_bajo=float(np.mean(a_clean < alpha_min)) if a_clean.size else np.nan,
        corr_asignacion=corr,
        n_eventos=len(events),
        eventos=events,
        divergio=bool(tr.diverged),
    )
