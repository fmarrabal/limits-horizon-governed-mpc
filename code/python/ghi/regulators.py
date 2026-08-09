"""Reguladores del peso: orden 0, 1 y 2, con igualacion EN TIEMPO DISCRETO.

DOS FALLOS QUE ESTE MODULO EXISTE PARA NO REPETIR
-------------------------------------------------
1. El anti-windup se disparaba en CADA muestra, no solo al bloquear. En la
   version anterior se comparaba el valor aplicado -que cae en una malla- con el
   estado continuo del campo, asi que la condicion se cumplia siempre: 200/200
   pasos con cero bloqueos inducidos. El zeta EFECTIVO resultante era ~0.868 en
   vez de 0.5. El campo analizado no era el campo ejecutado.
   AQUI: `sync` recibe explicitamente `blocked`.

2. La igualacion se hacia en tiempo continuo. tau se elegia con
   |H1(j*wn)| = |H2(j*wn)| sobre transferencias continuas, pero los reguladores
   se integran en discreto y w0*dt = 0.9 NO es << 1. El desajuste real era 1.69x.
   AQUI: `match_first_order` iguala sobre la respuesta en z de los integradores
   REALMENTE implementados, evaluada en z = exp(j*wn*dt).

POR QUE IMPORTA LA IGUALACION
-----------------------------
Sin ella la comparacion 1er vs 2o orden no dice nada: siempre se puede hacer un
filtro mas lento. Igualados en atenuacion a Nyquist, la comparacion pasa a ser
de FASE y de ganancia en la banda util, que es donde el segundo orden es
estructuralmente distinto (rolloff -40 dB/dec frente a -20).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


# --------------------------------------------------------------------------
#                   RESPUESTA EN z DE LOS INTEGRADORES REALES
# --------------------------------------------------------------------------

def ss_second_order(w0: float, zeta: float, dt: float = 1.0):
    """Espacio de estados EXACTO del Verlet de velocidad tal como se implementa.

    Estado (a, v), entrada ad, salida a. Se obtiene por linealidad evaluando el
    paso en los vectores base, asi que por construccion coincide con el codigo
    que corre en el lazo.
    """
    def paso(a, v, ad):
        acc = -2 * zeta * w0 * v - w0 ** 2 * (a - ad)
        a2 = a + dt * v + 0.5 * dt * dt * acc
        vh = v + 0.5 * dt * acc
        v2 = vh + 0.5 * dt * (-2 * zeta * w0 * vh - w0 ** 2 * (a2 - ad))
        return a2, v2

    a1, v1 = paso(1.0, 0.0, 0.0)
    a2, v2 = paso(0.0, 1.0, 0.0)
    a3, v3 = paso(0.0, 0.0, 1.0)
    Ad = np.array([[a1, a2], [v1, v2]])
    Bd = np.array([[a3], [v3]])
    Cd = np.array([[1.0, 0.0]])
    Dd = np.array([[0.0]])
    return Ad, Bd, Cd, Dd


def ss_first_order(tau: float, dt: float = 1.0):
    """a_{k+1} = a_k + (dt/tau)(ad - a_k)."""
    g = dt / tau
    return (np.array([[1.0 - g]]), np.array([[g]]),
            np.array([[1.0]]), np.array([[0.0]]))


def freq_response(Ad, Bd, Cd, Dd, zval: complex) -> complex:
    H = Cd @ np.linalg.solve(zval * np.eye(Ad.shape[0]) - Ad, Bd) + Dd
    return complex(H[0, 0])


def mag_at(Ad, Bd, Cd, Dd, w: float, dt: float = 1.0) -> float:
    return abs(freq_response(Ad, Bd, Cd, Dd, np.exp(1j * w * dt)))


def phase_at(Ad, Bd, Cd, Dd, w: float, dt: float = 1.0) -> float:
    return float(np.angle(freq_response(Ad, Bd, Cd, Dd, np.exp(1j * w * dt))))


def match_first_order(w0: float, zeta: float, dt: float = 1.0,
                      w_match: Optional[float] = None) -> Tuple[float, float]:
    """tau tal que |H1| = |H2| a la frecuencia de igualacion, AMBOS en discreto.

    Devuelve (tau, g) con g la atenuacion comun. Por defecto w_match = Nyquist.
    Con z = -1 hay forma cerrada:  |H1(-1)| = (dt/tau)/|dt/tau - 2|,
    que para tau > dt/2 vale dt/(2 tau - dt), luego tau = dt(1/g + 1)/2.
    """
    if w_match is None:
        w_match = np.pi / dt
    g2 = mag_at(*ss_second_order(w0, zeta, dt), w_match, dt)
    if not (0.0 < g2 < 1.0):
        raise ValueError(f"atenuacion fuera de rango: g2={g2}")
    zval = np.exp(1j * w_match * dt)
    if abs(zval + 1.0) < 1e-12:             # Nyquist exacto: forma cerrada
        tau = dt * (1.0 / g2 + 1.0) / 2.0
    else:                                    # biseccion generica
        lo, hi = dt * 0.5 + 1e-9, 1e6
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if mag_at(*ss_first_order(mid, dt), w_match, dt) > g2:
                lo = mid
            else:
                hi = mid
        tau = 0.5 * (lo + hi)
    return float(tau), float(g2)


# --------------------------------------------------------------------------
#                              REGULADORES
# --------------------------------------------------------------------------

class Regulator:
    """Interfaz. `step` propone; `sync` recibe lo que el filtro dejo aplicar."""

    name = "base"

    def step(self, ad: float, dt: float = 1.0) -> float:
        raise NotImplementedError

    def sync(self, a_applied: float, blocked: bool) -> None:
        pass

    def reset(self, a0: float = 0.5) -> None:
        pass


class Order0(Regulator):
    """Sin memoria: a = a_d. Es el alpha_d del TFM."""

    name = "orden-0"

    def __init__(self, a0: float = 0.5):
        self.a = a0

    def step(self, ad: float, dt: float = 1.0) -> float:
        self.a = float(ad)
        return self.a

    def sync(self, a_applied: float, blocked: bool) -> None:
        self.a = float(a_applied)

    def reset(self, a0: float = 0.5) -> None:
        self.a = a0


class Order1(Regulator):
    """Filtro de primer orden. Solo puede retrasarse: nunca adelanta fase."""

    name = "orden-1"

    def __init__(self, tau: float, a0: float = 0.5):
        if tau <= 0:
            raise ValueError("tau debe ser positivo")
        self.tau, self.a, self._a0 = float(tau), a0, a0

    def step(self, ad: float, dt: float = 1.0) -> float:
        self.a += dt / self.tau * (float(ad) - self.a)
        return self.a

    def sync(self, a_applied: float, blocked: bool) -> None:
        if blocked:
            self.a = float(a_applied)

    def reset(self, a0: Optional[float] = None) -> None:
        self.a = self._a0 if a0 is None else a0


class Order2(Regulator):
    """Campo homeostatico de segundo orden (rama de onda del HBP, un nodo),
    integrado con Verlet de velocidad.

        a'' + 2 zeta w0 a' + w0^2 (a - a_d) = 0

    `theta` es la ganancia de anti-windup: fraccion de velocidad que se CONSERVA
    cuando el filtro de seguridad recorta.
        theta = 1  -> arrastre (posicion sincronizada, velocidad intacta)
        theta = 0  -> reset ingenuo (mata toda la velocidad)
    El barrido en theta es el mando de diseno del compromiso seguimiento/bloqueo.
    """

    name = "orden-2"

    def __init__(self, w0: float, zeta: float, theta: float = 0.5, a0: float = 0.5):
        if w0 <= 0 or zeta < 0:
            raise ValueError("w0 > 0 y zeta >= 0")
        if not (0.0 <= theta <= 1.0):
            raise ValueError("theta en [0,1]")
        self.w0, self.zeta, self.theta = float(w0), float(zeta), float(theta)
        self.a, self.v, self._a0 = a0, 0.0, a0

    def _acc(self, a: float, v: float, ad: float) -> float:
        return -2 * self.zeta * self.w0 * v - self.w0 ** 2 * (a - ad)

    def step(self, ad: float, dt: float = 1.0) -> float:
        ad = float(ad)
        acc = self._acc(self.a, self.v, ad)
        self.a = self.a + dt * self.v + 0.5 * dt * dt * acc
        vh = self.v + 0.5 * dt * acc
        self.v = vh + 0.5 * dt * self._acc(self.a, vh, ad)
        return self.a

    def sync(self, a_applied: float, blocked: bool) -> None:
        if blocked:                       # SOLO al bloquear de verdad
            self.v *= self.theta
            self.a = float(a_applied)

    def reset(self, a0: Optional[float] = None) -> None:
        self.a = self._a0 if a0 is None else a0
        self.v = 0.0


class Frozen(Regulator):
    """Peso constante. Control nulo: aisla el mero hecho de tener parametros."""

    name = "congelado"

    def __init__(self, a: float = 0.5):
        self.a = float(a)

    def step(self, ad: float, dt: float = 1.0) -> float:
        return self.a

    def sync(self, a_applied: float, blocked: bool) -> None:
        pass


# --------------------------------------------------------------------------
#                    DIAGNOSTICO DEL REGULADOR EJECUTADO
# --------------------------------------------------------------------------

def effective_overshoot(reg: Regulator, dt: float = 1.0, T: int = 400) -> float:
    """Sobreoscilacion al escalon del regulador TAL COMO SE EJECUTA.

    Es el diagnostico que detecta el fallo 1: si el anti-windup se dispara
    cuando no debe, la sobreoscilacion medida no coincide con la del sistema
    discreto (con el fallo daba 0.41% frente al 5.66% del Verlet puro).

    OJO con el estado inicial: se resetea a 0 a proposito, para que el escalon
    tenga amplitud 1. Con el arranque por defecto (a=0.5) la amplitud seria 0.5
    y la sobreoscilacion relativa saldria justo la MITAD.
    """
    reg.reset(0.0)
    peak = -np.inf
    for _ in range(T):
        a = reg.step(1.0, dt)
        reg.sync(a, blocked=False)      # sin bloqueo: sync NO debe hacer nada
        peak = max(peak, a)
    return float(max(0.0, peak - 1.0))


def theoretical_overshoot(zeta: float) -> float:
    """Sobreoscilacion continua de un segundo orden: exp(-pi zeta/sqrt(1-zeta^2))."""
    if zeta >= 1.0:
        return 0.0
    return float(np.exp(-np.pi * zeta / np.sqrt(1.0 - zeta ** 2)))


def _state_of(reg: Regulator) -> tuple:
    return tuple(getattr(reg, k) for k in ("a", "v") if hasattr(reg, k))


def sync_fire_rate(reg: Regulator, T: int = 200, dt: float = 1.0,
                   seed: int = 0) -> float:
    """Fraccion de muestras en que `sync(..., blocked=False)` MODIFICA el estado.

    Debe salir 0.0 exactamente. Si no, el regulador analizado no es el ejecutado
    -- que es precisamente el fallo 1 de la cabecera de este modulo.
    """
    rng = np.random.default_rng(seed)
    reg.reset()
    fired = 0
    for _ in range(T):
        reg.step(float(rng.uniform(0.0, 1.0)), dt)
        antes = _state_of(reg)
        reg.sync(float(rng.uniform(0.0, 1.0)), blocked=False)
        if _state_of(reg) != antes:
            fired += 1
    return float(fired / T)
