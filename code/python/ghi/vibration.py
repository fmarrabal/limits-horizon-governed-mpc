"""La arena donde la regla refinada PREDICE que el campo inercial gana.

LA PREDICCION QUE SE PONE A PRUEBA
----------------------------------
Tres arenas han acotado la ventaja del segundo orden:

  * banco sintetico de pesos : demanda OSCILATORIA en RUIDO  -> GANA (t~+31 OOD)
  * invernadero              : demanda SUAVE y LIMPIA        -> pierde
  * alpha_N (horizonte)      : demanda de BANDA ANCHA (pulsos) -> pierde feo

De ahi la regla refinada: la inercia paga cuando la demanda es de BANDA ESTRECHA
(oscilatoria) ENTERRADA EN RUIDO. Pero esa regla se formulo mirando resultados;
mientras no prediga algo por adelantado no vale nada. Esta arena es su prueba:
un lazo de control REAL, con perturbacion periodica dominante y sensores
ruidosos, donde la regla predice victoria. Si el campo NO gana aqui, la regla
esta muerta y hay que decirlo.

POR QUE UN BATIDO Y NO UNA ENVOLVENTE ALEATORIA
-----------------------------------------------
Un proceso gaussiano de banda estrecha (oleaje JONSWAP, ruido filtrado) tiene
envolvente de RAYLEIGH, cuyo espectro es PASO-BAJO, no de banda estrecha. Esa
demanda seria del tipo "suave" -- el caso del invernadero, donde el campo YA
perdio. Para tener demanda genuinamente de banda estrecha OSCILATORIA hace
falta un BATIDO: dos fuentes armonicas de frecuencia proxima,

    d(t) = A1 sin(w1 t) + A2 sin(w2 t),   w1 ~ w2

que da portadora a (w1+w2)/2 y envolvente que OSCILA a |w1-w2|/2. Es
fisicamente canonico: dos maquinas rotativas ligeramente desincronizadas, dos
sistemas de mar de fondo, engrane con deslizamiento. Aqui se genera la
envolvente explicitamente para poder construir el NULO DE MECANISMO exacto.

POR QUE EL COMPROMISO OPTIMO ES NO ESTACIONARIO: FATIGA SUPERLINEAL
--------------------------------------------------------------------
Con coste verdadero cuadratico en todo, el peso optimo seria CONSTANTE y no
habria nada que modular (la leccion del invernadero). Aqui el coste real es

    L_real(x,u) = q * e^2  +  r * u^4

donde el termino de actuador es SUPERLINEAL: es la ley de dano por fatiga
(Miner/Basquin, dano ~ rango de tension^m con m = 3..5 en acero), canonica en
estructuras offshore y aerogeneradores. Con ella, el compromiso optimo depende
de la AMPLITUD de operacion.

DERIVACION DE alpha_d (se deduce, no se inventa)
------------------------------------------------
El MPC minimiza el subrogado cuadratico  (1-a) u^2 + a e^2. Desarrollando el
cuartico verdadero alrededor de la amplitud de operacion U:

    r u^4  ~  r (U^4 + 4U^3 (u-U) + 6U^2 (u-U)^2)

el coeficiente cuadratico efectivo del esfuerzo es 6 r U^2, frente a q para el
seguimiento. Igualando la RAZON del subrogado a la del coste real:

    (1-a)/a = 6 r U^2 / q      =>      a_d = 1 / (1 + kappa U^2),  kappa = 6r/q

Y U(t), la amplitud de esfuerzo necesaria, es proporcional a la envolvente de
la perturbacion. Luego

    a_d(t) = 1 / (1 + kappa' E(t)^2)

que es LA MISMA FORMA FUNCIONAL del alpha_d del TFM -- pero movida por una
envolvente fisica medida con ruido, en vez de por ||x||. Como E oscila a w_g,
a_d oscila a w_g y 2w_g: banda estrecha.

EL MPC NO PUEDE VER ESTO
------------------------
El MPC es NOMINAL: predice x+ = Ax + Bu sin modelo de perturbacion (lo
estandar, y lo honesto: si tuviera el modelo del oleaje lo compensaria por
prealimentacion y no habria nada que arbitrar). Luego su horizonte NO puede
arbitrar la envolvente. Cumple el principio de agosto: el campo modula lo que
el horizonte no ve.

EL NULO DE MECANISMO, QUE ES LO QUE HACE HONESTA LA ARENA
----------------------------------------------------------
Se genera la envolvente como  E(t) = E0 (1 + a * s(t))  con s(t) de media cero
y varianza FIJA, y solo se cambia su ESPECTRO:

    banda estrecha : s(t) = sqrt(2) sin(w_g t)      (oscilatoria)
    banda ancha    : s(t) = ruido filtrado ancho, MISMA media y MISMA varianza

Misma portadora, misma envolvente media, misma varianza de envolvente, misma
varianza de demanda: lo UNICO que cambia es si la demanda tiene o no una
frecuencia dominante. Si la ventaja del campo sobrevive al caso de banda ancha,
entonces "banda estrecha" no era el mecanismo y la regla es falsa.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

from .plant import Objective, Plant, Problem


# --------------------------------------------------------------------------
#                          LA ESTRUCTURA MECANICA
# --------------------------------------------------------------------------

def vibration_plant(dt: float = 1.0, wp: float = 0.70, zp: float = 0.04,
                    xmax: float = 40.0, vmax: float = 40.0,
                    umax: float = 6.0) -> Plant:
    """Masa-muelle-amortiguador poco amortiguada, discretizada exactamente.

        x'' + 2 zp wp x' + wp^2 x = u + d

    zp = 0.04 es una estructura ligera realista (acero soldado, offshore). Se
    sintoniza wp CERCA de la portadora para que la perturbacion realmente
    excite la estructura y el control tenga algo que hacer.
    """
    from scipy.linalg import expm
    Ac = np.array([[0.0, 1.0], [-wp ** 2, -2 * zp * wp]])
    Bc = np.array([[0.0], [1.0]])
    M = np.zeros((3, 3))
    M[:2, :2] = Ac
    M[:2, 2:] = Bc
    E = expm(M * dt)
    return Plant(A=E[:2, :2], B=E[:2, 2:], xmax=np.array([xmax, vmax]),
                 umax=np.array([umax]), name="estructura")


def problem_vibration(dt: float = 1.0, N: int = 10) -> Problem:
    """Dos objetivos: esfuerzo de actuador frente a desviacion estructural.

    Son las DOS caras del coste real: el termino de fatiga (que el MPC solo
    puede representar como cuadratico) y el de dano/servicio por desviacion.
    """
    J_esfuerzo = Objective(Q=np.diag([1e-6, 1e-6]), R=np.array([[1.0]]),
                           kind="quad", name="esfuerzo")
    J_estructura = Objective(Q=np.diag([1.0, 0.05]), R=np.array([[1e-6]]),
                             kind="quad", name="estructura")
    return Problem(vibration_plant(dt), [J_esfuerzo, J_estructura], N=N,
                   name="vibracion")


# --------------------------------------------------------------------------
#                    PERTURBACION Y ENVOLVENTE
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Excitation:
    """Perturbacion = portadora modulada por una envolvente de espectro
    controlado. El nulo de mecanismo se hace cambiando SOLO `band`."""

    E0: float = 2.6            # amplitud media de la envolvente
    depth: float = 0.75        # profundidad de modulacion (a)
    wc: float = 0.70           # portadora [rad/muestra]
    wg: float = 0.14           # frecuencia de batido [rad/muestra] (periodo ~45)
    band: str = "narrow"       # 'narrow' | 'broad'
    bw: float = 0.55           # ancho de banda relativo del caso 'broad'
    seed: int = 0

    def envelope(self, T: int) -> np.ndarray:
        """E(t) con media E0 y desviacion tipica E0*depth/sqrt(2) en AMBOS casos."""
        t = np.arange(T)
        if self.band == "narrow":
            s = np.sqrt(2.0) * np.sin(self.wg * t)          # var(s) = 1
        elif self.band == "broad":
            rng = np.random.default_rng(self.seed + 90001)
            w = rng.standard_normal(T + 400)
            # paso-bajo de 1er orden con corte ANCHO alrededor de wg: misma
            # escala temporal media, espectro sin pico
            a = np.exp(-self.bw)
            y = np.zeros_like(w)
            for k in range(1, len(w)):
                y[k] = a * y[k - 1] + (1 - a) * w[k]
            s = y[400:]
            s = s - s.mean()
            sd = s.std()
            s = s / sd if sd > 1e-12 else s                  # var(s) = 1, igual
        else:
            raise ValueError(self.band)
        E = self.E0 * (1.0 + self.depth * s)
        return np.maximum(E, 0.05 * self.E0)                 # envolvente positiva

    def signal(self, T: int) -> Tuple[np.ndarray, np.ndarray]:
        """(d(t), E(t)). La portadora es identica en los dos casos."""
        t = np.arange(T)
        E = self.envelope(T)
        d = E * np.sin(self.wc * t + 0.3)
        return d, E


def alpha_d_from_envelope(E: np.ndarray, kappa: float) -> np.ndarray:
    """a_d = 1/(1 + kappa E^2). Deducida del coste real (ver cabecera)."""
    return 1.0 / (1.0 + kappa * np.asarray(E, float) ** 2)


# --------------------------------------------------------------------------
#                            EL COSTE REAL
# --------------------------------------------------------------------------

def true_cost(x: np.ndarray, u: float, q: float = 1.0, r: float = 1.0,
              m: int = 4) -> float:
    """L_real = q e^2 + r |u|^m, con m = 4 (fatiga superlineal).

    Es la UNICA metrica de evaluacion y no depende de ningun peso: ni del que
    el regulador elige ni del que el experimentador desearia.
    """
    x = np.asarray(x, float).ravel()
    return float(q * x[0] ** 2 + r * abs(float(u)) ** m)


def kappa_from_costs(q: float = 1.0, r: float = 1.0,
                     gain: float = 1.0) -> float:
    """kappa = 6 r / q, escalado por la ganancia esfuerzo/envolvente.

    `gain` traduce envolvente de perturbacion a amplitud de esfuerzo (U ~ gain*E)
    y se calibra numericamente: es el unico numero libre, y se fija ANTES de
    mirar ningun resultado comparativo, buscando que el mejor peso CONSTANTE
    caiga en el INTERIOR del simplex (la leccion del invernadero).
    """
    return 6.0 * r / q * gain ** 2
