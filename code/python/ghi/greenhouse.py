"""Banco de invernadero con precio horario de la energia.

POR QUE ESTE BANCO EXISTE
-------------------------
El banco sintetico deja un resultado incomodo y honesto: en coste, el peso
CONGELADO es el mas barato de todos. La razon no es que la modulacion no sirva,
sino que en un juguete la demanda alpha_d(t) es una senal ARBITRARIA que el
experimentador inventa: seguirla no vale nada, y por tanto cualquier esfuerzo
en seguirla es esfuerzo perdido.

Aqui la demanda deja de ser arbitraria. El coste verdadero es UNO SOLO y esta
en euros:

    coste_real(t) = precio(t) * energia(t) + lambda * desviacion(t)^2

con precio(t) el precio horario de la electricidad. Ese coste es NO ESTACIONARIO
porque el precio lo es, y el compromiso optimo entre "gastar energia" y "mantener
la temperatura" cambia hora a hora.

LA CLAVE: alpha_d SE DEDUCE, NO SE INVENTA
------------------------------------------
Si el MPC escalariza  (1-a) * J_energia + a * J_desviacion  con
J_energia = u' R0 u  y  J_desviacion = x' Q1 x, para que el criterio escalarizado
sea PROPORCIONAL al coste real hace falta

    (1-a) * R0 = precio(t) * r      y      a * Q1 = lambda * Q

de donde, eliminando la constante de proporcionalidad,

    a_d(t) = 1 / (1 + kappa * precio(t)),      kappa = (r/R0) * (Q1/(lambda*Q))

Es decir: **el peso deseado es una funcion explicita del precio**, y seguirlo
bien reduce euros de verdad. La misma forma funcional que el alpha_d del TFM
-- 1/(1+c*algo) -- pero movida por una senal exogena real en lugar de por ||x||.

Con eso, el eje de coste vuelve a decir algo, que es justo lo que le faltaba al
banco sintetico.

MODELO
------
Dos estados, discretizado a Delta_t = 1 h:
    x1 = desviacion de la temperatura del aire interior respecto de la optima [C]
    x2 = desviacion de la masa termica (suelo + cultivo) [C]
    u  = potencia de calefaccion [kW, normalizada]

    C_air  dT_air/dt  = -k_v (T_air - T_out) - k_c (T_air - T_mass) + u + q_sol
    C_mass dT_mass/dt =  k_c (T_air - T_mass)

El aire es rapido y la masa termica lenta: es la separacion de escalas que hace
que la decision de cuando calentar dependa de lo que va a pasar despues, y por
tanto que el horizonte sirva de algo.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .plant import Objective, Plant, Problem


# --------------------------------------------------------------------------
#                               LA PLANTA
# --------------------------------------------------------------------------

def greenhouse_plant(dt: float = 1.0, C_air: float = 1.0, C_mass: float = 8.0,
                     k_v: float = 0.45, k_c: float = 0.35,
                     Tmax: float = 8.0, umax: float = 6.0) -> Plant:
    """Modelo lineal de dos capas, discretizado exactamente (Euler exponencial).

    Los valores por defecto dan una constante de tiempo del aire de ~1.2 h y de
    la masa termica de ~23 h, que es el orden correcto para un invernadero
    mediterraneo de plastico.
    """
    Ac = np.array([[-(k_v + k_c) / C_air, k_c / C_air],
                   [k_c / C_mass, -k_c / C_mass]])
    Bc = np.array([[1.0 / C_air], [0.0]])
    # discretizacion exacta por exponencial de la matriz aumentada
    n = 2
    Mexp = np.zeros((n + 1, n + 1))
    Mexp[:n, :n] = Ac
    Mexp[:n, n:] = Bc
    from scipy.linalg import expm
    E = expm(Mexp * dt)
    A = E[:n, :n]
    B = E[:n, n:]
    return Plant(A=A, B=B, xmax=np.array([Tmax, Tmax]), umax=np.array([umax]),
                 name="invernadero")


def problem_greenhouse(dt: float = 1.0) -> Problem:
    """Dos objetivos en euros-equivalentes: energia frente a desviacion agronomica.

    R0 y Q1 fijan las UNIDADES; el compromiso instantaneo lo decide alpha.
    """
    J_energia = Objective(Q=np.diag([1e-6, 1e-6]), R=np.array([[1.0]]),
                          kind="quad", name="energia")
    J_clima = Objective(Q=np.diag([1.0, 0.05]), R=np.array([[1e-6]]),
                        kind="quad", name="clima")
    return Problem(greenhouse_plant(dt), [J_energia, J_clima], N=8,
                   name="invernadero")


# --------------------------------------------------------------------------
#                      PRECIO, METEOROLOGIA Y DEMANDA
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Market:
    """Perfil diario de precio de la electricidad, en unidades arbitrarias
    proporcionales a EUR/kWh. Dos picos (manana y tarde-noche) y valle nocturno,
    que es la forma tipica del mercado diario peninsular."""

    base: float = 0.10
    amp_m: float = 0.09          # pico de manana (~8 h)
    amp_t: float = 0.13          # pico de tarde  (~20 h)
    period: float = 24.0

    def price(self, t: float) -> float:
        h = (t % self.period)
        p = (self.base
             + self.amp_m * np.exp(-0.5 * ((h - 8.0) / 2.2) ** 2)
             + self.amp_t * np.exp(-0.5 * ((h - 20.0) / 2.6) ** 2)
             - 0.035 * np.exp(-0.5 * ((h - 3.0) / 3.0) ** 2))
        return float(max(p, 0.02))


@dataclass(frozen=True)
class Weather:
    """Temperatura exterior y radiacion solar, ambas con ciclo de 24 h.

    Entran en el lazo como perturbacion aditiva sobre el estado. La radiacion
    calienta durante el dia (cuando la energia suele ser cara) y de noche el
    invernadero se enfria (cuando suele ser barata): ese desfase es lo que hace
    que el problema tenga contenido.
    """

    dT_out: float = 6.0          # amplitud del ciclo termico exterior
    q_sol: float = 3.2           # amplitud de la ganancia solar
    period: float = 24.0

    def disturbance(self, t: float, n: int = 2) -> np.ndarray:
        h = (t % self.period)
        w = np.zeros(n)
        # de noche pierde calor, de dia lo gana
        w[0] = (-self.dT_out * np.cos(2 * np.pi * h / self.period) * 0.12
                + self.q_sol * max(0.0, np.sin(np.pi * (h - 6.0) / 12.0)) * 0.18)
        return w


def alpha_d_from_price(price: float, kappa: float = 12.0) -> float:
    """El peso deseado DEDUCIDO del precio:  a_d = 1 / (1 + kappa * precio).

    Energia cara -> a_d pequeno -> pesa la economia.
    Energia barata -> a_d grande -> pesa el clima.
    """
    return float(1.0 / (1.0 + kappa * max(price, 1e-9)))


# --------------------------------------------------------------------------
#                        COSTE REAL, EN EUROS
# --------------------------------------------------------------------------

# lambda CALIBRADA. Convierte la desviacion agronomica a euros y es el mando que
# centra el banco. Barrido realizado sobre el escenario nominal, buscando que el
# mejor peso CONSTANTE caiga en el INTERIOR del simplex y no en la frontera:
#     lambda = 0.02 -> a* = 0.35 (interior)   <- adoptado
#     lambda = 0.05 -> a* = 0.85 (interior, pero casi en el borde)
#     lambda >= 0.10 -> a* = 0.95 (borde: el objetivo de clima domina siempre y
#                                  el banco pierde la capacidad de discriminar)
LAMBDA_CALIBRADA = 0.02


def true_cost(x: np.ndarray, u: float, price: float,
              lam: float = LAMBDA_CALIBRADA) -> float:
    """coste_real = precio * energia + lambda * desviacion agronomica.

    Es la UNICA metrica de evaluacion del banco, y no depende de ningun peso:
    ni del que el regulador elige ni del que el experimentador desearia. Eso es
    lo que la hace neutral.
    """
    x = np.asarray(x, float).ravel()
    energia = float(u) ** 2
    desviacion = float(x[0] ** 2 + 0.05 * x[1] ** 2)
    return float(price * energia + lam * desviacion)


def demand_fn(market: Market, kappa: float = 12.0, dt: float = 1.0
              ) -> Callable[[int], float]:
    """a_d(t) para el banco: se lee del precio, no se inventa."""
    return lambda t: alpha_d_from_price(market.price(t * dt), kappa)
