"""Banco experimental: lazo cerrado, escenarios y metricas.

EL LAZO, PASO A PASO
--------------------
  1. llega la demanda limpia a_d(t) y su version contaminada con ruido
  2. el regulador PROPONE  a_req
  3. se forma J_a con la secuencia desplazada y el peso del instante anterior
  4. se evalua el conjunto admisible A(x, J_a) sobre una malla
     (se BARRE porque el conjunto NO es convexo: V* es concava en alpha)
  5. se aplica el admisible mas cercano; se marca si hubo bloqueo
  6. se avisa al regulador SOLO si hubo bloqueo (anti-windup correcto)
  7. se resuelve el MPC con el peso aplicado y se ejecuta u_0
  8. la planta avanza con desajuste de modelo, excitacion persistente y ruido

METRICA DE COSTE NEUTRAL
------------------------
El coste de evaluacion se pondera con la DEMANDA LIMPIA del instante, no con el
peso aplicado. Si se ponderase con el peso aplicado, cada regulador se estaria
puntuando con su propio criterio y la comparacion no significaria nada.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .mompc import MOMPC, shifted_sequence, solve
from .plant import Problem
from .regulators import Regulator
from .terminal import Terminal
from .weights import Admissible, admissible_set, certified_radius, simplex_grid


# --------------------------------------------------------------------------
#                               ESCENARIOS
# --------------------------------------------------------------------------

@dataclass
class Scenario:
    """Un escenario = una demanda + una forma de salirse del sobre de diseno.

    `disturbance` y `cost` permiten enchufar un banco con fisica y metrica
    propias -- el invernadero con precio horario -- sin duplicar el lazo. Si se
    dejan a None se usan la excitacion senoidal y el coste neutral sinteticos.
    """

    name: str
    demand: Callable[[int], float]       # a_d(t) limpia, escalar en [0,1]
    mismatch: Optional[np.ndarray] = None   # se SUMA a A en la planta real
    dist_std: float = 0.0                   # ruido de proceso
    noise_std: float = 0.15                 # ruido en la medida de la demanda
    dist_amp: float = 0.6                   # excitacion persistente
    dist_period: float = 17.0
    x0: Sequence[float] = (5.0, 5.0)
    T: int = 60
    warmup: int = 10
    # ---- ganchos opcionales para bancos con fisica y metrica propias --------
    disturbance: Optional[Callable[[int, int], np.ndarray]] = None
    cost: Optional[Callable[[np.ndarray, float, int, float], float]] = None


def sine_demand(period: float, amp: float = 0.35, base: float = 0.5):
    return lambda t: base + amp * np.sin(2.0 * np.pi * t / period)


def constant_demand(value: float = 0.75):
    return lambda t: value


def default_scenarios() -> Dict[str, Scenario]:
    """Los cuatro escenarios del protocolo: uno cuasi-estatico (donde se ESPERA
    que el segundo orden pierda) y tres no estacionarios, dos de ellos fuera del
    sobre de diseno."""
    return {
        "A_cuasiestatico": Scenario("A_cuasiestatico", constant_demand(0.75)),
        "B_periodico": Scenario("B_periodico", sine_demand(20.0)),
        "C_OOD": Scenario(
            "C_OOD", sine_demand(20.0),
            mismatch=np.array([[0.0, 0.15], [0.0, 0.10]]), dist_std=0.08),
        "D_OOD_fuerte": Scenario(
            "D_OOD_fuerte", sine_demand(13.0),
            mismatch=np.array([[0.0, 0.25], [0.05, 0.18]]), dist_std=0.15),
    }


# --------------------------------------------------------------------------
#                               LAZO CERRADO
# --------------------------------------------------------------------------

@dataclass
class Trace:
    alpha: np.ndarray
    alpha_clean: np.ndarray
    alpha_req: np.ndarray
    x: np.ndarray
    u: np.ndarray
    stage: np.ndarray
    blocked: np.ndarray
    radius: np.ndarray
    Ja: np.ndarray
    V: np.ndarray

    def metrics(self, warmup: int = 10) -> Dict[str, float]:
        s = slice(warmup, None)
        err = self.alpha[s] - self.alpha_clean[s]
        return {
            "err_demanda": float(np.sqrt(np.mean(err ** 2))),
            "coste": float(np.sum(self.stage[s])),
            "esfuerzo": float(np.sum(np.abs(self.u[s]))),
            "bloqueos": int(np.sum(self.blocked[s])),
            "tasa_bloqueo": float(np.mean(self.blocked[s])),
            "radio_min": float(np.min(self.radius[s])),
            "radio_medio": float(np.mean(self.radius[s])),
            "x_rms": float(np.sqrt(np.mean(np.sum(self.x[s] ** 2, axis=1)))),
        }


def closed_loop(mompc: MOMPC, reg: Regulator, sc: Scenario, seed: int = 0,
                grid_points: int = 21) -> Optional[Trace]:
    """Una corrida completa. Devuelve None si el MPC se vuelve infactible."""
    prob = mompc.problem
    plant = prob.plant
    if prob.n_obj != 2:
        raise NotImplementedError("el banco esta escrito para 2 objetivos")

    rng = np.random.default_rng(seed)
    grid = simplex_grid(2, grid_points)
    Areal = plant.A if sc.mismatch is None else plant.A + sc.mismatch
    x = np.asarray(sc.x0, float).copy()

    reg.reset()
    a_prev = 0.5
    sol = solve(mompc, x, np.array([1 - a_prev, a_prev]))
    if sol is None:
        return None

    rec: List[dict] = []
    for t in range(sc.T):
        adc = float(np.clip(sc.demand(t), 0.0, 1.0))
        adr = float(np.clip(adc + rng.normal(0.0, sc.noise_std), 0.0, 1.0))
        a_req = float(np.clip(reg.step(adr, 1.0), 0.0, 1.0))

        # J_a: secuencia desplazada evaluada con el peso ANTERIOR
        Us = shifted_sequence(mompc, sol)
        Js = prob.costs(x, Us)
        Ja = float(np.array([1 - a_prev, a_prev]) @ Js)

        adm = admissible_set(mompc, x, Ja, grid=grid)
        a_vec, blocked = adm.nearest(np.array([1 - a_req, a_req]))
        a_app = float(a_vec[1])
        reg.sync(a_app, blocked)

        sol = solve(mompc, x, a_vec)
        if sol is None:
            return None
        u = float(sol.U[0, 0])
        r = certified_radius(sol.J, sol.V, Ja)

        if sc.cost is not None:
            # metrica propia del banco (p.ej. euros): NO depende de ningun peso
            stage = float(sc.cost(x, u, t, adc))
        else:
            # coste NEUTRAL sintetico: ponderado por la demanda limpia
            c0 = prob.objectives[0].stage(x, np.array([u]))
            c1 = prob.objectives[1].stage(x, np.array([u]))
            stage = (1.0 - adc) * c0 + adc * c1

        rec.append(dict(alpha=a_app, clean=adc, req=a_req, x=x.copy(), u=u,
                        stage=stage, blocked=blocked, radius=r, Ja=Ja, V=sol.V))

        if sc.disturbance is not None:
            wd = np.asarray(sc.disturbance(t, plant.n), float).ravel()
        else:
            wd = np.zeros(plant.n)
            wd[-1] = sc.dist_amp * np.sin(2.0 * np.pi * t / sc.dist_period)
        wn = sc.dist_std * rng.standard_normal(plant.n) if sc.dist_std > 0 else 0.0
        x = np.clip(Areal @ x + (plant.B @ np.array([u])) + wd + wn,
                    -plant.xmax, plant.xmax)
        a_prev = a_app

    return Trace(
        alpha=np.array([r["alpha"] for r in rec]),
        alpha_clean=np.array([r["clean"] for r in rec]),
        alpha_req=np.array([r["req"] for r in rec]),
        x=np.array([r["x"] for r in rec]),
        u=np.array([r["u"] for r in rec]),
        stage=np.array([r["stage"] for r in rec]),
        blocked=np.array([r["blocked"] for r in rec], bool),
        radius=np.array([r["radius"] for r in rec]),
        Ja=np.array([r["Ja"] for r in rec]),
        V=np.array([r["V"] for r in rec]),
    )


def run_bank(mompc: MOMPC, regulators: Dict[str, Callable[[], Regulator]],
             scenarios: Dict[str, Scenario], seeds: Sequence[int],
             grid_points: int = 21, verbose: bool = True) -> Dict[str, Dict[str, List[dict]]]:
    """Corre el banco completo y devuelve las metricas crudas por corrida."""
    out: Dict[str, Dict[str, List[dict]]] = {}
    for sname, sc in scenarios.items():
        out[sname] = {}
        if verbose:
            print(f"### {sname}")
            print(f"  {'regulador':<12}{'err_demanda':>18}{'coste':>17}{'bloqueos':>10}")
        for rname, make in regulators.items():
            rows: List[dict] = []
            for s in seeds:
                tr = closed_loop(mompc, make(), sc, seed=s, grid_points=grid_points)
                if tr is not None:
                    rows.append(tr.metrics(sc.warmup))
            out[sname][rname] = rows
            if verbose and rows:
                e = np.array([r["err_demanda"] for r in rows])
                c = np.array([r["coste"] for r in rows])
                b = np.array([r["bloqueos"] for r in rows], float)
                sd = e.std(ddof=1) if len(e) > 1 else 0.0
                sc_ = c.std(ddof=1) if len(c) > 1 else 0.0
                print(f"  {rname:<12}{e.mean():>9.4f}+-{sd:<7.4f}"
                      f"{c.mean():>10.2f}+-{sc_:<5.2f}{b.mean():>10.1f}")
        if verbose:
            print()
    return out
