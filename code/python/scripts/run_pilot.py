"""Banco principal: reguladores de orden 0, 1 y 2 sobre los cuatro escenarios.

PROTOCOLO
---------
  * ingredientes terminales CORRECTOS (S_i = 0 por construccion)
  * 1er y 2o orden IGUALADOS en atenuacion a Nyquist EN TIEMPO DISCRETO
  * anti-windup que actua SOLO cuando el filtro bloquea de verdad
  * coste de evaluacion NEUTRAL: ponderado por la demanda limpia, no por el
    peso que cada regulador decide (si no, cada uno se puntuaria con su propio
    criterio y la comparacion no significaria nada)
  * contrastes pareados por semilla, con correccion de Holm por familia

PREDICCION DECLARADA ANTES DE MIRAR NINGUN NUMERO
-------------------------------------------------
  A (demanda cuasi-estatica) : el 2o orden PIERDE frente al 1er orden. Debe
      perder: sobreoscila donde no hace falta, y su pico resonante amplifica
      ruido donde no hay senal util que recuperar.
  B, C, D (demanda no estacionaria y fuera del sobre de diseno) : el 2o orden
      gana. Es la misma firma que el paper del HBP: ninguna ventaja en el
      regimen facil, ventaja fuera de el.
  COSTE: seguir mejor la demanda cuesta mas. En un banco de juguete la demanda
      no vale lo que cuesta seguirla, asi que el peso congelado sera el mas
      barato. Eso NO invalida nada: senala que el banco definitivo necesita que
      seguir la demanda tenga valor real (p.ej. precio horario de la energia).
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ghi import experiment as gx
from ghi import plant as gplant
from ghi import regulators as greg
from ghi import stats as gs
from ghi import terminal as gterm
from ghi.mompc import MOMPC


def main(n_seeds: int = 10, w0: float = 0.9, zeta: float = 0.5,
         theta: float = 0.5, grid_points: int = 21) -> dict:
    t0 = time.time()
    prob = gplant.problem_conflict()
    prob, term = gterm.design(prob)
    M = MOMPC(prob, term)

    tau, g = greg.match_first_order(w0, zeta, 1.0)
    print("=" * 78)
    print("BANCO PRINCIPAL DEL GHI")
    print("=" * 78)
    print(f"planta      : {prob.name}, N = {prob.N}")
    print(f"terminal    : Kf = {np.round(term.Kf, 4).tolist()}, "
          f"Omega con {term.H.shape[0]} filas")
    print(f"igualacion  : w0={w0} zeta={zeta} -> |H| en Nyquist = {g:.6f}, tau = {tau:.4f}")
    print(f"              (EN DISCRETO; igualar en continuo daba un desajuste de 1.69x)")
    print(f"anti-windup : theta = {theta} (fraccion de velocidad conservada al bloquear)")
    print(f"semillas    : {n_seeds}\n")

    regs = {
        "orden-0": lambda: greg.Order0(),
        "orden-1": lambda: greg.Order1(tau),
        "orden-2": lambda: greg.Order2(w0, zeta, theta),
        "congelado": lambda: greg.Frozen(0.5),
    }
    bank = gx.run_bank(M, regs, gx.default_scenarios(), range(n_seeds),
                       grid_points=grid_points)

    table = gs.contrast_table(
        bank, [("orden-2", "orden-1"), ("orden-2", "orden-0"), ("orden-1", "orden-0")])
    print("=" * 78)
    print("CONTRASTES PAREADOS POR SEMILLA")
    gs.print_contrasts(table)

    print(f"\ntiempo: {time.time() - t0:.1f}s")
    return {"bank": bank,
            "contrasts": {m: [vars(p) for p in fam] for m, fam in table.items()},
            "config": {"w0": w0, "zeta": zeta, "theta": theta, "tau": tau,
                       "g_nyquist": g, "n_seeds": n_seeds}}


if __name__ == "__main__":
    res = main()
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dst = os.path.join(here, "results")
    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(dst, "pilot.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    print(f"escrito {os.path.join(dst, 'pilot.json')}")
