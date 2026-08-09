"""La correccion a Bemporad & Munoz de la Pena (2009): V* es CONCAVA en el peso.

    V*(x, alpha) = min_{U en U(x)}  alpha' J(U, x)

Para cada U fijo el corchete es AFIN en alpha, y el factible U(x) NO depende de
alpha. El infimo puntual de una familia de funciones afines es CONCAVO. Es el
mismo hecho que "la funcion dual de Lagrange es concava".

Bemporad & Munoz de la Pena, Automatica 45(12):2823-2830 (2009), afirman en su
Lema 4 que V* es "convex and piecewise affine w.r.t. mu", en su Teorema 6 la
escriben como el MAXIMO de sus piezas afines para reducir la seleccion de peso a
un LP, y en su Teorema 7 invocan Mangasarian & Rosen (1964), que es un resultado
sobre parametros en las RESTRICCIONES, para un parametro que esta en el COSTE.

CONSECUENCIA, y conviene enunciarla con cuidado: su LP no es una reformulacion
EQUIVALENTE sino una RESTRICCION INTERIOR conservadora. El algoritmo sigue siendo
SEGURO -- una restriccion interior nunca viola el certificado -- pero la
equivalencia y la optimalidad de alpha* tal como estan enunciadas no se sostienen.

ANTES DE CITAR NADA DE ESTO: verificar a mano sobre el PDF original que su mu es
el peso del coste y que su factible no depende de el.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ghi import plant as gplant
from ghi import terminal as gterm
from ghi import weights as gw
from ghi.mompc import MOMPC, shifted_sequence, solve


def main(n_states: int = 60, seed: int = 0) -> dict:
    prob = gplant.problem_conflict()
    prob, term = gterm.design(prob)
    M = MOMPC(prob, term)

    print("=" * 78)
    print("CONCAVIDAD DE V*(x, .) EN EL PESO")
    print("=" * 78)
    print("Argumento analitico, sin numeros:")
    print("  V*(x,a) = min_{U in U(x)} [ (1-a) J0(U,x) + a J1(U,x) ]")
    print("  Para cada U fijo el corchete es AFIN en a, y U(x) no depende de a.")
    print("  => V*(x,.) es el infimo puntual de una familia de afines => CONCAVA.")
    print("  => {a : V*(x,a) <= J_a} es un subnivel de una concava: NO convexo.\n")

    rng = np.random.default_rng(seed)
    states = []
    while len(states) < n_states:
        x = rng.uniform(-7, 7, prob.plant.n)
        if solve(M, x, np.array([0.5, 0.5])) is not None:
            states.append(x)

    ev = gw.concavity_evidence(M, states)
    print("--- test de cuerda (5 pares de pesos por estado) ---")
    print(f"  estados                      : {len(states)}")
    print(f"  tests                        : {ev['tests']}")
    print(f"  violaciones de CONCAVIDAD    : {ev['violaciones_concavidad']}")
    print(f"  violaciones de CONVEXIDAD    : {ev['violaciones_convexidad']}")
    print(f"  desviacion media respecto de la cuerda : {ev['d_media']:+.4e}")
    print(f"  (positiva = por ENCIMA de la cuerda = concava)")
    print(f"  veredicto                    : "
          f"{'CONCAVA' if ev['concava'] else 'NO CONCLUYENTE'}\n")

    print("--- geometria del conjunto admisible ---")
    tot_niv = tot_nc = pico_int = n_ok = 0
    for x in states[:20]:
        sw = gw.nonconvexity_sweep(M, x, grid_points=41, n_levels=25)
        if sw["niveles"] == 0:
            continue
        n_ok += 1
        tot_niv += sw["niveles"]
        tot_nc += sw["no_convexos"]
        pico_int += int(sw["pico_interior"])
    print(f"  estados evaluados                    : {n_ok}")
    print(f"  niveles de J_a barridos              : {tot_niv}")
    print(f"  niveles con conjunto NO convexo      : {tot_nc} "
          f"({100 * tot_nc / max(tot_niv, 1):.1f}%)")
    print(f"  estados con el maximo de V* INTERIOR : {pico_int}/{n_ok}")
    print()
    print("  Que el maximo de V* caiga en el interior del simplex es la firma de")
    print("  la concavidad, y es lo que obliga a que el subnivel sea la union de")
    print("  dos intervalos. No es una rareza numerica: es geometria forzada.")

    out = {"concavidad": ev,
           "no_convexidad": {"estados": n_ok, "niveles": tot_niv,
                             "no_convexos": tot_nc, "pico_interior": pico_int}}
    print("\n" + "=" * 78)
    return out


if __name__ == "__main__":
    res = main()
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dst = os.path.join(here, "results")
    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(dst, "concavidad.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"escrito {os.path.join(dst, 'concavidad.json')}")
