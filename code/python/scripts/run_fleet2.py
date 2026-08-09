"""La flota, comparada sobre el COMPUTO REALMENTE GASTADO.

QUE ESTABA MAL
--------------
La primera version imponia el presupuesto sobre la suma de horizontes
CONCEDIDOS, pero el gasto real del gobernador incluye ademas el solve del
instrumento cada vez que el horizonte cambia (el del paso anterior solo se
reutiliza si el horizonte se mantiene). Medido: con presupuesto 32 el gobernador
gastaba 40.2, un 25.6% MAS que el horizonte fijo con el que se le comparaba. El
"-9.3% a igual presupuesto" se conseguia gastando mas computo, asi que no era a
igual presupuesto.

LA CORRECCION
-------------
Se deja de comparar por presupuesto nominal y se compara sobre el plano
(computo REALMENTE GASTADO, coste), que es como ya se compara el gobernador de
horizonte en la Seccion V. Cada punto se coloca donde de verdad gasta:

  * horizonte fijo N: barrido de N, gasto = M*N por paso
  * gobernado: barrido de presupuesto, gasto = el medido

y la pregunta pasa a ser la unica honesta: A IGUAL COMPUTO GASTADO, quien
controla mejor. Se interpola linealmente la frontera de horizonte fijo en el
computo de cada punto gobernado, que es lo mismo que se hace en la Seccion V.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ghi import fleet as fl
from ghi.stats import paired

M, T = 8, 480
Q = np.diag([0.05, 1.0])
R = np.array([[0.5]])
N_MIN, N_MAX = 2, 16
SEEDS_TUNE = (100, 101, 102)
SEEDS_EVAL = (0, 1, 2, 3, 4, 5, 6, 7)
BUDGETS = [M * b for b in (2, 3, 4, 5, 6, 8, 12, 16)]
FIJOS = [2, 3, 4, 5, 6, 8, 10, 12, 16]


def campo(seed, rho):
    return fl.CloudField(M=M, T=T, rho=rho, depth=1.2, seed=seed).build()


def corre(plant, kind, N0, budget, rho, seeds, igual=False, tau=8.0, a_ref=0.3):
    cs, comp = [], []
    for s in seeds:
        r = fl.run_fleet(plant, Q, R, campo(s, rho), kind, budget, N_min=N_MIN,
                         N_max=N_MAX, N_fixed=N0 or 16, reparto_igual=igual,
                         tau=tau, a_ref=a_ref)
        if r.divergio or not np.isfinite(r.coste):
            return None
        cs.append(r.coste); comp.append(r.computo_medio)
    return {"coste": float(np.mean(cs)), "computo": float(np.mean(comp)),
            "por_semilla": [float(c) for c in cs]}


def frontera_en(comp, fijos):
    """Coste que el horizonte fijo alcanza gastando exactamente `comp`, por
    interpolacion lineal entre los dos puntos que lo rodean."""
    xs = np.array([f["computo"] for f in fijos])
    ys = np.array([f["coste"] for f in fijos])
    o = np.argsort(xs); xs, ys = xs[o], ys[o]
    if comp <= xs[0] or comp >= xs[-1]:
        return None
    return float(np.interp(comp, xs, ys))


def main() -> dict:
    plant = fl.thermal_loop()
    out = {"config": {"M": M, "T": T, "N_max": N_MAX, "seeds": len(SEEDS_EVAL)}}
    print("=" * 78)
    print(f"FLOTA DE M={M} LAZOS: COMPARACION SOBRE EL COMPUTO REALMENTE GASTADO")
    print("=" * 78)
    print("  el gobernador paga tambien el solve del instrumento cuando el")
    print("  horizonte cambia; comparar por presupuesto NOMINAL le regalaba")
    print("  entre un 18% y un 26% de computo\n")

    best = None
    for tau in (4.0, 8.0, 14.7, 25.0):
        for a_ref in (0.3, 0.6, 0.9):
            tot = sum(corre(plant, "fugas", None, B, 0.0, SEEDS_TUNE,
                            tau=tau, a_ref=a_ref)["coste"] for B in (24, 32, 48))
            if best is None or tot < best[0]:
                best = (tot, tau, a_ref)
    _, TAU, AREF = best
    print(f"  sintonia del gobernador (semillas 100-102): tau={TAU}, a_ref={AREF}")
    out["sintonia"] = {"tau": TAU, "a_ref": AREF}

    for rho, nombre in ((0.0, "INDEPENDIENTES"), (1.0, "FRENTE COMUN")):
        print("\n" + "-" * 78)
        print(f"  perturbaciones {nombre} (rho={rho:g})")
        print("-" * 78)
        fijos = []
        for N in FIJOS:
            r = corre(plant, "fijo", N, None, rho, SEEDS_EVAL)
            if r:
                fijos.append(dict(r, N=N))
        print(f"  frontera de horizonte fijo: " +
              ", ".join(f"N={f['N']}({f['computo']:.0f},{f['coste']:.0f})"
                        for f in fijos[:5]) + " ...")
        print(f"\n  {'presup.':>8}{'computo real':>14}{'coste gob.':>12}"
              f"{'frontera ahi':>14}{'diferencia':>12}")
        filas = []
        for B in BUDGETS:
            g = corre(plant, "fugas", None, B, rho, SEEDS_EVAL, tau=TAU, a_ref=AREF)
            if not g:
                continue
            fr = frontera_en(g["computo"], fijos)
            if fr is None:
                print(f"  {B:>8}{g['computo']:>14.1f}{g['coste']:>12.1f}"
                      f"{'fuera de rango':>14}{'':>12}")
                continue
            dif = 100.0 * (g["coste"] - fr) / fr
            print(f"  {B:>8}{g['computo']:>14.1f}{g['coste']:>12.1f}"
                  f"{fr:>14.1f}{dif:>+11.2f}%"
                  + ("   gana el gobernador" if dif < 0 else ""))
            filas.append({"B": B, "computo": g["computo"], "coste": g["coste"],
                          "frontera": fr, "dif_pct": float(dif),
                          "por_semilla": g["por_semilla"]})
        out[f"rho_{rho:g}"] = filas
        out[f"frontera_rho_{rho:g}"] = fijos
        gana = [f for f in filas if f["dif_pct"] < 0]
        if gana:
            mejor = min(filas, key=lambda f: f["dif_pct"])
            print(f"\n  gana en {len(gana)}/{len(filas)} puntos | mejor "
                  f"{mejor['dif_pct']:+.2f}% con computo {mejor['computo']:.0f}")
        else:
            print(f"\n  NO gana en ninguno de los {len(filas)} puntos evaluados")

    print("\n" + "=" * 78)
    print("LECTURA")
    print("=" * 78)
    g0 = [f for f in out["rho_0"] if f["dif_pct"] < 0]
    g1 = [f for f in out["rho_1"] if f["dif_pct"] < 0]
    print(f"  independientes: gana en {len(g0)}/{len(out['rho_0'])} puntos")
    print(f"  frente comun  : gana en {len(g1)}/{len(out['rho_1'])} puntos")
    if not g0:
        print("\n  A IGUAL COMPUTO GASTADO el gobernador NO bate al horizonte fijo.")
        print("  La ventaja que se media antes venia de gastar mas computo del que")
        print("  se le habia asignado. El caso de aplicacion NO se sostiene tal cual.")
    return out


if __name__ == "__main__":
    res = main()
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dst = os.path.join(here, "results")
    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(dst, "fleet2.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    print(f"\nescrito {os.path.join(dst, 'fleet2.json')}")
