"""CASO DE APLICACION: un controlador embebido para M lazos.

LA PREGUNTA INDUSTRIAL
----------------------
Un campo de colectores o un invernadero multizona tiene M lazos y UN controlador.
El periodo de muestreo fija un presupuesto duro de calculo por paso. Con
horizonte FIJO hay que dimensionar para que los M lazos quepan en su peor caso a
la vez, es decir M*N_max. La pregunta util no es "cuanto presupuesto hace falta
para igualar a un horizonte fijo sin restriccion" -- esa pregunta es tramposa,
porque en cuanto sobra presupuesto lo sensato es usar N_max y no gobernar nada --
sino la contraria:

    A IGUAL PRESUPUESTO, ¿quien controla mejor?

HIPOTESIS PRE-REGISTRADAS
-------------------------
  H1  Con presupuesto ESCASO y perturbaciones INDEPENDIENTES, gobernar el
      horizonte bate al mejor horizonte fijo, porque los picos de demanda de los
      distintos lazos no coinciden y el reparto puede moverse a quien lo necesita.
  H2  Con perturbaciones TOTALMENTE CORRELACIONADAS (un frente que barre el
      campo) la ventaja desaparece: los picos coinciden, no hay nada que
      multiplexar y todos los repartos son equivalentes.
  H3  Bajo escasez, repartir PROPORCIONALMENTE A LA DEMANDA declarada por el
      gobernador bate a repartir a partes iguales. Es la prueba de que la lectura
      de dificultad sirve para ORDENAR PRIORIDADES, no solo para elegir un numero.

H2 es la hipotesis incomoda y se reporta pase lo que pase.

SINTONIA
--------
El gobernador se sintoniza EN ESTA ARENA (constante de fugas y referencia del
mapa) sobre semillas 100-102, disjuntas de las de evaluacion. Heredar la
constante de otra arena lo pondria en desventaja, del mismo modo que los
horizontes fijos se benefician de barrer N.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ghi import fleet as fl
from ghi.stats import holm, paired

M, T = 8, 480
Q = np.diag([0.05, 1.0])          # se penaliza el ABSORBEDOR (ver fleet.py)
R = np.array([[0.5]])
N_MIN, N_MAX = 2, 16
SEEDS_TUNE = (100, 101, 102)
SEEDS_EVAL = (0, 1, 2, 3, 4, 5, 6, 7)
BUDGETS = [M * b for b in (2, 3, 4, 6, 8, 12, 16)]
FIJOS = [2, 4, 6, 8, 12, 16]


def campo(seed, rho):
    return fl.CloudField(M=M, T=T, rho=rho, depth=1.2, seed=seed).build()


def corre(plant, kind, N0, budget, rho, seeds, igual=False, tau=8.0, a_ref=0.3):
    cs, comp, rec = [], [], []
    for s in seeds:
        r = fl.run_fleet(plant, Q, R, campo(s, rho), kind, budget, N_min=N_MIN,
                         N_max=N_MAX, N_fixed=N0 or 16, reparto_igual=igual,
                         tau=tau, a_ref=a_ref)
        if r.divergio or not np.isfinite(r.coste):
            return None
        cs.append(r.coste); comp.append(r.computo_medio); rec.append(r.frac_recortado)
    return {"coste": float(np.mean(cs)), "sem": float(np.std(cs, ddof=1) / np.sqrt(len(cs))),
            "computo": float(np.mean(comp)), "recorte": float(np.mean(rec)),
            "por_semilla": [float(c) for c in cs]}


def mejor_fijo(plant, budget, rho, seeds):
    """El rival: el MEJOR horizonte fijo para ese presupuesto, elegido barriendo
    N. No se compara contra un fijo arbitrario."""
    best = None
    for N in FIJOS:
        r = corre(plant, "fijo", N, budget, rho, seeds)
        if r and (best is None or r["coste"] < best[1]["coste"]):
            best = (N, r)
    return best


def main() -> dict:
    plant = fl.thermal_loop()
    out = {"config": {"M": M, "T": T, "N_max": N_MAX,
                      "seeds_eval": len(SEEDS_EVAL)}}
    print("=" * 78)
    print(f"FLOTA DE M={M} LAZOS CON UN PRESUPUESTO DE COMPUTO COMPARTIDO")
    print("=" * 78)
    print("  lazo termico de colector: tau_f=2 min, tau_m=5 min, dt=30 s, "
          f"T={T} pasos (4 h)")
    print("  se penaliza la temperatura del ABSORBEDOR, a la que el caudal llega")
    print("  con retardo: es lo que hace que el horizonte importe")
    print(f"  presupuesto = suma de horizontes concedidos por paso; "
          f"peor caso M*N_max = {M*N_MAX}")

    # ---------------- sintonia del gobernador -----------------------------
    print("\n" + "-" * 78)
    print("  SINTONIA del gobernador (semillas 100-102, disjuntas de evaluacion)")
    print("-" * 78)
    best = None
    for tau in (4.0, 8.0, 14.7, 25.0):
        for a_ref in (0.3, 0.6, 0.9):
            tot = sum(corre(plant, "fugas", None, B, 0.0, SEEDS_TUNE,
                            tau=tau, a_ref=a_ref)["coste"] for B in (24, 32, 48))
            if best is None or tot < best[0]:
                best = (tot, tau, a_ref)
    _, TAU, AREF = best
    print(f"  tau = {TAU}, a_ref = {AREF}  (optimo interior de la rejilla)")
    out["sintonia"] = {"tau": TAU, "a_ref": AREF}

    # ---------------- H1 y H2 ---------------------------------------------
    for rho, nombre in ((0.0, "INDEPENDIENTES"), (1.0, "FRENTE COMUN")):
        print("\n" + "-" * 78)
        print(f"  perturbaciones {nombre} (rho = {rho:g}) -- coste A IGUAL PRESUPUESTO")
        print("-" * 78)
        print(f"  {'presup.':>8}{'B/(M N_max)':>13}{'mejor fijo':>13}{'coste fijo':>12}"
              f"{'gobernado':>11}{'mejora':>9}")
        filas = []
        for B in BUDGETS:
            Nf, rf = mejor_fijo(plant, B, rho, SEEDS_EVAL)
            rg = corre(plant, "fugas", None, B, rho, SEEDS_EVAL, tau=TAU, a_ref=AREF)
            mej = 100.0 * (rf["coste"] - rg["coste"]) / rf["coste"]
            st = paired([{"c": v} for v in rg["por_semilla"]],
                        [{"c": v} for v in rf["por_semilla"]], "c")
            print(f"  {B:>8}{B/(M*N_MAX):>13.2f}{('N='+str(Nf)):>13}"
                  f"{rf['coste']:>12.1f}{rg['coste']:>11.1f}{mej:>+8.2f}%"
                  + ("  *" if st.p < 0.05 else ""))
            filas.append({"B": B, "mejor_fijo": Nf, "coste_fijo": rf["coste"],
                          "coste_gob": rg["coste"], "mejora_pct": float(mej),
                          "t": float(st.t), "p": float(st.p),
                          "sem_fijo": rf["sem"], "sem_gob": rg["sem"],
                          "recorte": rg["recorte"]})
        out[f"rho_{rho:g}"] = filas
        pos = [f for f in filas if f["mejora_pct"] > 0 and f["p"] < 0.05]
        print(f"\n  el gobernador gana de forma significativa en {len(pos)}/{len(filas)} "
              f"presupuestos" + (f", el mayor margen {max(f['mejora_pct'] for f in pos):+.2f}%"
                                 if pos else ""))

    # ---------------- H3 ---------------------------------------------------
    print("\n" + "=" * 78)
    print("H3  REPARTIR POR DEMANDA FRENTE A REPARTIR A PARTES IGUALES")
    print("=" * 78)
    B = M * 4
    a = corre(plant, "fugas", None, B, 0.0, SEEDS_EVAL, igual=False, tau=TAU, a_ref=AREF)
    b = corre(plant, "fugas", None, B, 0.0, SEEDS_EVAL, igual=True, tau=TAU, a_ref=AREF)
    st = paired([{"c": v} for v in a["por_semilla"]],
                [{"c": v} for v in b["por_semilla"]], "c")
    print(f"  presupuesto escaso B = {B} (un cuarto del peor caso)")
    print(f"  proporcional a la demanda: {a['coste']:.1f}")
    print(f"  a partes iguales         : {b['coste']:.1f}")
    print(f"  mejora {100*(b['coste']-a['coste'])/b['coste']:+.2f}%  "
          f"t = {st.t:+.2f}  p = {st.p:.2e}")
    out["H3"] = {"proporcional": a["coste"], "igual": b["coste"],
                 "mejora_pct": float(100 * (b["coste"] - a["coste"]) / b["coste"]),
                 "t": float(st.t), "p": float(st.p)}

    print("\n" + "=" * 78)
    print("LECTURA")
    print("=" * 78)
    p0 = [f for f in out["rho_0"] if f["mejora_pct"] > 0 and f["p"] < 0.05]
    p1 = [f for f in out["rho_1"] if f["mejora_pct"] > 0 and f["p"] < 0.05]
    print(f"  H1 {'CONFIRMADA' if p0 else 'FALSADA'}: con lazos independientes el "
          f"gobernador gana en {len(p0)} presupuestos")
    print(f"  H2 {'CONFIRMADA' if len(p1) < len(p0) else 'FALSADA'}: con frente comun "
          f"gana solo en {len(p1)}")
    print(f"  H3 {'CONFIRMADA' if out['H3']['p'] < 0.05 and out['H3']['mejora_pct'] > 0 else 'FALSADA'}: "
          f"repartir por demanda mejora {out['H3']['mejora_pct']:+.2f}%")
    return out


if __name__ == "__main__":
    res = main()
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dst = os.path.join(here, "results")
    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(dst, "fleet.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    print(f"\nescrito {os.path.join(dst, 'fleet.json')}")
