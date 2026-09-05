"""La flota, evaluacion DEFINITIVA: sintonia auditable + computo gastado.

POR QUE EXISTE ESTE FICHERO (19-ago-2026)
-----------------------------------------
Una revision adversarial encontro que `fleet3.json` -- el fichero del que sale
el resultado de cabecera del Resultado II -- no tenia script que lo generase, y
que su sintonia (tau=30, a_ref -> 0) caia FUERA de la rejilla de `run_fleet2.py`
(tau in {4,8,14.7,25}, a_ref in {0.3,0.6,0.9}). Un arbitro no podia distinguir
"corregimos la fuga de contabilidad" de "extendimos la rejilla despues de ver la
metrica". Este script cierra ese agujero: barre una rejilla que CONTIENE la
sintonia publicada, archiva el barrido entero, y comprueba explicitamente las
dos afirmaciones que el manuscrito hace sobre ella.

SOBRE a_ref -> 0. El mapa deficit-alpha -> horizonte pide N_max cuando
alpha <= 0 y N_min cuando alpha >= a_ref. Al hacer a_ref -> 0+ el mapa degenera
en un ESCALON: N_max solo cuando el certificado se rompe. No es un punto de
rejilla cualquiera: es el LIMITE de la familia, y es el mecanismo que el paper
describe (detector de certificado). Como la disciplina del proyecto rechaza los
optimos en borde de rejilla, el script comprueba lo unico que legitima este:
que el coste sea PLANO en un entorno del limite (meseta), de modo que la
eleccion no dependa de donde se corte la rejilla.
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
B_TUNE = (24, 32, 48)
TAUS = (3.0, 4.5, 6.0, 8.0, 11.0, 15.0, 22.0, 30.0, 40.0, 55.0)
AREFS = (0.9, 0.6, 0.3, 0.1, 0.03, 0.01, 1e-6)


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
            "por_semilla": [float(c) for c in cs],
            "comp_por_semilla": [float(c) for c in comp]}


def frontera_en(comp, fijos):
    xs = np.array([f["computo"] for f in fijos]); ys = np.array([f["coste"] for f in fijos])
    o = np.argsort(xs); xs, ys = xs[o], ys[o]
    if comp <= xs[0] or comp >= xs[-1]:
        return None
    return float(np.interp(comp, xs, ys))


def frontera_por_semilla(comp_s, fijos, k):
    """Frontera de horizonte fijo interpolada EN LA SEMILLA k, evaluada en el
    computo que el gobernador gasto en esa misma semilla. Sin esto el contraste
    pareado no compara nada: compararia una semilla contra una media."""
    xs = np.array([f["computo"] for f in fijos])
    ys = np.array([f["por_semilla"][k] for f in fijos])
    o = np.argsort(xs); xs, ys = xs[o], ys[o]
    if comp_s <= xs[0] or comp_s >= xs[-1]:
        return None
    return float(np.interp(comp_s, xs, ys))


def main() -> dict:
    plant = fl.thermal_loop()
    out = {"config": {"M": M, "T": T, "N_max": N_MAX, "seeds": len(SEEDS_EVAL),
                      "taus": list(TAUS), "arefs": list(AREFS)}}
    print("=" * 78)
    print("FLOTA: SINTONIA AUDITABLE + COMPARACION A COMPUTO GASTADO")
    print("=" * 78)

    # ---------------- barrido de sintonia (semillas disjuntas) --------------
    # EL OBJETIVO ES LA METRICA DE EVALUACION, calculada de la misma manera:
    # hueco relativo del gobernador frente a la frontera de horizonte fijo DE
    # LA MISMA SEMILLA, interpolada en el computo que el gobernador GASTO en
    # esa semilla, promediado en semillas y sumado sobre B_TUNE.
    #
    # La version anterior (19-ago) sumaba coste bruto a presupuesto nominal.
    # Ese es exactamente el criterio que la Fuga 5 del manuscrito declara
    # haber abandonado ("the tuning objective must be the evaluation metric,
    # computed the same way"), y una revision adversarial (21-ago) encontro
    # que el script lo seguia usando mientras el texto afirmaba lo contrario.
    fijos_tune = []
    for N in FIJOS:
        r = corre(plant, "fijo", N, None, 0.0, SEEDS_TUNE)
        if r:
            fijos_tune.append(dict(r, N=N))

    def objetivo(g):
        """Hueco relativo medio frente a la frontera a computo GASTADO, semilla
        a semilla. Un punto cuyo gasto cae fuera del rango de la frontera no
        se puede puntuar y queda descartado (inf), igual que en la evaluacion."""
        huecos = []
        for k, (c, cm) in enumerate(zip(g["por_semilla"], g["comp_por_semilla"])):
            fk = frontera_por_semilla(cm, fijos_tune, k)
            if fk is None:
                return float("inf")
            huecos.append((c - fk) / fk)
        return float(np.mean(huecos))

    print(f"\n  barrido de sintonia en semillas {SEEDS_TUNE}, presupuestos "
          f"{B_TUNE}, rho=0")
    print(f"  objetivo = suma sobre B del hueco relativo medio frente a la "
          f"frontera a computo gastado (la metrica de evaluacion; menor es mejor)")
    print(f"  {'tau':>6}" + "".join(f"{a:>10.4g}" for a in AREFS))
    malla = {}
    for tau in TAUS:
        fila = []
        for a_ref in AREFS:
            tot = 0.0
            for B in B_TUNE:
                r = corre(plant, "fugas", None, B, 0.0, SEEDS_TUNE, tau=tau, a_ref=a_ref)
                tot += objetivo(r) if r else float("inf")
            malla[(tau, a_ref)] = tot
            fila.append(tot)
        print(f"  {tau:>6.1f}" + "".join(f"{100*v:>+10.2f}" for v in fila))
    (TAU, AREF), mejor = min(malla.items(), key=lambda kv: kv[1])
    print(f"\n  optimo del barrido: tau={TAU}, a_ref={AREF:g}  "
          f"(hueco acumulado {100*mejor:+.2f}%)")

    # las dos afirmaciones que el manuscrito hace sobre esta sintonia
    tau_int = TAUS[0] < TAU < TAUS[-1]
    fila_tau = [malla[(TAU, a)] for a in AREFS]
    monotona = all(fila_tau[i] >= fila_tau[i + 1] - 1e-9 for i in range(len(AREFS) - 1))
    meseta = [malla[(TAU, a)] for a in AREFS if a <= 0.03]
    disp = (max(meseta) - min(meseta)) / min(meseta)
    print(f"  tau interior a la rejilla: {tau_int}")
    print(f"  coste monotono no creciente al bajar a_ref: {monotona}")
    print(f"  MESETA en a_ref <= 0.03: dispersion relativa {disp:.2%} "
          f"({'plana: el borde es inmaterial' if disp < 0.01 else 'NO plana'})")
    out["sintonia"] = {"tau": TAU, "a_ref": AREF, "tau_interior": bool(tau_int),
                       "monotona_en_a_ref": bool(monotona),
                       "meseta_disp_rel": float(disp),
                       "malla": {f"{t}|{a:g}": v for (t, a), v in malla.items()}}

    # ---------------------------- evaluacion -------------------------------
    for rho in (0.0, 1.0):
        print("\n" + "-" * 78)
        print(f"  rho={rho:g}  ({'INDEPENDIENTES' if rho == 0 else 'FRENTE COMUN'})")
        print("-" * 78)
        fijos = []
        for N in FIJOS:
            r = corre(plant, "fijo", N, None, rho, SEEDS_EVAL)
            if r:
                fijos.append(dict(r, N=N))
        print(f"  {'B':>6}{'comp.real':>11}{'coste gob':>11}{'frontera':>10}"
              f"{'dif':>9}{'t':>8}{'p':>10}")
        filas, gana = [], 0
        for B in BUDGETS:
            g = corre(plant, "fugas", None, B, rho, SEEDS_EVAL, tau=TAU, a_ref=AREF)
            if not g:
                continue
            fr = frontera_en(g["computo"], fijos)
            if fr is None:
                continue
            dif = 100.0 * (g["coste"] - fr) / fr
            # contraste pareado REAL: en cada semilla, el gobernador contra la
            # frontera de horizonte fijo DE ESA SEMILLA, interpolada en el
            # computo que el gobernador gasto EN ESA SEMILLA
            par_g, par_f = [], []
            for k, (c, cm) in enumerate(zip(g["por_semilla"], g["comp_por_semilla"])):
                fk = frontera_por_semilla(cm, fijos, k)
                if fk is not None:
                    par_g.append({"c": c}); par_f.append({"c": fk})
            pr = paired(par_g, par_f, "c") if len(par_g) >= 2 else None
            sig = "*" if pr and pr.p < 0.05 else ""
            if dif < 0 and sig and pr.delta > 0:
                gana += 1
            print(f"  {B:>6}{g['computo']:>11.1f}{g['coste']:>11.1f}{fr:>10.1f}"
                  f"{dif:>+8.2f}%{(pr.t if pr else 0):>8.2f}{(pr.p if pr else 1):>10.2e}  {sig}")
            filas.append({"B": B, "computo": g["computo"], "coste": g["coste"],
                          "frontera": fr, "dif_pct": float(dif),
                          "t": float(pr.t) if pr else None,
                          "p": float(pr.p) if pr else None,
                          "n_pareado": len(par_g),
                          "comp_por_semilla": g["comp_por_semilla"],
                          "por_semilla": g["por_semilla"]})
        print(f"  gana significativamente en {gana}/{len(filas)} puntos")
        out[f"rho_{rho:g}"] = filas
        out[f"frontera_rho_{rho:g}"] = fijos
        out[f"gana_rho_{rho:g}"] = gana
    return out


if __name__ == "__main__":
    res = main()
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dst = os.path.join(here, "results")
    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(dst, "fleet3.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    print(f"\nescrito {os.path.join(dst, 'fleet3.json')}")
