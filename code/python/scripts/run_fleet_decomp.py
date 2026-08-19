"""Descomposicion del gobernador de flota: adaptacion TEMPORAL vs REPARTO.

QUE ESTABA MAL (encontrado por revision adversarial, 19-ago-2026)
------------------------------------------------------------------
El brazo "igualitario" fijaba Ng = [budget//M]*M en TODOS los pasos, ignorando
por completo la peticion del gobernador. Eso no es "el mismo gobernador con
otro reparto": es un HORIZONTE FIJO N = budget/M, y de hecho coincidia digito a
digito con la frontera fija (fleet_decomp rho_0 B=24 igual = 3053.79433...
identico a fleet3 frontera N=3). La diferencia proporcional-igualitario contenia
por tanto el mecanismo temporal Y el reparto, luego no aislaba nada, y ademas
los dos brazos gastaban computo distinto.

LA CORRECCION
-------------
El brazo igualitario conserva el gobernador y su adaptacion temporal: gasta en
cada paso el MISMO total de horizonte que concederia el reparto proporcional, y
solo cambia COMO lo distribuye (a partes iguales, con el resto rotando en t).
Asi la unica diferencia entre los dos brazos es el reparto entre lazos.

Y como el gasto REAL sigue difiriendo un poco (el instrumento se cobra cuando el
horizonte de un lazo cambia, y el reparto equitativo cambia menos), la
comparacion se hace donde el proyecto la hace siempre: sobre el computo
REALMENTE GASTADO, interpolando la curva del brazo igualitario en el computo del
punto proporcional. El contraste es pareado por semilla.
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
SEEDS = (0, 1, 2, 3, 4, 5, 6, 7)
BUDGETS = [24, 32, 40, 48, 64, 96]
# el brazo igualitario gasta menos por presupuesto (cambia menos de
# horizonte, luego paga menos solves de instrumento): su curva necesita
# presupuestos mayores para cubrir el computo de los puntos proporcionales
BUDGETS_IG = [24, 32, 40, 48, 64, 96, 128, 192, 256]
FIJOS = [2, 3, 4, 5, 6, 8, 10, 12, 16]
TAU, AREF = 11.0, 0.1          # la sintonia que el barrido de run_fleet3 elige


def campo(seed, rho):
    return fl.CloudField(M=M, T=T, rho=rho, depth=1.2, seed=seed).build()


def corre_fijo(plant, N, rho):
    cs, comp = [], []
    for s in SEEDS:
        r = fl.run_fleet(plant, Q, R, campo(s, rho), "fijo", None, N_min=N_MIN,
                         N_max=N_MAX, N_fixed=N)
        if r.divergio or not np.isfinite(r.coste):
            return None
        cs.append(r.coste); comp.append(r.computo_medio)
    return {"coste": float(np.mean(cs)), "computo": float(np.mean(comp)),
            "por_semilla": [float(c) for c in cs],
            "comp_por_semilla": [float(c) for c in comp], "N": N}


def corre(plant, budget, rho, igual):
    cs, comp, atan = [], [], []
    for s in SEEDS:
        r = fl.run_fleet(plant, Q, R, campo(s, rho), "fugas", budget,
                         N_min=N_MIN, N_max=N_MAX, reparto_igual=igual,
                         tau=TAU, a_ref=AREF)
        if r.divergio or not np.isfinite(r.coste):
            return None
        cs.append(r.coste); comp.append(r.computo_medio); atan.append(r.frac_recortado)
    return {"coste": float(np.mean(cs)), "computo": float(np.mean(comp)),
            "ata": float(np.mean(atan)),
            "por_semilla": [float(c) for c in cs],
            "comp_por_semilla": [float(c) for c in comp]}


def interp_semilla(curva, comp_s, k):
    xs = np.array([c["comp_por_semilla"][k] for c in curva])
    ys = np.array([c["por_semilla"][k] for c in curva])
    o = np.argsort(xs); xs, ys = xs[o], ys[o]
    if comp_s <= xs[0] or comp_s >= xs[-1]:
        return None
    return float(np.interp(comp_s, xs, ys))


def main() -> dict:
    plant = fl.thermal_loop()
    out = {"config": {"M": M, "T": T, "tau": TAU, "a_ref": AREF,
                      "seeds": len(SEEDS), "budgets": BUDGETS}}
    print("=" * 78)
    print("DESCOMPOSICION: adaptacion temporal vs reparto entre lazos")
    print("  brazo igualitario = MISMO gobernador, mismo total por paso,")
    print("  reparto a partes iguales; comparacion a computo GASTADO")
    print("=" * 78)
    for rho in (0.0, 1.0):
        prop = {B: corre(plant, B, rho, False) for B in BUDGETS}
        igual = {B: corre(plant, B, rho, True) for B in BUDGETS_IG}
        curva_ig = [v for v in igual.values() if v]
        print(f"\n--- rho={rho:g} ---")
        print(f"  {'B':>5}{'ata':>7}{'prop coste':>12}{'prop comp':>11}"
              f"{'igual@comp':>12}{'ventaja':>10}{'t':>8}{'p':>10}")
        filas = []
        for B in BUDGETS:
            p, g = prop[B], igual[B]
            if not p or not g:
                continue
            ig_en = interp_semilla(curva_ig, p["computo"], 0)  # solo informativo
            par_p, par_i = [], []
            for k, (c, cm) in enumerate(zip(p["por_semilla"], p["comp_por_semilla"])):
                ik = interp_semilla(curva_ig, cm, k)
                if ik is not None:
                    par_p.append({"c": c}); par_i.append({"c": ik})
            if len(par_p) < 2:
                # El brazo igualitario NO PUEDE gastar tanto computo: cambia de
                # horizonte menos veces y por tanto paga menos solves de
                # instrumento, de modo que su curva (coste vs computo gastado)
                # se agota antes. Donde no hay solape se reporta la comparacion
                # a igual PRESUPUESTO -- que en este diseno concede el mismo
                # total de horizonte por paso -- declarando el gasto de cada
                # brazo para que se vea la direccion del sesgo.
                gg = igual.get(B)
                pr2 = paired([{"c": c} for c in p["por_semilla"]],
                             [{"c": c} for c in gg["por_semilla"]], "c")
                vent2 = 100.0 * (gg["coste"] - p["coste"]) / gg["coste"]
                print(f"  {B:>5}{p['ata']:>6.0%}{p['coste']:>12.1f}"
                      f"{p['computo']:>11.1f}{gg['coste']:>12.1f}"
                      f"{vent2:>+9.2f}%{pr2.t:>8.2f}{pr2.p:>10.2e}  "
                      f"{'*' if pr2.p < 0.05 else ''} [a igual presupuesto; "
                      f"gasto {p['computo']:.1f} vs {gg['computo']:.1f}]")
                filas.append({"B": B, "ata": p["ata"], "coste_prop": p["coste"],
                              "computo_prop": p["computo"],
                              "coste_igual_en_comp": gg["coste"],
                              "computo_igual": gg["computo"],
                              "ventaja_pct": vent2, "t": float(pr2.t),
                              "p": float(pr2.p), "n": len(p["por_semilla"]),
                              "modo": "igual_presupuesto"})
                continue
            pr = paired(par_p, par_i, "c")
            m_ig = float(np.mean([r["c"] for r in par_i]))
            m_pr = float(np.mean([r["c"] for r in par_p]))
            vent = 100.0 * (m_ig - m_pr) / m_ig
            sig = "*" if pr and pr.p < 0.05 else ""
            print(f"  {B:>5}{p['ata']:>6.0%}{m_pr:>12.1f}{p['computo']:>11.1f}"
                  f"{m_ig:>12.1f}{vent:>+9.2f}%{pr.t:>8.2f}{pr.p:>10.2e}  {sig}")
            filas.append({"B": B, "ata": p["ata"], "coste_prop": m_pr,
                          "computo_prop": p["computo"], "coste_igual_en_comp": m_ig,
                          "ventaja_pct": vent, "t": float(pr.t), "p": float(pr.p),
                          "n": len(par_p), "modo": "igual_computo_gastado"})
        out[f"rho_{rho:g}"] = filas
        out[f"curva_igual_rho_{rho:g}"] = [
            {"B": B, "coste": v["coste"], "computo": v["computo"]}
            for B, v in igual.items() if v]
        out[f"budgets_igual"] = BUDGETS_IG

        # ---- COMPONENTE TEMPORAL: igualitario (gobernado, reparto plano)
        # ---- contra HORIZONTE FIJO, a igual computo gastado. Es la mitad que
        # ---- la comparacion proporcional-vs-igualitario NO puede ver, porque
        # ---- los dos brazos de aquella llevan adaptacion temporal.
        fijos = [f for f in (corre_fijo(plant, N, rho) for N in FIJOS) if f]
        print("")
        print("  componente TEMPORAL (igualitario vs fijo, a igual computo):")
        print(f"  {'B':>5}{'igual coste':>13}{'igual comp':>12}{'fijo@comp':>11}"
              f"{'ventaja':>10}{'t':>8}{'p':>10}")
        filas_t = []
        for B in BUDGETS:
            g = igual.get(B)
            if not g:
                continue
            par_g, par_f = [], []
            for k, (c, cm) in enumerate(zip(g["por_semilla"], g["comp_por_semilla"])):
                fk = interp_semilla(fijos, cm, k)
                if fk is not None:
                    par_g.append({"c": c}); par_f.append({"c": fk})
            if len(par_g) < 2:
                continue
            pr = paired(par_g, par_f, "c")
            m_f = float(np.mean([r["c"] for r in par_f]))
            m_g = float(np.mean([r["c"] for r in par_g]))
            vent = 100.0 * (m_f - m_g) / m_f
            print(f"  {B:>5}{m_g:>13.1f}{g['computo']:>12.1f}{m_f:>11.1f}"
                  f"{vent:>+9.2f}%{pr.t:>8.2f}{pr.p:>10.2e}  "
                  f"{'*' if pr.p < 0.05 else ''}")
            filas_t.append({"B": B, "coste_igual": m_g, "computo": g["computo"],
                            "coste_fijo_en_comp": m_f, "ventaja_pct": vent,
                            "t": float(pr.t), "p": float(pr.p), "n": len(par_g)})
        out[f"temporal_rho_{rho:g}"] = filas_t
    print("\n" + "=" * 78)
    print("LECTURA")
    print("=" * 78)
    v0 = [f["ventaja_pct"] for f in out["rho_0"]]
    v1 = [f["ventaja_pct"] for f in out["rho_1"]]
    t0 = [f["ventaja_pct"] for f in out["temporal_rho_0"]]
    t1 = [f["ventaja_pct"] for f in out["temporal_rho_1"]]
    print(f"  REPARTO   rho=0: {min(v0):+.2f}% a {max(v0):+.2f}%   "
          f"rho=1: {min(v1):+.2f}% a {max(v1):+.2f}%")
    print(f"  TEMPORAL  rho=0: {min(t0):+.2f}% a {max(t0):+.2f}%   "
          f"rho=1: {min(t1):+.2f}% a {max(t1):+.2f}%")
    print("  (el reparto solo puede pagar si los lazos se rompen en momentos")
    print("   DISTINTOS; con frente comun no hay a quien quitarle horizonte)")
    return out


if __name__ == "__main__":
    res = main()
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dst = os.path.join(here, "results")
    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(dst, "fleet_decomp.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    print(f"\nescrito {os.path.join(dst, 'fleet_decomp.json')}")
