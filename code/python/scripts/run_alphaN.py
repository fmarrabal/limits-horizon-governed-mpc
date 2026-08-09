"""alpha_N de Grune-Pannek como objetivo de modulacion del campo homeostatico.

LA PREGUNTA
-----------
El principio de agosto dice: el campo debe modular lo que el horizonte del MPC
no puede ver. alpha_N lo cumple por definicion -- mide la discrepancia entre el
paisaje de valor de horizonte finito y el lazo real, y solo se revela al cerrar
el lazo. Aqui se cierra por primera vez el lazo

    alpha_N observado  ->  campo  ->  horizonte N(t)

y se compara el campo inercial contra la regla de umbral de la literatura de
horizonte adaptativo, contra el mapa sin memoria, y contra el primer orden
IGUALADO en Nyquist.

HIPOTESIS PRE-REGISTRADAS (declaradas antes de mirar ningun numero)
-------------------------------------------------------------------
H1 (dentro del sobre): todos los gobernadores adaptativos ahorran computo
   grande frente a fijo-N_max con coste comparable; las diferencias ENTRE
   adaptativos son pequenas. Se espera un cuasi-nulo, como en el HBP.
H2 (fuera del sobre -- la hipotesis central, transferida del HBP): el campo de
   segundo orden mantiene la correlacion dificultad->computo asignado mejor que
   umbral / orden-0 / orden-1, pareado por semilla. PUEDE SALIR NEGATIVA: si la
   senal alpha es lo bastante limpia, el primer orden podria bastar. Se
   reportara tal cual salga.
H3: el fijo-N sintonizado al sobre de diseno (fijo-8, que cubre kicks <= 4
   segun la tabla de calibracion) se rompe fuera del sobre; los adaptativos no.
H4: el orden-0 puro reacciona mas rapido pero con mas varianza y mas
   sobreasignacion; el umbral con histeresis es competitivo dentro del sobre
   (por eso es el estandar de la literatura) y pierde trazado fino fuera.

CALIBRACION (tabla N*(v), medida en el sanity y reproducida aqui)
-----------------------------------------------------------------
Con umax = 1 el doble integrador necesita ~2v pasos de horizonte para frenar
un kick de velocidad v con certificado sano: v=1->N2, v=2->N4, v=3->N6,
v=4->N8, v=5->N10, v=6->N12. Kicks de diseno U[1.5,3.5] -> N* en [4,8];
kicks OOD U[3.5,6.5] -> N* en [8,14]. El rango [2,16] cubre ambos con margen.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
from scipy import stats as sps

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ghi.plant import Plant
from ghi import stats as gs
from ghi.suboptimality import (GovFixed, HorizonMPC, closed_loop, make_kicks,
                               matched_governors, trace_metrics)


def build() -> HorizonMPC:
    plant = Plant(A=np.array([[1.0, 1.0], [0.0, 1.0]]),
                  B=np.array([[0.5], [1.0]]),
                  xmax=np.array([1e6, 1e6]), umax=np.array([1.0]),
                  name="doble-integrador")
    return HorizonMPC(plant, Q=np.diag([1.0, 0.1]), R=np.array([[0.01]]),
                      N_min=2, N_max=16)


SCENARIOS = {
    # (size_lo, size_hi, A_real o None)
    "A_diseno":       (1.5, 3.5, None),
    "B_OOD_kick":     (3.5, 6.5, None),
    "C_OOD_modelo":   (1.5, 3.5, np.array([[1.0, 1.0], [0.0, 1.02]])),
    "D_OOD_ambos":    (3.5, 6.5, np.array([[1.0, 1.0], [0.0, 1.02]])),
}


def main(n_seeds: int = 10, T: int = 400, w0: float = 0.7, zeta: float = 0.5,
         theta: float = 0.5, a_ref: float = 0.6) -> dict:
    t0 = time.time()

    print("=" * 78)
    print("alpha_N COMO OBJETIVO DE MODULACION  --  campo inercial vs la literatura")
    print("=" * 78)
    print(__doc__.split("CALIBRACION")[0].split("HIPOTESIS")[1])

    govs = matched_governors(2, 16, w0=w0, zeta=zeta, theta=theta, a_ref=a_ref)
    govs["fijo-8"] = lambda: GovFixed(2, 16, 8)      # el sintonizado al sobre

    bank: dict = {}
    for sname, (lo, hi, A_real) in SCENARIOS.items():
        bank[sname] = {}
        print(f"\n### {sname}  (kicks U[{lo},{hi}]"
              + (", A22_real=1.02" if A_real is not None else "") + ")")
        print(f"  {'gobernador':<12}{'computo':>9}{'coste':>10}{'N_med':>7}"
              f"{'a_min':>8}{'fr<0.1':>8}{'corr':>7}{'div':>5}")
        for gname, mk in govs.items():
            rows = []
            for seed in range(n_seeds):
                # MPC NUEVO por corrida: los OSQP cacheados hacen warm-start del
                # solve anterior y el pareado por semilla dejaria de ser exacto
                # (hallazgo de la revision adversarial)
                mpc = build()
                kicks = make_kicks(T, seed=1000 * seed + 7, size_lo=lo, size_hi=hi)
                tr = closed_loop(mpc, mk(), T, kicks, A_real=A_real)
                rows.append(trace_metrics(tr, kicks))
            bank[sname][gname] = rows
            # los estadisticos de la tabla se calculan SOLO sobre corridas
            # completas: una traza divergida es una suma truncada, no una
            # observacion comparable (hallazgo FATAL de la revision)
            full = [r for r in rows if not r["divergio"]]
            dv = sum(r["divergio"] for r in rows)
            if full:
                c = np.array([r["computo"] for r in full], float)
                j = np.array([r["coste"] for r in full])
                nm = np.array([r["N_medio"] for r in full])
                am = np.array([r["alpha_min_limpio"] for r in full])
                fb = np.array([r["frac_alpha_bajo"] for r in full])
                co = np.array([r["corr_asignacion"] for r in full])
                print(f"  {gname:<12}{c.mean():>9.0f}{j.mean():>10.1f}{nm.mean():>7.2f}"
                      f"{np.nanmean(am):>8.2f}{np.nanmean(fb):>8.3f}"
                      f"{np.nanmean(co):>7.3f}{dv:>5d}")
            else:
                print(f"  {gname:<12}{'-':>9}{'-':>10}{'-':>7}{'-':>8}{'-':>8}{'-':>7}{dv:>5d}")

    # ----------------------------- frontera fijo-N -------------------------
    print("\n" + "=" * 78)
    print("FRONTERA DE HORIZONTE FIJO (computo medio, coste medio) por escenario")
    frontier: dict = {}
    for sname in ("A_diseno", "D_OOD_ambos"):
        lo, hi, A_real = SCENARIOS[sname]
        frontier[sname] = {}
        print(f"  {sname}:")
        for N in (2, 4, 6, 8, 10, 12, 16):
            cs, js, dv = [], [], 0
            for seed in range(n_seeds):
                kicks = make_kicks(T, seed=1000 * seed + 7, size_lo=lo, size_hi=hi)
                tr = closed_loop(mpc, GovFixed(2, 16, N), T, kicks, A_real=A_real)
                m = trace_metrics(tr, kicks)
                cs.append(m["computo"]); js.append(m["coste"]); dv += m["divergio"]
            frontier[sname][N] = dict(computo=float(np.mean(cs)),
                                      coste=float(np.mean(js)), divergencias=dv)
            print(f"    N={N:>2}: computo={np.mean(cs):>7.0f}  coste={np.mean(js):>9.1f}"
                  f"  divergencias={dv}")

    # ------------------------- contrastes pareados -------------------------
    # Politica de divergencias (revision adversarial, hallazgo FATAL): un par
    # (semilla) solo entra en el contraste si AMBOS brazos completaron la
    # corrida. Las divergencias se reportan aparte como conteos.
    def paired_bank(bank_in, comparisons, metrics):
        out = {}
        for met in metrics:
            fam = []
            for sname, regs in bank_in.items():
                for a_name, b_name in comparisons:
                    if a_name not in regs or b_name not in regs:
                        continue
                    ra, rb = regs[a_name], regs[b_name]
                    keep = [i for i in range(min(len(ra), len(rb)))
                            if not ra[i]["divergio"] and not rb[i]["divergio"]]
                    pr = gs.paired([ra[i] for i in keep], [rb[i] for i in keep],
                                   met, f"{sname}:{a_name}(n={len(keep)})", b_name)
                    if pr is not None:
                        fam.append(pr)
            out[met] = gs.holm(fam)
        return out

    print("\n" + "=" * 78)
    print("CONTRASTES PAREADOS POR SEMILLA (solo pares completos; Holm por familia)")
    print("DIVERGENCIAS POR GOBERNADOR Y ESCENARIO (fallo terminal, se reporta aparte):")
    for sname, regs in bank.items():
        divs = {g: sum(r["divergio"] for r in rows) for g, rows in regs.items()}
        divs = {g: d for g, d in divs.items() if d > 0}
        if divs:
            print(f"  {sname}: " + ", ".join(f"{g}={d}/{n_seeds}" for g, d in divs.items()))
    comparisons = [("orden-2", "orden-1-nyq"), ("orden-2", "orden-1-set"),
                   ("orden-2", "umbral-up4"), ("orden-2", "orden-0"),
                   ("orden-2-z1", "orden-2"), ("orden-1-nyq", "umbral-up4")]
    table = paired_bank(bank, comparisons,
                        ("coste", "computo", "frac_alpha_bajo", "corr_asignacion"))
    # OJO con el signo: en corr_asignacion MAS es MEJOR; en el resto menos es
    # mejor. gs.paired define delta = media(b) - media(a) (gana `a` si delta>0
    # para metricas a minimizar). Para corr hay que leerlo al reves; se marca.
    for met, fam in table.items():
        nota = "  [MAS es MEJOR: leer delta al reves]" if met == "corr_asignacion" else ""
        print(f"\n--- {met}{nota}")
        for pr in fam:
            print("  " + str(pr))

    print(f"\ntiempo: {time.time()-t0:.1f}s")
    return {"bank": bank, "frontier": frontier,
            "contrasts": {m: [vars(p) for p in fam] for m, fam in table.items()},
            "config": dict(n_seeds=n_seeds, T=T, w0=w0, zeta=zeta, theta=theta,
                           a_ref=a_ref, N_min=2, N_max=16)}


if __name__ == "__main__":
    res = main()
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dst = os.path.join(here, "results")
    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(dst, "alphaN.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    print(f"escrito {os.path.join(dst, 'alphaN.json')}")
