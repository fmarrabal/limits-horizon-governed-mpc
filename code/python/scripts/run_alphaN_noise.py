"""Barrido de ruido en alpha: la prediccion falsable de la regla de agosto.

POR QUE ESTE SEGUNDO EXPERIMENTO, Y POR QUE NO ES P-HACKING
-----------------------------------------------------------
El experimento principal salio NEGATIVO para H2: con la senal alpha LIMPIA, el
primer orden gana en coste, computo y correlacion de asignacion, y el campo de
segundo orden TIMBRA (sube a 12, baja a 2 en mitad de la recuperacion, rompe el
certificado otra vez, se re-excita). Dos mecanismos identificados en las trazas:

  (1) el mapa de demanda SATURA para todo alpha < 0: la amplitud del kick no
      esta codificada en la senal, solo su DURACION -- y un integrador con
      fugas es exactamente el decodificador de duraciones;
  (2) con zeta = 0.5 la respuesta al pulso sobreoscila y el columpio hacia
      abajo provoca nuevas violaciones del certificado (timbre re-excitado).

Ese negativo es CONSISTENTE con la regla establecida en agosto:

    la inercia paga cuando la senal de demanda es RUIDOSA o rapida respecto de
    la planta; no paga cuando es limpia.

Pero una regla que solo explica a posteriori no vale nada. Su prediccion
falsable aqui es:

    H5 (PRE-REGISTRADA): anadiendo ruido de medida al alpha observado -- que es
    lo realista: alpha es un cociente de diferencias de funciones de valor
    evaluadas en estados MEDIDOS, lo mas sensible al ruido que existe en el
    lazo -- debe aparecer un CRUCE: el orden-1 gana con ruido bajo y el orden-2
    lo alcanza y lo supera a partir de algun sigma_alpha. Si el cruce NO
    aparece a ningun nivel de ruido, la regla de agosto queda TOCADA en este
    escenario y asi se reportara.

H6 (mecanismo, secundaria): con zeta = 1 (amortiguamiento critico) el timbre
    desaparece y la correlacion de asignacion del 2o orden se repara aun sin
    ruido; a cambio pierde la ventaja de fase que da la subamortiguacion. Si
    zeta = 1 no repara la correlacion, el diagnostico del timbre esta mal.

Todos los gobernadores ven EXACTAMENTE el mismo alpha ruidoso (misma semilla);
la correlacion de asignacion se evalua contra el |kick| VERDADERO.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
from scipy import stats as sps

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ghi.suboptimality import (GovOrder1, GovOrder2, GovThreshold, HorizonMPC,
                               closed_loop, make_kicks, trace_metrics)
from ghi.regulators import match_first_order
from scripts.run_alphaN import build, SCENARIOS


class NoisyAlphaLoop:
    """Envuelve un gobernador para inyectarle ruido gaussiano en el alpha
    observado. El flujo de ruido depende SOLO de (semilla, t), no del
    gobernador: todos ven la misma realizacion."""

    def __init__(self, gov, sigma: float, seed: int):
        self.gov, self.sigma = gov, sigma
        self.rng = np.random.default_rng(seed)
        self.name = gov.name

    def update(self, alpha):
        if alpha is not None and self.sigma > 0:
            alpha = alpha + self.sigma * self.rng.standard_normal()
        elif alpha is None and self.sigma > 0:
            # el ruido tambien corrompe el "no hay informacion": un alpha
            # comodo medido con ruido puede parecer deficitario
            alpha = 1.0 + self.sigma * self.rng.standard_normal()
        return self.gov.update(alpha)

    def reset(self):
        self.gov.reset()


def main(n_seeds: int = 10, T: int = 400,
         sigmas=(0.0, 0.25, 0.5, 1.0, 2.0)) -> dict:
    t0 = time.time()
    mpc = build()
    lo, hi, A_real = SCENARIOS["B_OOD_kick"]      # el escenario donde el negativo fue mas nitido
    tau, _ = match_first_order(0.7, 0.5, 1.0)

    makers = {
        "umbral":        lambda: GovThreshold(2, 16),
        "orden-1":       lambda: GovOrder1(2, 16, tau),
        "orden-2":       lambda: GovOrder2(2, 16, 0.7, 0.5, 0.5),
        "orden-2-z1":    lambda: GovOrder2(2, 16, 0.7, 1.0, 0.5),   # H6: sin timbre
    }

    print("=" * 78)
    print("BARRIDO DE RUIDO EN alpha  --  la prediccion falsable de la regla de agosto")
    print("=" * 78)
    print("escenario B_OOD_kick (kicks U[3.5,6.5]); mismo alpha ruidoso para todos")
    print("H5: debe aparecer un CRUCE orden-1 -> orden-2 al crecer sigma_alpha")
    print("H6: zeta=1 repara la correlacion del 2o orden ya sin ruido\n")

    out: dict = {}
    for sig in sigmas:
        out[f"{sig}"] = {}
        print(f"### sigma_alpha = {sig}")
        print(f"  {'gobernador':<12}{'computo':>9}{'coste':>10}{'fr<0.1':>8}{'corr':>7}{'div':>5}")
        for gname, mk in makers.items():
            rows = []
            for seed in range(n_seeds):
                kicks = make_kicks(T, seed=1000 * seed + 7, size_lo=lo, size_hi=hi)
                gov = NoisyAlphaLoop(mk(), sig, seed=777 * seed + 13)
                tr = closed_loop(mpc, gov, T, kicks, A_real=A_real)
                rows.append(trace_metrics(tr, kicks))
            out[f"{sig}"][gname] = rows
            c = np.array([r["computo"] for r in rows], float)
            j = np.array([r["coste"] for r in rows])
            fb = np.array([r["frac_alpha_bajo"] for r in rows])
            co = np.array([r["corr_asignacion"] for r in rows])
            dv = sum(r["divergio"] for r in rows)
            print(f"  {gname:<12}{c.mean():>9.0f}{j.mean():>10.1f}"
                  f"{np.nanmean(fb):>8.3f}{np.nanmean(co):>7.3f}{dv:>5d}")
        print()

    # ------------------- el contraste que decide H5 ------------------------
    print("=" * 78)
    print("H5: coste orden-2 vs orden-1, pareado, por nivel de ruido")
    print("    (delta > 0 => gana el orden-2; el CRUCE es el cambio de signo)")
    for sig in sigmas:
        a = np.array([r["coste"] for r in out[f"{sig}"]["orden-2"]])
        b = np.array([r["coste"] for r in out[f"{sig}"]["orden-1"]])
        d = b - a
        tt = sps.ttest_rel(b, a)
        star = " *" if tt.pvalue < 0.05 else ""
        print(f"  sigma={sig:<5} delta={d.mean():+10.1f}+-{d.std(ddof=1)/np.sqrt(len(d)):<9.1f}"
              f" t={tt.statistic:+7.2f} p={tt.pvalue:.4f}{star}")

    print("\nH6: corr de asignacion sin ruido, orden-2 (z=0.5) vs orden-2-z1 (z=1.0)")
    a = np.array([r["corr_asignacion"] for r in out["0.0"]["orden-2"]])
    b = np.array([r["corr_asignacion"] for r in out["0.0"]["orden-2-z1"]])
    ok = ~np.isnan(a) & ~np.isnan(b)
    if ok.sum() >= 3:
        tt = sps.ttest_rel(b[ok], a[ok])
        print(f"  corr z=0.5: {np.nanmean(a):+.3f}   corr z=1.0: {np.nanmean(b):+.3f}"
              f"   delta={np.mean(b[ok]-a[ok]):+.3f}  t={tt.statistic:+.2f}  p={tt.pvalue:.4f}")

    print(f"\ntiempo: {time.time()-t0:.1f}s")
    return out


if __name__ == "__main__":
    res = main()
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dst = os.path.join(here, "results")
    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(dst, "alphaN_noise.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    print(f"escrito {os.path.join(dst, 'alphaN_noise.json')}")
