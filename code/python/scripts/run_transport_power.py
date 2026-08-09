"""Confirmacion con POTENCIA del contraste que decide, y sin bordes de rejilla.

QUE QUEDA POR RESOLVER
-----------------------
Con el operador corregido, la onda alcanza el 98.4% del oraculo y el rival
IGUALADO en grados de libertad (retardo + conformado de 2o orden, 7 parametros
como el campo) el 90.9%. Pero el contraste sale t = +0.96, p_holm = 0.36: con
10 semillas el estudio NO TIENE POTENCIA para resolver una diferencia del 1.2%
del coste. Y ambos contendientes tocaron borde de rejilla en `shift`.

Aqui se cierran las dos cosas a la vez:
  * rejillas extendidas en los ejes que apretaban (shift, gain, zs)
  * 30 semillas de evaluacion en vez de 10, que es lo que decide si el liderato
    numerico de la onda es real o es ruido

La regla de lectura se fija ANTES de ejecutar:
  * si la onda separa con p < 0.05 -> la estructura del campo aporta algo que
    un filtro conformado a mano con los mismos grados de libertad no alcanza
  * si NO separa -> la conclusion es que el campo IGUALA al mejor filtro
    conformado, que es exactamente lo que A17 predice (es un banco de filtros
    LTI: no puede ser mas expresivo), y su merito queda en que UN solo juego de
    parametros da el nucleo correcto para TODAS las distancias sin sintonizar
    por salto
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import run_transport_mechanism as base
import run_transport_matched as mm
from ghi.stats import holm, paired

SEEDS_EVAL30 = tuple(range(30))


def main() -> dict:
    plant, mpc = base.build_mpc()
    pares = [dict(a_lo=lo, a_hi=hi) for lo in (0.0, 0.2, 0.4)
             for hi in (0.67, 0.83, 1.0) if hi > lo]
    SH = (-12, -9, -6, -3, 0, 3)          # extendido hacia atras
    GS = (1.0, 2.0, 3.0, 5.0)             # extendido hacia arriba
    print("=" * 78)
    print("SINTONIA SIN BORDES (rejillas extendidas donde apretaban)")
    print("=" * 78)

    # --- campo de onda ---
    g1 = [dict(a_lo=0.0, a_hi=0.83, w0=w, zeta=z, c=c, D=0.0, gain=g, shift=s,
               gamma=1.0)
          for w in (0.08, 0.15, 0.3) for z in (0.5, 1.0, 1.6)
          for c in (0.08, 0.15, 0.3, 0.6) for g in GS for s in SH]
    best = None
    for par in g1:
        c_, _ = base.evaluate(mpc, plant, mm.make_field_mech_shift("wave"), par,
                              base.SEEDS_TUNE[:2], "wave")
        if best is None or c_ < best[0]:
            best = (c_, par)
    p_w, b_w = mm.tune(mpc, plant, mm.make_field_mech_shift("wave"),
                       [dict(best[1], **q) for q in pares], "wave", "onda",
                       {"gain": GS, "shift": SH, "c": (0.08, 0.15, 0.3, 0.6),
                        "zeta": (0.5, 1.0, 1.6), "w0": (0.08, 0.15, 0.3)})

    # --- rival igualado ---
    g1 = [dict(a_lo=0.0, a_hi=0.83, delta=d, ws=w, zs=z, gain=g, shift=s)
          for d in (0, 2, 4, 6, 8) for w in (0.08, 0.15, 0.3, 0.6)
          for z in (0.3, 0.7, 1.2, 2.0, 3.0) for g in GS for s in SH]
    best = None
    for par in g1:
        c_, _ = base.evaluate(mpc, plant, mm.mech_matched, par, base.SEEDS_TUNE[:2])
        if best is None or c_ < best[0]:
            best = (c_, par)
    p_m, b_m = mm.tune(mpc, plant, mm.mech_matched,
                       [dict(best[1], **q) for q in pares], None,
                       "retardo-conformado",
                       {"delta": (0, 2, 4, 6, 8), "ws": (0.08, 0.15, 0.3, 0.6),
                        "zs": (0.3, 0.7, 1.2, 2.0, 3.0), "gain": GS, "shift": SH})

    bg = [dict(a_lo=lo, a_hi=hi)
          for lo in np.round(np.linspace(0, .6, 7), 3)
          for hi in np.round(np.linspace(.3, 1., 8), 3) if hi > lo]
    p_r, _ = mm.tune(mpc, plant, base.mech_reactivo, bg, None, "reactivo")
    p_d, _ = mm.tune(mpc, plant, base.mech_retardo,
                     [dict(q, delta=d) for q in bg for d in (0, 2, 4, 6, 8)],
                     None, "retardo-puro", {"delta": (0, 2, 4, 6, 8)})
    p_o, _ = mm.tune(mpc, plant, base.mech_oraculo,
                     [dict(q, lead=L) for q in bg for L in (0, 2, 3, 4, 6, 8)],
                     None, "oraculo")

    print("\n" + "=" * 78)
    print(f"EVALUACION CON {len(SEEDS_EVAL30)} SEMILLAS (potencia)")
    print("=" * 78)
    filas = {
        "onda": base.per_seed(mpc, plant, mm.make_field_mech_shift("wave"), p_w,
                              SEEDS_EVAL30, "wave"),
        "retardo-conformado": base.per_seed(mpc, plant, mm.mech_matched, p_m, SEEDS_EVAL30),
        "retardo-puro": base.per_seed(mpc, plant, base.mech_retardo, p_d, SEEDS_EVAL30),
        "reactivo": base.per_seed(mpc, plant, base.mech_reactivo, p_r, SEEDS_EVAL30),
        "oraculo": base.per_seed(mpc, plant, base.mech_oraculo, p_o, SEEDS_EVAL30),
    }
    b = np.mean([f["coste"] for f in filas["reactivo"]])
    co = np.mean([f["coste"] for f in filas["oraculo"]])
    techo = b - co
    print(f"\n  {'mecanismo':<20}{'coste':>11}{'vs reactivo':>13}{'% del oraculo':>15}")
    tabla = {}
    for k in sorted(filas, key=lambda k: np.mean([f["coste"] for f in filas[k]])):
        v = np.array([f["coste"] for f in filas[k]], float)
        c_ = float(v.mean())
        sem = float(v.std(ddof=1) / np.sqrt(len(v)))
        print(f"  {k:<20}{c_:>11.1f}{100*(b-c_)/b:>+12.2f}%{100*(b-c_)/techo:>14.1f}%")
        # se guardan los valores POR SEMILLA: sin ellos no se pueden poner
        # barras de error honestas en las figuras
        tabla[k] = {"coste": c_, "sem": sem,
                    "pct_oraculo": float(100 * (b - c_) / techo),
                    "pct_oraculo_sem": float(100 * sem / techo),
                    "por_semilla": v.tolist()}

    print("\n" + "=" * 78)
    print("CONTRASTES (30 semillas, pareados, Holm)")
    print("=" * 78)
    pr = [("onda", "retardo-conformado", "onda vs RIVAL IGUALADO (7 vs 7)"),
          ("onda", "retardo-puro", "onda vs retardo puro (7 vs 3)"),
          ("retardo-conformado", "retardo-puro", "conformar ayuda al rival?"),
          ("oraculo", "onda", "cuanto le falta a la onda para el techo")]
    ts = holm([paired(filas[a], filas[bb], "coste", a_name=a, b_name=bb)
               for a, bb, _ in pr])
    contr = {}
    for (a, bb, txt), t in zip(pr, ts):
        print(f"  {txt:<44} delta={t.delta:+9.2f} t={t.t:+7.2f} "
              f"p_holm={t.p_holm:.4f} {'*' if t.significativo else ' '} "
              f"gana {a if t.delta > 0 else bb}")
        contr[f"{a}_vs_{bb}"] = {"delta": float(t.delta), "t": float(t.t),
                                 "p_holm": float(t.p_holm),
                                 "significativo": bool(t.significativo)}
    key = contr["onda_vs_retardo-conformado"]
    print("\n" + "=" * 78)
    if key["significativo"] and key["delta"] > 0:
        print("""  LA ONDA SEPARA del rival igualado con 30 semillas. La ventaja no es
  libertad de sintonia: es estructura.""")
    else:
        print("""  LA ONDA NO SEPARA del rival igualado ni con 30 semillas. La lectura
  honesta es que el campo IGUALA al mejor filtro conformado -- que es lo que
  A17 obliga, porque un operador LTI no crea informacion -- y que su merito
  esta en generar con UN solo juego de parametros el nucleo correcto para
  todas las distancias, sin sintonizar salto a salto.""")
    print("=" * 78)
    return {"onda": p_w, "matched": p_m, "bordes": {"onda": b_w, "matched": b_m},
            "tabla": tabla, "contrastes": contr}


if __name__ == "__main__":
    r = main()
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dst = os.path.join(here, "results")
    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(dst, "transport_power.json"), "w") as fh:
        json.dump(r, fh, indent=1, default=float)
    print(f"\nescrito {os.path.join(dst, 'transport_power.json')}")
