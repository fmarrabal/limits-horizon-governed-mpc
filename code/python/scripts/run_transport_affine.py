"""El transporte, con la familia AFIN dentro de la competicion.

QUE ESTABA MAL, Y ES EL FALLO MAS INSTRUCTIVO DE LOS TRES
----------------------------------------------------------
Los tres mecanismos que competian aplicaban al nodo i un retardo LINEAL en i:

    k_i = delta * i

El "oraculo clarividente" que figuraba como techo inalcanzable resulto ser, al
mirarlo, CAUSAL: abre en t=200/206/212 para llegadas en 206/212/218, es decir
nunca antes de que la fuente mida. Y se reproduce EXACTAMENTE (diferencia 0.0)
con un retardo AFIN que solo lee el pasado:

    k_i = max(0, TAU*i - lead)

O sea que el techo era alcanzable y el experimento habia dejado fuera al mejor
miembro de su propia familia.

POR QUE LA FAMILIA LINEAL ES ESTRUCTURALMENTE INCAPAZ
------------------------------------------------------
El nodo i sufre la perturbacion en t0 + i*TAU y la fuente la mide en t0, luego
la informacion esta disponible con i*TAU pasos de antelacion: CRECIENTE con la
distancia. Pero la antelacion UTIL son 2-4 pasos, la MISMA para todos. El
retardo que hay que aplicar es entonces i*TAU - lead: afin, con termino
independiente. Una familia lineal sin termino independiente solo puede acertar
en un nodo: con delta = TAU la antelacion es cero en todos, y con delta < TAU
crece con la distancia y a los nodos lejanos el aviso les llega demasiado
pronto.

LA PREGUNTA NUEVA
-----------------
Si la familia afin gana a la lineal, entonces la ENTREGA SI IMPORTA -- no la
forma del nucleo (onda, conformado o escalon), sino que la antelacion sea
CONSTANTE entre nodos. Eso es lo contrario de lo que concluia la version
anterior, y es un resultado de diseno, no una anecdota.

El techo pasa a ser un clarividente GENUINAMENTE acausal (antelacion libre por
nodo, mayor que la que la fuente puede dar), etiquetado como inalcanzable.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_transport_mechanism as base
import run_transport_matched as mm
import run_transport_causal2 as c2
from ghi import transport as tp
from ghi.stats import holm, paired

SEEDS_EVAL = tuple(range(30))
TAU = base.TAU_HOP


def _ret(src, k):
    k = int(round(k))
    return np.concatenate([np.zeros(k), src[:-k]]) if k > 0 else src.copy()


# ---------------------------------------------------------------- mecanismos

def mech_lineal(src, T, i, par):
    """La familia que competia: retardo proporcional al numero de saltos."""
    return base.alpha_from_gate(_ret(src, par["delta"] * i), par["a_lo"], par["a_hi"])


def mech_afin(src, T, i, par):
    """La familia que faltaba: retardo afin, antelacion CONSTANTE entre nodos."""
    k = max(0, int(round(par["delta"] * i - par["lead"])))
    return base.alpha_from_gate(_ret(src, k), par["a_lo"], par["a_hi"])


def mech_afin_conformado(src, T, i, par):
    """Afin + conformado de 2o orden (ganancia unidad en continua, sin normalizar)."""
    k = max(0, int(round(par["delta"] * i - par["lead"])))
    y = mm.second_order_lp(_ret(src, k), par["ws"], par["zs"])
    return base.alpha_from_gate(par["gain"] * y, par["a_lo"], par["a_hi"])


def mech_clarividente(src, T, i, par):
    """TECHO INALCANZABLE: antelacion libre respecto de la llegada al nodo i,
    sin la restriccion de derivarse de lo que la fuente puede medir. Para
    antelaciones mayores que TAU es ACAUSAL por construccion, y por eso se
    etiqueta como cota y no como mecanismo."""
    g = np.zeros(T)
    base_i = _ret(src, TAU * i)
    L = int(par["lead"])
    for t in range(T):
        if base_i[min(T - 1, t + L)] > 0.5:
            g[t] = 1.0
    return base.alpha_from_gate(g, par["a_lo"], par["a_hi"])


# ---------------------------------------------------------------- sintonia

def tune(mpc, plant, mech, grid, branch, etiqueta, rejillas, exigir_causal=True):
    best, n_ac = None, 0
    for par in grid:
        if exigir_causal and not tp.is_causal(c2.gate_of(mech, par, branch)):
            n_ac += 1
            continue
        c, _ = c2.evaluate(mpc, plant, mech, par, base.SEEDS_TUNE, branch)
        if np.isfinite(c) and (best is None or c < best[0]):
            best = (c, par)
    p = best[1]
    borde = [k for k, v in rejillas.items()
             if k in p and p[k] in (v[0], v[-1]) and len(v) > 1]
    print(f"  {etiqueta:<22} coste {best[0]:9.1f} | "
          + " ".join(f"{k}={p[k]}" for k in sorted(p) if not k.startswith("_"))
          + f"\n  {'':<22} {n_ac} acausales descartadas | "
          + (f"BORDE: {borde}" if borde else "optimo interior"))
    return p, borde


def main() -> dict:
    plant, mpc = base.build_mpc()
    AL = np.round(np.linspace(0.0, 0.9, 10), 3)     # AMPLIADA: antes llegaba a 0.6
    AH = np.round(np.linspace(0.3, 1.0, 8), 3)
    bg = [dict(a_lo=lo, a_hi=hi) for lo in AL for hi in AH if hi > lo]
    DL = (0, 2, 4, 6, 8)
    # extendida tras detectar lead=8 en el borde: ahora el optimo es interior
    LE = (0, 2, 3, 4, 6, 8, 10, 12)
    REJ = {"delta": DL, "lead": LE, "a_lo": AL, "a_hi": AH}

    print("=" * 78)
    print("SINTONIA (semillas 100-102). Todo causal salvo el techo, que se marca.")
    print("=" * 78)
    out, bordes = {}, {}

    p_r, b_r = tune(mpc, plant, base.mech_reactivo, bg, None, "reactivo (local)", REJ)
    # misma densidad de pesos (bg[::2]) para TODOS los competidores: sin esto
    # la comparacion mezcla presupuestos de sintonia distintos
    p_l, b_l = tune(mpc, plant, mech_lineal,
                    [dict(q, delta=d) for q in bg[::2] for d in DL], None,
                    "retardo LINEAL", REJ)
    p_a, b_a = tune(mpc, plant, mech_afin,
                    [dict(q, delta=d, lead=L) for q in bg[::2] for d in DL for L in LE],
                    None, "retardo AFIN", REJ)
    g_ac = [dict(q, delta=d, lead=L, ws=w, zs=z, gain=1.0)
            for q in bg[::2] for d in (4, 6) for L in LE
            for w in (0.15, 0.3, 0.6) for z in (0.7, 1.2, 2.0)]
    p_ac_, b_ac = tune(mpc, plant, mech_afin_conformado, g_ac, None,
                       "AFIN + conformado", REJ)

    g_w = [dict(a_lo=0.0, a_hi=0.83, w0=w, zeta=z, c=c, D=0.0, gain=g, shift=s,
                gamma=1.0)
           for w in (0.08, 0.15, 0.3) for z in (0.5, 1.0, 1.6)
           for c in (0.08, 0.15, 0.3, 0.6) for g in (1.0, 2.0, 3.0, 5.0)
           for s in (0, 2, 4, 6, 9, 12)]
    mech_w = c2.make_field_mech_causal("wave")
    best = None
    for par in g_w:
        if not tp.is_causal(c2.gate_of(mech_w, par, "wave")):
            continue
        c_, _ = c2.evaluate(mpc, plant, mech_w, par, base.SEEDS_TUNE[:2], "wave")
        if np.isfinite(c_) and (best is None or c_ < best[0]):
            best = (c_, par)
    p_w, b_w = tune(mpc, plant, mech_w,
                    [dict(best[1], **q) for q in bg[::2]], "wave", "onda", REJ)

    p_c, _ = tune(mpc, plant, mech_clarividente,
                  [dict(q, lead=L) for q in bg for L in (8, 10, 12, 16, 20, 24)],
                  None, "clarividente (techo)", REJ, exigir_causal=False)
    print("  (el clarividente usa antelaciones > TAU, luego es ACAUSAL a proposito:")
    print("   es una cota superior, no un mecanismo implementable)")

    print("\n" + "=" * 78)
    print(f"EVALUACION CON {len(SEEDS_EVAL)} SEMILLAS")
    print("=" * 78)
    filas = {
        "reactivo": base.per_seed(mpc, plant, base.mech_reactivo, p_r, SEEDS_EVAL),
        "retardo-lineal": base.per_seed(mpc, plant, mech_lineal, p_l, SEEDS_EVAL),
        "retardo-afin": base.per_seed(mpc, plant, mech_afin, p_a, SEEDS_EVAL),
        "afin-conformado": base.per_seed(mpc, plant, mech_afin_conformado, p_ac_, SEEDS_EVAL),
        "onda": c2.per_seed(mpc, plant, mech_w, p_w, SEEDS_EVAL, "wave"),
        "clarividente": base.per_seed(mpc, plant, mech_clarividente, p_c, SEEDS_EVAL),
    }
    b = np.mean([f["coste"] for f in filas["reactivo"]])
    co = np.mean([f["coste"] for f in filas["clarividente"]])
    techo = b - co
    print(f"\n  {'mecanismo':<20}{'coste':>11}{'sem':>8}{'vs local':>10}{'% techo':>10}")
    tabla = {}
    for k in sorted(filas, key=lambda k: np.mean([f["coste"] for f in filas[k]])):
        v = np.array([f["coste"] for f in filas[k]], float)
        c_, sem = float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)))
        print(f"  {k:<20}{c_:>11.1f}{sem:>8.1f}{100*(b-c_)/b:>+9.2f}%"
              f"{100*(b-c_)/techo:>9.1f}%")
        tabla[k] = {"coste": c_, "sem": sem, "por_semilla": v.tolist(),
                    "vs_local_pct": float(100 * (b - c_) / b),
                    "pct_techo": float(100 * (b - c_) / techo),
                    "pct_techo_sem": float(100 * sem / techo)}

    print("\n" + "=" * 78)
    print("CONTRASTES (pareados, Holm) CON MINIMO DETECTABLE")
    print("=" * 78)
    pr = [("retardo-afin", "retardo-lineal", "AFIN vs LINEAL  <== el que decide"),
          ("retardo-afin", "onda", "afin vs onda"),
          ("retardo-afin", "afin-conformado", "conformar ayuda al afin?"),
          ("retardo-afin", "reactivo", "afin vs solo local"),
          ("clarividente", "retardo-afin", "cuanto falta para el techo")]
    ts = holm([paired(filas[a], filas[bb], "coste", a_name=a, b_name=bb)
               for a, bb, _ in pr])
    contr = {}
    for (a, bb, txt), t in zip(pr, ts):
        mde = 2.8 * t.sem
        print(f"  {txt:<36} delta={t.delta:+9.1f} t={t.t:+7.2f} "
              f"p={t.p_holm:.4f}{'*' if t.significativo else ' '} | "
              f"detectable >= {mde:.0f} ({100*mde/b:.2f}%)")
        contr[f"{a}_vs_{bb}"] = {"delta": float(t.delta), "t": float(t.t),
                                 "p_holm": float(t.p_holm),
                                 "significativo": bool(t.significativo),
                                 "mde": float(mde), "mde_pct": float(100 * mde / b)}

    print("\n" + "=" * 78)
    print("LECTURA")
    print("=" * 78)
    key = contr["retardo-afin_vs_retardo-lineal"]
    if key["significativo"] and key["delta"] > 0:
        print("  LA ENTREGA SI IMPORTA. El retardo AFIN bate al LINEAL: lo que")
        print("  cuenta no es la forma del nucleo sino que la antelacion sea")
        print("  CONSTANTE entre nodos, y eso exige termino independiente.")
    else:
        print("  el afin NO separa del lineal; habria que revisar el diagnostico")
    out = {"sintonia": {"reactivo": p_r, "lineal": p_l, "afin": p_a,
                        "afin_conformado": p_ac_, "onda": p_w, "clarividente": p_c},
           "bordes": {"reactivo": b_r, "lineal": b_l, "afin": b_a,
                      "afin_conformado": b_ac, "onda": b_w},
           "tabla": tabla, "contrastes": contr}
    return out


if __name__ == "__main__":
    r = main()
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dst = os.path.join(here, "results")
    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(dst, "transport_affine.json"), "w") as fh:
        json.dump(r, fh, indent=1, default=float)
    print(f"\nescrito {os.path.join(dst, 'transport_affine.json')}")
