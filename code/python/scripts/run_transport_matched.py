"""EL CONTRASTE DECISIVO: campo frente a un rival IGUALADO EN GRADOS DE LIBERTAD.

POR QUE ESTE SCRIPT DECIDE EL PROGRAMA
---------------------------------------
Con el operador del campo CORREGIDO (`forcing="source"`, ver A18) la onda pasa a
batir a la linea de retardo pura: 87.1% del oraculo frente al 71.4%, t=+137.5.
Es el primer resultado positivo del programa GHI en seis arenas, y por eso es el
que mas hay que atacar.

El ataque obvio, y es correcto: LA COMPARACION NO ESTABA IGUALADA. La linea de
retardo pura tiene TRES parametros (delta, a_lo, a_hi); el campo tiene SIETE
(c, zeta, w0, gain, shift, a_lo, a_hi). Ganar con cuatro grados de libertad de
mas no demuestra nada sobre el operador de onda.

Y hay un argumento teorico que obliga a hacerlo bien (A17): con beta = 0 el
campo es EXACTAMENTE un banco de filtros LTI, luego lo que entrega a cada nodo
es la fuente convolucionada con un nucleo fijo de retardo-mas-ensanchamiento. Un
rival que parametrice libremente esa misma familia CONTIENE al campo. Por tanto:

  * si el rival igualado EMPATA o gana -> la ventaja de la onda era libertad de
    sintonia, no estructura, y el programa GHI se cierra en negativo
  * si el campo sigue ganando con el rival igualado -> la ventaja esta en la
    ESTRUCTURA: un solo juego de parametros genera simultaneamente el nucleo
    correcto para TODAS las distancias, mientras que el rival tiene que acertar
    con una familia impuesta a mano

RIVAL IGUALADO: retardo por salto + conformado de 2o orden
-----------------------------------------------------------
    g_i = normaliza( filtro2(  src retrasado delta*i ,  ws, zs ) ),  desplazado
Parametros: delta, ws, zs, gain, shift, a_lo, a_hi = SIETE, los mismos que el
campo, y con la misma normalizacion por nodo, de modo que ninguna de las dos
partes gana por escala.

Se corrige ademas el borde de rejilla en c (antes el optimo se pegaba a 0.3, el
minimo del barrido) extendiendo hacia abajo.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import run_transport_mechanism as base
from ghi.stats import holm, paired


def second_order_lp(x, ws, zs):
    """Filtro paso-bajo de 2o orden discreto, la MISMA estructura de polos que
    la rama de onda del campo. Es lo que hace justa la comparacion: no se
    enfrenta un 2o orden contra un 1er orden."""
    a = np.zeros_like(x); v = 0.0; y = 0.0
    for t in range(len(x)):
        acc = ws * ws * (x[t] - y) - 2.0 * zs * ws * v
        v += acc
        y += v
        a[t] = y
    return a


def _norm(h):
    r = h.max() - h.min()
    return (h - h.min()) / r if r > 1e-9 else np.zeros_like(h)


def _shift(h, k):
    k = int(round(k))
    if k > 0:
        return np.concatenate([np.zeros(k), h[:-k]])
    if k < 0:
        return np.concatenate([h[-k:], np.zeros(-k)])
    return h


def mech_matched(src, T, i, par):
    """RIVAL IGUALADO: retardo por salto + conformado de 2o orden + ganancia +
    desplazamiento. Siete parametros, los mismos que el campo."""
    k = int(round(par["delta"] * i))
    x = np.concatenate([np.zeros(k), src[:-k]]) if k > 0 else src.copy()
    y = second_order_lp(x, par["ws"], par["zs"])
    g = _norm(_shift(y, par["shift"]))
    return base.alpha_from_gate(par["gain"] * g, par["a_lo"], par["a_hi"])


def make_field_mech_shift(branch):
    def mech(src, T, i, par):
        g = _norm(_shift(par["_H"][:, i], par.get("shift", 0)))
        return base.alpha_from_gate(par["gain"] * g, par["a_lo"], par["a_hi"])
    return mech


def tune(mpc, plant, mech, grid, branch=None, etiqueta="", bordes=None):
    best = None
    for par in grid:
        c, _ = base.evaluate(mpc, plant, mech, par, base.SEEDS_TUNE, branch)
        if best is None or c < best[0]:
            best = (c, par)
    p = best[1]
    enborde = [k for k, vals in (bordes or {}).items()
               if p.get(k) in (vals[0], vals[-1])]
    print(f"  {etiqueta:<20} coste {best[0]:9.1f} | "
          + " ".join(f"{k}={p[k]}" for k in sorted(p) if not k.startswith('_'))
          + f" | borde: {enborde if enborde else 'ninguno'}")
    return p, enborde


def main() -> dict:
    plant, mpc = base.build_mpc()
    AL = (0.0, 0.2, 0.4)
    AH = (0.67, 0.83, 1.0)
    pares = [dict(a_lo=lo, a_hi=hi) for lo in AL for hi in AH if hi > lo]

    CS = (0.08, 0.15, 0.3, 0.6, 1.0)      # extendido HACIA ABAJO (antes minimo 0.3)
    ZS = (0.2, 0.5, 1.0, 1.6)
    WS = (0.08, 0.15, 0.5)
    GS = (1.0, 2.0, 3.0)
    SH = (-6, -3, 0, 3, 6)
    DL = (0, 2, 4, 6, 8)
    WSF = (0.08, 0.15, 0.3, 0.6)
    ZSF = (0.3, 0.7, 1.2)

    print("=" * 78)
    print("SINTONIA IGUALADA (semillas 100-102). Siete parametros en ambos bandos.")
    print("=" * 78)
    out = {}

    # --- campo, con c extendido hacia abajo -------------------------------
    for branch, nombre in (("wave", "onda"), ("diffusion", "difusion")):
        mech = make_field_mech_shift(branch)
        g1 = [dict(a_lo=0.20, a_hi=0.83, w0=w, zeta=z, c=c,
                   D=(0.4 if branch == "diffusion" else 0.0), gain=g, shift=s,
                   gamma=1.0)
              for w in WS for z in ZS for c in CS for g in GS for s in SH]
        best = None
        for par in g1:
            c_, _ = base.evaluate(mpc, plant, mech, par, base.SEEDS_TUNE[:2], branch)
            if best is None or c_ < best[0]:
                best = (c_, par)
        g2 = [dict(best[1], **pp) for pp in pares]
        p, borde = tune(mpc, plant, mech, g2, branch, nombre,
                        {"c": CS, "zeta": ZS, "gain": GS, "shift": SH, "w0": WS})
        out[nombre] = {"par": {k: v for k, v in p.items() if not k.startswith('_')},
                       "borde": borde}

    # --- rival IGUALADO ----------------------------------------------------
    g1 = [dict(a_lo=0.20, a_hi=0.83, delta=d, ws=w, zs=z, gain=g, shift=s)
          for d in DL for w in WSF for z in ZSF for g in GS for s in SH]
    best = None
    for par in g1:
        c_, _ = base.evaluate(mpc, plant, mech_matched, par, base.SEEDS_TUNE[:2])
        if best is None or c_ < best[0]:
            best = (c_, par)
    g2 = [dict(best[1], **pp) for pp in pares]
    p_m, borde_m = tune(mpc, plant, mech_matched, g2, None, "retardo-conformado",
                        {"delta": DL, "ws": WSF, "zs": ZSF, "gain": GS, "shift": SH})
    out["retardo-conformado"] = {"par": p_m, "borde": borde_m}

    # --- rivales de referencia --------------------------------------------
    bg = [dict(a_lo=lo, a_hi=hi)
          for lo in np.round(np.linspace(0, .6, 7), 3)
          for hi in np.round(np.linspace(.3, 1., 8), 3) if hi > lo]
    p_r, _ = tune(mpc, plant, base.mech_reactivo, bg, None, "reactivo")
    p_d, _ = tune(mpc, plant, base.mech_retardo,
                  [dict(q, delta=d) for q in bg for d in DL], None, "retardo-puro",
                  {"delta": DL})
    p_o, _ = tune(mpc, plant, base.mech_oraculo,
                  [dict(q, lead=L) for q in bg for L in (0, 2, 3, 4, 6, 8)], None,
                  "oraculo")

    print("\n" + "=" * 78)
    print("EVALUACION (semillas 0-9, no usadas en la sintonia)")
    print("=" * 78)
    filas = {
        "onda": base.per_seed(mpc, plant, make_field_mech_shift("wave"),
                              out["onda"]["par"], base.SEEDS_EVAL, "wave"),
        "difusion": base.per_seed(mpc, plant, make_field_mech_shift("diffusion"),
                                  out["difusion"]["par"], base.SEEDS_EVAL, "diffusion"),
        "retardo-conformado": base.per_seed(mpc, plant, mech_matched, p_m, base.SEEDS_EVAL),
        "retardo-puro": base.per_seed(mpc, plant, base.mech_retardo, p_d, base.SEEDS_EVAL),
        "reactivo": base.per_seed(mpc, plant, base.mech_reactivo, p_r, base.SEEDS_EVAL),
        "oraculo": base.per_seed(mpc, plant, base.mech_oraculo, p_o, base.SEEDS_EVAL),
    }
    b = np.mean([f["coste"] for f in filas["reactivo"]])
    co = np.mean([f["coste"] for f in filas["oraculo"]])
    techo = b - co
    print(f"\n  {'mecanismo':<20}{'coste':>11}{'vs reactivo':>13}{'% del oraculo':>15}")
    tabla = {}
    for k in sorted(filas, key=lambda k: np.mean([f["coste"] for f in filas[k]])):
        c_ = np.mean([f["coste"] for f in filas[k]])
        print(f"  {k:<20}{c_:>11.1f}{100*(b-c_)/b:>+12.2f}%{100*(b-c_)/techo:>14.1f}%")
        tabla[k] = {"coste": float(c_), "pct_oraculo": float(100 * (b - c_) / techo)}

    print("\n" + "=" * 78)
    print("EL CONTRASTE QUE DECIDE")
    print("=" * 78)
    pr = [("onda", "retardo-conformado", "onda vs RIVAL IGUALADO (7 vs 7 parametros)"),
          ("onda", "retardo-puro", "onda vs retardo puro (7 vs 3)"),
          ("onda", "difusion", "onda vs difusion (nulo de mecanismo)"),
          ("retardo-conformado", "retardo-puro", "conformar ayuda al rival?")]
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

    tt = contr["onda_vs_retardo-conformado"]
    print("\n" + "=" * 78)
    if tt["delta"] > 0 and tt["significativo"]:
        print("""  EL CAMPO SOBREVIVE AL RIVAL IGUALADO. La ventaja no era libertad de
  sintonia: con los mismos siete grados de libertad, un retardo mas conformado
  de 2o orden ajustado a mano NO alcanza al campo. Lo que el campo aporta es
  ESTRUCTURA: un solo juego de parametros genera a la vez el nucleo correcto
  para todas las distancias de la cadena.""")
    else:
        print("""  EL CAMPO NO SOBREVIVE AL RIVAL IGUALADO. Su ventaja sobre la linea de
  retardo pura era libertad de sintonia, no estructura: un retardo conformado
  con los mismos grados de libertad lo alcanza. Coherente con A17 (el campo es
  un banco de filtros LTI y no crea informacion).""")
    print("=" * 78)
    return {"sintonia": out, "tabla": tabla, "contrastes": contr}


if __name__ == "__main__":
    r = main()
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dst = os.path.join(here, "results")
    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(dst, "transport_matched.json"), "w") as fh:
        json.dump(r, fh, indent=1, default=float)
    print(f"\nescrito {os.path.join(dst, 'transport_matched.json')}")
