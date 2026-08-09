"""Re-sintonia del CAMPO con rejilla ampliada: la derrota no puede venir de un
borde de rejilla.

POR QUE ESTE SCRIPT EXISTE
---------------------------
En `run_transport_mechanism.py` el campo de onda perdio contra la linea de
retardo pura (t=-5.19, p_holm=0.0023). Pero DOS de sus parametros sintonizados
cayeron en el BORDE de la rejilla: c = 0.60 (maximo del barrido) y gain = 1.6
(maximo). Una derrota con el perdedor pegado al borde de su propia rejilla no
vale: hay que ampliarla hasta que el optimo quede en el INTERIOR, y solo
entonces aceptar el resultado.

Se amplia el barrido del campo en las cuatro direcciones que pueden importar
(velocidad de propagacion c, amortiguamiento zeta, rigidez w0, ganancia del
mapa h -> alpha) y se anade un DESPLAZAMIENTO explicito del gate, para que el
campo pueda colocar su entrega en el instante que quiera igual que puede
hacerlo la linea de retardo. Si aun asi pierde, la derrota es real.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import run_transport_mechanism as base           # reutiliza arena y mecanismos
from ghi.stats import holm, paired


def make_field_mech_shift(branch):
    """Como el mecanismo de campo, pero con un DESPLAZAMIENTO temporal libre:
    se le concede al campo exactamente la misma libertad de colocacion que
    tiene la linea de retardo, para que la comparacion sea de FORMA de la
    senal, no de calibracion del instante."""
    def mech(src, T, i, par):
        H = par["_H"]
        h = H[:, i]
        k = int(round(par.get("shift", 0)))
        if k > 0:
            h = np.concatenate([np.zeros(k), h[:-k]])
        elif k < 0:
            h = np.concatenate([h[-k:], np.zeros(-k)])
        rng = h.max() - h.min()
        g = (h - h.min()) / rng if rng > 1e-9 else np.zeros_like(h)
        return base.alpha_from_gate(par["gain"] * g, par["a_lo"], par["a_hi"])
    return mech


def main() -> dict:
    plant, mpc = base.build_mpc()
    AL = np.round(np.linspace(0.0, 0.6, 4), 3)
    AH = np.round(np.linspace(0.5, 1.0, 4), 3)
    pares = [dict(a_lo=lo, a_hi=hi) for lo in AL for hi in AH if hi > lo]

    CS = (0.3, 0.6, 1.0, 1.6, 2.5, 4.0)          # antes llegaba solo a 0.6
    ZS = (0.2, 0.5, 1.0, 1.6)
    WS = (0.15, 0.5)
    GS = (1.0, 2.0, 4.0)                         # antes llegaba solo a 1.6
    SH = (-6, -3, 0, 3, 6)

    print("=" * 78)
    print("RE-SINTONIA DEL CAMPO CON REJILLA AMPLIADA (semillas de sintonia 100-102)")
    print("=" * 78)
    print(f"  c    in {CS}   (antes: hasta 0.6, el optimo se pego al borde)")
    print(f"  gain in {GS}   (antes: hasta 1.6, idem)")
    print(f"  zeta in {ZS} | w0 in {WS} | desplazamiento in {SH}")

    # Busqueda en DOS ETAPAS para que la rejilla ampliada sea ejecutable sin
    # perder cobertura: (1) parametros del campo con el par de pesos fijado al
    # que sintonizo la linea de retardo -- de modo que el campo no compite en
    # desventaja de pesos --, (2) refinado del par de pesos con el campo ya fijo.
    res = {}
    for branch, nombre in (("wave", "onda"), ("diffusion", "difusion")):
        mech = make_field_mech_shift(branch)
        g1 = [dict(a_lo=0.30, a_hi=0.90, w0=w, zeta=z, c=c,
                   D=(0.4 if branch == "diffusion" else 0.0), gain=g, shift=s,
                   gamma=1.0)
              for w in WS for z in ZS for c in CS for g in GS for s in SH]
        print(f"\n  {nombre}: etapa 1 = {len(g1)} configuraciones de campo")
        best = None
        for par in g1:
            c_, _ = base.evaluate(mpc, plant, mech, par, base.SEEDS_TUNE[:2], branch)
            if best is None or c_ < best[0]:
                best = (c_, par)
        g2 = [dict(best[1], a_lo=p["a_lo"], a_hi=p["a_hi"]) for p in pares]
        print(f"    etapa 2 = {len(g2)} pares de pesos con el campo ya fijado")
        best = None
        for par in g2:
            c_, _ = base.evaluate(mpc, plant, mech, par, base.SEEDS_TUNE, branch)
            if best is None or c_ < best[0]:
                best = (c_, par)
        p = best[1]
        borde = []
        if p["c"] in (CS[0], CS[-1]): borde.append("c")
        if p["gain"] in (GS[0], GS[-1]): borde.append("gain")
        if p["zeta"] in (ZS[0], ZS[-1]): borde.append("zeta")
        if p["shift"] in (SH[0], SH[-1]): borde.append("shift")
        print(f"    optimo: c={p['c']} zeta={p['zeta']} w0={p['w0']} gain={p['gain']} "
              f"shift={p['shift']} a=({p['a_lo']:.2f},{p['a_hi']:.2f})")
        print(f"    coste sintonia {best[0]:.1f} | parametros EN EL BORDE: "
              f"{borde if borde else 'ninguno (optimo interior)'}")
        res[nombre] = {"par": {k: v for k, v in p.items() if not k.startswith('_')},
                       "coste_sintonia": float(best[0]), "borde": borde}

    print("\n" + "=" * 78)
    print("EVALUACION en semillas 0-9 con el campo RE-SINTONIZADO")
    print("=" * 78)
    filas = {}
    for nombre, branch in (("onda", "wave"), ("difusion", "diffusion")):
        filas[nombre] = base.per_seed(mpc, plant, make_field_mech_shift(branch),
                                      res[nombre]["par"], base.SEEDS_EVAL, branch)
    # rivales, sintonizados como en el experimento original
    rivales = {}
    AL2 = np.round(np.linspace(0.0, 0.6, 7), 3)
    AH2 = np.round(np.linspace(0.3, 1.0, 8), 3)
    bg = [dict(a_lo=lo, a_hi=hi) for lo in AL2 for hi in AH2 if hi > lo]
    for nombre, mech, grid in (
            ("reactivo", base.mech_reactivo, bg),
            ("retardo-puro", base.mech_retardo,
             [dict(p, delta=d) for p in bg for d in (0, 1, 2, 3, 4, 5, 6, 8)]),
            ("oraculo", base.mech_oraculo,
             [dict(p, lead=L) for p in bg for L in (0, 2, 3, 4, 6, 8)])):
        best = None
        for par in grid:
            c_, _ = base.evaluate(mpc, plant, mech, par, base.SEEDS_TUNE)
            if best is None or c_ < best[0]:
                best = (c_, par)
        rivales[nombre] = best[1]
        filas[nombre] = base.per_seed(mpc, plant, mech, best[1], base.SEEDS_EVAL)

    b = np.mean([f["coste"] for f in filas["reactivo"]])
    print(f"\n  {'mecanismo':<16}{'coste':>11}{'vs reactivo':>13}{'% del oraculo':>15}")
    co = np.mean([f["coste"] for f in filas["oraculo"]])
    techo = b - co
    orden = sorted(filas, key=lambda k: np.mean([f["coste"] for f in filas[k]]))
    tabla = {}
    for k in orden:
        c_ = np.mean([f["coste"] for f in filas[k]])
        frac = 100.0 * (b - c_) / techo if techo > 0 else np.nan
        print(f"  {k:<16}{c_:>11.1f}{100*(b-c_)/b:>+12.2f}%{frac:>14.1f}%")
        tabla[k] = {"coste": float(c_), "vs_reactivo_pct": float(100 * (b - c_) / b),
                    "pct_del_oraculo": float(frac)}

    print("\n" + "=" * 78)
    print("CONTRASTES (pareados, Holm)")
    print("=" * 78)
    pares_t = [("onda", "retardo-puro", "T1: onda RE-SINTONIZADA vs retardo puro"),
               ("onda", "difusion", "T2: onda vs difusion"),
               ("onda", "reactivo", "T4b: onda vs solo local")]
    ts = holm([paired(filas[a], filas[bb], "coste", a_name=a, b_name=bb)
               for a, bb, _ in pares_t])
    contr = {}
    for (a, bb, txt), t in zip(pares_t, ts):
        print(f"  {txt:<44} delta={t.delta:+9.2f} t={t.t:+6.2f} "
              f"p_holm={t.p_holm:.4f} {'*' if t.significativo else ' '} "
              f"gana {a if t.delta > 0 else bb}")
        contr[f"{a}_vs_{bb}"] = {"delta": float(t.delta), "t": float(t.t),
                                 "p_holm": float(t.p_holm),
                                 "significativo": bool(t.significativo)}
    return {"resintonia": res, "tabla": tabla, "contrastes": contr}


if __name__ == "__main__":
    out = main()
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dst = os.path.join(here, "results")
    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(dst, "transport_retune.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(f"\nescrito {os.path.join(dst, 'transport_retune.json')}")
