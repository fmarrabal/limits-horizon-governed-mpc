"""El experimento del transporte, REHECHO con guarda de causalidad.

POR QUE HAY QUE REHACERLO
--------------------------
La revision adversarial encontro una FUGA ACAUSAL en la version anterior. El
mapa de la senal al peso admitia un desplazamiento libre, y la rejilla de
sintonia incluia valores NEGATIVOS, que ADELANTAN la senal. Los dos mecanismos
ganadores explotaron eso: con la sonda de un pulso que la fuente mide en t=200,
el retardo conformado encendia su puerta en t=194 y la onda en t=192. Es decir,
ambos usaban informacion que todavia no existia. La linea de retardo pura, en
cambio, era causal (subia en t=202): el unico que jugaba limpio era el rival
mas simple.

Aqui toda configuracion pasa por `tp.is_causal` antes de entrar en el barrido,
exactamente igual que ya se descartaban las integraciones de Verlet inestables.
Lo que salga de aqui sustituye a la Seccion VI del manuscrito.
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
from ghi import transport as tp
from ghi.stats import holm, paired

SEEDS_EVAL = tuple(range(30))
SH = (0, 2, 4, 6, 9, 12)          # SOLO retardos: un desplazamiento negativo
GS = (1.0, 2.0, 3.0, 5.0)         # adelantaria la senal (ver run_transport_causal)


def gate_of(mech, par, branch=None):
    """Envuelve un mecanismo para poder someterlo a la sonda de causalidad."""
    def g(src, T, i):
        p = dict(par)
        if branch is not None:
            H = base._field_gate(src, T, branch, p)
            if H is None:
                return np.full(T, p["a_lo"])
            p["_H"] = H
        return mech(src, T, i, p)
    return g


def tune(mpc, plant, mech, grid, branch, etiqueta, bordes):
    """Sintonia con DOS filtros previos: estabilidad del integrador y
    causalidad. Se reporta cuantas configuraciones se descartan por cada uno,
    porque ese conteo es parte del resultado."""
    best, n_ac, n_in = None, 0, 0
    for par in grid:
        if not tp.is_causal(gate_of(mech, par, branch)):
            n_ac += 1
            continue
        c, _ = base.evaluate(mpc, plant, mech, par, base.SEEDS_TUNE, branch)
        if not np.isfinite(c):
            n_in += 1
            continue
        if best is None or c < best[0]:
            best = (c, par)
    p = best[1]
    enborde = [k for k, vals in bordes.items() if p.get(k) in (vals[0], vals[-1])]
    print(f"  {etiqueta:<20} coste {best[0]:9.1f} | "
          + " ".join(f"{k}={p[k]}" for k in sorted(p) if not k.startswith("_"))
          + f"\n  {'':<20} descartadas: {n_ac} acausales, {n_in} inestables | "
            f"borde: {enborde if enborde else 'ninguno'}")
    return p, enborde, n_ac


def main() -> dict:
    plant, mpc = base.build_mpc()
    pares = [dict(a_lo=lo, a_hi=hi) for lo in (0.0, 0.2, 0.4)
             for hi in (0.67, 0.83, 1.0) if hi > lo]
    print("=" * 78)
    print("SINTONIA CON GUARDA DE CAUSALIDAD (semillas 100-102)")
    print("=" * 78)
    print("  toda configuracion cuya puerta se mueva antes de que la fuente mida")
    print("  el evento queda descartada, igual que una integracion inestable\n")
    out = {}

    g1 = [dict(a_lo=0.0, a_hi=0.83, w0=w, zeta=z, c=c, D=0.0, gain=g, shift=s,
               gamma=1.0)
          for w in (0.08, 0.15, 0.3) for z in (0.5, 1.0, 1.6)
          for c in (0.08, 0.15, 0.3, 0.6) for g in GS for s in SH]
    best = None
    for par in g1:
        if not tp.is_causal(gate_of(mm.make_field_mech_shift("wave"), par, "wave")):
            continue
        c_, _ = base.evaluate(mpc, plant, mm.make_field_mech_shift("wave"), par,
                              base.SEEDS_TUNE[:2], "wave")
        if np.isfinite(c_) and (best is None or c_ < best[0]):
            best = (c_, par)
    p_w, b_w, ac_w = tune(mpc, plant, mm.make_field_mech_shift("wave"),
                          [dict(best[1], **q) for q in pares], "wave", "onda",
                          {"gain": GS, "shift": SH, "c": (0.08, 0.15, 0.3, 0.6),
                           "zeta": (0.5, 1.0, 1.6), "w0": (0.08, 0.15, 0.3)})

    g1 = [dict(a_lo=0.0, a_hi=0.83, delta=d, ws=w, zs=z, gain=g, shift=s)
          for d in (0, 2, 4, 6, 8) for w in (0.08, 0.15, 0.3, 0.6)
          for z in (0.3, 0.7, 1.2, 2.0, 3.0) for g in GS for s in SH]
    best = None
    for par in g1:
        if not tp.is_causal(gate_of(mm.mech_matched, par)):
            continue
        c_, _ = base.evaluate(mpc, plant, mm.mech_matched, par, base.SEEDS_TUNE[:2])
        if np.isfinite(c_) and (best is None or c_ < best[0]):
            best = (c_, par)
    p_m, b_m, ac_m = tune(mpc, plant, mm.mech_matched,
                          [dict(best[1], **q) for q in pares], None,
                          "retardo-conformado",
                          {"delta": (0, 2, 4, 6, 8), "ws": (0.08, 0.15, 0.3, 0.6),
                           "zs": (0.3, 0.7, 1.2, 2.0, 3.0), "gain": GS, "shift": SH})

    bg = [dict(a_lo=lo, a_hi=hi)
          for lo in np.round(np.linspace(0, .6, 7), 3)
          for hi in np.round(np.linspace(.3, 1., 8), 3) if hi > lo]
    p_r, _, _ = tune(mpc, plant, base.mech_reactivo, bg, None, "reactivo", {})
    p_d, _, _ = tune(mpc, plant, base.mech_retardo,
                     [dict(q, delta=d) for q in bg for d in (0, 2, 4, 6, 8)],
                     None, "retardo-puro", {"delta": (0, 2, 4, 6, 8)})
    p_o, _, _ = tune(mpc, plant, base.mech_oraculo,
                     [dict(q, lead=L) for q in bg for L in (0, 2, 3, 4, 6, 8)],
                     None, "oraculo(*)", {})
    print("  (*) el oraculo es CLARIVIDENTE por definicion: es la referencia de")
    print("      techo, no un mecanismo implementable, y por eso no se le exige causalidad")

    print("\n" + "=" * 78)
    print(f"EVALUACION CON {len(SEEDS_EVAL)} SEMILLAS")
    print("=" * 78)
    filas = {
        "onda": base.per_seed(mpc, plant, mm.make_field_mech_shift("wave"), p_w,
                              SEEDS_EVAL, "wave"),
        "retardo-conformado": base.per_seed(mpc, plant, mm.mech_matched, p_m, SEEDS_EVAL),
        "retardo-puro": base.per_seed(mpc, plant, base.mech_retardo, p_d, SEEDS_EVAL),
        "reactivo": base.per_seed(mpc, plant, base.mech_reactivo, p_r, SEEDS_EVAL),
        "oraculo": base.per_seed(mpc, plant, base.mech_oraculo, p_o, SEEDS_EVAL),
    }
    b = np.mean([f["coste"] for f in filas["reactivo"]])
    co = np.mean([f["coste"] for f in filas["oraculo"]])
    techo = b - co
    print(f"\n  {'mecanismo':<20}{'coste':>11}{'sem':>8}{'vs reactivo':>13}{'% oraculo':>11}")
    tabla = {}
    for k in sorted(filas, key=lambda k: np.mean([f["coste"] for f in filas[k]])):
        v = np.array([f["coste"] for f in filas[k]], float)
        c_, sem = float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)))
        print(f"  {k:<20}{c_:>11.1f}{sem:>8.1f}{100*(b-c_)/b:>+12.2f}%"
              f"{100*(b-c_)/techo:>10.1f}%")
        tabla[k] = {"coste": c_, "sem": sem, "por_semilla": v.tolist(),
                    "vs_reactivo_pct": float(100 * (b - c_) / b),
                    "pct_oraculo": float(100 * (b - c_) / techo),
                    "pct_oraculo_sem": float(100 * sem / techo)}

    print("\n" + "=" * 78)
    print("CONTRASTES (pareados, Holm)")
    print("=" * 78)
    pr = [("onda", "retardo-conformado", "onda vs rival igualado (7 vs 7)"),
          ("onda", "retardo-puro", "onda vs retardo puro (7 vs 3)"),
          ("retardo-conformado", "retardo-puro", "conformar ayuda al rival?"),
          ("retardo-puro", "reactivo", "info de aguas arriba vs solo local")]
    ts = holm([paired(filas[a], filas[bb], "coste", a_name=a, b_name=bb)
               for a, bb, _ in pr])
    contr = {}
    for (a, bb, txt), t in zip(pr, ts):
        print(f"  {txt:<40} delta={t.delta:+9.2f} t={t.t:+7.2f} "
              f"p_holm={t.p_holm:.4f} {'*' if t.significativo else ' '}")
        contr[f"{a}_vs_{bb}"] = {"delta": float(t.delta), "t": float(t.t),
                                 "p_holm": float(t.p_holm),
                                 "significativo": bool(t.significativo)}
    out = {"sintonia": {"onda": p_w, "matched": p_m, "retardo": p_d,
                        "reactivo": p_r, "oraculo": p_o},
           "descartadas_acausales": {"onda": ac_w, "matched": ac_m},
           "tabla": tabla, "contrastes": contr}

    print("\n" + "=" * 78)
    print("LECTURA")
    print("=" * 78)
    cf = contr["retardo-conformado_vs_retardo-puro"]
    up = contr["retardo-puro_vs_reactivo"]
    print(f"  informacion de aguas arriba frente a deteccion local: "
          f"{'SIGUE SIENDO significativa' if up['significativo'] else 'YA NO es significativa'}"
          f" (t={up['t']:+.2f}, p={up['p_holm']:.4f})")
    print(f"  'conformar ayuda': {'sobrevive' if cf['significativo'] else 'NO SOBREVIVE'}"
          f" a la guarda de causalidad (t={cf['t']:+.2f}, p={cf['p_holm']:.4f})")
    return out


if __name__ == "__main__":
    r = main()
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dst = os.path.join(here, "results")
    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(dst, "transport_causal.json"), "w") as fh:
        json.dump(r, fh, indent=1, default=float)
    print(f"\nescrito {os.path.join(dst, 'transport_causal.json')}")
