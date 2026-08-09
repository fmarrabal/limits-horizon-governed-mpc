"""El transporte, con normalizacion CAUSAL: la tercera fuga de la misma clase.

QUE FALTABA TAPAR
------------------
Tras corregir la fuga del desplazamiento negativo, quedaba una tercera via por
la que dos de los tres mecanismos veian el futuro: la NORMALIZACION.

    g = (h - min(h)) / (max(h) - min(h))

toma el minimo y el maximo de la TRAZA ENTERA, de modo que la escala de la
puerta en el instante t depende de valores posteriores a t. Medido con la sonda
de un pulso: truncando la traza despues del evento, la puerta de la onda cambia
hasta 0.418 en instantes ANTERIORES al truncamiento. La linea de retardo pura no
usaba esa normalizacion, asi que -- igual que en la fuga anterior -- el sesgo
favorecia exactamente a los dos mecanismos cuyo merito se discutia.

LA CORRECCION, QUE ADEMAS QUITA UN MANDO
-----------------------------------------
No hace falta normalizar por la traza: la escala correcta se conoce fuera de
linea.

  * el conformado de 2o orden tiene GANANCIA UNIDAD EN CONTINUA (en equilibrio
    ws^2 (x - y) = 0 => y = x), luego su salida ya esta en la escala de la
    fuente y no necesita normalizacion ninguna;
  * el campo tiene ganancia en continua conocida y calculable: (w0^2 K^-1)[i,0]
    (Prop. 3 del manuscrito), asi que dividir por ella da una puerta de ganancia
    unidad, sin mirar un solo dato futuro.

Esto no solo tapa la fuga: elimina un grado de libertad arbitrario de ambos
mecanismos, con lo que la comparacion queda mas limpia que antes.
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
from ghi.distributed import GraphField
from ghi.field import FieldParams
from ghi.stats import holm, paired

SEEDS_EVAL = tuple(range(30))
SH = (0, 2, 4, 6, 9, 12)
GS = (1.0, 2.0, 3.0, 5.0)


def _shift_causal(h, k):
    """Solo RETARDO. Un desplazamiento negativo adelantaria la senal."""
    k = int(round(k))
    if k <= 0:
        return h
    return np.concatenate([np.zeros(k), h[:-k]])


def dc_gain_node(par, i, branch):
    """Ganancia en continua del campo en el nodo i: (w0^2 K^-1)[i,0].
    Es un numero que se calcula OFFLINE a partir de los parametros, sin datos."""
    p = FieldParams(w0=par["w0"], zeta=par["zeta"], c=par["c"], D=par["D"],
                    b=0.0, beta=0.0, gamma=par.get("gamma", 1.0))
    f = GraphField(base.M, p, branch=branch, a0=0.0, forcing="source")
    return float(f.dc_gain()[i, 0])


def mech_matched_causal(src, T, i, par):
    """Retardo por salto + conformado de 2o orden. SIN normalizacion: el
    conformado tiene ganancia unidad en continua."""
    k = int(round(par["delta"] * i))
    x = _shift_causal(src, k)
    y = mm.second_order_lp(x, par["ws"], par["zs"])
    g = _shift_causal(y, par.get("shift", 0))
    return base.alpha_from_gate(par["gain"] * g, par["a_lo"], par["a_hi"])


def make_field_mech_causal(branch):
    def mech(src, T, i, par):
        h = par["_H"][:, i]
        dc = par.get("_dc", [None] * base.M)[i]
        if dc is None or abs(dc) < 1e-12:
            return np.full(T, par["a_lo"])
        g = _shift_causal(h / dc, par.get("shift", 0))
        return base.alpha_from_gate(par["gain"] * g, par["a_lo"], par["a_hi"])
    return mech


def gate_of(mech, par, branch=None):
    def g(src, T, i):
        p = dict(par)
        if branch is not None:
            H = base._field_gate(src, T, branch, p)
            if H is None:
                return np.full(T, p["a_lo"])
            p["_H"] = H
            p["_dc"] = [dc_gain_node(p, j, branch) for j in range(base.M)]
        return mech(src, T, i, p)
    return g


def evaluate(mpc, plant, mech, par, seeds, branch=None):
    tot, rot = 0.0, []
    for s in seeds:
        tr = base.traffic(s)
        d0, starts = tr.build()
        src = base.source_pulse(tr.T, starts, tr.width)
        p = dict(par)
        if branch is not None:
            H = base._field_gate(src, tr.T, branch, p)
            if H is None:
                return np.inf, 1.0
            p["_H"] = H
            p["_dc"] = [dc_gain_node(p, j, branch) for j in range(base.M)]
        for i in range(1, base.M):
            a = mech(src, tr.T, i, p)
            r = tp.run_storage(mpc, plant, base.node_signal(d0, i), a,
                               b_econ=base.B_ECON, b_crit=base.B_CRIT, pen=base.PEN)
            if not np.isfinite(r["coste"]):
                return np.inf, 1.0
            tot += r["coste"]; rot.append(r["roturas"])
    return tot / len(seeds), float(np.mean(rot))


def per_seed(mpc, plant, mech, par, seeds, branch=None):
    return [{"coste": evaluate(mpc, plant, mech, par, (s,), branch)[0]} for s in seeds]


def tune(mpc, plant, mech, grid, branch, etiqueta):
    best, n_ac = None, 0
    for par in grid:
        if not tp.is_causal(gate_of(mech, par, branch)):
            n_ac += 1
            continue
        c, _ = evaluate(mpc, plant, mech, par, base.SEEDS_TUNE, branch)
        if np.isfinite(c) and (best is None or c < best[0]):
            best = (c, par)
    p = best[1]
    print(f"  {etiqueta:<20} coste {best[0]:9.1f} | "
          + " ".join(f"{k}={p[k]}" for k in sorted(p) if not k.startswith("_"))
          + f" | {n_ac} descartadas por acausales")
    return p


def main() -> dict:
    plant, mpc = base.build_mpc()
    pares = [dict(a_lo=lo, a_hi=hi) for lo in (0.0, 0.2, 0.4)
             for hi in (0.67, 0.83, 1.0) if hi > lo]
    print("=" * 78)
    print("SINTONIA con guarda de causalidad Y normalizacion CAUSAL")
    print("=" * 78)
    print("  la escala de la puerta se fija con la ganancia en continua, que se")
    print("  calcula offline; ya no se mira el maximo de la traza completa\n")

    g1 = [dict(a_lo=0.0, a_hi=0.83, w0=w, zeta=z, c=c, D=0.0, gain=g, shift=s,
               gamma=1.0)
          for w in (0.08, 0.15, 0.3) for z in (0.5, 1.0, 1.6)
          for c in (0.08, 0.15, 0.3, 0.6) for g in GS for s in SH]
    best = None
    mech_w = make_field_mech_causal("wave")
    for par in g1:
        if not tp.is_causal(gate_of(mech_w, par, "wave")):
            continue
        c_, _ = evaluate(mpc, plant, mech_w, par, base.SEEDS_TUNE[:2], "wave")
        if np.isfinite(c_) and (best is None or c_ < best[0]):
            best = (c_, par)
    p_w = tune(mpc, plant, mech_w, [dict(best[1], **q) for q in pares], "wave", "onda")

    g1 = [dict(a_lo=0.0, a_hi=0.83, delta=d, ws=w, zs=z, gain=g, shift=s)
          for d in (0, 2, 4, 6, 8) for w in (0.08, 0.15, 0.3, 0.6)
          for z in (0.3, 0.7, 1.2, 2.0, 3.0) for g in GS for s in SH]
    best = None
    for par in g1:
        if not tp.is_causal(gate_of(mech_matched_causal, par)):
            continue
        c_, _ = evaluate(mpc, plant, mech_matched_causal, par, base.SEEDS_TUNE[:2])
        if np.isfinite(c_) and (best is None or c_ < best[0]):
            best = (c_, par)
    p_m = tune(mpc, plant, mech_matched_causal,
               [dict(best[1], **q) for q in pares], None, "retardo-conformado")

    bg = [dict(a_lo=lo, a_hi=hi)
          for lo in np.round(np.linspace(0, .6, 7), 3)
          for hi in np.round(np.linspace(.3, 1., 8), 3) if hi > lo]
    p_r = tune(mpc, plant, base.mech_reactivo, bg, None, "reactivo")
    p_d = tune(mpc, plant, base.mech_retardo,
               [dict(q, delta=d) for q in bg for d in (0, 2, 4, 6, 8)], None,
               "retardo-puro")
    p_o = tune(mpc, plant, base.mech_oraculo,
               [dict(q, lead=L) for q in bg for L in (0, 2, 3, 4, 6, 8)], None,
               "oraculo(*)")

    print("\n" + "=" * 78)
    print(f"EVALUACION CON {len(SEEDS_EVAL)} SEMILLAS")
    print("=" * 78)
    filas = {
        "onda": per_seed(mpc, plant, mech_w, p_w, SEEDS_EVAL, "wave"),
        "retardo-conformado": per_seed(mpc, plant, mech_matched_causal, p_m, SEEDS_EVAL),
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
    print("CONTRASTES (pareados, Holm) Y EQUIVALENCIA")
    print("=" * 78)
    pr = [("onda", "retardo-conformado", "onda vs rival igualado"),
          ("onda", "retardo-puro", "onda vs retardo puro"),
          ("retardo-conformado", "retardo-puro", "conformar ayuda?"),
          ("retardo-puro", "reactivo", "info de aguas arriba vs solo local")]
    ts = holm([paired(filas[a], filas[bb], "coste", a_name=a, b_name=bb)
               for a, bb, _ in pr])
    contr = {}
    for (a, bb, txt), t in zip(pr, ts):
        # margen de equivalencia: la MENOR diferencia que este estudio habria
        # podido detectar con potencia 0.8. Declarar "no separa" sin esto es
        # confundir ausencia de evidencia con evidencia de ausencia.
        n = t.n
        mde = 2.8 * t.sem            # ~ (1.96+0.84) errores tipicos
        print(f"  {txt:<38} delta={t.delta:+9.2f} t={t.t:+7.2f} "
              f"p={t.p_holm:.4f}{'*' if t.significativo else ' '} | "
              f"detectable >= {mde:.0f} ({100*mde/b:.2f}% del coste)")
        contr[f"{a}_vs_{bb}"] = {"delta": float(t.delta), "t": float(t.t),
                                 "p_holm": float(t.p_holm), "n": int(n),
                                 "significativo": bool(t.significativo),
                                 "mde": float(mde), "mde_pct": float(100 * mde / b)}
    return {"sintonia": {"onda": p_w, "matched": p_m, "retardo": p_d,
                         "reactivo": p_r, "oraculo": p_o},
            "tabla": tabla, "contrastes": contr}


if __name__ == "__main__":
    r = main()
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dst = os.path.join(here, "results")
    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(dst, "transport_causal2.json"), "w") as fh:
        json.dump(r, fh, indent=1, default=float)
    print(f"\nescrito {os.path.join(dst, 'transport_causal2.json')}")
