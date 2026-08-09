"""EL EXPERIMENTO DEL TRANSPORTE: onda, difusion, o una simple linea de retardo.

QUE SE SABE YA AL LLEGAR AQUI
------------------------------
  * la arena con ALMACENAMIENTO es viable: anticipar compra +4.6% sobre el
    limite reactivo, 9/10 semillas, p = 2e-4 (`run_transport_viability2.py`)
  * la antelacion util es lead* en [2,4] en todo el rango fisico barrido, y
    crece con la lentitud del actuador tal y como predice el mecanismo
  * pero LA VENTANA ES ANCHA: todo lead entre 1 y 6 (hasta 20 en el caso de
    menos autoridad) queda dentro del 1% del optimo (`run_lead_window.py`)

LA PREDICCION QUE ESTE SCRIPT PONE A PRUEBA
--------------------------------------------
Si la ventana de antelacion util es ANCHA, entonces el INSTANTE EXACTO de
entrega no importa, y la propiedad que distingue a la ecuacion de onda -- fase
lineal, retardo de grupo bien definido, forma preservada -- no tiene nada que
comprar. La prediccion, registrada antes de ejecutar:

  T1  onda ~= linea de retardo pura sintonizada  (diferencia < 1%)
      -> si se cumple, el campo de onda es "una linea de retardo con pasos de
         mas" y el transporte queda cerrado en negativo
  T2  onda > difusion (la difusion emborrona y llega tarde y atenuada)
  T3  instantaneo ~= retardo sintonizado (llegar pronto no penaliza, porque la
      ventana es ancha). Es el NULO DE MECANISMO: si se cumple, confirma que lo
      que importa es TENER la informacion, no ENTREGARLA en el instante justo.
  T4  todos los que tienen informacion de aguas arriba baten a los que no

Cada mecanismo se sintoniza en un conjunto de semillas de SINTONIA y se evalua
en semillas DISTINTAS, para que nadie gane por sobreajuste.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ghi import transport as tp
from ghi.distributed import GraphField, Independent
from ghi.field import FieldParams
from ghi.stats import holm, paired

M = 4                     # nodos de la cadena
TAU_HOP = 6               # retardo de transporte por tramo (pasos)
N_MPC = 16
ALPHAS = np.round(np.linspace(0.0, 1.0, 21), 4)
B_ECON, B_ALTA, B_CRIT, PEN = 0.0, 3.0, -2.5, 5.0
TAU_ACT, UMAX, AMP, WIDTH = 8.0, 0.35, 0.80, 16
SEEDS_TUNE = (100, 101, 102)
SEEDS_EVAL = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)


def build_mpc():
    plant = tp.storage_plant(tau_act=TAU_ACT, umax=UMAX)
    objs = [(np.diag([1.0, 0.05]), np.array([[1.0]]), np.array([B_ECON, 0.0])),
            (np.diag([1.0, 0.05]), np.array([[1.0]]), np.array([B_ALTA, 0.0]))]
    return plant, tp.SetpointMPC(plant, objs, N_MPC, ALPHAS)


def traffic(seed):
    return tp.Traffic(seed=seed, sigma_bg=0.020, amp=AMP, width=WIDTH,
                      gap_lo=80, gap_hi=120, T=800)


def node_signal(d0, i):
    """La perturbacion del nodo i es la del nodo 0 RETRASADA i*TAU_HOP pasos:
    es la misma masa de fluido/carga que viaja aguas abajo."""
    k = i * TAU_HOP
    return np.concatenate([np.zeros(k), d0[:-k]]) if k > 0 else d0.copy()


def source_pulse(T, starts, width, lead0=0):
    """Senal binaria que el nodo 0 puede medir: 1 mientras el evento le afecta."""
    s = np.zeros(T)
    for t0 in starts:
        s[max(0, t0 - lead0):min(T, t0 + width)] = 1.0
    return s


# ------------------------------------------------------------------
#   MECANISMOS: cada uno convierte la medida del nodo 0 en alpha_i(t)
# ------------------------------------------------------------------

def alpha_from_gate(g, a_lo, a_hi):
    return a_lo + (a_hi - a_lo) * np.clip(g, 0.0, 1.0)


def mech_constante(src, T, i, par):
    return np.full(T, par["a_hi"])


def mech_reactivo(src, T, i, par):
    """Detector local IDEAL: se entera del evento en el instante exacto en que
    llega a SU nodo. Es el mejor limite reactivo posible, mas fuerte que
    cualquier detector real."""
    k = i * TAU_HOP
    g = np.concatenate([np.zeros(k), src[:-k]]) if k > 0 else src.copy()
    return alpha_from_gate(g, par["a_lo"], par["a_hi"])


def mech_retardo(src, T, i, par):
    """LINEA DE RETARDO PURA sintonizada: el nodo i recibe la medida del nodo 0
    retrasada `delta` pasos por tramo. Es la linea base que puede matar al campo."""
    k = int(round(par["delta"] * i))
    g = np.concatenate([np.zeros(k), src[:-k]]) if k > 0 else src.copy()
    return alpha_from_gate(g, par["a_lo"], par["a_hi"])


def mech_instantaneo(src, T, i, par):
    """Difusion de la informacion sin ningun retardo: todos los nodos ven la
    medida del nodo 0 a la vez. Llega TAU_HOP*i pasos ANTES de tiempo."""
    return alpha_from_gate(src, par["a_lo"], par["a_hi"])


def _field_gate(src, T, branch, par):
    """Propaga la medida del nodo 0 por el campo del grafo y devuelve h (T, M)."""
    p = FieldParams(w0=par["w0"], zeta=par["zeta"], c=par["c"], D=par["D"],
                    b=0.0, beta=0.0, gamma=par.get("gamma", 1.0))
    fld = GraphField(M, p, branch=branch, a0=0.0, forcing="source")
    if not fld.verlet_stable(1.0):
        return None                      # configuracion desbordada: no es candidata
    H = np.zeros((T, M))
    hd = np.zeros(M)
    for t in range(T):
        hd[0] = src[t]
        H[t] = fld.step(hd, 1.0)
    return H


def make_field_mech(branch):
    def mech(src, T, i, par):
        H = par["_H"]
        h = H[:, i]
        rng = h.max() - h.min()
        g = (h - h.min()) / rng if rng > 1e-9 else np.zeros_like(h)
        return alpha_from_gate(par["gain"] * g, par["a_lo"], par["a_hi"])
    return mech


def mech_oraculo(src, T, i, par):
    """Cota superior: programa clarividente con la antelacion optima medida."""
    g = np.zeros(T)
    k = i * TAU_HOP
    base = np.concatenate([np.zeros(k), src[:-k]]) if k > 0 else src.copy()
    L = int(par["lead"])
    for t in range(T):
        if base[min(T - 1, t + L)] > 0.5:
            g[t] = 1.0
    return alpha_from_gate(g, par["a_lo"], par["a_hi"])


# ------------------------------------------------------------------

def evaluate(mpc, plant, mech, par, seeds, field_branch=None):
    """Coste total AGUAS ABAJO (nodos 1..M-1): el nodo 0 no tiene informacion
    anticipada por definicion y su coste solo anadiria ruido comun."""
    tot, rot = 0.0, []
    for s in seeds:
        tr = traffic(s)
        d0, starts = tr.build()
        src = source_pulse(tr.T, starts, tr.width)
        if field_branch is not None:
            H = _field_gate(src, tr.T, field_branch, par)
            if H is None:
                return np.inf, 1.0
            par = dict(par); par["_H"] = H
        for i in range(1, M):
            a = mech(src, tr.T, i, par)
            r = tp.run_storage(mpc, plant, node_signal(d0, i), a,
                               b_econ=B_ECON, b_crit=B_CRIT, pen=PEN)
            if not np.isfinite(r["coste"]):
                return np.inf, 1.0
            tot += r["coste"]; rot.append(r["roturas"])
    return tot / len(seeds), float(np.mean(rot))


def per_seed(mpc, plant, mech, par, seeds, field_branch=None):
    filas = []
    for s in seeds:
        c, r = evaluate(mpc, plant, mech, par, (s,), field_branch)
        filas.append({"coste": c, "roturas": r})
    return filas


def main() -> dict:
    plant, mpc = build_mpc()
    print("=" * 78)
    print("SINTONIA (en semillas 100-102, DISTINTAS de las de evaluacion)")
    print("=" * 78)
    print(f"  cadena de M={M} nodos, retardo de transporte TAU_HOP={TAU_HOP} pasos/tramo")
    print(f"  actuador: umax={UMAX} tau_act={TAU_ACT} | evento amp={AMP} ancho={WIDTH}")

    AL = np.round(np.linspace(0.0, 0.6, 7), 3)
    AH = np.round(np.linspace(0.3, 1.0, 8), 3)
    tuned = {}

    def tune(name, mech, grid, branch=None):
        best = None
        for par in grid:
            c, _ = evaluate(mpc, plant, mech, par, SEEDS_TUNE, branch)
            if best is None or c < best[0]:
                best = (c, par)
        tuned[name] = best[1]
        extra = {k: v for k, v in best[1].items()
                 if k not in ("a_lo", "a_hi") and not k.startswith("_")}
        print(f"  {name:<16} a=({best[1]['a_lo']:.2f},{best[1]['a_hi']:.2f}) "
              f"{extra if extra else ''}  coste sintonia {best[0]:.1f}")
        return best[1]

    base_grid = [dict(a_lo=lo, a_hi=hi) for lo in AL for hi in AH if hi > lo]
    tune("constante", mech_constante, [dict(a_lo=0.0, a_hi=h) for h in ALPHAS])
    tune("reactivo", mech_reactivo, base_grid)
    tune("retardo-puro", mech_retardo,
         [dict(p, delta=dl) for p in base_grid for dl in (0, 1, 2, 3, 4, 5, 6, 8)])
    tune("instantaneo", mech_instantaneo, base_grid)
    tune("oraculo", mech_oraculo,
         [dict(p, lead=L) for p in base_grid for L in (0, 2, 3, 4, 6, 8)])
    fgrid_w = [dict(p, w0=w, zeta=z, c=c, D=0.0, gain=g)
               for p in base_grid[::3] for w in (0.25, 0.5) for z in (0.5, 1.0)
               for c in (0.15, 0.30, 0.60) for g in (1.0, 1.6)]
    fgrid_d = [dict(p, w0=w, zeta=1.0, c=c, D=dd, gain=g, gamma=1.0)
               for p in base_grid[::3] for w in (0.25, 0.5)
               for c in (0.15, 0.30, 0.60) for dd in (0.1, 0.4) for g in (1.0, 1.6)]
    tune("onda", make_field_mech("wave"), fgrid_w, branch="wave")
    tune("difusion", make_field_mech("diffusion"), fgrid_d, branch="diffusion")

    print("\n" + "=" * 78)
    print("EVALUACION (semillas 0-9, no usadas en la sintonia)")
    print("=" * 78)
    branches = {"onda": "wave", "difusion": "diffusion"}
    mechs = {"constante": mech_constante, "reactivo": mech_reactivo,
             "retardo-puro": mech_retardo, "instantaneo": mech_instantaneo,
             "onda": make_field_mech("wave"), "difusion": make_field_mech("diffusion"),
             "oraculo": mech_oraculo}
    filas = {}
    print(f"\n  {'mecanismo':<16}{'coste':>11}{'vs reactivo':>13}{'% rotura':>11}")
    for name, mech in mechs.items():
        filas[name] = per_seed(mpc, plant, mech, tuned[name], SEEDS_EVAL,
                               branches.get(name))
    base = np.mean([f["coste"] for f in filas["reactivo"]])
    orden = sorted(filas, key=lambda k: np.mean([f["coste"] for f in filas[k]]))
    res = {}
    for name in orden:
        c = np.mean([f["coste"] for f in filas[name]])
        r = np.mean([f["roturas"] for f in filas[name]])
        print(f"  {name:<16}{c:>11.1f}{100*(base-c)/base:>+12.2f}%{100*r:>10.1f}%")
        res[name] = {"coste": float(c), "vs_reactivo_pct": float(100 * (base - c) / base),
                     "roturas": float(r)}

    print("\n" + "=" * 78)
    print("CONTRASTES PRE-REGISTRADOS (pareados por semilla, Holm)")
    print("=" * 78)
    pares = [("onda", "retardo-puro", "T1: onda vs linea de retardo pura"),
             ("onda", "difusion", "T2: onda vs difusion"),
             ("instantaneo", "retardo-puro", "T3: instantaneo vs retardo (nulo de timing)"),
             ("retardo-puro", "reactivo", "T4: informacion de aguas arriba vs solo local"),
             ("onda", "reactivo", "T4b: onda vs solo local"),
             ("oraculo", "reactivo", "cota: oraculo vs solo local")]
    tests = [paired(filas[a], filas[b], "coste", a_name=a, b_name=b) for a, b, _ in pares]
    tests = holm([t for t in tests if t is not None])
    for (a, b, txt), t in zip(pares, tests):
        signo = "gana " + (a if t.delta > 0 else b)
        print(f"  {txt:<46} delta={t.delta:+9.2f} t={t.t:+6.2f} "
              f"p_holm={t.p_holm:.4f} {'*' if t.significativo else ' '} {signo}")
        res.setdefault("_contrastes", {})[f"{a}_vs_{b}"] = {
            "delta": float(t.delta), "t": float(t.t), "p_holm": float(t.p_holm),
            "significativo": bool(t.significativo)}

    print("\n" + "=" * 78)
    print("LECTURA")
    print("=" * 78)
    t1 = res["_contrastes"]["onda_vs_retardo-puro"]
    dif = abs(res["onda"]["coste"] - res["retardo-puro"]["coste"]) / res["retardo-puro"]["coste"]
    if not t1["significativo"] or dif < 0.01:
        print(f"""  T1 SE CUMPLE: la onda y la linea de retardo pura sintonizada difieren un
  {100*dif:.2f}% y el contraste {'no ' if not t1['significativo'] else ''}es significativo. El campo de onda es una
  LINEA DE RETARDO CON PASOS DE MAS. Era lo predicho por la anchura de la
  ventana de antelacion: si el instante exacto de entrega no importa, la fase
  lineal de la onda no compra nada.""")
    else:
        print("  T1 SE FALSA: la onda y la linea de retardo NO son equivalentes.")
    return {"sintonia": {k: {kk: vv for kk, vv in v.items() if not kk.startswith('_')}
                         for k, v in tuned.items()}, "evaluacion": res}


if __name__ == "__main__":
    out = main()
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dst = os.path.join(here, "results")
    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(dst, "transport_mechanism.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(f"\nescrito {os.path.join(dst, 'transport_mechanism.json')}")
