"""TEST DE VIABILIDAD de la arena de transporte. Se ejecuta ANTES de construir
ningun mecanismo de transporte, y puede matar la arena entera.

LA PREGUNTA, Y POR QUE ES DECISIVA
-----------------------------------
Si un gobernador CLARIVIDENTE -- que conoce el instante exacto de llegada de
cada evento y puede actuar con `lead` pasos de antelacion -- no bate al limite
reactivo (lead = 0, lo mejor que puede lograr cualquier detector local), entonces
la informacion anticipada NO COMPRA NADA por el canal del meta-parametro, y da
igual como se transporte: campo de onda, difusion, linea de retardo o telepatia.
La arena estaria muerta y no habria que implementar nada.

Se comprueban ademas las tres condiciones necesarias:
  C1 excitacion de fondo -> el nodo nunca esta en reposo (si lo estuviera,
     u* = 0 para todo peso y el meta-parametro no tendria efecto ninguno)
  C2 saturacion ACTIVA -> sin ella el lazo es lineal y A15 vacia la arena
  C3 la accion preventiva CUESTA -> si no, el mejor peso constante ya la haria

ORDEN DE EJECUCION
------------------
  P0  calibracion: el mejor peso CONSTANTE cae en el interior, y hay saturacion
  P1  el barrido de `lead`: LA PRUEBA. Coste frente a antelacion.
  P2  control de gratuidad (C3): si el peso alto fuera gratis, el constante alto
      ganaria; se comprueba que no lo hace
  P3  robustez del hallazgo: varias semillas, y el contraste pareado
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ghi import transport as tp
from ghi.stats import paired

N_MPC = 12
ALPHAS = np.round(np.linspace(0.05, 0.95, 19), 4)
Q = np.diag([1.0, 0.02])
R = np.array([[1.0]])
LEADS = [0, 1, 2, 3, 5, 8, 12, 18, 25]


def build(umax=1.0):
    plant = tp.transport_plant(umax=umax)
    mpc = tp.WeightedMPC(plant, Q, R, N_MPC, ALPHAS)
    return plant, mpc


def best_constant(mpc, plant, d):
    res = [(tp.run_schedule(mpc, plant, d, a), a) for a in ALPHAS]
    k = int(np.argmin([r[0]["coste"] for r in res]))
    return res[k][1], res[k][0], [r[0]["coste"] for r in res]


def main() -> dict:
    out = {}
    plant, mpc = build()
    traf = tp.Traffic(seed=0)
    d, starts = traf.build()

    print("=" * 78)
    print("P0  CALIBRACION: hay compromiso que arbitrar, y la no linealidad esta ACTIVA")
    print("=" * 78)
    a_c, r_c, curva = best_constant(mpc, plant, d)
    print(f"\n  eventos en la traza: {len(starts)}  (anchura {traf.width}, "
          f"amplitud {traf.amp}, ruido de fondo {traf.sigma_bg})")
    print(f"  {'alpha':>7}{'coste':>12}{'% saturado':>13}")
    for a in (0.05, 0.25, 0.5, 0.75, 0.95):
        rr = tp.run_schedule(mpc, plant, d, a)
        mark = "  <== MEJOR CONSTANTE" if abs(a - a_c) < 1e-9 else ""
        print(f"  {a:>7.2f}{rr['coste']:>12.2f}{100*rr['sat']:>12.1f}%{mark}")
    print(f"\n  mejor peso constante = {a_c:.2f}  (interior: {0.05 < a_c < 0.95})")
    print(f"  saturacion en el optimo constante: {100*r_c['sat']:.1f}%")
    c2 = r_c["sat"] > 0.01
    print(f"  C2 (saturacion ACTIVA, >1% de los pasos): {'SI' if c2 else 'NO -- A15 vaciaria la arena'}")
    out["P0"] = {"alpha_const": float(a_c), "coste_const": float(r_c["coste"]),
                 "saturacion": float(r_c["sat"]), "C2": bool(c2),
                 "n_eventos": len(starts)}

    print("\n" + "=" * 78)
    print("P1  LA PRUEBA: coste frente a ANTELACION del gobernador clarividente")
    print("=" * 78)
    print("  lead = 0 es el limite REACTIVO ideal (saber del evento justo al empezar).")
    print("  Todo lead > 0 exige informacion que solo el vecino de aguas arriba tiene.")
    # el par (a_lo, a_hi) se optimiza CON lead = 0, para que la ventaja de la
    # anticipacion no venga de una sintonia regalada al clarividente
    mejor = None
    for a_lo in ALPHAS[::2]:
        for a_hi in ALPHAS[::2]:
            if a_hi <= a_lo:
                continue
            sch = tp.clairvoyant_schedule(traf.T, starts, traf.width, a_lo, a_hi, 0)
            c = tp.run_schedule(mpc, plant, d, sch)["coste"]
            if mejor is None or c < mejor[0]:
                mejor = (c, float(a_lo), float(a_hi))
    c0, a_lo, a_hi = mejor
    print(f"\n  par (a_lo, a_hi) sintonizado CON lead=0: ({a_lo:.2f}, {a_hi:.2f})"
          f"  coste {c0:.2f}")
    print(f"\n  {'lead':>6}{'coste':>12}{'vs lead=0':>12}{'vs constante':>14}{'% sat':>9}")
    filas = []
    for L in LEADS:
        sch = tp.clairvoyant_schedule(traf.T, starts, traf.width, a_lo, a_hi, L)
        rr = tp.run_schedule(mpc, plant, d, sch)
        g0 = 100.0 * (c0 - rr["coste"]) / c0
        gc = 100.0 * (r_c["coste"] - rr["coste"]) / r_c["coste"]
        print(f"  {L:>6d}{rr['coste']:>12.2f}{g0:>+11.2f}%{gc:>+13.2f}%"
              f"{100*rr['sat']:>8.1f}%")
        filas.append({"lead": L, "coste": float(rr["coste"]),
                      "ganancia_vs_lead0": float(g0), "ganancia_vs_const": float(gc)})
    best = max(filas, key=lambda f: f["ganancia_vs_lead0"])
    plano = best["ganancia_vs_lead0"] < 1.0
    print(f"\n  mejor antelacion: lead = {best['lead']}  "
          f"({best['ganancia_vs_lead0']:+.2f}% sobre el limite reactivo)")
    print(f"  VEREDICTO P1: {'PLANO -> la anticipacion NO compra nada -> ARENA MUERTA' if plano else 'HAY VALOR EN LA ANTICIPACION -> la arena es viable'}")
    out["P1"] = {"a_lo": a_lo, "a_hi": a_hi, "coste_lead0": float(c0),
                 "filas": filas, "lead_optimo": best["lead"],
                 "ganancia_max_pct": best["ganancia_vs_lead0"],
                 "plano": bool(plano)}

    print("\n" + "=" * 78)
    print("P2  CONTROL DE GRATUIDAD (C3): la accion preventiva TIENE que costar")
    print("=" * 78)
    r_hi = tp.run_schedule(mpc, plant, d, a_hi)
    print(f"\n  peso alto SIEMPRE (a={a_hi:.2f}): {r_hi['coste']:.2f}")
    print(f"  mejor peso constante   (a={a_c:.2f}): {r_c['coste']:.2f}")
    c3 = r_hi["coste"] > r_c["coste"] * 1.005
    print(f"  C3 (mantener el peso alto es PEOR que el mejor constante): "
          f"{'SI' if c3 else 'NO -- entonces no hay nada que programar'}")
    out["P2"] = {"coste_alto_siempre": float(r_hi["coste"]), "C3": bool(c3)}

    print("\n" + "=" * 78)
    print("P3  ROBUSTEZ: 8 semillas, contraste pareado lead* frente a lead=0")
    print("=" * 78)
    L_star = best["lead"] if best["lead"] > 0 else max(LEADS)
    A, B = [], []
    for s in range(8):
        tr = tp.Traffic(seed=s)
        ds, st = tr.build()
        s0 = tp.clairvoyant_schedule(tr.T, st, tr.width, a_lo, a_hi, 0)
        sL = tp.clairvoyant_schedule(tr.T, st, tr.width, a_lo, a_hi, L_star)
        A.append(tp.run_schedule(mpc, plant, ds, s0)["coste"])
        B.append(tp.run_schedule(mpc, plant, ds, sL)["coste"])
    A, B = np.array(A), np.array(B)
    # delta = media(lead=0) - media(lead*) > 0 significa que la ANTICIPACION gana
    st_ = paired([{"c": v} for v in B], [{"c": v} for v in A], "c",
                 a_name=f"lead={L_star}", b_name="lead=0")
    print(f"\n  lead=0 : media {A.mean():.2f}")
    print(f"  lead={L_star:<2d}: media {B.mean():.2f}")
    print(f"  mejora media {100*(A-B).mean()/A.mean():+.2f}%  |  "
          f"t = {st_.t:+.2f}  p = {st_.p:.2e}  ({(B<A).sum()}/8 semillas a favor)")
    out["P3"] = {"lead": int(L_star), "media_lead0": float(A.mean()),
                 "media_leadL": float(B.mean()), "delta": float(st_.delta),
                 "t": float(st_.t), "p": float(st_.p),
                 "semillas_a_favor": int((B < A).sum())}

    print("\n" + "=" * 78)
    if plano:
        print("VEREDICTO GLOBAL: ARENA MUERTA. La informacion anticipada no compra")
        print("nada por el canal del meta-parametro, luego ningun mecanismo de")
        print("transporte puede pagar. No implementar el campo.")
    else:
        print("VEREDICTO GLOBAL: ARENA VIABLE. Existe valor en la anticipacion, luego")
        print("la pregunta de COMO transportarla (onda / difusion / linea de retardo")
        print("pura sintonizada) es legitima y hay que responderla con las lineas base")
        print("mas fuertes posibles.")
    print("=" * 78)
    return out


if __name__ == "__main__":
    res = main()
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dst = os.path.join(here, "results")
    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(dst, "transport_viability.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    print(f"\nescrito {os.path.join(dst, 'transport_viability.json')}")
