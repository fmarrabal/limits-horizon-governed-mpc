"""TEST DE VIABILIDAD, segunda version: la arena CON ALMACENAMIENTO.

POR QUE HAY UNA SEGUNDA VERSION
--------------------------------
La primera (`run_transport_viability.py`) fallo la condicion C3: el mejor peso
constante cayo en el extremo del simplex y "peso alto siempre" resulto ser
exactamente el mejor constante, de modo que no habia nada que programar en el
tiempo. Su veredicto de "arena muerta" NO es valido -- las condiciones
necesarias no se cumplian. Pero al buscar el porque aparecio algo mas general:

  PROPOSICION (inutilidad de la anticipacion sin almacenamiento). Con una planta
  asintoticamente estable regulada al origen, el equilibrio en reposo es x = 0
  PARA TODO peso y PARA TODO horizonte. El punto de operacion optimo no depende
  del meta-parametro, luego no hay nada que pre-posicionar y el aviso anticipado
  es estructuralmente inutilizable por ese canal, por mucha antelacion que tenga.

Es el mismo tipo de argumento que A15 y descarta otra familia entera de arenas.
La consecuencia constructiva: hace falta un estado de ALMACENAMIENTO cuyo nivel
deseado dependa del peso. Aqui alpha mezcla dos objetivos con CONSIGNAS
DISTINTAS (operar barato en b_econ / mantener reserva en b_alta), asi que alpha
fija el punto de operacion y anticipar significa literalmente CARGAR ANTES.

PREDICCION CUANTITATIVA PRE-REGISTRADA
---------------------------------------
La antelacion optima debe ser aproximadamente el tiempo que cuesta cargar el
deficit, limitado por la autoridad del actuador y por su retardo:

    lead* ~ (b_objetivo - b_econ)/umax + tau_act

Si el barrido de `lead` sale plano, la arena esta muerta pese al almacenamiento.
Si sale con un minimo interior cerca de esa prediccion, es viable Y el mecanismo
esta identificado.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ghi import transport as tp
from ghi.stats import paired

N_MPC = 16
ALPHAS = np.round(np.linspace(0.0, 1.0, 21), 4)
B_ECON, B_ALTA, B_CRIT = 0.0, 3.0, -1.2
TAU_ACT, UMAX = 3.0, 0.6
LEADS = [0, 1, 2, 3, 5, 7, 10, 14, 20, 28]


def build():
    plant = tp.storage_plant(tau_act=TAU_ACT, umax=UMAX)
    objs = [(np.diag([1.0, 0.05]), np.array([[1.0]]), np.array([B_ECON, 0.0])),
            (np.diag([1.0, 0.05]), np.array([[1.0]]), np.array([B_ALTA, 0.0]))]
    return plant, tp.SetpointMPC(plant, objs, N_MPC, ALPHAS)


def calibrate(plant, mpc):
    """Barrido de calibracion. NO se busca que gane ningun gobernador: se busca
    la region donde las condiciones NECESARIAS se cumplen, que es un requisito
    del banco, no un resultado. Se reporta la rejilla entera.

      C2  la no linealidad esta ACTIVA (hay roturas de servicio si no se previene)
      C3  la carga preventiva CUESTA (peso alto siempre es peor que el mejor
          constante), porque si fuese gratis no habria nada que programar
    """
    print("=" * 78)
    print("P0  CALIBRACION: buscar donde se cumplen las condiciones NECESARIAS")
    print("=" * 78)
    print("  (no se busca que gane nadie: se busca que el banco no este vacio)")
    print(f"\n  {'amp':>6}{'b_crit':>8}{'pen':>6}{'a*':>6}{'coste*':>11}"
          f"{'%rot(a*)':>10}{'%rot(a=0)':>11}{'alto/const':>12}  cond.")
    mejor = None
    grid = []
    for amp in (0.75, 0.90, 1.05):
        for b_crit in (-2.5, -3.5, -4.5):
            for pen in (5.0, 15.0):
                traf = tp.Traffic(seed=0, sigma_bg=0.020, amp=amp, width=16,
                                  gap_lo=80, gap_hi=120, T=900)
                d, starts = traf.build()
                kw = dict(b_econ=B_ECON, b_crit=b_crit, pen=pen)
                res = [tp.run_storage(mpc, plant, d, a, **kw) for a in ALPHAS]
                cs = [r["coste"] for r in res]
                j = int(np.argmin(cs))
                a_c, r_c = float(ALPHAS[j]), res[j]
                r_hi = res[-1]
                interior = 0.0 < a_c < 1.0
                c2 = res[0]["roturas"] > 0.002          # sin prevenir SI rompe
                c3 = r_hi["coste"] > r_c["coste"] * 1.005
                ratio = r_hi["coste"] / max(r_c["coste"], 1e-9)
                cond = f"{'i' if interior else '-'}{'2' if c2 else '-'}{'3' if c3 else '-'}"
                print(f"  {amp:>6.2f}{b_crit:>8.1f}{pen:>6.1f}{a_c:>6.2f}"
                      f"{r_c['coste']:>11.1f}{100*r_c['roturas']:>9.1f}%"
                      f"{100*res[0]['roturas']:>10.1f}%{ratio:>12.2f}  {cond}")
                fila = dict(amp=amp, b_crit=b_crit, pen=pen, alpha=a_c,
                            coste=float(r_c["coste"]), c2=bool(c2), c3=bool(c3),
                            interior=bool(interior))
                grid.append(fila)
                if c2 and c3 and (mejor is None or res[0]["roturas"] > mejor[1]):
                    mejor = ((amp, b_crit, pen), res[0]["roturas"], a_c, r_c)
    return mejor, grid


def main() -> dict:
    out = {}
    plant, mpc = build()
    mejor, grid = calibrate(plant, mpc)
    out["calibracion"] = grid
    if mejor is None:
        print("\n  NINGUNA configuracion cumple C2 y C3 -> no hay banco que construir.")
        out["viable"] = False
        return out
    (amp, b_crit, pen), _, a_c, r_c = mejor
    print(f"\n  configuracion elegida: amp={amp} b_crit={b_crit} pen={pen}"
          f"  -> mejor constante a*={a_c:.2f}, coste {r_c['coste']:.1f}")
    traf = tp.Traffic(seed=0, sigma_bg=0.020, amp=amp, width=16,
                      gap_lo=80, gap_hi=120, T=900)
    d, starts = traf.build()
    kw = dict(b_econ=B_ECON, b_crit=b_crit, pen=pen)
    interior = 0.0 < a_c < 1.0
    c2 = True
    out["P0"] = {"amp": amp, "b_crit": b_crit, "pen": pen, "alpha_const": a_c,
                 "coste_const": float(r_c["coste"]), "roturas": float(r_c["roturas"]),
                 "interior": bool(interior), "C2": True, "n_eventos": len(starts)}

    print("\n" + "=" * 78)
    print("P1  LA PRUEBA: coste frente a ANTELACION del gobernador clarividente")
    print("=" * 78)
    mejor = None
    for a_lo in ALPHAS:
        for a_hi in ALPHAS:
            if a_hi <= a_lo:
                continue
            sch = tp.clairvoyant_schedule(traf.T, starts, traf.width, a_lo, a_hi, 0)
            c = tp.run_storage(mpc, plant, d, sch, **kw)["coste"]
            if mejor is None or c < mejor[0]:
                mejor = (c, float(a_lo), float(a_hi))
    c0, a_lo, a_hi = mejor
    print(f"\n  par (a_lo,a_hi) sintonizado CON lead=0: ({a_lo:.2f},{a_hi:.2f}) "
          f"coste {c0:.2f}   [la antelacion no recibe sintonia regalada]")
    pred = (B_ALTA * a_hi - B_ECON) / UMAX + TAU_ACT
    print(f"  prediccion pre-registrada de lead*: ~{pred:.1f} pasos")
    print(f"\n  {'lead':>6}{'coste':>12}{'vs lead=0':>12}{'vs const':>11}{'% rotura':>11}")
    filas = []
    for L in LEADS:
        sch = tp.clairvoyant_schedule(traf.T, starts, traf.width, a_lo, a_hi, L)
        rr = tp.run_storage(mpc, plant, d, sch, **kw)
        g0 = 100.0 * (c0 - rr["coste"]) / c0
        gc = 100.0 * (r_c["coste"] - rr["coste"]) / r_c["coste"]
        print(f"  {L:>6d}{rr['coste']:>12.2f}{g0:>+11.2f}%{gc:>+10.2f}%"
              f"{100*rr['roturas']:>10.1f}%")
        filas.append({"lead": L, "coste": float(rr["coste"]),
                      "ganancia_vs_lead0": float(g0), "ganancia_vs_const": float(gc),
                      "roturas": float(rr["roturas"])})
    best = max(filas, key=lambda f: f["ganancia_vs_lead0"])
    plano = best["ganancia_vs_lead0"] < 1.0
    print(f"\n  mejor antelacion: lead = {best['lead']} "
          f"({best['ganancia_vs_lead0']:+.2f}% sobre el limite reactivo) | "
          f"prediccion {pred:.1f}")
    print(f"  VEREDICTO P1: {'PLANO -> ARENA MUERTA' if plano else 'HAY VALOR EN LA ANTICIPACION -> VIABLE'}")
    out["P1"] = {"a_lo": a_lo, "a_hi": a_hi, "coste_lead0": float(c0),
                 "prediccion_lead": float(pred), "filas": filas,
                 "lead_optimo": best["lead"], "ganancia_max_pct": best["ganancia_vs_lead0"],
                 "plano": bool(plano)}

    print("\n" + "=" * 78)
    print("P2  CONTROL DE GRATUIDAD (C3): cargar SIEMPRE tiene que ser peor")
    print("=" * 78)
    r_hi = tp.run_storage(mpc, plant, d, a_hi, **kw)
    c3 = r_hi["coste"] > r_c["coste"] * 1.005
    print(f"\n  peso alto SIEMPRE (a={a_hi:.2f}): {r_hi['coste']:.2f}")
    print(f"  mejor constante     (a={a_c:.2f}): {r_c['coste']:.2f}")
    print(f"  C3 (la carga preventiva CUESTA): {'SI' if c3 else 'NO -- nada que programar'}")
    out["P2"] = {"coste_alto_siempre": float(r_hi["coste"]), "C3": bool(c3)}

    print("\n" + "=" * 78)
    print("P3  ROBUSTEZ: 10 semillas, contraste pareado lead* frente a lead=0")
    print("=" * 78)
    L_star = best["lead"] if best["lead"] > 0 else max(LEADS)
    A, Bv = [], []
    # semillas 1..10: la 0 se uso en la calibracion y no puede evaluar
    for s in range(1, 11):
        tr = tp.Traffic(seed=s, sigma_bg=0.020, amp=amp, width=16,
                        gap_lo=80, gap_hi=120, T=900)
        ds, st = tr.build()
        A.append(tp.run_storage(mpc, plant, ds,
                 tp.clairvoyant_schedule(tr.T, st, tr.width, a_lo, a_hi, 0), **kw)["coste"])
        Bv.append(tp.run_storage(mpc, plant, ds,
                  tp.clairvoyant_schedule(tr.T, st, tr.width, a_lo, a_hi, L_star), **kw)["coste"])
    A, Bv = np.array(A), np.array(Bv)
    st_ = paired([{"c": v} for v in Bv], [{"c": v} for v in A], "c",
                 a_name=f"lead={L_star}", b_name="lead=0")
    print(f"\n  lead=0 : media {A.mean():.2f}")
    print(f"  lead={L_star:<2d}: media {Bv.mean():.2f}")
    print(f"  mejora media {100*(A-Bv).mean()/A.mean():+.2f}%  |  t = {st_.t:+.2f}  "
          f"p = {st_.p:.2e}  ({(Bv<A).sum()}/10 semillas a favor)")
    out["P3"] = {"lead": int(L_star), "media_lead0": float(A.mean()),
                 "media_leadL": float(Bv.mean()), "t": float(st_.t),
                 "p": float(st_.p), "semillas_a_favor": int((Bv < A).sum())}

    print("\n" + "=" * 78)
    ok = (not plano) and c3 and interior and c2
    if ok:
        print("VEREDICTO GLOBAL: ARENA VIABLE. Las tres condiciones se cumplen y la")
        print("anticipacion compra coste real. Procede la pregunta del MECANISMO:")
        print("onda contra difusion contra LINEA DE RETARDO PURA SINTONIZADA.")
    else:
        print("VEREDICTO GLOBAL: NO VIABLE con esta calibracion. Condiciones:")
        print(f"  interior={interior}  C2(no lin. activa)={c2}  C3(carga cuesta)={c3}  "
              f"anticipacion util={not plano}")
    print("=" * 78)
    out["viable"] = bool(ok)
    return out


if __name__ == "__main__":
    res = main()
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dst = os.path.join(here, "results")
    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(dst, "transport_viability2.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    print(f"\nescrito {os.path.join(dst, 'transport_viability2.json')}")
