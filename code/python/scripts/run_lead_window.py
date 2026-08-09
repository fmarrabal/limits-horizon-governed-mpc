"""De que depende la ANTELACION UTIL, y por que decide el destino del transporte.

EL PROBLEMA QUE ESTE SCRIPT RESUELVE
-------------------------------------
El test de viabilidad establecio que en la arena con almacenamiento la
anticipacion SI compra coste (+4.6% sobre el limite reactivo, 9/10 semillas,
p=2e-4). Pero midio una antelacion optima de solo lead* = 2 pasos, frente a los
~7 que predecia el argumento de carga. Y una ventana de 2 pasos AMENAZA LA
HISTORIA ENTERA DEL TRANSPORTE: si la informacion util cabe en dos pasos, una
linea de retardo trivial la entrega y no hay nada que un campo de onda con
velocidad de propagacion ajustada pueda aportar.

Antes de implementar ningun mecanismo hay que saber DE QUE DEPENDE lead*. La
hipotesis mecanica es que la antelacion util la fija el tiempo que cuesta
mover el almacenamiento, es decir la AUTORIDAD y la INERCIA del actuador:

    lead* crece al bajar umax (menos caudal para cargar)
    lead* crece al subir tau_act (el caudal tarda mas en establecerse)
    lead* crece con la severidad del evento (hay mas deficit que cubrir)

Si lead* resulta ser estructuralmente PEQUENO en todo el rango fisico
razonable, el transporte esta muerto por la misma clase de argumento que A15 y
que la proposicion de almacenamiento: no porque el campo este mal sintonizado,
sino porque no hay ventana temporal que llenar.

Si lead* CRECE con la lentitud del actuador de forma predecible, entonces existe
un regimen fisico (actuadores lentos y de poca autoridad -- valvulas grandes,
inercia termica, bombas de gran porte) donde la antelacion util es larga y la
pregunta del mecanismo de transporte es legitima.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ghi import transport as tp

ALPHAS = np.round(np.linspace(0.0, 1.0, 21), 4)
B_ECON, B_ALTA = 0.0, 3.0
N_MPC = 16
LEADS = [0, 1, 2, 3, 4, 6, 8, 11, 15, 20, 26, 34]


def lead_star(umax, tau_act, amp, width=16, b_crit=-2.5, pen=5.0,
              seeds=(0, 1, 2)):
    """Devuelve (lead*, ganancia sobre reactivo, curva). El par (a_lo,a_hi) se
    sintoniza SIEMPRE con lead=0, de modo que la anticipacion nunca recibe una
    sintonia que el rival reactivo no tenga."""
    plant = tp.storage_plant(tau_act=tau_act, umax=umax)
    objs = [(np.diag([1.0, 0.05]), np.array([[1.0]]), np.array([B_ECON, 0.0])),
            (np.diag([1.0, 0.05]), np.array([[1.0]]), np.array([B_ALTA, 0.0]))]
    mpc = tp.SetpointMPC(plant, objs, N_MPC, ALPHAS)
    kw = dict(b_econ=B_ECON, b_crit=b_crit, pen=pen)
    trafs = [tp.Traffic(seed=s, sigma_bg=0.020, amp=amp, width=width,
                        gap_lo=80, gap_hi=120, T=700) for s in seeds]
    data = [t.build() for t in trafs]

    def cost(a_lo, a_hi, L):
        tot = 0.0
        for (d, st), tr in zip(data, trafs):
            sch = tp.clairvoyant_schedule(tr.T, st, tr.width, a_lo, a_hi, L)
            c = tp.run_storage(mpc, plant, d, sch, **kw)["coste"]
            if not np.isfinite(c):
                return np.inf
            tot += c
        return tot

    best = None
    for a_lo in ALPHAS[::2]:
        for a_hi in ALPHAS[::2]:
            if a_hi <= a_lo:
                continue
            c = cost(a_lo, a_hi, 0)
            if best is None or c < best[0]:
                best = (c, float(a_lo), float(a_hi))
    c0, a_lo, a_hi = best
    curva = [(L, cost(a_lo, a_hi, L)) for L in LEADS]
    j = int(np.argmin([c for _, c in curva]))
    Ls, cs = curva[j]
    # anchura de la ventana: leads dentro del 1% del optimo
    dentro = [L for L, c in curva if c <= cs * 1.01]
    ganancia = 100.0 * (c0 - cs) / c0
    # roturas sin prevenir, para saber que la no linealidad estaba activa
    rot = np.mean([tp.run_storage(mpc, plant, d,
                   tp.clairvoyant_schedule(tr.T, st, tr.width, a_lo, a_lo, 0),
                   **kw)["roturas"] for (d, st), tr in zip(data, trafs)])
    return dict(lead=Ls, ganancia=ganancia, a_lo=a_lo, a_hi=a_hi,
                ventana=(min(dentro), max(dentro)), roturas_base=float(rot),
                curva=[(int(L), float(c)) for L, c in curva])


def main() -> dict:
    out = {"celdas": []}
    print("=" * 78)
    print("DE QUE DEPENDE LA ANTELACION UTIL lead*")
    print("=" * 78)
    print("  ganancia = % de coste que la anticipacion compra sobre el limite REACTIVO")
    print("  ventana  = leads dentro del 1% del optimo (si es ancha, el instante")
    print("             exacto de entrega da igual y el transporte fino no aporta)")
    print(f"\n  {'umax':>6}{'tau_act':>9}{'amp':>6}{'lead*':>7}{'ventana':>11}"
          f"{'ganancia':>10}{'%rot base':>11}")
    for umax in (0.6, 0.35, 0.20):
        for tau_act in (3.0, 8.0):
            amp = round(umax + 0.45, 3)      # severidad relativa CONSTANTE
            r = lead_star(umax, tau_act, amp)
            print(f"  {umax:>6.2f}{tau_act:>9.1f}{amp:>6.2f}{r['lead']:>7d}"
                  f"{str(r['ventana']):>11}{r['ganancia']:>+9.2f}%"
                  f"{100*r['roturas_base']:>10.1f}%")
            out["celdas"].append(dict(umax=umax, tau_act=tau_act, amp=amp,
                                      lead=int(r["lead"]), ganancia=float(r["ganancia"]),
                                      ventana=[int(v) for v in r["ventana"]],
                                      roturas=float(r["roturas_base"])))
    leads = [c["lead"] for c in out["celdas"]]
    gan = [c["ganancia"] for c in out["celdas"]]
    print(f"\n  rango de lead*: {min(leads)} .. {max(leads)}  |  "
          f"ganancia {min(gan):+.2f}% .. {max(gan):+.2f}%")

    print("\n" + "=" * 78)
    print("LECTURA")
    print("=" * 78)
    if max(leads) <= 4:
        print("""  lead* es PEQUENO en todo el rango fisico barrido. La informacion
  anticipada util cabe en unos pocos pasos, luego una linea de retardo trivial
  la entrega y no queda nada que un campo de onda con velocidad de propagacion
  ajustada pueda aportar. El transporte estaria muerto por falta de VENTANA, no
  por mala sintonia del campo.""")
    else:
        print(f"""  lead* CRECE hasta {max(leads)} pasos al hacer el actuador mas lento y con
  menos autoridad. Existe entonces un regimen fisico donde la ventana de
  anticipacion es larga, y ahi la pregunta de COMO entregar la senal en el
  instante correcto (onda contra difusion contra linea de retardo pura
  sintonizada) es legitima.""")
    return out


if __name__ == "__main__":
    res = main()
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dst = os.path.join(here, "results")
    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(dst, "lead_window.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    print(f"\nescrito {os.path.join(dst, 'lead_window.json')}")
