"""Invariancia de escala: por que la demanda por AMPLITUD no puede modular nada.

EL RESULTADO
------------
En un lazo lineal con coste cuadratico y SIN restricciones activas, escalar la
perturbacion por lambda escala la trayectoria optima por lambda y el coste por
lambda^2, PARA CADA valor fijo del meta-parametro. Luego

    argmin_alpha  y  argmin_N   son INVARIANTES a la amplitud.

Demostracion, una linea: si d -> lambda d, por linealidad u* -> lambda u*,
x* -> lambda x*, y J -> lambda^2 J para todo (alpha, N) fijo. Un factor comun
positivo no mueve el argmin.

CONSECUENCIA, Y ES LA QUE IMPORTA PARA EL PROGRAMA GHI
-------------------------------------------------------
Una envolvente que module la AMPLITUD de la perturbacion -- que es la forma
canonica de obtener una demanda de BANDA ESTRECHA OSCILATORIA (batido, grupos
de oleaje, desequilibrio rotativo) -- NO genera ningun meta-parametro optimo
variable en el tiempo mientras el lazo se mantenga lineal y sin saturar. No hay
nada que modular, y por tanto no hay ventaja posible para NINGUN gobernador,
inercial o no.

Para que aparezca valor hace falta cruzar una NO LINEALIDAD (saturacion del
actuador, barrera de restriccion, coste superlineal). Y ahi esta la tension que
este script documenta: la region donde la no linealidad esta MARGINALMENTE
activa -- la unica donde el meta-parametro optimo depende del nivel -- es
estrecha y esta pegada al margen de autoridad del actuador. Por debajo el
horizonte corto basta; por encima no hay horizonte que salve el lazo.

Esto se comprueba en dos partes:
  P1  invariancia de escala del alpha optimo y del N optimo (regimen lineal)
  P2  la ventana de saturacion donde N* SI depende del nivel, y su estrechez
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ghi import terminal as gterm
from ghi import vibration as vb
from ghi.mompc import MOMPC, solve
from ghi.plant import Problem
from ghi.suboptimality import HorizonMPC


def loop_weight(M, plant, d, a, T=500, warm=80):
    x = np.zeros(2); tot = 0.0
    for t in range(T):
        s = solve(M, x, np.array([1 - a, a]))
        if s is None:
            return np.inf
        u = float(s.U[0, 0])
        if t >= warm:
            tot += float(x[0] ** 2 + 0.05 * x[1] ** 2 + u * u)
        w = np.zeros(2); w[1] = d[t]
        x = plant.A @ x + plant.B.ravel() * u + w
        if np.max(np.abs(x)) > 1e6:
            return np.inf
    return tot


def loop_horizon(mpc, plant, d, N, T=500, warm=80, clip=True):
    x = np.zeros(2); tot = 0.0
    for t in range(T):
        u = mpc.solve(x, N).u0
        if t >= warm:
            tot += float(x[0] ** 2 + 0.05 * x[1] ** 2 + u * u)
        w = np.zeros(2); w[1] = d[t]
        x = plant.A @ x + plant.B.ravel() * u + w
        if clip:
            x = np.clip(x, -plant.xmax, plant.xmax)
        if np.max(np.abs(x)) > 1e7:
            return np.inf
    return tot


def main() -> dict:
    print("=" * 78)
    print("P1  INVARIANCIA DE ESCALA EN EL REGIMEN LINEAL (sin saturar)")
    print("=" * 78)

    # actuador amplio: nunca satura en el rango barrido -> regimen lineal puro
    p_lin = vb.vibration_plant(umax=1e4, xmax=1e5, vmax=1e5)
    prob = Problem(p_lin, vb.problem_vibration().objectives, N=10, name="lin")
    prob, term = gterm.design(prob)
    Mw = MOMPC(prob, term)
    mpc = HorizonMPC(p_lin, Q=np.diag([1.0, 0.05]), R=np.array([[1.0]]),
                     N_min=2, N_max=20)

    lambdas = [0.5, 1.0, 2.0, 4.0, 8.0]
    alphas = np.linspace(0.1, 0.9, 9)
    Ns = [2, 3, 4, 6, 8, 12, 20]

    print(f"\n  alpha optimo frente a la escala de la perturbacion:")
    print(f"  {'lambda':>7}{'alpha*':>9}{'coste*':>14}{'coste*/lambda^2':>18}")
    a_stars, cost_scaled = [], []
    for lam in lambdas:
        d, _ = vb.Excitation(E0=1.0 * lam, depth=0.6).signal(500)
        cs = [loop_weight(Mw, p_lin, d, a) for a in alphas]
        j = int(np.argmin(cs))
        a_stars.append(float(alphas[j])); cost_scaled.append(cs[j] / lam ** 2)
        print(f"  {lam:>7.1f}{alphas[j]:>9.2f}{cs[j]:>14.1f}{cs[j]/lam**2:>18.4f}")
    inv_a = len(set(a_stars)) == 1
    spread_a = float(np.std(cost_scaled) / max(np.mean(cost_scaled), 1e-12))

    print(f"\n  N optimo frente a la escala de la perturbacion:")
    print(f"  {'lambda':>7}{'N*':>6}{'coste*':>14}{'coste*/lambda^2':>18}")
    N_stars, costN_scaled = [], []
    for lam in lambdas:
        d, _ = vb.Excitation(E0=1.0 * lam, depth=0.6).signal(500)
        cs = [loop_horizon(mpc, p_lin, d, N, clip=False) for N in Ns]
        j = int(np.argmin(cs))
        N_stars.append(Ns[j]); costN_scaled.append(cs[j] / lam ** 2)
        print(f"  {lam:>7.1f}{Ns[j]:>6d}{cs[j]:>14.1f}{cs[j]/lam**2:>18.4f}")
    inv_N = len(set(N_stars)) == 1
    spread_N = float(np.std(costN_scaled) / max(np.mean(costN_scaled), 1e-12))

    print(f"\n  alpha* invariante: {inv_a}  (valores {sorted(set(a_stars))})")
    print(f"  N* invariante    : {inv_N}  (valores {sorted(set(N_stars))})")
    print(f"  coste/lambda^2 constante: dispersion relativa {spread_a:.2e} (peso), "
          f"{spread_N:.2e} (horizonte)")
    print("\n  => en regimen lineal NO HAY NADA QUE MODULAR por amplitud.")

    print("\n" + "=" * 78)
    print("P2  LA VENTANA DONDE LA SATURACION HACE QUE N* DEPENDA DEL NIVEL")
    print("=" * 78)
    print("  (ratio = coste(N=2)/coste(N*): cuanto se gana usando el horizonte correcto)")
    p_sat = vb.vibration_plant(umax=1.2)
    mpc_s = HorizonMPC(p_sat, Q=np.diag([1.0, 0.05]), R=np.array([[0.05]]),
                       N_min=2, N_max=24)
    Ns2 = [2, 4, 6, 8, 12, 16, 24]
    print(f"\n  {'E':>6}{'N*':>5}{'N sufic.':>10}{'ratio N2/N*':>14}{'coste*':>13}  regimen")
    window = []
    for Ec in [0.5, 1.0, 1.5, 1.8, 2.0, 2.2, 2.5, 3.0, 3.5, 5.0]:
        d, _ = vb.Excitation(E0=Ec, depth=0.0).signal(600)
        cs = [loop_horizon(mpc_s, p_sat, d, N, T=600) for N in Ns2]
        j = int(np.argmin(cs))
        suf = [n for n, c in zip(Ns2, cs) if c <= cs[j] * 1.01][0]
        ratio = cs[0] / cs[j] if cs[j] > 0 else np.nan
        # el criterio es el RATIO (cuanto se gana eligiendo bien N), no el coste
        # absoluto: un nivel caro donde N no cambia nada no es ventana
        if ratio > 1.2:
            reg = "<<< VENTANA: el horizonte IMPORTA"
            window.append(Ec)
        elif cs[j] > 1e4:
            reg = "saturado (caro, pero ningun N lo arregla)"
        else:
            reg = "holgado (N_min basta)"
        print(f"  {Ec:>6.1f}{Ns2[j]:>5d}{suf:>10d}{ratio:>14.2f}{cs[j]:>13.1f}  {reg}")

    print(f"\n  ventana medida: E en {window if window else '(vacia)'}")
    print("  => la region donde el meta-parametro optimo depende del nivel es")
    print("     ESTRECHA y esta pegada al margen de autoridad del actuador.")

    print("\n" + "=" * 78)
    print("LA TENSION ESTRUCTURAL QUE ESTO DEJA AL DESCUBIERTO")
    print("=" * 78)
    print("""  Para que MODULAR pague hace falta cruzar una no linealidad -> la demanda
  resulta EVENTUAL (pulsos) -> banda ANCHA -> gana el integrador con fugas.
  Para que la INERCIA pague hace falta demanda de banda ESTRECHA (tonos) ->
  la forma canonica de obtenerla es modular AMPLITUDES -> y por invariancia
  de escala eso no genera ningun optimo variable en el tiempo.

  Las dos condiciones que el campo inercial necesita estan EN TENSION.""")

    return {"P1": {"lambdas": lambdas, "alpha_star": a_stars, "N_star": N_stars,
                   "alpha_invariante": bool(inv_a), "N_invariante": bool(inv_N),
                   "dispersion_coste_escalado_peso": spread_a,
                   "dispersion_coste_escalado_horizonte": spread_N},
            "P2": {"ventana_E": window}}


if __name__ == "__main__":
    res = main()
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dst = os.path.join(here, "results")
    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(dst, "scale_invariance.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    print(f"\nescrito {os.path.join(dst, 'scale_invariance.json')}")
