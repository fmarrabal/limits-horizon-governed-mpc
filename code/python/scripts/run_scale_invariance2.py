"""Invariancia de escala, medida bien: optimos INTERIORES y la hipotesis que faltaba.

QUE ESTABA MAL EN LA PRIMERA MEDIDA
------------------------------------
  1. LOS OPTIMOS CAIAN EN EL BORDE de la rejilla (alpha* = 0.9 con rejilla hasta
     0.9; N* = 20 con rejilla hasta 20). Un argmin pegado al extremo es el caso
     en que MENOS puede moverse, luego era la prueba menos informativa posible
     de que el argmin no se mueve. El propio manuscrito prohibe aceptar optimos
     de borde; no podia saltarse su propia regla aqui.
  2. LAS DOS CIFRAS QUE SE CITABAN (9.15e-14 y 8.86e-17) no miden el argmin:
     miden la dispersion del COSTE normalizado. Son la verificacion de la
     homogeneidad J -> lambda^2 J, que es otra afirmacion. Hay que separarlas.
  3. FALTABA UNA HIPOTESIS: x_0 = 0. Con estado inicial no nulo la respuesta
     libre NO escala con lambda, el coste deja de ser homogeneo y el argmin SI
     se mueve. Los scripts la usaban (todos arrancan en x = 0) pero el enunciado
     no la decia.

Aqui se miden las tres cosas por separado, y la tercera se convierte en una
comprobacion positiva: se muestra que al violar la hipotesis el resultado se
rompe, que es la unica forma de demostrar que la hipotesis no es decorativa.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ghi import terminal as gterm, vibration as vb
from ghi.mompc import MOMPC
from ghi.plant import Problem
from ghi.suboptimality import HorizonMPC
import run_scale_invariance as rs

LAMBDAS = [0.5, 1.0, 2.0, 4.0, 8.0]


# Peso del esfuerzo en el coste de EVALUACION. Se elige r = 40 porque con r
# menor el optimo del peso cae en el extremo del simplex, y un argmin de borde
# es el caso en que menos puede moverse: seria la prueba menos informativa
# posible de que el argmin no se mueve. El coste tiene que ser CUADRATICO: la
# proposicion no dice nada sobre costes de otro grado (con el cuartico de fatiga
# de la arena de vibracion el argmin SI se mueve, y con razon).
R_EVAL = 40.0


def loop_w(Mw, p, d, a, x0, T=400, warm=60, r=R_EVAL):
    x = np.array(x0, float)
    tot = 0.0
    from ghi.mompc import solve
    for t in range(T):
        s = solve(Mw, x, np.array([1 - a, a]))
        if s is None:
            return np.inf
        u = float(s.U[0, 0])
        if t >= warm:
            tot += float(x[0] ** 2 + r * u * u)
        w = np.zeros(2); w[1] = d[t]
        x = p.A @ x + p.B.ravel() * u + w
        if np.max(np.abs(x)) > 1e7:
            return np.inf
    return tot


def loop_N(hm, p, d, N, x0, T=400, warm=60):
    x = np.array(x0, float)
    tot = 0.0
    for t in range(T):
        u = hm.solve(x, N).u0
        if t >= warm:
            tot += float(x[0] ** 2 + 0.05 * x[1] ** 2 + u * u)
        w = np.zeros(2); w[1] = d[t]
        x = p.A @ x + p.B.ravel() * u + w
        if np.max(np.abs(x)) > 1e7:
            return np.inf
    return tot


def main() -> dict:
    p = vb.vibration_plant(umax=1e4, xmax=1e5, vmax=1e5)   # nunca satura
    prob = Problem(p, vb.problem_vibration().objectives, N=10, name="lin")
    prob, term = gterm.design(prob)
    Mw = MOMPC(prob, term)
    hm = HorizonMPC(p, Q=np.diag([1.0, 0.05]), R=np.array([[1.0]]),
                    N_min=2, N_max=40)
    # rejillas AMPLIADAS hasta que el optimo quede dentro
    ALS = np.round(np.linspace(0.05, 0.95, 19), 4)
    NS = [2, 3, 4, 6, 8, 12, 16, 20, 26, 32, 40]
    out = {}

    print("=" * 78)
    print("P1  ARGMIN INVARIANTE, con rejillas ampliadas y x_0 = 0")
    print("=" * 78)
    print(f"  peso en [{ALS[0]}, {ALS[-1]}] ({len(ALS)} puntos) | "
          f"horizonte en {NS[0]}..{NS[-1]}")
    print(f"\n  {'lambda':>7}{'alpha*':>9}{'interior?':>11}{'N*':>6}{'interior?':>11}"
          f"{'J*/lambda^2 (peso)':>21}")
    aa, nn, cw, cn = [], [], [], []
    for L in LAMBDAS:
        d, _ = vb.Excitation(E0=L, depth=0.6).signal(400)
        vs = [loop_w(Mw, p, d, a, (0.0, 0.0)) for a in ALS]
        j = int(np.argmin(vs)); aa.append(float(ALS[j])); cw.append(vs[j] / L ** 2)
        vs2 = [loop_N(hm, p, d, N, (0.0, 0.0)) for N in NS]
        k = int(np.argmin(vs2)); nn.append(NS[k]); cn.append(vs2[k] / L ** 2)
        ia = 0 < j < len(ALS) - 1
        iN = 0 < k < len(NS) - 1
        print(f"  {L:>7.1f}{ALS[j]:>9.3f}{'si' if ia else 'NO':>11}{NS[k]:>6}"
              f"{'si' if iN else 'NO':>11}{vs[j]/L**2:>21.6f}")
    inv_a, inv_N = len(set(aa)) == 1, len(set(nn)) == 1
    disp_w = float(np.std(cw) / max(np.mean(cw), 1e-12))
    disp_n = float(np.std(cn) / max(np.mean(cn), 1e-12))
    print(f"\n  AFIRMACION 1 -- el argmin no se mueve: peso {inv_a} (valores "
          f"{sorted(set(aa))}), horizonte {inv_N} (valores {sorted(set(nn))})")
    print(f"  AFIRMACION 2 -- homogeneidad J -> lambda^2 J: dispersion relativa "
          f"del coste normalizado {disp_w:.2e} (peso), {disp_n:.2e} (horizonte)")
    print("  (son afirmaciones DISTINTAS: la segunda no implica la primera)")
    out["P1"] = {"alphas": aa, "Ns": nn, "argmin_invariante_peso": bool(inv_a),
                 "argmin_invariante_horizonte": bool(inv_N),
                 "dispersion_coste_peso": disp_w,
                 "dispersion_coste_horizonte": disp_n,
                 "interior_peso": bool(0 < ALS.tolist().index(aa[0]) < len(ALS) - 1),
                 "interior_horizonte": bool(0 < NS.index(nn[0]) < len(NS) - 1)}

    print("\n" + "=" * 78)
    print("P2  LA HIPOTESIS x_0 = 0 NO ES DECORATIVA")
    print("=" * 78)
    print("  con estado inicial NO nulo la respuesta libre no escala con lambda,")
    print("  el coste deja de ser homogeneo y el argmin SI se mueve\n")
    print(f"  {'x_0':>12}{'lambda':>9}{'N*':>6}{'J*/lambda^2':>16}")
    rot = {}
    for x0 in ((0.0, 0.0), (0.0, -4.0)):
        Nl, cl = [], []
        for L in (0.01, 10.0):
            d, _ = vb.Excitation(E0=L, depth=0.6).signal(60)
            NS2 = (2, 3, 4, 6, 8)      # rejilla ampliada: N*=4 quedaba en el borde
            vs = [loop_N(hm, p, d, N, x0, T=60, warm=0) for N in NS2]
            k = int(np.argmin(vs)); Nl.append(NS2[k]); cl.append(vs[k] / L ** 2)
            print(f"  {str(x0):>12}{L:>9.2f}{Nl[-1]:>6}{cl[-1]:>16.4f}")
        rot[str(x0)] = {"N": Nl, "invariante": len(set(Nl)) == 1,
                        "dispersion": float(np.std(cl) / max(np.mean(cl), 1e-12))}
        print(f"  {'':>12}-> argmin invariante: {rot[str(x0)]['invariante']}, "
              f"dispersion del coste normalizado {rot[str(x0)]['dispersion']:.2e}\n")
    out["P2_hipotesis"] = rot

    print("=" * 78)
    ok = (out["P1"]["argmin_invariante_peso"] and out["P1"]["argmin_invariante_horizonte"]
          and rot["(0.0, 0.0)"]["invariante"] and not rot["(0.0, -4.0)"]["invariante"])
    print("VEREDICTO:", "la proposicion se sostiene CON su hipotesis, y se rompe SIN ella"
          if ok else "revisar: el patron esperado no se reproduce")
    return out


if __name__ == "__main__":
    res = main()
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dst = os.path.join(here, "results")
    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(dst, "scale_invariance2.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    print(f"\nescrito {os.path.join(dst, 'scale_invariance2.json')}")
