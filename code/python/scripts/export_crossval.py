"""Exporta los casos canonicos que MATLAB debe reproducir.

LA IDEA DE LA VALIDACION CRUZADA
--------------------------------
No basta con que las dos implementaciones den resultados "parecidos": eso lo
consigue cualquier par de programas que resuelvan problemas distintos pero
similares. Lo que se compara aqui es la FORMULACION:

  1. los ingredientes terminales (Kf, P_i, Omega) -- construccion, no solucion
  2. las matrices del QP condensado (G, g, Ain, bin) elemento a elemento
  3. la solucion del QP y los costes J
  4. la respuesta en z de los reguladores y el tau de igualacion
  5. los operadores del campo sobre el grafo y el umbral de flutter

Si (1) y (2) coinciden a 1e-12, las dos implementaciones son literalmente el
mismo problema y solo puede diferir el solver. Si ademas (3) coincide a la
tolerancia del solver, la validacion es completa.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ghi import field as gfield
from ghi import plant as gplant
from ghi import regulators as greg
from ghi import terminal as gterm
from ghi.mompc import MOMPC, shifted_sequence, solve


def _l(a):
    """np -> lista anidada de floats (json no traga arrays de numpy)."""
    return np.asarray(a, float).tolist()


def main(out_path: str) -> dict:
    prob = gplant.problem_conflict()
    prob, term = gterm.design(prob)
    M = MOMPC(prob, term)

    data = {
        "meta": {
            "descripcion": "Casos canonicos GHI para validacion cruzada Python<->MATLAB",
            "problema": prob.name,
            "N": prob.N,
            "n_obj": prob.n_obj,
        },
        "plant": {
            "A": _l(prob.plant.A), "B": _l(prob.plant.B),
            "xmax": _l(prob.plant.xmax), "umax": _l(prob.plant.umax),
        },
        "objectives": [
            {"name": ob.name, "kind": ob.kind, "Q": _l(ob.Q), "R": _l(ob.R), "P": _l(ob.P)}
            for ob in prob.objectives
        ],
        "terminal": {
            "Kf": _l(term.Kf), "P": [_l(p) for p in term.P],
            "H": _l(term.H), "k": _l(term.k), "n_iter": term.n_iter,
            "rho_Acl": float(np.max(np.abs(np.linalg.eigvals(
                prob.plant.A + prob.plant.B @ term.Kf)))),
        },
        "prediction": {"Sx": _l(M.Sx), "Su": _l(M.Su)},
        "constraints": {"Ain": _l(M.Ain), "b_const": _l(M._b_const), "B_x": _l(M._B_x)},
        "cases": [],
        "regulators": {},
        "field": {},
    }

    # --------------------------- casos de QP ---------------------------
    rng = np.random.default_rng(12345)
    x_list = [np.array([5.0, 5.0]), np.array([3.0, -2.0]), np.array([-4.0, 1.0]),
              np.array([8.0, -3.0]), np.array([-1.5, 2.5]), np.array([0.0, 0.0])]
    while len(x_list) < 12:
        x = rng.uniform(-7, 7, 2)
        if solve(M, x, np.array([0.5, 0.5])) is not None:
            x_list.append(x)
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]

    for x0 in x_list:
        for a in alphas:
            av = np.array([1.0 - a, a])
            qp = M.build(x0, av)
            sol = solve(M, x0, av, backend="CLARABEL")
            if sol is None:
                continue
            Us = shifted_sequence(M, sol)
            case = {
                "x0": _l(x0), "alpha": _l(av),
                "G": _l(qp.G), "g": _l(qp.g), "bin": _l(qp.bin),
                "U": _l(sol.U), "X": _l(sol.X), "J": _l(sol.J), "V": float(sol.V),
                "U_shift": _l(Us), "J_shift": _l(prob.costs(x0, Us)),
            }
            data["cases"].append(case)

    # ------------------------- reguladores ------------------------------
    regs = []
    for w0, zeta in ((0.9, 0.5), (0.6, 0.35), (1.2, 0.7)):
        tau, g = greg.match_first_order(w0, zeta, 1.0)
        Ad, Bd, Cd, Dd = greg.ss_second_order(w0, zeta)
        A1, B1, C1, D1 = greg.ss_first_order(tau)
        ws = [np.pi, np.pi / 2, 2 * np.pi / 20, 2 * np.pi / 13]
        regs.append({
            "w0": w0, "zeta": zeta, "tau": tau, "g_nyquist": g,
            "A2": _l(Ad), "B2": _l(Bd), "C2": _l(Cd), "D2": _l(Dd),
            "A1": _l(A1), "B1": _l(B1),
            "w": ws,
            "mag2": [greg.mag_at(Ad, Bd, Cd, Dd, w) for w in ws],
            "mag1": [greg.mag_at(A1, B1, C1, D1, w) for w in ws],
            "pha2": [greg.phase_at(Ad, Bd, Cd, Dd, w) for w in ws],
            "pha1": [greg.phase_at(A1, B1, C1, D1, w) for w in ws],
            "overshoot_exec": greg.effective_overshoot(greg.Order2(w0, zeta, 0.5)),
        })
    data["regulators"]["matching"] = regs

    # ---------------------------- campo ---------------------------------
    L, A = gfield.chain_graph(6)
    p = gfield.FieldParams(w0=1.0, zeta=0.15, c=0.35, D=0.05, b=0.4, beta=0.02)
    K, C, G = gfield.operators(L, A, p)
    data["field"] = {
        "M": 6, "L": _l(L), "A": _l(A),
        "params": {"w0": p.w0, "zeta": p.zeta, "c": p.c, "D": p.D, "b": p.b, "beta": p.beta},
        "K": _l(K), "C": _l(C), "G": _l(G),
        "rho_G": gfield.rho_G(A, p.b, p.beta),
        "merkin": gfield.merkin_threshold(p),
        "abscissa_gyro": float(np.max(np.linalg.eigvals(
            gfield.wave_state(K, C, G, "gyroscopic")).real)),
        "abscissa_circ": float(np.max(np.linalg.eigvals(
            gfield.wave_state(K, C, G, "circulatory")).real)),
    }
    deg = gfield.FieldParams(w0=1.0, zeta=0.15, c=0.0, D=0.0, b=0.0)
    st = gfield.collocation_study(M=6, p=deg)
    data["field"]["merkin_degenerado"] = {
        "beta_pred": st["beta_pred"], "beta_obs": st["beta_obs"],
        "ratio": st["ratio_obs_pred"], "rho_A3": st["rho_A3"],
    }

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1)
    print(f"escrito {out_path}")
    print(f"  casos de QP: {len(data['cases'])}")
    print(f"  Ain: {np.asarray(data['constraints']['Ain']).shape}")
    print(f"  reguladores: {len(regs)}")
    return data


if __name__ == "__main__":
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out = os.path.join(here, "crossval", "python_reference.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    main(out)
