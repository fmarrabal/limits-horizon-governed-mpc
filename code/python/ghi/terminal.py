"""Ingredientes terminales: ley auxiliar, costes terminales y region terminal.

POR QUE ESTE MODULO EXISTE
--------------------------
La revision adversarial encontro que los ingredientes heredados del TFM NO
satisfacen la desigualdad terminal. Con los pesos del banco de conflicto:

    S_0 = P_0 - Acl' P_0 Acl - Q_0 - Kf' R_0 Kf   ->  eig = [-11.04, +0.02]
    S_1 = P_1 - Acl' P_1 Acl - Q_1 - Kf' R_1 Kf   ->  eig = [ -3.38, +9.03]

violada para AMBOS objetivos; y la "region terminal" H x_N <= 10 es una CAJA,
no un conjunto invariante. Sin esos ingredientes, la restriccion contractiva
V* <= J_a NO certifica decrecimiento de Lyapunov: es solo una restriccion de
consistencia entre muestras.

Aqui se construyen bien:
  * Kf COMUN a todos los objetivos (la secuencia desplazada debe ser factible
    para todos a la vez, asi que la ley terminal no puede depender de i).
  * P_i por ecuacion de Lyapunov en tiempo discreto -> S_i = 0 EXACTAMENTE,
    que es la version mas apretada posible de la desigualdad.
  * Omega = conjunto maximo positivamente invariante de Acl dentro de las
    restricciones (algoritmo de Gilbert-Tan).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy.linalg import solve_discrete_are, solve_discrete_lyapunov
from scipy.optimize import linprog

from .plant import Objective, Plant, Problem


@dataclass
class Terminal:
    """Ingredientes terminales de un problema multiobjetivo."""

    Kf: np.ndarray           # (m, n)  ley auxiliar COMUN
    P: List[np.ndarray]      # coste terminal por objetivo
    H: np.ndarray            # Omega = {x : H x <= k}
    k: np.ndarray
    n_iter: int = 0          # iteraciones de Gilbert-Tan hasta converger

    @property
    def Acl_gain(self) -> np.ndarray:
        return self.Kf

    def contains(self, x: np.ndarray, tol: float = 1e-9) -> bool:
        return bool(np.all(self.H @ np.asarray(x, float).ravel() <= self.k + tol))


# --------------------------------------------------------------------------
#                            LEY AUXILIAR
# --------------------------------------------------------------------------

def lqr_gain(plant: Plant, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    """u = -K x  optimo LQR discreto (convenio: devuelve K con el signo YA
    incorporado, es decir  u = Kf x  con Kf = -K)."""
    P = solve_discrete_are(plant.A, plant.B, Q, R)
    K = np.linalg.solve(R + plant.B.T @ P @ plant.B, plant.B.T @ P @ plant.A)
    return -K


def common_gain(problem: Problem, weights: np.ndarray | None = None) -> np.ndarray:
    """Ley terminal COMUN. Por defecto, el LQR de la mezcla uniforme de los
    objetivos cuadraticos (los de norma infinito se convierten a su cota
    cuadratica Q'Q para este calculo, que solo fija Kf)."""
    n_obj = problem.n_obj
    w = np.full(n_obj, 1.0 / n_obj) if weights is None else np.asarray(weights, float)
    if w.size != n_obj or np.any(w < 0):
        raise ValueError("weights debe ser un vector de tamano n_obj y no negativo")
    Q = np.zeros((problem.plant.n, problem.plant.n))
    R = np.zeros((problem.plant.m, problem.plant.m))
    for wi, ob in zip(w, problem.objectives):
        if ob.kind == "quad":
            Q += wi * ob.Q
            R += wi * ob.R
        else:  # ||Qx||_inf esta acotada por debajo por x'Q'Qx / ||Q||; basta para fijar Kf
            Q += wi * (ob.Q.T @ ob.Q)
            R += wi * (ob.R.T @ ob.R)
    # regularizacion minima para que el ARE este bien puesto
    Q = Q + 1e-9 * np.eye(Q.shape[0])
    R = R + 1e-9 * np.eye(R.shape[0])
    return lqr_gain(problem.plant, Q, R)


# --------------------------------------------------------------------------
#                          COSTES TERMINALES
# --------------------------------------------------------------------------

def lyapunov_P(plant: Plant, Kf: np.ndarray, ob: Objective) -> np.ndarray:
    """P tal que  P = Acl' P Acl + Q + Kf' R Kf, es decir S = 0 exactamente.

    Solo para objetivos cuadraticos. `solve_discrete_lyapunov(a, q)` resuelve
    a X a' - X + q = 0, asi que hay que pasarle Acl' para obtener la forma
    Acl' P Acl - P + W = 0.
    """
    if ob.kind != "quad":
        raise ValueError("lyapunov_P solo vale para objetivos cuadraticos")
    Acl = plant.A + plant.B @ Kf
    W = ob.Q + Kf.T @ ob.R @ Kf
    P = solve_discrete_lyapunov(Acl.T, W)
    return 0.5 * (P + P.T)


def terminal_slack(plant: Plant, Kf: np.ndarray, ob: Objective, P: np.ndarray) -> np.ndarray:
    """S = P - Acl' P Acl - Q - Kf' R Kf. Debe ser semidefinida positiva."""
    Acl = plant.A + plant.B @ Kf
    S = P - Acl.T @ P @ Acl - ob.Q - Kf.T @ ob.R @ Kf
    return 0.5 * (S + S.T)


# --------------------------------------------------------------------------
#                    REGION TERMINAL (Gilbert-Tan)
# --------------------------------------------------------------------------

def _box_polytope(plant: Plant, Kf: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """{x : |x| <= xmax, |Kf x| <= umax} en forma H x <= k."""
    n = plant.n
    H = np.vstack([np.eye(n), -np.eye(n), Kf, -Kf])
    k = np.concatenate([plant.xmax, plant.xmax, plant.umax, plant.umax])
    return H, k


def _remove_redundant(H: np.ndarray, k: np.ndarray, tol: float = 1e-9
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """Elimina desigualdades redundantes resolviendo un LP por fila."""
    keep: List[int] = []
    for j in range(H.shape[0]):
        idx = [i for i in range(H.shape[0]) if i != j and i in keep or (i != j and i > j)]
        if not idx:
            keep.append(j)
            continue
        # max H_j x  s.a.  H_idx x <= k_idx  (linprog minimiza)
        res = linprog(-H[j], A_ub=H[idx], b_ub=k[idx], bounds=[(None, None)] * H.shape[1],
                      method="highs")
        if not res.success:      # no acotado o infactible -> la fila hace falta
            keep.append(j)
        elif -res.fun > k[j] + tol:
            keep.append(j)
    if not keep:
        keep = list(range(H.shape[0]))
    return H[keep], k[keep]


def invariant_set(plant: Plant, Kf: np.ndarray, max_iter: int = 60, tol: float = 1e-9
                  ) -> Tuple[np.ndarray, np.ndarray, int]:
    """Conjunto maximo positivamente invariante de x+ = Acl x dentro de las
    restricciones de estado y de la entrada terminal u = Kf x.

    Algoritmo clasico (Gilbert & Tan 1991): O_0 = X, O_{j+1} = O_j ^ Acl^{-1} O_j,
    parar cuando anadir Acl^{j+1} no recorta nada.
    """
    Acl = plant.A + plant.B @ Kf
    H0, k0 = _box_polytope(plant, Kf)
    H, k = H0.copy(), k0.copy()
    Apow = np.eye(plant.n)
    for it in range(1, max_iter + 1):
        Apow = Acl @ Apow
        Hn, kn = H0 @ Apow, k0
        # ya esta contenido?  max (Hn_j x - kn_j) sobre {H x <= k} <= 0 para todo j
        redundant = True
        for j in range(Hn.shape[0]):
            res = linprog(-Hn[j], A_ub=H, b_ub=k, bounds=[(None, None)] * plant.n,
                          method="highs")
            if res.success and (-res.fun) > kn[j] + tol:
                redundant = False
                break
            if not res.success:
                redundant = False
                break
        if redundant:
            Hr, kr = _remove_redundant(H, k)
            return Hr, kr, it
        H = np.vstack([H, Hn])
        k = np.concatenate([k, kn])
    raise RuntimeError(f"Gilbert-Tan no convergio en {max_iter} iteraciones")


# --------------------------------------------------------------------------
#                                 DISENO
# --------------------------------------------------------------------------

def design(problem: Problem, weights: np.ndarray | None = None,
           max_iter: int = 60) -> Tuple[Problem, Terminal]:
    """Calcula (Kf, P_i, Omega) correctos y devuelve el problema con los P puestos.

    Los objetivos de norma infinito NO admiten la construccion por Lyapunov; se
    dejan con el P que trajeran y `audit` reporta si cumplen o no. Es deliberado:
    preferimos un fallo declarado a un ingrediente inventado.
    """
    Kf = common_gain(problem, weights)
    P: List[np.ndarray] = []
    new_objs: List[Objective] = []
    for ob in problem.objectives:
        if ob.kind == "quad":
            Pi = lyapunov_P(problem.plant, Kf, ob)
        else:
            if ob.P is None:
                raise ValueError(
                    f"el objetivo '{ob.name}' es de norma infinito y no trae P; "
                    "la construccion por Lyapunov no aplica, hay que darlo a mano")
            Pi = ob.P
        P.append(Pi)
        new_objs.append(ob.with_P(Pi))
    H, k, it = invariant_set(problem.plant, Kf, max_iter=max_iter)
    prob = Problem(problem.plant, new_objs, problem.N, problem.name)
    return prob, Terminal(Kf=Kf, P=P, H=H, k=k, n_iter=it)


# --------------------------------------------------------------------------
#                                AUDITORIA
# --------------------------------------------------------------------------

def audit(problem: Problem, term: Terminal, n_samples: int = 400,
          seed: int = 0, tol: float = 1e-8) -> dict:
    """Comprueba TODO lo que la revision adversarial senalo como no comprobado."""
    plant = problem.plant
    Acl = plant.A + plant.B @ term.Kf
    rng = np.random.default_rng(seed)
    out: dict = {"Kf": term.Kf, "rho_Acl": float(np.max(np.abs(np.linalg.eigvals(Acl))))}

    # (1) desigualdad terminal por objetivo
    slacks = []
    for ob, Pi in zip(problem.objectives, term.P):
        if ob.kind == "quad":
            S = terminal_slack(plant, term.Kf, ob, Pi)
            ev = np.linalg.eigvalsh(S)
            slacks.append({"name": ob.name, "kind": ob.kind,
                           "eig_min": float(ev.min()), "eig": ev.tolist(),
                           "ok": bool(ev.min() >= -tol)})
        else:
            # version por muestreo: F(Acl x) + l(x, Kf x) <= F(x) para x en Omega
            worst, bad = -np.inf, 0
            for _ in range(n_samples):
                x = _sample_polytope(term.H, term.k, plant.n, rng)
                if x is None:
                    continue
                u = term.Kf @ x
                d = ob.terminal(Acl @ x) + ob.stage(x, u) - ob.terminal(x)
                worst = max(worst, d)
                bad += int(d > tol)
            slacks.append({"name": ob.name, "kind": ob.kind,
                           "peor_violacion": float(worst),
                           "muestras_malas": bad, "muestras": n_samples,
                           "ok": bool(worst <= tol)})
    out["terminal"] = slacks
    out["terminal_ok"] = all(s["ok"] for s in slacks)

    # (2) invariancia positiva de Omega bajo Acl
    worst_inv, n_ok = -np.inf, 0
    for _ in range(n_samples):
        x = _sample_polytope(term.H, term.k, plant.n, rng)
        if x is None:
            continue
        n_ok += 1
        worst_inv = max(worst_inv, float(np.max(term.H @ (Acl @ x) - term.k)))
    out["invariancia"] = {"peor_violacion": float(worst_inv), "muestras": n_ok,
                          "ok": bool(worst_inv <= tol)}

    # (3) Omega respeta las restricciones de estado y de entrada
    worst_con = -np.inf
    for _ in range(n_samples):
        x = _sample_polytope(term.H, term.k, plant.n, rng)
        if x is None:
            continue
        worst_con = max(worst_con,
                        float(np.max(np.abs(x) - plant.xmax)),
                        float(np.max(np.abs(term.Kf @ x) - plant.umax)))
    out["restricciones"] = {"peor_violacion": float(worst_con), "ok": bool(worst_con <= tol)}

    out["ok"] = bool(out["terminal_ok"] and out["invariancia"]["ok"]
                     and out["restricciones"]["ok"])
    return out


def _sample_polytope(H: np.ndarray, k: np.ndarray, n: int, rng, tries: int = 200):
    """Rechazo simple dentro de la caja que envuelve al politopo."""
    # cota de caja por LP en cada eje
    lo, hi = np.zeros(n), np.zeros(n)
    for j in range(n):
        e = np.zeros(n); e[j] = 1.0
        r1 = linprog(e, A_ub=H, b_ub=k, bounds=[(None, None)] * n, method="highs")
        r2 = linprog(-e, A_ub=H, b_ub=k, bounds=[(None, None)] * n, method="highs")
        lo[j] = r1.fun if r1.success else -1.0
        hi[j] = -r2.fun if r2.success else 1.0
    for _ in range(tries):
        x = rng.uniform(lo, hi)
        if np.all(H @ x <= k + 1e-12):
            return x
    return None
