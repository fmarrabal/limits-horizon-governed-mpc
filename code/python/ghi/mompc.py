"""MPC multiobjetivo escalarizado, en forma de QP denso explicito.

    U*(x, alpha) = argmin_U  sum_i alpha_i J_i(U, x)
    s.a.  x_{k+1} = A x_k + B u_k,  |x| <= xmax, |u| <= umax,  x_N en Omega

DECISION DE DISENO: el problema se CONDENSA a mano a la forma estandar

    min_z  1/2 z' G z + g' z   s.a.  Ain z <= bin

y esas cuatro matrices son la interfaz. Python las resuelve con cvxpy/CLARABEL
y MATLAB con quadprog, pero AMBOS construyen las mismas (G, g, Ain, bin). Eso
convierte la validacion cruzada entre lenguajes en una comparacion de matrices
elemento a elemento, no en una comparacion de resultados finales que podria
esconder dos formulaciones distintas que casualmente se parecen.

VARIABLE DE DECISION
    z = [ U ; t ]
con U = [u_0; ...; u_{N-1}] (m*N) y t las variables de epigrafo que necesitan
los objetivos de norma infinito (2N+1 por cada uno). Los objetivos cuadraticos
no aportan epigrafo.

NOTA SOBRE LOS COSTES: J(U,x) NUNCA se lee del epigrafo. Si alpha_i = 0, las
variables de epigrafo del objetivo i no aparecen en la funcion objetivo y el
solver las deja en cualquier punto factible por encima del valor verdadero.
Se recalcula siempre con `Problem.costs`, que es exacta y es la MISMA funcion
que produce J_a a partir de la secuencia desplazada.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .plant import Objective, Plant, Problem
from .terminal import Terminal


# --------------------------------------------------------------------------
#                        MATRICES DE PREDICCION
# --------------------------------------------------------------------------

def prediction_matrices(plant: Plant, N: int):
    """X = Sx x0 + Su U, con X = [x_0; x_1; ...; x_N]."""
    n, m = plant.n, plant.m
    Sx = np.zeros((n * (N + 1), n))
    Su = np.zeros((n * (N + 1), m * N))
    Ak = np.eye(n)
    for i in range(N + 1):
        Sx[i * n:(i + 1) * n, :] = Ak
        for j in range(i):
            # bloque (i,j) = A^{i-1-j} B
            Su[i * n:(i + 1) * n, j * m:(j + 1) * m] = np.linalg.matrix_power(
                plant.A, i - 1 - j) @ plant.B
        Ak = plant.A @ Ak
    return Sx, Su


# --------------------------------------------------------------------------
#                            CONSTRUCCION DEL QP
# --------------------------------------------------------------------------

@dataclass
class QPData:
    G: np.ndarray
    g: np.ndarray
    Ain: np.ndarray
    bin: np.ndarray
    nU: int
    nz: int
    N: int
    m: int

    def as_dict(self) -> Dict[str, np.ndarray]:
        return {"G": self.G, "g": self.g, "Ain": self.Ain, "bin": self.bin}


class MOMPC:
    """MPC multiobjetivo con matrices de prediccion precalculadas."""

    def __init__(self, problem: Problem, terminal: Terminal):
        self.problem = problem
        self.terminal = terminal
        p = problem.plant
        self.Sx, self.Su = prediction_matrices(p, problem.N)
        self.N, self.n, self.m = problem.N, p.n, p.m
        self.nU = self.m * problem.N

        # indices de epigrafo por objetivo (solo los de norma infinito)
        self.epi: List[Optional[slice]] = []
        cursor = self.nU
        for ob in problem.objectives:
            if ob.kind == "inf":
                width = 2 * problem.N + 1     # s_k (N), r_k (N), p (1)
                self.epi.append(slice(cursor, cursor + width))
                cursor += width
            else:
                self.epi.append(None)
        self.nz = cursor
        self._Gi, self._fi, self._ci_mat = self._cost_blocks()
        self._A_epi, self._Bx_epi, self._b0_epi = self._epigraph_rows()
        # Ain NO depende de x0: se precomputa una vez. Solo el lado derecho es
        # afin en x0,  bin = b_const + B_x x0.  Esto permite cachear el solver.
        self.Ain, self._b_const, self._B_x = self._constraint_data()
        self._osqp = None
        self._osqp_pattern = None

    # ---------------- costes por objetivo, en forma cuadratica/lineal ------
    def _cost_blocks(self):
        """Para cada objetivo devuelve (G_i, f_i(x0)->vector, C_i) tales que
        J_i = 1/2 z' G_i z + f_i(x0)' z + x0' C_i x0   (los 'inf' tienen G_i=0)."""
        N, n, m, nz, nU = self.N, self.n, self.m, self.nz, self.nU
        Gis, fis, cis = [], [], []
        for idx, ob in enumerate(self.problem.objectives):
            G = np.zeros((nz, nz))
            Fmat = np.zeros((nz, n))     # f_i = Fmat @ x0
            C = np.zeros((n, n))
            if ob.kind == "quad":
                # Qbar sobre X = [x_0..x_N]: etapas 0..N-1 con Q, terminal con P
                Qbar = np.zeros((n * (N + 1), n * (N + 1)))
                for k in range(N):
                    Qbar[k * n:(k + 1) * n, k * n:(k + 1) * n] = ob.Q
                Qbar[N * n:(N + 1) * n, N * n:(N + 1) * n] = ob.P
                Rbar = np.kron(np.eye(N), ob.R)
                Hu = self.Su.T @ Qbar @ self.Su + Rbar          # J = U'Hu U + 2 x0'Sx'Qbar Su U + ...
                G[:nU, :nU] = 2.0 * (0.5 * (Hu + Hu.T))          # 1/2 z'G z = U'Hu U
                Fmat[:nU, :] = 2.0 * (self.Su.T @ Qbar @ self.Sx)
                C = self.Sx.T @ Qbar @ self.Sx
            else:
                sl = self.epi[idx]
                lin = np.zeros(nz)
                lin[sl] = 1.0                                    # J = sum s_k + sum r_k + p
                Fmat[:, :] = 0.0
                G[:, :] = 0.0
                # el termino lineal no depende de x0
                fis.append(("lin", lin))
                Gis.append(G)
                cis.append(C)
                continue
            Gis.append(G)
            fis.append(("aff", Fmat))
            cis.append(C)
        return Gis, fis, cis

    def cost_quadform(self, idx: int, x0: np.ndarray):
        """(G_i, g_i, c_i) del objetivo idx para un x0 dado."""
        kind, M = self._fi[idx]
        if kind == "lin":
            return self._Gi[idx], M.copy(), 0.0
        x0 = np.asarray(x0, float).ravel()
        return self._Gi[idx], M @ x0, float(x0 @ self._ci_mat[idx] @ x0)

    # ---------------- filas de epigrafo (dependen de x0) -------------------
    def _epigraph_rows(self):
        """Filas  A_epi z <= b_epi_const + B_epi x0  para los objetivos 'inf'."""
        rows, bconst, bx = [], [], []
        N, n, m, nz, nU = self.N, self.n, self.m, self.nz, self.nU
        for idx, ob in enumerate(self.problem.objectives):
            if ob.kind != "inf":
                continue
            sl = self.epi[idx]
            base = sl.start
            nq = ob.Q.shape[0]
            nr = ob.R.shape[0]
            npp = ob.P.shape[0]
            for k in range(N):
                Sxk = self.Sx[k * n:(k + 1) * n, :]
                Suk = self.Su[k * n:(k + 1) * n, :]
                # +-(Q x_k) <= s_k
                for sgn in (+1.0, -1.0):
                    A = np.zeros((nq, nz))
                    A[:, :nU] = sgn * (ob.Q @ Suk)
                    A[:, base + k] = -1.0
                    rows.append(A)
                    bconst.append(np.zeros(nq))
                    bx.append(-sgn * (ob.Q @ Sxk))
                # +-(R u_k) <= r_k
                for sgn in (+1.0, -1.0):
                    A = np.zeros((nr, nz))
                    Sel = np.zeros((m, nU)); Sel[:, k * m:(k + 1) * m] = np.eye(m)
                    A[:, :nU] = sgn * (ob.R @ Sel)
                    A[:, base + N + k] = -1.0
                    rows.append(A)
                    bconst.append(np.zeros(nr))
                    bx.append(np.zeros((nr, n)))
            SxN = self.Sx[N * n:(N + 1) * n, :]
            SuN = self.Su[N * n:(N + 1) * n, :]
            for sgn in (+1.0, -1.0):
                A = np.zeros((npp, nz))
                A[:, :nU] = sgn * (ob.P @ SuN)
                A[:, base + 2 * N] = -1.0
                rows.append(A)
                bconst.append(np.zeros(npp))
                bx.append(-sgn * (ob.P @ SxN))
        if not rows:
            return (np.zeros((0, self.nz)), np.zeros((0, self.n)), np.zeros(0))
        return np.vstack(rows), np.vstack(bx), np.concatenate(bconst)

    # ---------------- QP completo -----------------------------------------
    def build(self, x0: np.ndarray, alpha: np.ndarray) -> QPData:
        x0 = np.asarray(x0, float).ravel()
        alpha = np.asarray(alpha, float).ravel()
        if alpha.size != self.problem.n_obj:
            raise ValueError("alpha no tiene tamano n_obj")
        if np.any(alpha < -1e-12):
            raise ValueError("alpha debe ser no negativo")

        G = np.zeros((self.nz, self.nz))
        g = np.zeros(self.nz)
        for i, ai in enumerate(alpha):
            Gi, gi, _ = self.cost_quadform(i, x0)
            G += ai * Gi
            g += ai * gi
        G = 0.5 * (G + G.T)
        return QPData(G=G, g=g, Ain=self.Ain, bin=self.rhs(x0),
                      nU=self.nU, nz=self.nz, N=self.N, m=self.m)

    def rhs(self, x0: np.ndarray) -> np.ndarray:
        """bin = b_const + B_x x0."""
        return self._b_const + self._B_x @ np.asarray(x0, float).ravel()

    def _constraint_data(self):
        """(Ain, b_const, B_x) con  Ain z <= b_const + B_x x0.  Ain es constante."""
        N, n, m, nz, nU = self.N, self.n, self.m, self.nz, self.nU
        p = self.problem.plant
        rows, bc, bx = [], [], []

        # caja de estado para k = 1..N  (x_0 esta dado, no se restringe)
        for k in range(1, N + 1):
            Sxk = self.Sx[k * n:(k + 1) * n, :]
            Suk = self.Su[k * n:(k + 1) * n, :]
            for sgn in (+1.0, -1.0):
                A = np.zeros((n, nz)); A[:, :nU] = sgn * Suk
                rows.append(A); bc.append(p.xmax); bx.append(-sgn * Sxk)

        # caja de entrada
        for k in range(N):
            Sel = np.zeros((m, nU)); Sel[:, k * m:(k + 1) * m] = np.eye(m)
            for sgn in (+1.0, -1.0):
                A = np.zeros((m, nz)); A[:, :nU] = sgn * Sel
                rows.append(A); bc.append(p.umax); bx.append(np.zeros((m, n)))

        # region terminal
        SxN = self.Sx[N * n:(N + 1) * n, :]
        SuN = self.Su[N * n:(N + 1) * n, :]
        Ht, kt = self.terminal.H, self.terminal.k
        A = np.zeros((Ht.shape[0], nz)); A[:, :nU] = Ht @ SuN
        rows.append(A); bc.append(kt); bx.append(-Ht @ SxN)

        # epigrafo (solo si hay objetivos de norma infinito)
        if self._A_epi.shape[0] > 0:
            rows.append(self._A_epi); bc.append(self._b0_epi); bx.append(self._Bx_epi)

        return np.vstack(rows), np.concatenate(bc), np.vstack(bx)


# --------------------------------------------------------------------------
#                               SOLUCION
# --------------------------------------------------------------------------

def solve_qp_cvxpy(qp: QPData, solver: str = "CLARABEL"):
    """Referencia lenta pero muy fiable. Se usa para validar la via rapida."""
    import cvxpy as cp
    z = cp.Variable(qp.nz)
    Gs = 0.5 * (qp.G + qp.G.T)
    # psd_wrap evita que cvxpy rechace un G con autovalores -1e-16 por redondeo
    obj = cp.Minimize(0.5 * cp.quad_form(z, cp.psd_wrap(Gs)) + qp.g @ z)
    prob = cp.Problem(obj, [qp.Ain @ z <= qp.bin])
    try:
        prob.solve(solver=getattr(cp, solver))
    except Exception:
        return None
    if z.value is None or prob.status not in ("optimal", "optimal_inaccurate"):
        return None
    return np.asarray(z.value, float).ravel()


def solve_qp_osqp(qp: QPData, cache: Optional[dict] = None, eps: float = 1e-10):
    """Via rapida. OSQP resuelve  min 1/2 z'Pz + q'z  s.a.  l <= Az <= u.

    Ain es constante para un problema dado, asi que el setup se cachea y en cada
    llamada solo se actualizan (Px, q, u). Es lo que hace viable el banco.
    """
    import osqp
    from scipy import sparse

    P = sparse.triu(sparse.csc_matrix(0.5 * (qp.G + qp.G.T)), format="csc")
    A = sparse.csc_matrix(qp.Ain)
    l = np.full(qp.Ain.shape[0], -np.inf)

    key = "osqp"
    if cache is not None and key in cache:
        m, P_pat = cache[key]
        if P_pat.nnz == P.nnz and np.array_equal(P_pat.indices, P.indices) \
                and np.array_equal(P_pat.indptr, P.indptr):
            m.update(Px=P.data, q=qp.g, u=qp.bin)
            res = m.solve()
            if res.info.status_val not in (1, 2):     # solved / solved inaccurate
                return None
            return np.asarray(res.x, float).ravel()

    m = osqp.OSQP()
    # polish=False a proposito: medido contra CLARABEL en 40 instancias, sin
    # pulido el error es 1.4e-6 y con pulido 1.7e-5. Ademas el pulido imprime
    # por stdout aunque verbose=False.
    m.setup(P=P, q=qp.g, A=A, l=l, u=qp.bin, verbose=False,
            eps_abs=eps, eps_rel=eps, max_iter=20000, polish=False)
    if cache is not None:
        cache[key] = (m, P)
    res = m.solve()
    if res.info.status_val not in (1, 2):
        return None
    return np.asarray(res.x, float).ravel()


def solve_qp(qp: QPData, backend: str = "osqp", cache: Optional[dict] = None):
    if backend == "osqp":
        z = solve_qp_osqp(qp, cache=cache)
        if z is not None:
            return z
        return solve_qp_cvxpy(qp)          # respaldo si OSQP no converge
    return solve_qp_cvxpy(qp, solver=backend)


@dataclass
class Solution:
    U: np.ndarray        # (m, N)
    X: np.ndarray        # (n, N+1)
    J: np.ndarray        # vector de costes, recalculado exactamente
    V: float             # alpha' J
    alpha: np.ndarray
    status: str = "optimal"


def solve(mompc: MOMPC, x0: np.ndarray, alpha: np.ndarray,
          backend: str = "osqp") -> Optional[Solution]:
    x0 = np.asarray(x0, float).ravel()
    alpha = np.asarray(alpha, float).ravel()
    qp = mompc.build(x0, alpha)
    if mompc._osqp is None:
        mompc._osqp = {}
    z = solve_qp(qp, backend=backend, cache=mompc._osqp)
    if z is None:
        return None
    U = z[:mompc.nU].reshape(mompc.N, mompc.m).T
    X = (mompc.Sx @ x0 + mompc.Su @ z[:mompc.nU]).reshape(mompc.N + 1, mompc.n).T
    J = mompc.problem.costs(x0, U)          # exacto, no leido del epigrafo
    return Solution(U=U, X=X, J=J, V=float(alpha @ J), alpha=alpha)


def shifted_sequence(mompc: MOMPC, sol: Solution) -> np.ndarray:
    """U_s = [u_1*, ..., u_{N-1}*, Kf x_N*] -- la cola desplazada que produce J_a."""
    Kf = mompc.terminal.Kf
    xN = sol.X[:, mompc.N]
    tail = (Kf @ xN).reshape(mompc.m, 1)
    Us = np.hstack([sol.U[:, 1:], tail])
    umax = mompc.problem.plant.umax.reshape(-1, 1)
    return np.clip(Us, -umax, umax)
