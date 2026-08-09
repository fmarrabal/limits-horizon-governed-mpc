"""Geometria del peso: conjunto admisible, radio certificado y seleccion.

EL PUNTO CENTRAL, Y ES UNA CORRECCION A LA LITERATURA
-----------------------------------------------------
    V*(x, alpha) = min_{U en U(x)}  alpha' J(U, x)

Para cada U fijo el corchete es AFIN en alpha, y el factible U(x) NO depende de
alpha. El infimo puntual de una familia de funciones afines es CONCAVO. (Es el
mismo hecho que "la funcion dual de Lagrange es concava".)

Por tanto:
  * V*(x, .) es CONCAVA en alpha, no convexa;
  * A(x, J_a) = {alpha en Delta : V*(x,alpha) <= J_a} es un SUBNIVEL DE UNA
    CONCAVA y por tanto NO es convexo en general.

Bemporad & Munoz de la Pena (Automatica 45(12):2823-2830, 2009) afirman en su
Lema 4 que V* es "convex and piecewise affine w.r.t. mu" y en su Teorema 6 la
escriben como el MAXIMO de sus piezas afines, lo que reduce la seleccion de peso
a un LP. Ese LP es una RESTRICCION INTERIOR conservadora, no una reformulacion
equivalente: el algoritmo sigue siendo SEGURO, pero la equivalencia y la
optimalidad de alpha* enunciadas no se sostienen.

`concavity_evidence` produce la evidencia numerica de todo esto.

EL RADIO CERTIFICADO
--------------------
Por concavidad, J(U*(x,alpha), x) es un SUPERGRADIENTE de V*(x,.) en alpha
(teorema de la envolvente / Danskin para un minimo). Luego para todo alpha':

    V*(x, alpha') <= V*(x, alpha) + J' (alpha' - alpha)

y como alpha, alpha' viven en el simplex, 1'(alpha'-alpha) = 0, de modo que J
puede sustituirse por su version CENTRADA J^perp = J - mean(J)*1 sin cambiar el
producto. Cauchy-Schwarz da entonces la condicion suficiente

    ||alpha' - alpha||_2 <= r := (J_a - V*(x,alpha)) / ||J^perp||_2   =>   alpha' admisible

La observacion fina es el CENTRADO: lo que limita el movimiento del peso es la
DISPERSION de los costes entre objetivos, no su magnitud. Con todos los costes
iguales el radio es infinito, que es justo cuando el problema multiobjetivo deja
de tener interes.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .mompc import MOMPC, Solution, solve


# --------------------------------------------------------------------------
#                          MALLAS SOBRE EL SIMPLEX
# --------------------------------------------------------------------------

def simplex_grid(n_obj: int, points: int = 21) -> np.ndarray:
    """Malla regular sobre el simplex. Para n_obj=2 son `points` pesos."""
    if n_obj < 2:
        raise ValueError("hacen falta al menos 2 objetivos")
    if n_obj == 2:
        a = np.linspace(0.0, 1.0, points)
        return np.stack([1.0 - a, a], axis=1)
    # composiciones de `points-1` en n_obj partes
    div = points - 1
    out = []
    for comb in product(range(div + 1), repeat=n_obj - 1):
        s = sum(comb)
        if s <= div:
            out.append(np.array(list(comb) + [div - s], float) / div)
    return np.array(out)


# --------------------------------------------------------------------------
#                          CONJUNTO ADMISIBLE
# --------------------------------------------------------------------------

@dataclass
class Admissible:
    alphas: np.ndarray          # malla evaluada, (M, n_obj)
    V: np.ndarray               # V*(x, alpha) en cada punto (inf si infactible)
    feasible: np.ndarray        # mascara booleana V <= Ja
    Ja: float

    @property
    def any_feasible(self) -> bool:
        return bool(self.feasible.any())

    def nearest(self, alpha_req: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Punto admisible mas cercano a la propuesta. Devuelve (alpha, bloqueado)."""
        alpha_req = np.asarray(alpha_req, float).ravel()
        if self.any_feasible:
            cand = self.alphas[self.feasible]
            d = np.linalg.norm(cand - alpha_req, axis=1)
            j = int(np.argmin(d))
            # bloqueo = la propuesta NO cae en el conjunto admisible salvo por
            # la resolucion de la malla
            step = self._grid_step()
            return cand[j], bool(d[j] > 0.51 * step)
        j = int(np.argmin(self.V))          # fallback: el mas estabilizante
        return self.alphas[j], True

    def _grid_step(self) -> float:
        if len(self.alphas) < 2:
            return np.inf
        d = np.linalg.norm(self.alphas[1] - self.alphas[0])
        return float(d) if d > 0 else np.inf


def admissible_set(mompc: MOMPC, x0: np.ndarray, Ja: float,
                   grid: Optional[np.ndarray] = None, tol: float = 1e-7,
                   cache: Optional[dict] = None) -> Admissible:
    """Evalua V*(x, .) sobre la malla y marca el subnivel <= Ja.

    Se barre porque el conjunto NO es convexo (ver cabecera del modulo): no se
    puede proyectar con una proyeccion convexa ordinaria.
    """
    if grid is None:
        grid = simplex_grid(mompc.problem.n_obj, 21)
    V = np.full(len(grid), np.inf)
    for j, a in enumerate(grid):
        sol = solve(mompc, x0, a)
        if sol is not None:
            V[j] = sol.V
            if cache is not None:
                cache[j] = sol
    return Admissible(alphas=grid, V=V, feasible=V <= Ja + tol, Ja=float(Ja))


# --------------------------------------------------------------------------
#                          RADIO CERTIFICADO
# --------------------------------------------------------------------------

def certified_radius(J: np.ndarray, V: float, Ja: float) -> float:
    """r = (J_a - V) / ||J - mean(J)||_2 ; +inf si todos los costes coinciden."""
    J = np.asarray(J, float).ravel()
    Jc = J - J.mean()
    nrm = float(np.linalg.norm(Jc))
    slack = float(Ja - V)
    if slack < 0:
        return 0.0
    if nrm < 1e-14:
        return np.inf
    return slack / nrm


def radius_is_valid(mompc: MOMPC, x0: np.ndarray, alpha: np.ndarray,
                    Ja: float, n_dirs: int = 24, seed: int = 0,
                    backend: str = "CLARABEL") -> dict:
    """Comprueba empiricamente que todo alpha' a distancia <= r es admisible.

    El paso se recorta al MINIMO entre r y la distancia a la frontera del
    simplex: si no, con r grande todas las direcciones se saldrian del simplex
    y el test no llegaria a ejecutarse nunca (que es justo lo que pasaba antes).
    """
    alpha = np.asarray(alpha, float).ravel()
    sol = solve(mompc, x0, alpha, backend=backend)
    if sol is None:
        return {"ok": False, "motivo": "infactible en alpha", "n_tests": 0}
    r = certified_radius(sol.J, sol.V, Ja)
    rng = np.random.default_rng(seed)
    n = alpha.size
    peor, n_tests = -np.inf, 0
    for _ in range(n_dirs):
        d = rng.normal(size=n)
        d -= d.mean()                       # tangente al simplex
        nd = np.linalg.norm(d)
        if nd < 1e-12:
            continue
        d = d / nd
        # mayor paso que mantiene alpha' >= 0
        neg = d < -1e-15
        t_max = np.min(-alpha[neg] / d[neg]) if neg.any() else np.inf
        t_lim = min(r, t_max) if np.isfinite(r) else t_max
        if not np.isfinite(t_lim) or t_lim <= 1e-12:
            continue
        for frac in (0.5, 0.9, 0.999):
            ap = np.clip(alpha + frac * t_lim * d, 0.0, None)
            ap = ap / ap.sum()
            s2 = solve(mompc, x0, ap, backend=backend)
            if s2 is None:
                continue
            n_tests += 1
            peor = max(peor, s2.V - Ja)
    return {"ok": bool(n_tests > 0 and peor <= 1e-6), "r": float(r),
            "peor_exceso": float(peor), "n_tests": n_tests}


# --------------------------------------------------------------------------
#           EVIDENCIA DE CONCAVIDAD  (la correccion a Bemporad 2009)
# --------------------------------------------------------------------------

def concavity_evidence(mompc: MOMPC, states: Sequence[np.ndarray],
                       pairs: Optional[Sequence[Tuple[float, float]]] = None,
                       rtol: float = 1e-8, backend: str = "CLARABEL") -> dict:
    """Test de cuerda sobre V*(x, .) para n_obj = 2.

    Para cada estado y cada par (a1, a2) compara V* en el punto medio con la
    media de los extremos:
        d = V*(am) - [V*(a1) + V*(a2)]/2
        d > 0  =>  por ENCIMA de la cuerda  =>  concava
        d < 0  =>  por DEBAJO de la cuerda  =>  convexa

    La tolerancia es RELATIVA a la escala de V en cada test: el error del solver
    escala con V, y una tolerancia absoluta produciria falsos positivos en los
    estados grandes y seria ciega en los pequenos. Por defecto se usa el backend
    exacto (CLARABEL), no el rapido.
    """
    if mompc.problem.n_obj != 2:
        raise NotImplementedError("el test de cuerda esta escrito para 2 objetivos")
    if pairs is None:
        pairs = [(0.0, 1.0), (0.1, 0.9), (0.2, 0.8), (0.3, 0.7), (0.25, 0.75)]
    viol_conc = viol_conv = n = 0
    peor_conc, peor_conv = 0.0, 0.0
    ds: List[float] = []
    for x in states:
        for a1, a2 in pairs:
            am = 0.5 * (a1 + a2)
            s = [solve(mompc, x, np.array([1 - a, a]), backend=backend)
                 for a in (a1, a2, am)]
            if any(si is None for si in s):
                continue
            n += 1
            d = s[2].V - 0.5 * (s[0].V + s[1].V)
            escala = max(abs(s[0].V), abs(s[1].V), abs(s[2].V), 1.0)
            tol = rtol * escala
            ds.append(float(d))
            if d < -tol:
                viol_conc += 1
                peor_conc = min(peor_conc, d)
            if d > tol:
                viol_conv += 1
                peor_conv = max(peor_conv, d)
    return {
        "tests": n,
        "violaciones_concavidad": viol_conc,
        "violaciones_convexidad": viol_conv,
        "peor_desviacion_negativa": float(peor_conc),
        "mayor_desviacion_positiva": float(peor_conv),
        "d_media": float(np.mean(ds)) if ds else 0.0,
        "concava": bool(viol_conc == 0 and viol_conv > 0),
    }


def nonconvexity_evidence(mompc: MOMPC, x0: np.ndarray, Ja: float,
                          grid_points: int = 41, tol: float = 1e-7,
                          backend: str = "CLARABEL") -> dict:
    """Fraccion de puntos medios de pares admisibles que NO son admisibles.

    Si el conjunto fuese convexo esa fraccion seria exactamente 0.
    """
    grid = simplex_grid(mompc.problem.n_obj, grid_points)
    V = np.array([(lambda s: np.inf if s is None else s.V)(
        solve(mompc, x0, a, backend=backend)) for a in grid])
    feas = V <= Ja + tol
    idx = np.where(feas)[0]
    if len(idx) < 2:
        return {"pares": 0, "fuera": 0, "fraccion": 0.0, "no_convexo": False,
                "nota": "menos de dos puntos admisibles"}
    fuera = pares = 0
    for ii in range(len(idx)):
        for jj in range(ii + 1, len(idx)):
            if idx[jj] - idx[ii] < 2:
                continue                    # sin punto de malla intermedio
            mid = (idx[ii] + idx[jj]) // 2
            pares += 1
            if not feas[mid]:
                fuera += 1
    return {"pares": pares, "fuera": fuera,
            "fraccion": float(fuera / pares) if pares else 0.0,
            "no_convexo": bool(fuera > 0),
            "V_min": float(V[np.isfinite(V)].min()) if np.isfinite(V).any() else np.nan,
            "V_max": float(V[np.isfinite(V)].max()) if np.isfinite(V).any() else np.nan}


def nonconvexity_sweep(mompc: MOMPC, x0: np.ndarray, grid_points: int = 41,
                       n_levels: int = 25, backend: str = "CLARABEL") -> dict:
    """Barre J_a entre min V* y max V* y mide para que niveles el conjunto
    admisible es NO convexo.

    Es la demostracion limpia. Como V*(x,.) es CONCAVA, su maximo esta en el
    interior del simplex, asi que para todo J_a entre ese maximo y el mayor de
    los dos valores en los extremos el subnivel es la UNION DE DOS INTERVALOS.
    No es una rareza numerica: es la geometria obligada de un subnivel concavo.
    """
    grid = simplex_grid(mompc.problem.n_obj, grid_points)
    V = np.array([(lambda s: np.inf if s is None else s.V)(
        solve(mompc, x0, a, backend=backend)) for a in grid])
    fin = np.isfinite(V)
    if fin.sum() < 3:
        return {"niveles": 0, "no_convexos": 0, "fraccion": 0.0}
    vmin, vmax = float(V[fin].min()), float(V[fin].max())
    j_peak = int(np.nanargmax(np.where(fin, V, -np.inf)))
    interior_peak = 0 < j_peak < len(grid) - 1
    niveles = np.linspace(vmin, vmax, n_levels + 2)[1:-1]
    nc = 0
    ejemplos = []
    for Ja in niveles:
        feas = V <= Ja + 1e-9
        if feas.sum() < 2:
            continue
        idx = np.where(feas)[0]
        # no convexo <=> hay un hueco entre el primer y el ultimo admisible
        hueco = bool(np.any(~feas[idx[0]:idx[-1] + 1]))
        nc += int(hueco)
        if hueco and len(ejemplos) < 3:
            ejemplos.append(float(Ja))
    return {"niveles": int(len(niveles)), "no_convexos": nc,
            "fraccion": float(nc / len(niveles)) if len(niveles) else 0.0,
            "pico_interior": interior_peak, "j_pico": j_peak,
            "V_min": vmin, "V_max": vmax, "ejemplos_Ja": ejemplos}
