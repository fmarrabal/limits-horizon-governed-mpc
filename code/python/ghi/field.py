"""El campo homeostatico sobre el grafo de subsistemas de la planta.

    K = w0^2 I + c^2 L        (rigidez: reaccion + difusion espacial)
    C = 2 zeta w0 I + D L     (disipacion: uniforme + estructural)
    G = b A + beta A^3        (antisimetricos: adveccion + dispersion tipo KdV)

    rama de ONDA (2o orden):     u'' + (C + G) u' + K u = f
    rama de DIFUSION (1er orden): gamma u' + (K + G) u = f

REGLA DE COLOCACION (clasica; NO es un resultado propio)
--------------------------------------------------------
El operador antisimetrico va sobre la VELOCIDAD (giroscopico, seguro por
Kelvin-Tait-Chetaev, porque u'^T G u' = 0) y nunca sobre la POSICION
(circulatorio, produce flutter). Es mecanica no conservativa clasica:
Thomson & Tait (1879), Chetaev, Merkin, Ziegler (1952); tratamiento moderno en
Kirillov, "Gyroscopic Stabilization in the Presence of Nonconservative Forces",
Doklady Mathematics 76(2):780-785 (2007), y Udwadia, "Stability of Gyroscopic
Circulatory Systems", ASME J. Appl. Mech. 86(2):021002 (2019).

Kirillov reduce el sistema a una escalar de coeficientes complejos resuelta con
el criterio de Bilharz, y la literatura llama a esa ecuacion "ecuacion de
Maxwell-Bloch modificada" -- conviene saberlo antes de presentar la analogia
con RMN como aportacion.

QUE SI ES APORTABLE AQUI
------------------------
Comprobar si el umbral  rho(G) < 2 zeta w0^2, deducido para el caso degenerado
de Merkin (frecuencias iguales), sigue valiendo sobre el grafo de una PLANTA,
donde K y C dejan de ser multiplos de la identidad. Respuesta medida: es EXACTO
en el caso degenerado y CONSERVADOR en cuanto la planta aporta estructura --
tanto D*L como c^2*L aumentan el margen real. Es decir, se puede usar como cota
segura de diseno.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np

Placement = Literal["gyroscopic", "circulatory"]


# --------------------------------------------------------------------------
#                                 GRAFOS
# --------------------------------------------------------------------------

def chain_graph(M: int) -> Tuple[np.ndarray, np.ndarray]:
    """Cadena dirigida 1 -> 2 -> ... -> M.

    Analogo de un lazo solar, una linea de proceso o una hilera de modulos de
    invernadero. La adveccion apunta en el sentido del flujo del proceso, que en
    una planta es un DATO FISICO, no una convencion.
    """
    if M < 2:
        raise ValueError("hacen falta al menos 2 nodos")
    Adj = np.zeros((M, M))
    for i in range(M - 1):
        Adj[i, i + 1] = Adj[i + 1, i] = 1.0
    L = np.diag(Adj.sum(1)) - Adj
    A = np.zeros((M, M))
    for i in range(M - 1):
        A[i, i + 1] = 1.0
        A[i + 1, i] = -1.0
    return L, A


def ring_graph(M: int) -> Tuple[np.ndarray, np.ndarray]:
    """Anillo dirigido (recirculacion)."""
    Adj = np.zeros((M, M))
    A = np.zeros((M, M))
    for i in range(M):
        j = (i + 1) % M
        Adj[i, j] = Adj[j, i] = 1.0
        A[i, j] += 1.0
        A[j, i] -= 1.0
    L = np.diag(Adj.sum(1)) - Adj
    return L, A


# --------------------------------------------------------------------------
#                               OPERADORES
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class FieldParams:
    w0: float = 1.0
    zeta: float = 0.15
    c: float = 0.0        # rigidez espacial
    D: float = 0.0        # amortiguamiento estructural
    b: float = 0.0        # adveccion
    beta: float = 0.0     # dispersion tipo KdV
    gamma: float = 1.0    # tasa propia de la rama difusiva


def operators(L: np.ndarray, A: np.ndarray, p: FieldParams):
    M = L.shape[0]
    I = np.eye(M)
    K = p.w0 ** 2 * I + p.c ** 2 * L
    C = 2 * p.zeta * p.w0 * I + p.D * L
    G = p.b * A + p.beta * (A @ A @ A)
    return K, C, G


def companion(K: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Matriz de estado de  u'' + C u' + K u = 0."""
    M = K.shape[0]
    Z, I = np.zeros((M, M)), np.eye(M)
    return np.block([[Z, I], [-K, -C]])


def spectral_abscissa(K: np.ndarray, C: np.ndarray) -> float:
    return float(np.max(np.linalg.eigvals(companion(K, C)).real))


def wave_state(K: np.ndarray, C: np.ndarray, G: np.ndarray,
               placement: Placement = "gyroscopic") -> np.ndarray:
    """Matriz de estado segun donde se coloque el operador antisimetrico."""
    if placement == "gyroscopic":
        return companion(K, C + G)        # G sobre la velocidad: seguro
    if placement == "circulatory":
        return companion(K + G, C)        # G sobre la posicion: flutter
    raise ValueError(placement)


def rho_G(A: np.ndarray, b: float, beta: float) -> float:
    """Radio espectral de G = bA + beta A^3.

    A antisimetrica tiene autovalores i*mu_k, luego A^3 los tiene -i*mu_k^3 y G
    los tiene i*(b mu_k - beta mu_k^3).
    """
    mu = np.abs(np.linalg.eigvals(A).imag)
    return float(np.max(np.abs(b * mu - beta * mu ** 3)))


def merkin_threshold(p: FieldParams) -> float:
    """Cota del caso degenerado:  rho(G) < 2 zeta w0^2."""
    return float(2.0 * p.zeta * p.w0 ** 2)


# --------------------------------------------------------------------------
#                       ESTUDIO DE COLOCACION Y FLUTTER
# --------------------------------------------------------------------------

def collocation_study(M: int = 6, p: FieldParams = FieldParams(),
                      beta_max: Optional[float] = None, n: int = 2001,
                      graph=chain_graph) -> dict:
    """Barre beta y localiza el umbral REAL de flutter de la colocacion
    circulatoria, comparandolo con el que predice la cota de Merkin."""
    L, A = graph(M)
    K, C, _ = operators(L, A, p)
    mu = np.abs(np.linalg.eigvals(A).imag)
    thr = merkin_threshold(p)

    # beta al que rho(G) alcanza el umbral
    bb = np.linspace(0.0, 5.0, 200001)
    rr = np.array([rho_G(A, p.b, x) for x in bb])
    cruce = np.where(rr >= thr)[0]
    beta_pred = float(bb[cruce[0]]) if len(cruce) else np.inf

    if beta_max is None:
        beta_max = max(2.0 * beta_pred, 0.5) if np.isfinite(beta_pred) else 1.0
    betas = np.linspace(0.0, beta_max, n)
    A3 = A @ A @ A
    gyro = np.empty(n)
    circ = np.empty(n)
    for j, be in enumerate(betas):
        G = p.b * A + be * A3
        gyro[j] = float(np.max(np.linalg.eigvals(wave_state(K, C, G, "gyroscopic")).real))
        circ[j] = float(np.max(np.linalg.eigvals(wave_state(K, C, G, "circulatory")).real))
    idx = np.where(circ > 1e-9)[0]
    beta_obs = float(betas[idx[0]]) if len(idx) else np.inf

    return {
        "M": M, "mu_max": float(mu.max()), "rho_A3": float(mu.max() ** 3),
        "umbral_2zw0^2": thr, "beta_pred": beta_pred, "beta_obs": beta_obs,
        "ratio_obs_pred": float(beta_obs / beta_pred) if np.isfinite(beta_pred) and beta_pred > 0 else np.nan,
        "gyro_max_re": float(gyro.max()),
        "gyro_estable": bool(gyro.max() < 0.0),
        "betas": betas, "gyro": gyro, "circ": circ,
    }


# --------------------------------------------------------------------------
#                   PROPAGACION: ONDA FRENTE A DIFUSION
# --------------------------------------------------------------------------

def propagate(M: int = 8, p: FieldParams = FieldParams(w0=1.0, zeta=0.15, c=0.6, D=0.6),
              dt: float = 0.02, T: float = 40.0, graph=chain_graph) -> dict:
    """Impulso de interocepcion en el nodo 1: cuando y con cuanta amplitud se
    entera cada nodo, por la rama de onda y por la de difusion."""
    L, A = graph(M)
    K, C, G = operators(L, A, p)
    nT = int(T / dt)

    # ONDA: Verlet de velocidad, u'' + (C+G) u' + K u = 0
    u = np.zeros(M); v = np.zeros(M); u[0] = 1.0
    Uw = np.empty((nT, M))
    CG = C + G
    for k in range(nT):
        acc = -CG @ v - K @ u
        u = u + dt * v + 0.5 * dt ** 2 * acc
        vh = v + 0.5 * dt * acc
        v = vh + 0.5 * dt * (-CG @ vh - K @ u)
        Uw[k] = u

    # DIFUSION: IMEX backward-Euler, gamma u' + (K+G) u = 0
    lam = dt / p.gamma
    Minv = np.linalg.inv(np.eye(M) + lam * (K + G))
    u = np.zeros(M); u[0] = 1.0
    Ud = np.empty((nT, M))
    for k in range(nT):
        u = Minv @ u
        Ud[k] = u

    def first_cross(U, frac=0.05):
        t = np.full(M, np.nan)
        for j in range(M):
            pk = np.max(np.abs(U[:, j]))
            if pk < 1e-14:
                continue
            idx = np.where(np.abs(U[:, j]) >= frac * pk)[0]
            if len(idx):
                t[j] = idx[0] * dt
        return t

    tw, td = first_cross(Uw), first_cross(Ud)
    pw = np.max(np.abs(Uw), axis=0)
    pd = np.max(np.abs(Ud), axis=0)
    d = np.arange(M)
    ok = ~np.isnan(tw)
    slope = np.polyfit(d[ok][1:], tw[ok][1:], 1)[0] if ok.sum() > 2 else np.nan
    corr = float(np.corrcoef(d[ok][1:], tw[ok][1:])[0, 1]) if ok.sum() > 2 else np.nan
    return {
        "M": M, "t_onda": tw, "t_difusion": td, "pico_onda": pw, "pico_difusion": pd,
        "pendiente_frente": float(slope), "velocidad_frente": float(1.0 / slope) if slope else np.nan,
        "corr_distancia_tiempo": corr,
        "atenuacion_onda": float(pw[-1] / pw[0]),
        "atenuacion_difusion": float(pd[-1] / pd[0]),
        "ventaja_transporte": float((pw[-1] / pw[0]) / (pd[-1] / pd[0])),
        "Uw": Uw, "Ud": Ud,
    }
