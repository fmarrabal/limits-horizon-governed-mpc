"""Plantas, objetivos y bancos de prueba del Gobernador Homeostatico Inercial.

Todo el paquete usa la misma convencion que el TFM de 2012:
    x_{k+1} = A x_k + B u_k,   |x| <= xmax,  |u| <= umax
y un VECTOR de objetivos J = [J_0, ..., J_l], cada uno de la forma
    J_i(U, x) = sum_{k=0}^{N-1} l_i(x_k, u_k) + F_i(x_N).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal

import numpy as np

Kind = Literal["quad", "inf"]


@dataclass(frozen=True)
class Plant:
    """Sistema LTI discreto con restricciones de caja."""

    A: np.ndarray
    B: np.ndarray
    xmax: np.ndarray
    umax: np.ndarray
    name: str = ""

    def __post_init__(self) -> None:
        A, B = np.atleast_2d(self.A), np.atleast_2d(self.B)
        if A.shape[0] != A.shape[1]:
            raise ValueError(f"A debe ser cuadrada, es {A.shape}")
        if B.shape[0] != A.shape[0]:
            raise ValueError(f"B tiene {B.shape[0]} filas y A tiene {A.shape[0]}")
        object.__setattr__(self, "A", A)
        object.__setattr__(self, "B", B)
        object.__setattr__(self, "xmax", np.atleast_1d(np.asarray(self.xmax, float)))
        object.__setattr__(self, "umax", np.atleast_1d(np.asarray(self.umax, float)))
        if self.xmax.size != self.n or self.umax.size != self.m:
            raise ValueError("xmax/umax no casan con las dimensiones de la planta")
        if np.any(self.xmax <= 0) or np.any(self.umax <= 0):
            raise ValueError("las cotas de caja deben ser estrictamente positivas")

    @property
    def n(self) -> int:
        return self.A.shape[0]

    @property
    def m(self) -> int:
        return self.B.shape[1]

    def step(self, x: np.ndarray, u: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
        x = np.asarray(x, float).ravel()
        u = np.atleast_1d(np.asarray(u, float)).ravel()
        xn = self.A @ x + self.B @ u
        if w is not None:
            xn = xn + np.asarray(w, float).ravel()
        return xn


@dataclass(frozen=True)
class Objective:
    """Un objetivo del vector J.

    kind='quad' -> l(x,u) = x'Qx + u'Ru,      F(x) = x'Px
    kind='inf'  -> l(x,u) = ||Qx||_inf + ||Ru||_inf,  F(x) = ||Px||_inf
    """

    Q: np.ndarray
    R: np.ndarray
    P: np.ndarray | None = None
    kind: Kind = "quad"
    name: str = ""

    def __post_init__(self) -> None:
        Q = np.atleast_2d(np.asarray(self.Q, float))
        R = np.atleast_2d(np.asarray(self.R, float))
        if self.kind == "quad":
            # solo la parte simetrica define la forma cuadratica; la asimetrica es ruido
            Q = 0.5 * (Q + Q.T)
            R = 0.5 * (R + R.T)
        object.__setattr__(self, "Q", Q)
        object.__setattr__(self, "R", R)
        if self.P is not None:
            P = np.atleast_2d(np.asarray(self.P, float))
            if self.kind == "quad":
                P = 0.5 * (P + P.T)
            object.__setattr__(self, "P", P)
        if self.kind not in ("quad", "inf"):
            raise ValueError(f"kind desconocido: {self.kind}")

    def with_P(self, P: np.ndarray) -> "Objective":
        return Objective(self.Q, self.R, P, self.kind, self.name)

    def stage(self, x: np.ndarray, u: np.ndarray) -> float:
        x = np.asarray(x, float).ravel()
        u = np.atleast_1d(np.asarray(u, float)).ravel()
        if self.kind == "quad":
            return float(x @ self.Q @ x + u @ self.R @ u)
        return float(np.max(np.abs(self.Q @ x)) + np.max(np.abs(self.R @ u)))

    def terminal(self, x: np.ndarray) -> float:
        if self.P is None:
            return 0.0
        x = np.asarray(x, float).ravel()
        if self.kind == "quad":
            return float(x @ self.P @ x)
        return float(np.max(np.abs(self.P @ x)))


@dataclass
class Problem:
    """Planta + vector de objetivos + horizonte. Los ingredientes terminales
    se anaden despues, con `ghi.terminal.design`."""

    plant: Plant
    objectives: List[Objective]
    N: int = 5
    name: str = ""

    @property
    def n_obj(self) -> int:
        return len(self.objectives)

    def costs(self, x0: np.ndarray, U: np.ndarray) -> np.ndarray:
        """J(U, x0) evaluado sobre una secuencia de control DADA.

        U tiene forma (m, N). Es la funcion que produce J_a a partir de la
        secuencia desplazada, asi que debe ser exacta y coincidir con el QP.
        """
        U = np.asarray(U, float).reshape(self.plant.m, self.N)
        x = np.asarray(x0, float).ravel().copy()
        J = np.zeros(self.n_obj)
        for k in range(self.N):
            u = U[:, k]
            for i, ob in enumerate(self.objectives):
                J[i] += ob.stage(x, u)
            x = self.plant.step(x, u)
        for i, ob in enumerate(self.objectives):
            J[i] += ob.terminal(x)
        return J


# --------------------------------------------------------------------------
#                              BANCOS DE PRUEBA
# --------------------------------------------------------------------------

def plant_tfm() -> Plant:
    """La planta del TFM 2012 (doble integrador muestreado)."""
    return Plant(
        A=np.array([[1.0, 1.0], [0.0, 1.0]]),
        B=np.array([[0.5], [1.0]]),
        xmax=np.array([10.0, 10.0]),
        umax=np.array([10.0]),
        name="TFM-2012",
    )


def problem_tfm() -> Problem:
    """Reproduccion literal del ejemplo del TFM: J0 con normas infinito y J1
    cuadratica. Los P que trae el TFM VIOLAN la desigualdad terminal; se dejan
    tal cual a proposito, y `ghi.terminal.audit` lo documenta."""
    J0 = Objective(
        Q=np.diag([0.1, 1.0]), R=np.array([[0.2]]),
        P=np.array([[0.5649, 0.4054], [0.4054, 1.6027]]),
        kind="inf", name="J0-inf (TFM)",
    )
    J1 = Objective(
        Q=np.diag([1.0, 0.1]), R=np.array([[0.1]]),
        P=np.array([[9.6085, 1.1401], [-0.2965, 9.4107]]),
        kind="quad", name="J1-quad (TFM)",
    )
    return Problem(plant_tfm(), [J0, J1], N=5, name="TFM-2012")


def problem_conflict() -> Problem:
    """Banco principal: dos objetivos que SI compiten.

    J0 = economia   (penaliza casi solo el esfuerzo de control)
    J1 = prestaciones (penaliza casi solo la desviacion de estado)

    En el banco del TFM los dos objetivos casi no compiten -- el coste en lazo
    cerrado sale identico para cualquier regulador de alpha -- asi que no puede
    discriminar. Este si. Los P se calculan con `ghi.terminal.design`.
    """
    J0 = Objective(Q=np.diag([0.01, 0.01]), R=np.array([[5.0]]), kind="quad", name="economia")
    J1 = Objective(Q=np.diag([5.0, 1.0]), R=np.array([[0.01]]), kind="quad", name="prestaciones")
    return Problem(plant_tfm(), [J0, J1], N=5, name="conflicto")
