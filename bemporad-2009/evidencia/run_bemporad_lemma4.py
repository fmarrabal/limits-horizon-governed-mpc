"""Verificacion directa del Lema 4 de Bemporad & Munoz de la Pena (Automatica 2009).

Este script NO usa el MPC del paquete. Ataca la afirmacion tal como esta escrita,
sobre su propio problema (13), para que la evidencia sea independiente de como
yo haya instanciado el controlador.

EL ENUNCIADO, LITERAL (Automatica 45(12):2823-2830, 2009, p. 2826)
------------------------------------------------------------------
  "Lemma 4. Consider the multiparametric linear problem (13) with parameters
   mu in R^l IN THE COST FUNCTION and x in R^n in the rhs of the constraints.
   Then the set F* of parameters (mu,x) for which (13) has a solution is a
   convex polyhedron, the value function V*: F* -> R is continuous w.r.t.
   (mu,x), CONVEX and piecewise affine w.r.t. mu for any given x and w.r.t. x
   for any given mu."

y su problema (13), literal:

       min_z  (c'_0 + mu' C_mu) z      s.a.   G z <= b + S x

EL ARGUMENTO, EN DOS LINEAS
---------------------------
El objetivo es lineal en z con coeficientes AFINES en mu, y el factible
{z : Gz <= b + Sx} NO depende de mu. Luego, para cada z factible fijo, el
objetivo es una funcion AFIN de mu, y

       V*(mu, x) = min_z  [afin en mu]

es el infimo puntual de una familia de funciones afines: CONCAVO en mu.

Es el hecho clasico de programacion parametrica: la funcion de valor de un LP
es CONVEXA en el lado derecho y CONCAVA en los coeficientes del coste. El Lema 4
acierta en x (que esta en el rhs) y tiene el signo cambiado en mu (que esta en
el coste). Y la prueba de su Teorema 6 repite el error:

  "the value function V*(mu,x) can be evaluated as the MAXIMUM of the affine
   functions {(phi_i x + gamma_i)'(c'_0 + C'_mu mu)}"

siendo que es el MINIMO.

QUE COMPRUEBA ESTE SCRIPT
-------------------------
  T1  test de cuerda en mu   -> debe salir CONCAVA
  T2  test de cuerda en x    -> debe salir CONVEXA (la parte que ellos aciertan)
  T3  V* es el MINIMO de las piezas afines, no el maximo
  T4  el conjunto {mu : V*(mu) <= J_a} puede ser NO convexo
  T5  la restriccion de su LP (17) es un SUBCONJUNTO ESTRICTO del admisible real
      -> el algoritmo es SEGURO, pero no equivalente ni optimo
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.optimize import linprog

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# --------------------------------------------------------------------------
#              UN mp-LP ALEATORIO CON LA ESTRUCTURA DE SU (13)
# --------------------------------------------------------------------------

def random_mplp(d=6, l=2, n=2, ncon=8, seed=0):
    """min_z (c0 + Cmu' mu)' z   s.a.   G z <= b + S x, con el factible acotado."""
    rng = np.random.default_rng(seed)
    c0 = rng.normal(size=d)
    Cmu = rng.normal(size=(l, d))                 # fila i = c_i - c_0
    Grand = rng.normal(size=(ncon, d))
    brand = 2.0 + rng.uniform(0, 2, ncon)
    # caja para que el LP sea siempre acotado (si no, V* = -inf y no hay nada que medir)
    G = np.vstack([Grand, np.eye(d), -np.eye(d)])
    b = np.concatenate([brand, np.full(d, 3.0), np.full(d, 3.0)])
    S = np.vstack([rng.normal(size=(ncon, n)) * 0.3, np.zeros((2 * d, n))])
    return c0, Cmu, G, b, S


def V_star(mu, x, c0, Cmu, G, b, S):
    """Valor optimo y optimizador de (13). None si es infactible."""
    c = c0 + Cmu.T @ np.asarray(mu, float).ravel()
    r = linprog(c, A_ub=G, b_ub=b + S @ np.asarray(x, float).ravel(),
                bounds=[(None, None)] * len(c0), method="highs")
    if not r.success:
        return None, None
    return float(r.fun), np.asarray(r.x, float)


# --------------------------------------------------------------------------

def chord_test(fun, p1, p2, tol_rel=1e-9):
    """d = f(medio) - [f(p1)+f(p2)]/2.  d>0 => concava ; d<0 => convexa."""
    v1, v2 = fun(p1), fun(p2)
    vm = fun(0.5 * (np.asarray(p1, float) + np.asarray(p2, float)))
    if None in (v1, v2, vm):
        return None
    d = vm - 0.5 * (v1 + v2)
    tol = tol_rel * max(abs(v1), abs(v2), abs(vm), 1.0)
    return d, tol


def main(n_problems: int = 200, n_pairs: int = 6, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    print("=" * 78)
    print("VERIFICACION DIRECTA DEL LEMA 4 DE BEMPORAD & MUNOZ DE LA PENA (2009)")
    print("=" * 78)
    print("Problema (13):   min_z (c0 + Cmu' mu)' z   s.a.  G z <= b + S x")
    print("El factible NO depende de mu; el objetivo es AFIN en mu para cada z.")
    print("=> V*(.,x) = min de afines = CONCAVA en mu.\n")

    # ---------------------------------------------------------------- T1, T2
    nmu = nc_mu = nv_mu = 0
    nx = nc_x = nv_x = 0
    dmu_sum = dx_sum = 0.0
    for p in range(n_problems):
        c0, Cmu, G, b, S = random_mplp(seed=1000 + p)
        l, n = Cmu.shape[0], S.shape[1]
        x0 = rng.normal(size=n) * 0.5
        f_mu = lambda m: V_star(m, x0, c0, Cmu, G, b, S)[0]
        for _ in range(n_pairs):
            m1, m2 = rng.uniform(0, 1, l), rng.uniform(0, 1, l)
            r = chord_test(f_mu, m1, m2)
            if r is None:
                continue
            d, tol = r
            nmu += 1; dmu_sum += d
            if d < -tol: nc_mu += 1          # por debajo de la cuerda -> convexa
            if d >  tol: nv_mu += 1          # por encima -> concava
        mu0 = rng.uniform(0, 1, l)
        f_x = lambda xx: V_star(mu0, xx, c0, Cmu, G, b, S)[0]
        for _ in range(n_pairs):
            x1, x2 = rng.normal(size=n), rng.normal(size=n)
            r = chord_test(f_x, x1, x2)
            if r is None:
                continue
            d, tol = r
            nx += 1; dx_sum += d
            if d < -tol: nc_x += 1
            if d >  tol: nv_x += 1

    print("--- T1: curvatura de V* en MU (parametro en el COSTE) ---")
    print(f"  tests                                 : {nmu}")
    print(f"  por ENCIMA de la cuerda (concava)     : {nv_mu}")
    print(f"  por DEBAJO de la cuerda (convexa)     : {nc_mu}")
    print(f"  desviacion media                      : {dmu_sum / max(nmu,1):+.4e}")
    t1 = nc_mu == 0 and nv_mu > 0
    print(f"  => {'CONCAVA' if t1 else 'NO CONCLUYENTE'}  "
          f"(el Lema 4 dice CONVEXA: signo cambiado)\n")

    print("--- T2: curvatura de V* en X (parametro en el LADO DERECHO) ---")
    print(f"  tests                                 : {nx}")
    print(f"  por DEBAJO de la cuerda (convexa)     : {nc_x}")
    print(f"  por ENCIMA de la cuerda (concava)     : {nv_x}")
    print(f"  desviacion media                      : {dx_sum / max(nx,1):+.4e}")
    t2 = nv_x == 0 and nc_x > 0
    print(f"  => {'CONVEXA' if t2 else 'NO CONCLUYENTE'}  "
          f"(aqui el Lema 4 acierta; el fallo es solo en mu)\n")

    # ------------------------------------------------------------------- T3
    print("--- T3: V* es el MINIMO de las piezas afines, no el maximo ---")
    c0, Cmu, G, b, S = random_mplp(l=1, seed=7)
    x0 = np.zeros(S.shape[1])
    mus = np.linspace(0.0, 4.0, 241).reshape(-1, 1)
    vals, verts = [], []
    for m in mus:
        v, z = V_star(m, x0, c0, Cmu, G, b, S)
        vals.append(v); verts.append(z)
    vals = np.array(vals)
    # vertices distintos = piezas afines
    uniq = []
    for z in verts:
        if not any(np.allclose(z, u, atol=1e-6) for u in uniq):
            uniq.append(z)
    A = np.array([[(c0 + Cmu.T @ m) @ z for z in uniq] for m in mus])
    err_min = float(np.max(np.abs(vals - A.min(axis=1))))
    err_max = float(np.max(np.abs(vals - A.max(axis=1))))
    print(f"  piezas afines distintas encontradas   : {len(uniq)}")
    print(f"  max |V* - MIN de las piezas|          : {err_min:.3e}   <-- es el minimo")
    print(f"  max |V* - MAX de las piezas|          : {err_max:.3e}")
    t3 = err_min < 1e-7 < err_max
    print(f"  => {'V* = MINIMO de las piezas' if t3 else 'NO CONCLUYENTE'}  "
          f"(el Teorema 6 usa el MAXIMO)\n")

    # --------------------------------------------------------------- T4, T5
    print("--- T4/T5: el conjunto admisible de pesos y su LP ---")
    nc_found = 0; strict = 0; ntried = 0
    ejemplo = None
    for p in range(60):
        c0, Cmu, G, b, S = random_mplp(l=1, seed=500 + p)
        x0 = np.zeros(S.shape[1])
        grid = np.linspace(0.0, 4.0, 161).reshape(-1, 1)
        V = np.array([V_star(m, x0, c0, Cmu, G, b, S)[0] for m in grid])
        Z = [V_star(m, x0, c0, Cmu, G, b, S)[1] for m in grid]
        uq = []
        for z in Z:
            if not any(np.allclose(z, u, atol=1e-6) for u in uq):
                uq.append(z)
        # barrer J_a entre el minimo y el maximo de V*
        for Ja in np.linspace(V.min(), V.max(), 40)[1:-1]:
            ntried += 1
            feas = V <= Ja + 1e-9
            if feas.sum() < 2:
                continue
            i1, i2 = np.where(feas)[0][[0, -1]]
            hueco = bool(np.any(~feas[i1:i2 + 1]))
            # el conjunto que impone SU LP: TODAS las piezas afines <= J_a
            allp = np.array([[(c0 + Cmu.T @ m) @ z for z in uq] for m in grid])
            lp_feas = np.all(allp <= Ja + 1e-9, axis=1)
            if hueco:
                nc_found += 1
                if ejemplo is None:
                    ejemplo = (p, float(Ja), int(feas.sum()), int(lp_feas.sum()))
            if lp_feas.sum() < feas.sum():
                strict += 1
            assert not np.any(lp_feas & ~feas), "el LP deberia ser un SUBCONJUNTO"
    print(f"  niveles de J_a barridos                       : {ntried}")
    print(f"  conjunto admisible NO convexo                 : {nc_found} "
          f"({100 * nc_found / max(ntried,1):.1f}%)")
    print(f"  su LP es SUBCONJUNTO ESTRICTO del admisible   : {strict} "
          f"({100 * strict / max(ntried,1):.1f}%)")
    print(f"  su LP nunca acepta un mu inadmisible          : SI (comprobado en todos)")
    if ejemplo:
        print(f"  ejemplo: problema {ejemplo[0]}, J_a={ejemplo[1]:.4f}, "
              f"{ejemplo[2]} puntos admisibles de los que su LP ve {ejemplo[3]}")
    print()
    print("  LECTURA: el LP (17) impone TODAS las piezas afines <= J_a. Como V* es el")
    print("  MINIMO de esas piezas, exigirlas todas es MAS FUERTE que V* <= J_a. Su LP")
    print("  es por tanto una RESTRICCION INTERIOR conservadora: el algoritmo SIGUE")
    print("  SIENDO SEGURO -- nunca acepta un peso que viole el certificado -- pero la")
    print("  equivalencia con (6) y la optimalidad de alpha* enunciadas son falsas.")

    print("\n" + "=" * 78)
    ok = t1 and t2 and t3 and nc_found > 0
    print("VEREDICTO: " + ("CONFIRMADO en las cinco comprobaciones" if ok
                           else "REVISAR: alguna comprobacion no concluye"))
    print("=" * 78)
    return {"T1_concava_en_mu": {"tests": nmu, "concava": nv_mu, "convexa": nc_mu,
                                 "d_media": dmu_sum / max(nmu, 1), "ok": bool(t1)},
            "T2_convexa_en_x": {"tests": nx, "convexa": nc_x, "concava": nv_x,
                                "d_media": dx_sum / max(nx, 1), "ok": bool(t2)},
            "T3_minimo_de_piezas": {"piezas": len(uniq), "err_min": err_min,
                                    "err_max": err_max, "ok": bool(t3)},
            "T4_T5_conjunto": {"niveles": ntried, "no_convexos": nc_found,
                               "lp_subconjunto_estricto": strict},
            "veredicto_ok": bool(ok)}


if __name__ == "__main__":
    res = main()
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dst = os.path.join(here, "results")
    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(dst, "bemporad_lemma4.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"\nescrito {os.path.join(dst, 'bemporad_lemma4.json')}")
