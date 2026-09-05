"""Evidencia del REMEDIO EXACTO propuesto en el Comment, sobre el problema (13)
de Bemporad & Munoz de la Pena (Automatica 2009).

QUE SE VERIFICA
---------------
Por el Lema del Comment, V*(.,x) es el MINIMO de las piezas afines
{p_i(mu) = (phi_i x + gamma_i)'(c_0 + C_mu' mu)}_{i in I(x)}. Por tanto

    A(x,J_a) = {mu in Delta : V*(mu,x) <= J_a}
             = UNION_{i in I(x)} {mu in Delta : p_i(mu) <= J_a},                (R)

que es una union de a lo sumo card(I(x)) POLIEDROS, cada uno definido por UNA
desigualdad afin sobre el mismo simplex. Es decir: el conjunto admisible exacto
se obtiene resolviendo card(I(x)) programas lineales del MISMO tamano que su
(17), sin renunciar a la exactitud. La (17) publicada, en cambio, impone TODAS
las piezas a la vez, que es la INTERSECCION: una aproximacion interior.

  T1  la union (R) reproduce el conjunto admisible verdadero (rejilla fina)
  T2  la (17) publicada es un subconjunto de (R), y con que frecuencia ESTRICTO
  T3  con que frecuencia (17) es INFACTIBLE mientras A(x,J_a) NO es vacio
      (ahi el esquema publicado no encuentra peso admisible aunque exista;
       el remedio si, y devuelve ademas el testigo de factibilidad recursiva)
  T4  con que frecuencia A(x,J_a) es NO CONVEXO, medido SOLO sobre el simplex
      declarado en el articulo comentado (sum mu_i <= 1), no sobre un rango
      mas ancho

Todo con el mismo generador de mp-LP aleatorios que run_bemporad_lemma4.py, sin
usar el MPC del proyecto: la evidencia es independiente de como instanciemos
nosotros el controlador.
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy.optimize import linprog

from run_bemporad_lemma4 import random_mplp, V_star


def piezas_activas(x, c0, Cmu, G, b, S, mus):
    """Vertices optimos visitados al barrer mu: cada uno define una pieza afin
    p_i(mu) = (c_0 + C_mu' mu)' z_i. Devuelve la lista de z_i distintos."""
    Z = []
    for mu in mus:
        c = c0 + Cmu.T @ np.asarray(mu).ravel()
        r = linprog(c, A_ub=G, b_ub=b + S @ x, bounds=(None, None),
                    method="highs")
        if not r.success:
            continue
        z = np.round(r.x, 9)
        if not any(np.allclose(z, w, atol=1e-7) for w in Z):
            Z.append(z)
    return Z


def main(n_problems: int = 120, seed: int = 11) -> dict:
    rng = np.random.default_rng(seed)
    l = 2                                   # dos objetivos: simplex 2D
    # rejilla del SIMPLEX declarado en el articulo: mu >= 0, sum mu <= 1
    g = np.linspace(0.0, 1.0, 41)
    MU = np.array([[a, b_] for a in g for b_ in g if a + b_ <= 1.0 + 1e-12])

    t1_ok = t1_tot = 0
    t2_sub = t2_estricto = t2_tot = 0
    t3_lp_vacio_con_admisible = t3_tot = 0
    t4_noconvex = t4_tot = 0
    peor_err = 0.0
    # por cuantil: el protocolo de niveles J_a no estaba revelado en el
    # Comment ni desglosado en el archivo (revision adversarial del 21-ago)
    CUANTILES = (0.15, 0.35, 0.55, 0.75)
    por_q = {q: {"casos": 0, "infactible_con_admisible": 0,
                 "estrictamente_menor": 0, "no_convexos": 0} for q in CUANTILES}

    for p in range(n_problems):
        c0, Cmu, G, b, S = random_mplp(d=6, l=l, n=2, ncon=8, seed=2000 + p)
        x = rng.normal(size=2) * 0.5
        Vt = [V_star(mu, x, c0, Cmu, G, b, S)[0] for mu in MU]
        if any(v is None for v in Vt):
            continue
        V = np.array(Vt, float)
        Z = piezas_activas(x, c0, Cmu, G, b, S, MU)
        if not Z:
            continue
        # piezas afines evaluadas en toda la rejilla
        P = np.array([[float((c0 + Cmu.T @ mu) @ z) for z in Z] for mu in MU])
        for q in CUANTILES:
            Ja = float(np.quantile(V, q))
            verdad = V <= Ja + 1e-9
            union = (P.min(axis=1) <= Ja + 1e-9)        # remedio exacto (R)
            inter = (P.max(axis=1) <= Ja + 1e-9)        # la (17) publicada
            pq = por_q[q]
            pq["casos"] += 1
            t1_tot += 1
            if np.array_equal(verdad, union):
                t1_ok += 1
            peor_err = max(peor_err, float(np.max(np.abs(
                P.min(axis=1) - V))))
            t2_tot += 1
            if np.all(inter <= verdad):
                t2_sub += 1
            if inter.sum() < verdad.sum():
                t2_estricto += 1
                pq["estrictamente_menor"] += 1
            t3_tot += 1
            if inter.sum() == 0 and verdad.sum() > 0:
                t3_lp_vacio_con_admisible += 1
                pq["infactible_con_admisible"] += 1
            # no convexidad: algun punto medio de dos admisibles cae fuera
            t4_tot += 1
            idx = np.where(verdad)[0]
            if len(idx) >= 2:
                pares = rng.choice(idx, size=(min(60, len(idx)), 2))
                fuera = False
                for i, j in pares:
                    m = 0.5 * (MU[i] + MU[j])
                    vm = V_star(m, x, c0, Cmu, G, b, S)[0]
                    if vm is None or vm > Ja + 1e-7:
                        fuera = True
                        break
                if fuera:
                    t4_noconvex += 1
                    pq["no_convexos"] += 1

    out = {
        "protocolo": {"problemas": n_problems, "l": l,
                      "rejilla": "simplex mu>=0, sum mu<=1, paso 1/40",
                      "niveles_por_problema": 4, "seed": seed,
                      "niveles": "cuantiles de V*(., x) sobre la rejilla del "
                                 "simplex: 0.15, 0.35, 0.55, 0.75",
                      "cuantiles": list(CUANTILES)},
        "por_cuantil": {str(q): v for q, v in por_q.items()},
        "T1_union_reproduce_el_conjunto": {
            "casos": t1_tot, "exactos": t1_ok,
            "peor_error_min_piezas_vs_V": peor_err},
        "T2_la_17_es_subconjunto": {
            "casos": t2_tot, "subconjunto": t2_sub,
            "estrictamente_menor": t2_estricto},
        "T3_17_infactible_con_admisible_no_vacio": {
            "casos": t3_tot, "veces": t3_lp_vacio_con_admisible},
        "T4_no_convexidad_en_el_simplex": {
            "casos": t4_tot, "no_convexos": t4_noconvex},
    }
    print(json.dumps(out, indent=1))
    dst = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "bemporad_remedy.json")
    with open(dst, "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print("escrito", dst)
    return out


if __name__ == "__main__":
    main()
