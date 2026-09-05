# -*- coding: utf-8 -*-
"""Incertidumbre de la fraccion de techo recuperada por el retardo afin.

El manuscrito daba ``$90.9\\%$ de una cota clarividente'' como si fuera una
cifra puntual. No lo es: el denominador (local menos techo) es una diferencia
entre dos medias ruidosas, y su s.e.m. archivada vale $30.6$ puntos. Este
script remuestrea las treinta semillas emparejadas y publica el intervalo,
que es lo que permite decir --- o no decir --- que el mecanismo causal esta
``en el techo''.

Publica code/results/ceiling_ci.json.
"""
from __future__ import annotations

import json
import os

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(AQUI)))
RES = os.path.join(os.path.dirname(os.path.dirname(AQUI)), "results")

SEMILLA = 20260805          # fija: el intervalo tiene que ser reproducible
REMUESTREOS = 20000


def fraccion(loc, cei, mec, idx):
    """Fraccion del hueco local-techo que cierra el mecanismo, en tanto por
    ciento. Se remuestrean las *semillas*, no los tres brazos por separado:
    el diseno es emparejado y romper el emparejamiento inflaria el intervalo.
    """
    return 100.0 * (loc[idx].mean() - mec[idx].mean()) / \
                   (loc[idx].mean() - cei[idx].mean())


def main():
    tabla = json.load(open(os.path.join(RES, "transport_affine.json")))["tabla"]
    loc = np.array(tabla["reactivo"]["por_semilla"])
    cei = np.array(tabla["clarividente"]["por_semilla"])
    n = len(loc)
    rng = np.random.default_rng(SEMILLA)
    idx0 = np.arange(n)

    salida = {"semilla": SEMILLA, "remuestreos": REMUESTREOS, "n_semillas": n,
              "mecanismos": {}}
    for nombre in ("retardo-afin", "afin-conformado", "onda", "retardo-lineal"):
        mec = np.array(tabla[nombre]["por_semilla"])
        punt = fraccion(loc, cei, mec, idx0)
        muestras = np.array([fraccion(loc, cei, mec, rng.integers(0, n, n))
                             for _ in range(REMUESTREOS)])
        lo, hi = np.percentile(muestras, [2.5, 97.5])
        salida["mecanismos"][nombre] = {
            "pct_techo": round(float(punt), 2),
            "ic95": [round(float(lo), 1), round(float(hi), 1)],
            "frac_por_encima_del_techo": round(float((muestras >= 100).mean()), 4),
        }
        print(f"  {nombre:16s} {punt:6.2f}%  IC95 [{lo:.1f}, {hi:.1f}]  "
              f"P(>=100%) = {(muestras >= 100).mean():.3f}")

    # El hueco de COSTE entre el afin y la referencia acausal, que es lo que el
    # texto debe decir en lugar de "statistically at the ceiling": intervalo
    # bootstrap del hueco relativo, semillas en que el causal es MAS barato, y
    # el contraste pareado crudo (el p impreso en la tabla es el de Holm).
    # Va DESPUES del bucle para no alterar la secuencia aleatoria de los
    # intervalos de arriba, que el manuscrito ya cita.
    mec = np.array(tabla["retardo-afin"]["por_semilla"])
    hueco = lambda idx: 100.0 * (mec[idx].mean() - cei[idx].mean()) / cei[idx].mean()
    bs = np.array([hueco(rng.integers(0, n, n)) for _ in range(REMUESTREOS)])
    d = mec - cei
    t_cr = float(d.mean() / (d.std(ddof=1) / np.sqrt(n)))
    from scipy import stats as _st
    p_cr = float(2 * _st.t.sf(abs(t_cr), n - 1))
    salida["hueco_afin_vs_referencia"] = {
        "hueco_pct": round(float(hueco(idx0)), 2),
        "ic95_pct": [round(float(np.percentile(bs, 2.5)), 1),
                     round(float(np.percentile(bs, 97.5)), 1)],
        "semillas_causal_mas_barato": int((mec < cei).sum()),
        "t_pareado_crudo": round(t_cr, 2), "p_crudo_bilateral": round(p_cr, 3)}
    print(f"  hueco afin - referencia: {hueco(idx0):+.2f}%  IC95 "
          f"[{np.percentile(bs, 2.5):+.1f}, {np.percentile(bs, 97.5):+.1f}]  "
          f"causal mas barato en {(mec < cei).sum()}/{n} semillas  "
          f"t={t_cr:+.2f} p={p_cr:.3f}")

    with open(os.path.join(RES, "ceiling_ci.json"), "w") as f:
        json.dump(salida, f, indent=1)
    print("escrito", os.path.join(RES, "ceiling_ci.json"))


if __name__ == "__main__":
    main()
