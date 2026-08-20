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

    with open(os.path.join(RES, "ceiling_ci.json"), "w") as f:
        json.dump(salida, f, indent=1)
    print("escrito", os.path.join(RES, "ceiling_ci.json"))


if __name__ == "__main__":
    main()
