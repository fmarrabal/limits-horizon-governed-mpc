# -*- coding: utf-8 -*-
"""Anota en fleet3.json y fleet_decomp.json los p ajustados por Holm.

POR QUE EXISTE. El manuscrito decia "Holm-adjusted p < 1.5e-2 throughout" y
ningun script aplicaba Holm a la flota: los p impresos eran crudos (revision
adversarial del 21-ago, hallazgo G-7). Este paso lee los p crudos archivados
por run_fleet3.py / run_fleet_decomp.py, aplica Holm dentro de cada familia
(los ocho presupuestos de un mismo rho) y escribe p_holm al lado, sin tocar
ningun otro campo. Es post-proceso: no vuelve a simular nada.

  python holm_fleet3.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ghi.stats import holm

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "results")


def anota(ruta, familias):
    d = json.load(open(ruta))
    for fam in familias:
        filas = [r for r in d.get(fam, []) if r.get("p") is not None]
        if not filas:
            continue
        # ghi.stats.holm trabaja sobre objetos con .p y anota .p_holm
        class _T:
            def __init__(self, p):
                self.p = p
                self.p_holm = None
        ts = [_T(r["p"]) for r in filas]
        holm(ts)
        for r, t in zip(filas, ts):
            r["p_holm"] = float(t.p_holm)
        print(f"  {os.path.basename(ruta)} / {fam}: "
              + ", ".join(f"B={r['B']}: p={r['p']:.2e} -> {r['p_holm']:.2e}"
                          for r in filas))
    with open(ruta, "w") as fh:
        json.dump(d, fh, indent=1, default=float)


if __name__ == "__main__":
    anota(os.path.join(RES, "fleet3.json"), ["rho_0", "rho_1"])
    anota(os.path.join(RES, "fleet_decomp.json"),
          ["rho_0", "rho_1", "temporal_rho_0", "temporal_rho_1"])
    print("escrito p_holm en fleet3.json y fleet_decomp.json")
