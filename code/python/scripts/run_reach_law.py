"""La ley de alcance, medida SIN censura y SIN el efecto del nodo final.

QUE ESTABA MAL EN LA PRIMERA VERSION
-------------------------------------
La primera medida se hizo sobre una cadena de M = 5 nodos y produjo dos defectos
que invalidaban la comprobacion de la ley:

  1. EL ALCANCE ES UN ENTERO. La ley n_max = log(D)/log(rho) se traslado a la
     figura como una CURVA CONTINUA, cuando lo que se mide es un numero de
     saltos. Comparar un entero con una curva hace que la ley parezca
     sistematicamente desplazada hacia arriba: hay que tomar el suelo.

  2. EL PUNTO MAS FINO ESTABA CENSURADO. Con D = 0.005 la ley predice alcance 5,
     pero una cadena de 5 nodos solo tiene saltos 0..4, asi que 4 es el maximo
     OBSERVABLE. Ese punto no confirma ni desmiente la ley: no se puede usar.

  3. EL NODO FINAL FALSEA rho. En una cadena el ultimo nodo tiene grado 1, luego
     no cede senal aguas abajo y retiene mas de la que le corresponde: la razon
     medida entre el ultimo salto y el anterior es 0.52 frente a 0.35 en el
     interior. Ajustar rho incluyendo ese salto lo sesga de 0.349 a 0.395.

Aqui se corrige lo tercero midiendo rho SOLO en nodos interiores, y lo segundo
alargando la cadena hasta que el alcance predicho quepa dentro de ella.
"""
from __future__ import annotations

import io
import json
import os
import sys

import contextlib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_distributed as rd


M = 8                      # cadena larga: el alcance predicho cabe dentro
GRIDS = [51, 101, 201, 401]
SEEDS = 3


def main() -> dict:
    print("=" * 78)
    print(f"LEY DE ALCANCE, cadena de M={M} nodos (sin censura por longitud)")
    print("=" * 78)
    print("  amplitud relativa transportada por salto, y alcance = ultimo salto")
    print("  cuya amplitud supera un paso de rejilla\n")
    print(f"  {'rejilla':>8}{'paso D':>9}   perfil por salto (0..%d)" % (M - 1))
    filas = []
    for g in GRIDS:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            r = rd.main(M=M, n_seeds=SEEDS, grid_points=g, forcing="source")
        amp = np.array(r["onda"]["amplitud_relativa"], float)
        D = 1.0 / (g - 1)
        alcance = int(np.max(np.flatnonzero(amp > 0))) if np.any(amp > 0) else 0
        filas.append({"rejilla": g, "paso": D, "amplitud": amp.tolist(),
                      "alcance": alcance})
        print(f"  {g:>8}{D:>9.4f}   " + " ".join(f"{v:6.4f}" for v in amp))

    # rho SOLO en nodos INTERIORES: el ultimo nodo de la cadena tiene grado 1 y
    # retiene senal que no cede aguas abajo, asi que sesga el ajuste
    ref = np.array(filas[-1]["amplitud"], float)
    interior = [n for n in range(1, M - 1) if ref[n] > 0]
    rho = float(np.exp(np.polyfit(interior, np.log(ref[interior]), 1)[0]))
    rho_todo = float(np.exp(np.polyfit([n for n in range(1, M) if ref[n] > 0],
                                       np.log(ref[[n for n in range(1, M) if ref[n] > 0]]), 1)[0]))
    print(f"\n  rho ajustado en nodos INTERIORES (saltos {interior[0]}..{interior[-1]}): {rho:.4f}")
    print(f"  rho ajustado incluyendo el nodo FINAL: {rho_todo:.4f}  <- sesgado")

    print(f"\n  {'paso D':>9}{'ley log D/log rho':>19}{'floor':>7}{'medido':>8}{'censurado?':>12}")
    ok = True
    for f in filas:
        ley = np.log(f["paso"]) / np.log(rho)
        pred = int(np.floor(ley))
        cens = pred > M - 1
        coincide = (f["alcance"] == pred) if not cens else None
        f.update(ley=float(ley), prediccion=pred, censurado=bool(cens))
        if coincide is False:
            ok = False
        print(f"  {f['paso']:>9.4f}{ley:>19.2f}{pred:>7}{f['alcance']:>8}"
              f"{'SI' if cens else 'no':>12}"
              + ("" if cens else ("  OK" if coincide else "  DISCREPA")))

    usables = [f for f in filas if not f["censurado"]]
    print(f"\n  puntos utilizables: {len(usables)}/{len(filas)} | "
          f"la ley con suelo {'ACIERTA en todos' if ok else 'FALLA en alguno'}")
    return {"M": M, "rho_interior": rho, "rho_sesgado": rho_todo, "filas": filas,
            "ley_valida": bool(ok)}


if __name__ == "__main__":
    res = main()
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dst = os.path.join(here, "results")
    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(dst, "reach_law.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    print(f"\nescrito {os.path.join(dst, 'reach_law.json')}")
