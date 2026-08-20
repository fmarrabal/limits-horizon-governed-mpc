"""Verifica que cada cifra del Comment coincide con la evidencia archivada.

Uso:  python _check_numbers.py      (codigo de salida 0 = todo verde)

La evidencia vive en ../evidencia/ y la generan dos scripts independientes del
MPC del proyecto: run_bemporad_lemma4.py (el Lema y sus consecuencias sobre el
problema (13) del articulo comentado) y run_bemporad_remedy.py (el remedio
exacto por union de piezas, sobre el simplex que ese articulo declara).
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EV = os.path.join(os.path.dirname(HERE), "evidencia")
import re as _re
_RAW = open(os.path.join(HERE, "comment.tex"), encoding="utf-8").read()
# insensible al ajuste de lineas: el texto se compara con espacios normalizados
TEX = _re.sub(r"\s+", " ", _RAW)

L4 = json.load(open(os.path.join(EV, "bemporad_lemma4.json")))
RM = json.load(open(os.path.join(EV, "bemporad_remedy.json")))

fallos = 0


def chk(tag, cond, detalle=""):
    global fallos
    print(f"  [{'OK ' if cond else 'FALLA'}] {tag}{'  ' + detalle if detalle else ''}")
    if not cond:
        fallos += 1


def en_texto(*cadenas):
    return all(c in TEX for c in cadenas)


print("Lema y verificacion directa (bemporad_lemma4.json)")
t1, t2, t3 = L4["T1_concava_en_mu"], L4["T2_convexa_en_x"], L4["T3_minimo_de_piezas"]
chk("1.200 cuerdas en mu", t1["tests"] == 1200 and en_texto("$1{,}200$"))
chk("791 estrictamente concavas", t1["concava"] == 791 and en_texto("$791$"))
chk("ninguna convexa en mu", t1["convexa"] == 0)
chk("1.200 cuerdas en x, 320 estrictas", t2["tests"] == 1200 and t2["convexa"] == 320
    and t2["concava"] == 0 and en_texto("$320$"))
chk("el maximo de piezas sobrestima hasta 5.15",
    abs(t3["err_max"] - 5.15) < 0.02 and en_texto("$5.15$"))
chk("el minimo reproduce a 7e-15",
    t3["err_min"] < 1e-14 and en_texto("7", "10^{-15}"))

print("\nRemedio exacto y gap de la (17) (bemporad_remedy.json)")
r1, r2, r3, r4 = (RM["T1_union_reproduce_el_conjunto"],
                  RM["T2_la_17_es_subconjunto"],
                  RM["T3_17_infactible_con_admisible_no_vacio"],
                  RM["T4_no_convexidad_en_el_simplex"])
n = r2["casos"]
chk("480 pares instancia-nivel", n == 480 and en_texto("$480$"))
chk("(17) siempre contenida", r2["subconjunto"] == n)
chk("estrictamente menor en 468 (97.5%)",
    r2["estrictamente_menor"] == 468 and en_texto("$468$", "97.5"))
chk("infactible con admisible no vacio en 302 (62.9%)",
    r3["veces"] == 302 and en_texto("$302$", "62.9"))
chk("no convexo en 232 (48.3%)",
    r4["no_convexos"] == 232 and en_texto("$232$", "48.3"))
chk("la union reproduce el conjunto en 466 (97.1%)",
    r1["exactos"] == 466 and en_texto("$466$", "97.1"))
chk("minimo de piezas = V* a 2e-9",
    r1["peor_error_min_piezas_vs_V"] < 1e-8 and en_texto("2", "10^{-9}"))

print("\nReglas de honestidad comprometidas")
chk("el claim de seguridad va acotado a la rama afin a trozos",
    "rest on that constraint" in TEX and "piecewise affine scheme is safe" in TEX)
chk("no se caracteriza el alcance de Mangasarian-Rosen",
    "Mangasarian and Rosen" in TEX and "situation covered by" not in TEX)
chk("el remedio NO llama exacto al mallado",
    "an approximation, not an exact method" in TEX)
chk("fecha del contacto con los autores", "7~August 2026" in TEX)

print(f"\n{'TODAS LAS CIFRAS DEL COMMENT COINCIDEN' if not fallos else f'{fallos} FALLOS'}")
sys.exit(1 if fallos else 0)
