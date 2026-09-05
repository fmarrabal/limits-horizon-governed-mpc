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
chk("200 problemas x 6 cuerdas = 1.200 pruebas",
    L4["protocolo"]["problemas"] == 200
    and L4["protocolo"]["cuerdas"] == 6
    and t1["tests"] == 1200 and en_texto("$200$", "$1{,}200$"))
chk("746 estrictamente concavas (simplex)", t1["concava"] == 746 and en_texto("$746$"))
chk("ninguna convexa en mu", t1["convexa"] == 0)
chk("1.200 cuerdas en x, 295 estrictas", t2["tests"] == 1200 and t2["convexa"] == 295
    and t2["concava"] == 0 and en_texto("$295$"))
chk("el maximo de piezas sobrestima hasta 1.34 (simplex)",
    abs(t3["err_max"] - 1.34) < 0.01 and en_texto("$1.34$"))
chk("el minimo reproduce a 4e-15",
    t3["err_min"] < 1e-14 and en_texto("$4\\times10^{-15}$"))
chk("los pesos se muestrean en el simplex",
    "sampled in the simplex" in TEX)

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
chk("la union reproduce el conjunto en 466 (14 desacuerdos sobre el nivel)",
    r1["exactos"] == 466 and r1["casos"] - r1["exactos"] == 14
    and en_texto("$466$", "$14$ disagreements"))
_pq = RM["por_cuantil"]
_tasa = lambda q: 100.0 * _pq[q]["infactible_con_admisible"] / _pq[q]["casos"]
chk("protocolo de niveles revelado: cuantiles 15/35/55/75",
    RM["protocolo"]["cuantiles"] == [0.15, 0.35, 0.55, 0.75]
    and en_texto("$15$th, $35$th, $55$th and $75$th percentiles"))
chk("infactibilidad por nivel: 79.2% -> 45.0%",
    abs(_tasa("0.15") - 79.2) < 0.05 and abs(_tasa("0.75") - 45.0) < 0.05
    and en_texto("$79.2\\%$", "$45.0\\%$"))
chk("el remedio ya no se vende 'al mismo coste'",
    "at the same cost" not in TEX and "the price of exactness" in TEX)
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


# --- trazabilidad: ninguna cifra del cuerpo sin origen ----------------------
# Las comprobaciones de arriba van del archivo al texto. Este pase va al reves:
# cada literal informativo del Comment tiene que salir de la evidencia
# archivada o estar declarado como constante que no viene de ningun computo.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)),
                                "code", "python"))
from ghi.anclaje import trazabilidad as _traza

DECLARADOS = {
    "080002": "codigo postal de la afiliacion",
    "4.6": "version del sistema de IA, declarada en agradecimientos",
    "5.1": "version del sistema de IA, declarada en agradecimientos",
    "04120": "codigo postal de la afiliacion",
    "2009": "ano del articulo comentado",
    "2026": "ano del contacto con los autores",
    "13": "numero de ecuacion del articulo comentado",
    "17": "numero de ecuacion del articulo comentado",
    "0.05": "nivel nominal, constante de diseno",
    "1964": "ano de la referencia citada en prosa",
    "2823": "pagina inicial del articulo comentado, en el titulo",
    "2830": "pagina final del articulo comentado, en el titulo",
}
# los porcentajes se derivan de los recuentos archivados: se pasan como
# formas adicionales para no tener que declararlos como si no tuvieran origen
from ghi.anclaje import variantes as _var
_derivados = set()
for _q, _v in RM["por_cuantil"].items():          # tasas por nivel (G-17)
    _derivados |= _var(100.0 * _v["infactible_con_admisible"] / _v["casos"])
for _num, _den in ((r2["estrictamente_menor"], n), (r3["veces"], n),
                   (r4["no_convexos"], n), (r1["exactos"], n),
                   (t1["concava"], t1["tests"]), (t2["convexa"], t2["tests"])):
    _derivados |= _var(100.0 * _num / _den)
_inform, _huerf = _traza(os.path.join(HERE, "comment.tex"),
                         [os.path.join(EV, "*.json")], declarados=DECLARADOS,
                         extra=_derivados)
for _h in _huerf:
    print(f"  [FALLA] sin origen archivado: {_h}")
fallos += len(_huerf)
print(f"trazabilidad: {len(_inform)} literales informativos, "
      f"{len(_inform) - len(_huerf) - len(set(DECLARADOS) & _inform)} "
      f"con origen, {len(set(DECLARADOS) & _inform)} declarados, "
      f"{len(_huerf)} sin origen")

print(f"\n{'TODAS LAS CIFRAS DEL COMMENT COINCIDEN' if not fallos else f'{fallos} FALLOS'}")
sys.exit(1 if fallos else 0)
