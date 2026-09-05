# -*- coding: utf-8 -*-
"""Ancla las comprobaciones al manuscrito.

Un verificador que compara cifras transcritas a mano contra los JSON verifica
la transcripcion, no el manuscrito: si alguien edita main.tex, sigue en verde.
Este modulo cierra el lazo. Para cada comprobacion exige que el valor deje
rastro literal en main.tex, admitiendo las formas en que LaTeX lo puede
renderizar (porcentaje, separador de millares, un redondeo mas corto).

El rastro tiene que ser *especifico*. Una version anterior admitia cualquier
redondeo, y entonces un ``0.6`` cualquiera del texto anclaba un archivado
$0.5956$: el anclaje pasaba en verde sin decir nada. Aqui una forma solo vale
si reconstruye el valor con error relativo por debajo de 5e-4, y la version en
tanto por ciento solo se admite para fracciones, que es cuando el texto la usa.

Los valores que el manuscrito expresa con palabras --- ``zero violations'',
``exact'' --- no dejan rastro numerico y se declaran uno a uno en PERMITIDOS
de cada verificador, nunca por omision silenciosa.
"""
from __future__ import annotations

import io
import re

BARRA = chr(92)


def sin_comentarios(t: str) -> str:
    """Quita los comentarios de LaTeX respetando el escape ``\\%``."""
    fuera = []
    for linea in t.split("\n"):
        corte, escapado = None, False
        for k, ch in enumerate(linea):
            if escapado:
                escapado = False
                continue
            if ch == BARRA:
                escapado = True
                continue
            if ch == "%":
                corte = k
                break
        fuera.append(linea if corte is None else linea[:corte])
    return "\n".join(fuera)


def normaliza(tex: str) -> str:
    tex = sin_comentarios(tex)
    return tex.replace("{,}", "").replace(BARRA + ",", "")   # 49\,650 -> 49650


def literales_de(tex: str) -> set[str]:
    return set(re.findall(r"[0-9]+(?:[.][0-9]+)?", normaliza(tex)))


def literales(ruta_tex: str) -> set[str]:
    return literales_de(io.open(ruta_tex, encoding="utf-8").read())


def variantes(v: float) -> set[str]:
    """Formas fieles en que el manuscrito puede escribir ``v``.

    Una forma vale si es el redondeo *correcto* de ``v`` a la precision con
    que esta impresa y conserva al menos dos cifras significativas. Lo
    primero admite lo que un articulo escribe de verdad ($4.64$ por
    $4.6449$); lo segundo impide que un ``0.6`` cualquiera del texto ancle un
    archivado $0.5956$, que es como una version anterior de este modulo
    pasaba en verde sin comprobar nada.
    """
    v = abs(float(v))
    if v == 0:
        return set()
    escalas = [v] + ([v * 100.0] if v < 1.0 else [])     # 0.122 -> ``12.2\%''
    salida = set()
    for x in escalas:
        for d in range(7):
            s = f"{x:.{d}f}"
            if float(s) == 0.0 or round(x, d) != float(s):
                continue
            if len(s.replace(".", "").lstrip("0")) < 2 and float(s) != x:
                continue
            salida.add(s)
            if "." in s:                                  # 0.59560 -> 0.5956
                salida.add(s.rstrip("0").rstrip("."))
    return {s for s in salida if s}


def _fmt(x: float) -> str:
    """Forma decimal limpia, sin notacion cientifica ni ceros finales."""
    s = f"{abs(float(x)):.10f}".rstrip("0").rstrip(".")
    return s or "0"


def formas_impresas(v: float) -> set[str]:
    """Las formas en que el manuscrito puede imprimir EXACTAMENTE ``v``.

    A diferencia de ``variantes``, no admite redondeos mas cortos: ``man`` es lo
    que el autor afirma haber impreso, y el anclaje tiene que encontrar eso
    mismo. Un ``0.15`` del verificador no puede anclarse a un ``15`` suelto
    del texto (21-ago: con ``variantes`` pasaba, y 1946 mutantes sobrevivian).
    Solo se toleran ceros finales (``14`` / ``14.0``; ``0.15`` / ``0.150``).
    """
    s = _fmt(v)
    out = {s, s + "0" if "." in s else s + ".0"}
    if "." in s:
        out.add(s + "00")
    return out


def formas_impresas_pct(v: float) -> set[str]:
    """La forma en tanto por ciento, admisible SOLO junto a ``\\%``."""
    if abs(float(v)) >= 1.0:
        return set()
    return formas_impresas(round(float(v) * 100.0, 8))


def literales_pct_de(tex: str) -> set[str]:
    """Los numeros que el texto imprime como tanto por ciento: seguidos de
    ``\\%``, o como extremo inferior de un rango ``$a$--$b\\%$``."""
    t = normaliza(tex)
    num = r"([0-9]+(?:[.][0-9]+)?)"
    pct = re.escape(BARRA) + "%"
    sueltos = set(re.findall(num + pct, t))
    rangos = set(re.findall(num + r"\$?--\$?[0-9]+(?:[.][0-9]+)?" + pct, t))
    return sueltos | rangos


def presente(man: float, lits: set[str], lits_pct: set[str]) -> set[str]:
    """Formas de ``man`` presentes en el texto (vacio = huerfano)."""
    return (formas_impresas(man) & lits) | (formas_impresas_pct(man) & lits_pct)


def redondeo_correcto(man: float, arch: float) -> bool:
    """``man`` tiene que ser el redondeo correcto de ``arch`` a la precision con
    que esta impreso: 1.83 no vale para un archivado 1.8249 aunque una
    tolerancia relativa lo deje pasar (G-24b)."""
    s = _fmt(man)
    d = len(s.split(".")[1]) if "." in s else 0
    return abs(abs(float(man)) - abs(float(arch))) <= 0.5 * 10.0 ** (-d) + 1e-12


def sin_rastro(checks, ruta_tex: str, permitidos=()):
    """Comprobaciones cuyo valor impreso no aparece en el manuscrito."""
    tex = io.open(ruta_tex, encoding="utf-8").read()
    lits, lits_pct = literales_de(tex), literales_pct_de(tex)
    permitidos = set(permitidos)
    return [(tag, man) for tag, man, _a, _t in checks
            if tag not in permitidos and not presente(man, lits, lits_pct)]


def informe(checks, ruta_tex: str, permitidos=()) -> int:
    """Imprime el estado del anclaje y devuelve el numero de huerfanos."""
    huerfanos = sin_rastro(checks, ruta_tex, permitidos)
    anclados = len(checks) - len(permitidos) - len(huerfanos)
    for tag, man in huerfanos:
        print(f"  HUERFANO  {tag}: {man} no aparece en main.tex")
    print(f"anclaje: {anclados}/{len(checks)} cifras localizadas en el "
          f"manuscrito, {len(permitidos)} declaradas en palabras, "
          f"{len(huerfanos)} huerfanas")
    return len(huerfanos)

def _hojas(o):
    if isinstance(o, dict):
        for v in o.values():
            yield from _hojas(v)
    elif isinstance(o, (list, tuple)):
        for v in o:
            yield from _hojas(v)
    elif isinstance(o, bool):
        pass
    elif isinstance(o, (int, float)):
        yield float(o)
    elif isinstance(o, str):
        # algunos veredictos se archivan como texto ("F3 VIVE C-plana:1.37+-0.97",
        # "refractario 170-206 muestras"): la cifra esta archivada aunque el
        # tipo del JSON sea una cadena, y ocultarla obligaria a declararla como
        # si no tuviera origen
        for t in re.findall(r"[0-9]+(?:[.][0-9]+)?", o):
            try:
                yield float(t)
            except ValueError:
                pass


def formas_archivadas(patrones):
    """Todas las formas en que los JSON archivados pueden aparecer impresas."""
    import glob
    import json
    salida = set()
    for patron in patrones:
        for fichero in glob.glob(patron):
            try:
                datos = json.load(io.open(fichero, encoding="utf-8"))
            except Exception:
                continue
            for v in _hojas(datos):
                salida |= variantes(v)
    return salida


def cuerpo(ruta_tex: str) -> str:
    """El texto del articulo sin preambulo, bibliografia ni argumentos de
    comandos (etiquetas, citas, rutas de figura llevan numeros que no son
    resultados)."""
    t = normaliza(io.open(ruta_tex, encoding="utf-8").read())
    t = t.split(BARRA + "begin{document}")[-1]
    t = t.split(BARRA + "bibliography")[0]
    t = re.split(re.escape(BARRA + "begin{thebibliography}"), t)[0]
    for cmd in ("citep", "citet", "citealp", "cite", "label", "ref",
                "eqref", "url", "href", "includegraphics", "bibitem"):
        t = re.sub(re.escape(BARRA + cmd) + r"[*]?(\[[^]]*\])?\{[^}]*\}", " ", t)
    return t


def trazabilidad(ruta_tex, patrones_json, declarados=(), extra=()):
    """Literales del cuerpo que no se regeneran de ningun fichero archivado.

    ``extra`` son formas adicionales que el verificador calcula (agregados,
    cocientes) y que por tanto tampoco son hojas de un JSON.
    """
    formas = formas_archivadas(patrones_json) | set(extra)
    lits = set(re.findall(r"(?<![0-9.])[0-9]+(?:[.][0-9]+)?(?![0-9])",
                          cuerpo(ruta_tex)))
    inform = {x for x in lits if ("." in x) or len(x) >= 3}
    huerfanos = sorted(inform - formas - set(declarados))
    return inform, huerfanos
