# -*- coding: utf-8 -*-
"""Prueba de mutacion del verificador de cifras.

Un verificador en verde no prueba nada si no puede ponerse rojo. Aqui se
altera, una a una, cada cifra que el verificador regenera del archivo, y se
ejecuta el verificador *entero* contra el manuscrito alterado: tiene que
salir en rojo. Un mutante superviviente es una cifra que se podria editar en
main.tex sin que nadie se entere.

Se mutan a la vez todas las apariciones de esa cifra (cuerpo y tabla, con y
sin separador de millares): mutar una sola no probaria nada si se repite.

Uso:  python prueba_mutacion.py        (0 = ningun superviviente)
"""
from __future__ import annotations

import io
import json
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
BARRA = chr(92)


def regenerados() -> list[str]:
    """Cifras que el verificador saca del archivo, tal y como las escribe."""
    fd, ruta = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    env = dict(os.environ, LISTAR=ruta)
    subprocess.run([sys.executable, "_check_numbers.py"], cwd=HERE, env=env,
                   capture_output=True)
    valores = json.load(open(ruta, encoding="utf-8"))
    os.remove(ruta)
    return valores


def muta(token: str) -> str:
    return token[:-1] + str((int(token[-1]) + 1) % 10)


def main() -> int:
    original = io.open(os.path.join(HERE, "main.tex"), encoding="utf-8").read()
    vivos, probados, saltados = [], 0, []
    tmp = os.path.join(HERE, "_mutante.tex")

    try:
        for crudo in regenerados():
            # el verificador escribe algunas cifras con signo o separador
            token = (crudo.replace("{+}", "").replace("{-}", "")
                     .lstrip("+-").replace("{,}", "")
                     .replace(BARRA + ",", ""))
            if not re.fullmatch(r"[0-9]+(?:[.][0-9]+)?", token):
                saltados.append(crudo)
                continue
            # el separador de millares parte el numero en el fuente: se admite
            # cualquier separador entre digito y digito
            partes = [re.escape(ch) for ch in token]
            hueco = r"(?:" + re.escape(BARRA + ",") + r"|\{,\}|,)?"
            cuerpo = hueco.join(partes)
            pat = r"(?<![0-9.])" + cuerpo + r"(?![0-9])"
            mutado, cuantos = re.subn(
                pat, lambda m: muta(m.group(0).replace(BARRA + ",", "")
                                   .replace("{,}", "").replace(",", "")),
                original)
            if not cuantos:
                vivos.append((crudo, "no aparece en el manuscrito"))
                continue
            io.open(tmp, "w", encoding="utf-8", newline="\n").write(mutado)
            r = subprocess.run([sys.executable, "_check_numbers.py"], cwd=HERE,
                               env=dict(os.environ, MANUSCRITO="_mutante.tex"),
                               capture_output=True)
            probados += 1
            if r.returncode == 0:
                vivos.append((crudo, f"sobrevive tras mutar {cuantos} apariciones"))
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

    print(f"{probados} mutantes ejecutados, {len(vivos)} supervivientes, "
          f"{len(saltados)} no numericos omitidos")
    for v, por in vivos:
        print(f"  VIVO  {v}: {por}")
    return len(vivos)


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
