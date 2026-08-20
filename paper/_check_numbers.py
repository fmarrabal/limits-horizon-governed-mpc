"""Contrasta cada cifra del manuscrito contra su fichero de resultados.

Version para el manuscrito reescrito (encuadre de control): las fuentes son
transport_affine.json, fleet3.json, fleet_decomp.json, alphaN.json,
scale_invariance2.json, reach_law.json, vibration.json, concavidad.json y
transport_viability2.json. Si una cifra deja de coincidir, el script lo dice.
"""
import io
import json
import os
import re

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "code", "results")
# La prueba de mutacion inyecta aqui una copia alterada del manuscrito.
MANUSCRITO = os.environ.get("MANUSCRITO", "main.tex")
tex = io.open(MANUSCRITO, encoding="utf-8").read()
ok = True


def J(n):
    return json.load(open(os.path.join(RES, n)))


def _en_tex(s):
    """Presencia como *token* numerico, no como subcadena.

    Con la busqueda por subcadena un ``8`` casaba dentro de ``0.85`` y de
    ``128``: el verificador daba por comprobada una cifra que el manuscrito
    no contiene. Aqui se exige que a los lados no haya digitos, y por la
    izquierda tampoco un punto decimal (para que ``85`` no case en ``0.85``).
    El separador de millares de LaTeX se normaliza antes.
    """
    limpio = tex.replace("{,}", "").replace(chr(92) + ",", "")
    tok = s.replace("{,}", "").replace(chr(92) + ",", "")
    if not re.fullmatch(r"-?[0-9]+(?:[.][0-9]+)?", tok):
        return tok in limpio          # no es una cifra: comparacion literal
    return re.search(r"(?<![0-9.])" + re.escape(tok) + r"(?![0-9])",
                     limpio) is not None


REGENERADOS = set()


def chk(nombre, valor):
    global ok
    s = str(valor)
    REGENERADOS.add(s)
    hit = _en_tex(s)
    ok = ok and hit
    print(f"  [{'OK ' if hit else 'MAL'}] {nombre}: {s}")


print("Transporte afin (transport_affine.json)")
t = J("transport_affine.json")
for k, lab in (("clarividente", "ceiling"), ("afin-conformado", "shaped"),
               ("retardo-afin", "affine"), ("onda", "wave"),
               ("retardo-lineal", "prop"), ("reactivo", "local")):
    chk(f"coste {lab}", f'{t["tabla"][k]["coste"]:,.0f}'.replace(",", r"\,"))
    chk(f"% techo {lab}", f'{t["tabla"][k]["pct_techo"]:.1f}')
for k, lab, fld in (("retardo-afin_vs_retardo-lineal", "afin vs lineal", "t"),
                    ("retardo-afin_vs_onda", "afin vs onda", "t"),
                    ("retardo-afin_vs_afin-conformado", "conformar", "t"),
                    ("retardo-afin_vs_reactivo", "afin vs local", "t"),
                    ("clarividente_vs_retardo-afin", "techo vs afin", "t")):
    chk(f"t de {lab}", f'{t["contrastes"][k][fld]:+.2f}'.replace("+", "{+}").replace("-", "{-}"))

# el contraste contra la mejor sintonia densa del lineal: recomputado aqui
c2 = J("transport_causal2.json")
a = np.array(t["tabla"]["retardo-afin"]["por_semilla"])
b = np.array(c2["tabla"]["retardo-puro"]["por_semilla"])
d = b - a
tt = float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d))))
chk("t vs mejor sintonia densa del lineal", f"{tt:+.2f}")

print("\nFlota (fleet3.json)")
f3 = J("fleet3.json")
r0 = {r["B"]: r for r in f3["rho_0"]}
r1 = {r["B"]: r for r in f3["rho_1"]}
for B, rr in ((24, r0), (32, r0)):
    chk(f"dif rho0 B={B}", f'{abs(rr[B]["dif_pct"]):.1f}')
chk("dif rho1 B=40", f'{abs(r1[40]["dif_pct"]):.1f}')
chk("dif rho1 B=48", f'{abs(r1[48]["dif_pct"]):.1f}')

print("\nDescomposicion (fleet_decomp.json)")
dd = J("fleet_decomp.json")
rep0 = [r["ventaja_pct"] for r in dd["rho_0"]]
rep1 = [r["ventaja_pct"] for r in dd["rho_1"]]
tmp0 = [r["ventaja_pct"] for r in dd["temporal_rho_0"]]
tmp1 = [r["ventaja_pct"] for r in dd["temporal_rho_1"]]
chk("reparto rho0 minimo", f"{min(rep0):.1f}")
chk("reparto rho0 maximo", f"{max(rep0):.1f}")
chk("temporal rho1 maximo", f"{max(tmp1):.1f}")
# las dos afirmaciones cualitativas del texto, aseveradas y no solo impresas
assert min(rep0) > 4.0, "el reparto deberia pagar con rho=0"
assert abs(max(rep1)) < 1.0 and abs(min(rep1)) < 1.0, "reparto nulo con rho=1"
assert min(tmp0) < -3.0, "el temporal deberia perjudicar con rho=0"
assert max(tmp1) > 6.0, "el temporal deberia pagar con rho=1"
print("  [OK ] complementariedad: reparto solo con rho=0, temporal solo con rho=1")
todos = all(x < 0 for x in tmp0)
print(f"  [{'OK ' if todos else 'MAL'}] vibracion pierde 8/8: {todos}")
ok = ok and todos


print("\nTabla de gobernadores (alphaN.json, escenario fuera de sobre)")
# La tabla promedia SOLO las semillas que no divergieron y da la cuenta de
# divergencias aparte; con la media sobre las diez la tabla no sale, y esa
# regla no estaba escrita en ninguna parte hasta esta revision.
import statistics as _st
_al = J("alphaN.json")


def _vivas(filas):
    v = [f for f in filas if not f["divergio"]]
    cor = [f["corr_asignacion"] for f in v if f["corr_asignacion"] is not None]
    return (round(_st.mean(f["computo"] for f in v)),
            round(_st.mean(f["coste"] for f in v)),
            _st.mean(cor) if cor else None,
            sum(1 for f in filas if f["divergio"]))


for _et, _op in (("leaky integrator", "orden-1-nyq"), ("threshold", "umbral-up4"),
                 ("2nd-order field", "orden-2"), ("memoryless map", "orden-0")):
    _c, _k, _r, _d = _vivas(_al["bank"]["D_OOD_ambos"][_op])
    chk(f"{_et}: computo", _c)
    chk(f"{_et}: coste", f"{_k:,}".replace(",", r"\,"))
    chk(f"{_et}: corr", f"{_r:.3f}")
    chk(f"{_et}: divergencias", _d)
for _N in ("2", "4", "6", "8", "16"):
    _f = _al["frontier"]["D_OOD_ambos"][_N]
    chk(f"frontera N={_N}: computo", round(_f["computo"]))
    chk(f"frontera N={_N}: coste", f'{round(_f["coste"]):,}'.replace(",", r"\,"))


print("\nContrastes emparejados (transport_affine.json, Holm)")
_ta = J("transport_affine.json")["contrastes"]
for _k, _et in (("retardo-afin_vs_retardo-lineal", "afin vs lineal"),
                ("retardo-afin_vs_onda", "afin vs onda"),
                ("retardo-afin_vs_afin-conformado", "conformar"),
                ("retardo-afin_vs_reactivo", "afin vs local"),
                ("clarividente_vs_retardo-afin", "techo vs afin")):
    _c = _ta[_k]
    chk(f"{_et}: delta", f"{abs(_c['delta']):.0f}")
    chk(f"{_et}: MDE %", f"{_c['mde_pct']:.1f}")
    if _c["p_holm"] >= 1e-4:
        chk(f"{_et}: p Holm", f"{_c['p_holm']:.4f}" if _c["p_holm"] < 0.01
            else f"{_c['p_holm']:.2f}")
    else:
        print(f"  [OK ] {_et}: p Holm < 1e-4 ({_c['p_holm']:.1e})")

print("\nInterpolacion de la frontera al computo del integrador con fuga")
_fr = J("alphaN.json")["frontier"]["D_OOD_ambos"]
_pts = sorted((v["computo"], v["coste"]) for v in _fr.values())
_x = 1775.0
for (_x0, _y0), (_x1, _y1) in zip(_pts, _pts[1:]):
    if _x0 <= _x <= _x1:
        _y = _y0 + (_y1 - _y0) * (_x - _x0) / (_x1 - _x0)
        chk("frontera interpolada en 1775", f"{round(_y, -1):,.0f}".replace(",", r"\,"))
        break


print(chr(10) + "Techo clarividente: intervalo bootstrap (ceiling_ci.json)")
_cc = J("ceiling_ci.json")["mecanismos"]["retardo-afin"]
chk("fraccion de techo", f"{_cc['pct_techo']:.1f}")
chk("fraccion de techo redondeada", f"{_cc['pct_techo']:.0f}")
chk("IC95 inferior", f"{_cc['ic95'][0]:.1f}")
chk("IC95 superior", f"{_cc['ic95'][1]:.1f}")
chk("P(causal bajo el techo)", f"{100 * _cc['frac_por_encima_del_techo']:.1f}")
assert _cc["ic95"][1] > 100.0, "el intervalo deberia contener el techo"


print(chr(10) + "Ley de alcance (reach_law.json)")
_rl = J("reach_law.json")
_falla = [f for f in _rl["filas"] if f["alcance"] != f["prediccion"]]
assert len(_falla) == 1, "se esperaba exactamente un fallo de la ley"
chk("paso donde falla la ley", _falla[0]["paso"])
chk("rho interior", f"{_rl['rho_interior']:.3f}")
_disputada = _rl["rho_interior"] ** _falla[0]["prediccion"]
chk("amplitud en el salto disputado", f"{_disputada:.4f}")
assert _rl["ley_valida"] is False, "el archivo dice que la ley SI valida"
print("  [OK ] veredicto pre-registrado archivado: ley no validada")


print(chr(10) + "Bordes de rejilla (transport_affine.json)")
_bd = J("transport_affine.json")["bordes"]
_sn = J("transport_affine.json")["sintonia"]
_en_borde = {k: v for k, v in _bd.items() if v}
# la regla pre-registrada: un optimo en el borde se rechaza salvo que el borde
# sea un limite fisico. Aqui se comprueba, no se promete.
for _k, _campos in _en_borde.items():
    for _c in _campos:
        _v = _sn[_k][_c]
        assert _v in (0.0, 1.0), (
            f"{_k}.{_c} = {_v} esta en el borde y NO es un limite fisico")
        print(f"  [OK ] {_k}.{_c} = {_v:g}: borde del simplex, limite fisico")
assert set(_en_borde) == {"lineal"}, (
    f"el manuscrito declara una sola excepcion; hay {sorted(_en_borde)}")
print(f"  [OK ] una sola excepcion declarada, y es del rival: {sorted(_en_borde)}")

c = J("concavidad.json")
frac = c["no_convexidad"]["no_convexos"] / c["no_convexidad"]["niveles"]
quinto = abs(frac - 0.2) < 0.03 and "one fifth" in tex
# (los lotes de curvatura se verifican ahora en el checker del Comment)

print(f"  [{'OK ' if quinto else 'MAL'}] 'one fifth' respaldado ({100*frac:.1f}%)")
ok = ok and quinto


# --- trazabilidad: ninguna cifra del cuerpo sin origen ----------------------
# El texto afirmaba que un chequeo verificaba "cada cifra". No era cierto:
# comprobaba 26 de las 151 del cuerpo. Este pase cierra el hueco: cada
# literal informativo tiene que (a) coincidir con un valor hoja de algun
# fichero de resultados, (b) haber pasado por chk(), o (c) estar declarado
# abajo como constante que no viene de ningun experimento.
import glob as _glob
import sys as _sys
_sys.path.insert(0, os.path.join(ROOT, "code", "python"))
from ghi.anclaje import normaliza as _norm, variantes as _var

DECLARADOS = {
    "080002": "codigo postal de la afiliacion",
    "04120": "codigo postal de la afiliacion",
    "0.0": "diferencia nula, enunciada como exactitud",
    "1.45": "longitud de columna de LaTeX",
    "1.85": "longitud de columna de LaTeX",
    "3.8": "longitud de columna de LaTeX",
    "3.6": "longitud de tabcolsep de LaTeX",
    "0.85": "ejemplo ilustrativo del anclaje por subcadena",
    "160": "cuenta de literales, comprobada abajo contra el propio pase",
    "150": "cuenta de regenerados, comprobada abajo",
}


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


_formas = set()
for _f in _glob.glob(os.path.join(RES, "*.json")):
    try:
        _d = json.load(open(_f))
    except Exception:
        continue
    for _v in _hojas(_d):
        _formas |= _var(_v)
for _v in REGENERADOS:
    _t = _v.replace("{,}", "").replace(chr(92) + ",", "").lstrip("+-")
    _formas.add(_t)
    try:
        _formas |= _var(float(_t))
    except ValueError:
        pass

_c = _norm(tex).split(chr(92) + "begin{document}")[-1]
_c = _c.split(chr(92) + "bibliography")[0]
for _cmd in ("cite", "label", "ref", "eqref", "url", "href", "includegraphics"):
    _c = re.sub(re.escape(chr(92) + _cmd) + r"[*]?(\[[^]]*\])?\{[^}]*\}", " ", _c)
_lits = set(re.findall(r"(?<![0-9.])[0-9]+(?:[.][0-9]+)?(?![0-9])", _c))
_inform = {x for x in _lits if ("." in x) or len(x) >= 3}
_huerf = sorted(_inform - _formas - set(DECLARADOS))
print(f"\nTrazabilidad: {len(_inform)} literales informativos en el cuerpo, "
      f"{len(_inform) - len(_huerf) - len(set(DECLARADOS) & _inform)} regenerados "
      f"del archivo, {len(set(DECLARADOS) & _inform)} declarados, "
      f"{len(_huerf)} sin origen")
for _h in _huerf:
    print(f"  [MAL] sin origen archivado: {_h}")
ok = ok and not _huerf

print("\n" + ("TODAS LAS CIFRAS DEL TEXTO COINCIDEN" if ok else
             "HAY CIFRAS QUE NO COINCIDEN -- revisar las marcadas MAL"))

_dec = len(set(DECLARADOS) & _inform)
_m = re.search(r"of the [$]([0-9]+)[$] numeric literals in the body, "
               r"[$]([0-9]+)[$] are", _norm(tex))
if _m is None:
    print("  [MAL] la seccion de reproducibilidad no declara sus cuentas")
    ok = False
else:
    _dicho = (int(_m.group(1)), int(_m.group(2)))
    _real = (len(_inform), len(_inform) - len(_huerf) - _dec)
    _bien = _dicho == _real
    marca = "OK " if _bien else "MAL"
    print(f"  [{marca}] cuentas declaradas {_dicho} contra las medidas {_real}")
    ok = ok and _bien

if os.environ.get("LISTAR"):
    io.open(os.environ["LISTAR"], "w", encoding="utf-8").write(
        json.dumps(sorted(REGENERADOS)))
import sys as _s
_s.exit(0 if ok else 1)
