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
import sys as _sys0
_sys0.path.insert(0, os.path.join(ROOT, "code", "python"))
from ghi.anclaje import cuerpo as _cuerpo
# lo que se comprueba es el CUERPO del articulo: sin preambulo, sin
# bibliografia y sin los argumentos de \cite, \ref y \label, donde una cifra
# no es un resultado. Una cifra pegada tras \end{document} no cuenta.
CUERPO = _cuerpo(MANUSCRITO)
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
    limpio = CUERPO
    tok = s.replace("{,}", "").replace(chr(92) + ",", "")
    if not re.fullmatch(r"-?[0-9]+(?:[.][0-9]+)?", tok):
        # no es una cifra: comparacion literal, con los espacios de alineacion
        # de las tablas colapsados (una fila "$610$  & $29653$" es la misma)
        _col = lambda t: re.sub(r"[ 	]+", " ", t)
        return _col(tok) in _col(limpio)
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
    chk(f"s.e.m. {lab}", f'{t["tabla"][k]["sem"]:.0f}')
    chk(f"vs local {lab}", f'{t["tabla"][k]["vs_local_pct"]:.1f}')
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


def _celda(r, negrita):
    d = f"{r['dif_pct']:+.1f}\\%"
    return f"$\\mathbf{{{d}}}$" if negrita else f"${d}$"


for rho, rr in (("0", r0), ("1", r1)):
    rows = [rr[B] for B in sorted(rr) if B != 16]
    win2 = sorted([r["dif_pct"] for r in rows if r["dif_pct"] < 0])[:2]
    vistos = set()
    for r in rows:
        clave = (round(r["dif_pct"], 3), round(r["computo"], 1))
        if clave in vistos:
            continue                      # filas fundidas en una sola ($\ge$)
        vistos.add(clave)
        # la fila entera, con signo: una cadena que no casa con ninguna otra
        chk(f"fila rho{rho} B={r['B']}",
            f"${r['computo']:.1f}$ & ${round(r['coste'])}$ & "
            f"${round(r['frontera'])}$ & " + _celda(r, r["dif_pct"] in win2))
        chk(f"dif rho{rho} B={r['B']}", f"{r['dif_pct']:.1f}")

g0 = [r for B, r in r0.items() if B != 16 and r["dif_pct"] < 0 and r["p_holm"] < 0.05]
g1 = [r for B, r in r1.items() if B != 16 and r["dif_pct"] < 0 and r["p_holm"] < 0.05]
chk("hasta (rho0)", f"{-min(r['dif_pct'] for B, r in r0.items() if B != 16):.1f}")
chk("hasta (rho1)", f"{-min(r['dif_pct'] for B, r in r1.items() if B != 16):.1f}")
# con contexto: la misma cifra esta en la tabla, y mutar la copia de la prosa
# tiene que ponerse rojo por si sola
chk("frase: hasta rho0",
    f"by up to ${-min(r['dif_pct'] for B, r in r0.items() if B != 16):.1f}\\%$ (Holm-adjusted")
chk("frase: hasta rho1",
    f"adjustment, by up to ${-min(r['dif_pct'] for B, r in r1.items() if B != 16):.1f}\\%$, and")
chk("frase: laxos",
    "favour it by\n$" + f"{-max(r1[96]['dif_pct'], r1[128]['dif_pct']):.1f}" + "\\%$ at raw")
chk("frase: se asienta",
    f"settles at ${-r0[128]['dif_pct']:.1f}\\%$ once it stops binding")
chk("gana en todos los que atan (rho0)", "at every\nbudget that binds" if len(g0) == 7 else "NO-7")
chk("cuantos de ocho tras Holm (rho1)", f"{len(g1)} of eight")
chk("se asienta al dejar de atar", f"{-r0[128]['dif_pct']:.1f}")
chk("B=16 coincide", f"{abs(r0[16]['dif_pct']):.2f}")
chk("laxos rho1", f"{-max(r1[96]['dif_pct'], r1[128]['dif_pct']):.1f}")
chk("laxos rho1 p crudo", f"{max(r1[96]['p'], r1[128]['p']):.2f}")
chk("laxos rho1 p Holm", f"{min(r1[96]['p_holm'], r1[128]['p_holm']):.3f}")
assert all(r["p_holm"] < 1e-4 for r in g0), "el texto dice Holm p<1e-4 en rho=0"
assert all(r1[B]["p_holm"] >= 0.05 for B in (96, 128)), "los laxos no deberian pasar Holm"
chk("tau de la flota", f"{f3['sintonia']['tau']:.0f}")
chk("meseta", f"{abs(f3['sintonia']['meseta_disp_rel']) * 100:.2f}")
assert f3["sintonia"]["a_ref"] <= 1e-3 and f3["sintonia"]["tau_interior"], "sintonia"
fr = {r["N"]: r["coste"] for r in f3["frontera_rho_0"]}
chk("J(2)/J(16)", f"{fr[2] / fr[16]:.2f}")

print("\nDescomposicion (fleet_decomp.json)")
dd = J("fleet_decomp.json")
rep0 = [r["ventaja_pct"] for r in dd["rho_0"]]
rep1 = [r["ventaja_pct"] for r in dd["rho_1"]]
tmp0 = [r["ventaja_pct"] for r in dd["temporal_rho_0"]]
tmp1 = [r["ventaja_pct"] for r in dd["temporal_rho_1"]]
rep1_sig = sum(1 for r in dd["rho_1"] if r["p_holm"] < 0.05 and r["ventaja_pct"] > 0)
tmp1_sig = [r["ventaja_pct"] for r in dd["temporal_rho_1"] if r["p_holm"] < 0.05 and r["ventaja_pct"] > 0]
chk("reparto rho0 minimo", f"{min(rep0):.1f}")
chk("reparto rho0 maximo", f"{max(rep0):.1f}")
chk("reparto rho1 minimo", f"{min(rep1):.1f}")
chk("reparto rho1 maximo", f"{max(rep1):.1f}")
chk("reparto rho1 significativo", f"{rep1_sig} of six budgets")
chk("temporal rho1 minimo sig", f"{min(tmp1_sig):.1f}")
chk("temporal rho1 maximo", f"{max(tmp1):.1f}")
chk("temporal rho1 en cuantos", f"{len(tmp1_sig)} middle budgets")
chk("temporal rho0 minimo", f"{min(tmp0):.1f}")
assert all(r["modo"] == "igual_computo_gastado" and r["n"] == 8 for r in dd["rho_0"]), \
    "el texto dice: todos a computo gastado igual, ocho semillas"
# las afirmaciones cualitativas del texto, aseveradas y no solo impresas
assert min(rep0) > 5.0, "el reparto deberia pagar con rho=0"
assert max(rep1) < 2.0, "reparto casi nulo con rho=1"
assert min(tmp0) < -3.0, "el temporal deberia perjudicar con rho=0"
assert max(tmp1) > 5.0, "el temporal deberia pagar con rho=1"
print("  [OK ] complementariedad: reparto solo con rho=0, temporal solo con rho=1")
# vibracion 8/8: de vibration.json (antes esta linea miraba, mal, tmp0)
vb = J("vibration.json")["modulacion"]
todos = len(vb) == 8 and all(c["mejora_pct"] < 0 for c in vb)
print(f"  [{'OK ' if todos else 'MAL'}] vibracion pierde 8/8 ({len(vb)} configuraciones): {todos}")
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
    if _et == "leaky integrator":
        chk(f"fila {_et}", f"$\\mathbf{{{_c}}}$ & $\\mathbf{{{_k}}}$ & ${_r:.3f}$ & ${_d}$")
    else:
        chk(f"fila {_et}", f"${_c}$ & ${_k}$ & ${_r:.3f}$ & ${_d}$")
for _N in ("2", "4", "6", "8", "16"):
    _f = _al["frontier"]["D_OOD_ambos"][_N]
    chk(f"frontera N={_N}: computo", round(_f["computo"]))
    chk(f"frontera N={_N}: coste", f'{round(_f["coste"]):,}'.replace(",", r"\,"))
    chk(f"fila fija N={_N}", f"${round(_f['computo'])}$ & ${round(_f['coste'])}$ & ---")


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
_hu = J("ceiling_ci.json")["hueco_afin_vs_referencia"]
chk("hueco IC inferior", f"{_hu['ic95_pct'][0]:.1f}")
chk("hueco IC superior", f"{_hu['ic95_pct'][1]:.1f}")
chk("semillas causal mas barato", f"${_hu['semillas_causal_mas_barato']}$ of\n$30$ seeds")
chk("t pareado crudo", f"{_hu['t_pareado_crudo']:+.2f}")
chk("p crudo bilateral", f"{_hu['p_crudo_bilateral']:.2f}")
_lo, _hi = _hu["ic95_pct"]
chk("frase: IC en el pie de figura",
    f"within ${_lo:+.1f}\\%$ to ${_hi:+.1f}\\%$ ($95\\%$" + "\ninterval) of the acausal reference, and neither")
chk("frase: IC en resultados",
    f"is within ${_lo:+.1f}\\%$ to ${_hi:+.1f}\\%$ of the acausal reference ($95\\%$" + "\ninterval; it is cheaper")
chk("frase: IC en conclusiones",
    f"within ${_lo:+.1f}\\%$ to ${_hi:+.1f}\\%$ of the acausal" + "\nclairvoyant reference, worth")
chk("frase: Holm p en prosa",
    f"Holm $p={_ta['clarividente_vs_retardo-afin']['p_holm']:.2f}$")
# las dos filas de la tabla de contrastes cuyo p y MDE se repiten en la prosa
for _k, _et in (("retardo-afin_vs_afin-conformado", "conformar"), ("clarividente_vs_retardo-afin", "techo vs afin")):
    _c = _ta[_k]
    chk(f"fila de contraste: {_et}", f"$p{{=}}{_c['p_holm']:.2f}$,\ ${_c['mde_pct']:.1f}\%$")
chk("frase: MDE del conformado en prosa",
    f"against an MDE of ${_ta['retardo-afin_vs_afin-conformado']['mde_pct']:.1f}\\%$")
assert _hu["ic95_pct"][0] < 0 < _hu["ic95_pct"][1], "el intervalo del hueco deberia contener 0"

print(chr(10) + "Amortiguamiento critico (alphaN.json) y viabilidad P3")
_al = J("alphaN.json")
_bk = _al["bank"]["B_OOD_kick"]
import statistics as _st2
for _op in ("orden-2", "orden-2-z1"):
    _cor = [f["corr_asignacion"] for f in _bk[_op] if f["corr_asignacion"] is not None]
    chk(f"corr {_op} (gran patada)", f"{_st2.mean(_cor):+.2f}")
_ct = [x for x in _al["contrasts"]["corr_asignacion"]
       if x["a_name"].startswith("B_OOD_kick:orden-2-z1") and x["b_name"] == "orden-2"]
assert len(_ct) == 1, f"contraste de amortiguamiento no encontrado ({len(_ct)})"
chk("t amortiguamiento", f"{abs(_ct[0]['t']):+.2f}")
_p3 = J("transport_viability2.json")["P3"]
chk("P3 t", f"{_p3['t']:+.2f}")
chk("P3 ganancia %", f"{100 * (_p3['media_lead0'] - _p3['media_leadL']) / _p3['media_lead0']:.1f}")
_pal = {9: "nine", 10: "ten", 8: "eight"}.get(_p3["semillas_a_favor"], str(_p3["semillas_a_favor"]))
chk("P3 semillas", f"{_pal} of ten seeds")
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
_c = _c.split(chr(92) + "section*{Reproducibility}")[0]   # sus cuentas no se auditan a si mismas
for _cmd in ("cite", "label", "ref", "eqref", "url", "href", "includegraphics"):
    _c = re.sub(re.escape(chr(92) + _cmd) + r"[*]?(\[[^]]*\])?\{[^}]*\}", " ", _c)
_lits = set(re.findall(r"(?<![0-9.])[0-9]+(?:[.][0-9]+)?(?![0-9])", _c))
_inform = {x for x in _lits if ("." in x) or len(x) >= 3}
_huerf = sorted(_inform - _formas - set(DECLARADOS))
print(f"\nTrazabilidad: {len(_inform)} literales informativos en el cuerpo, "
      f"{len(_inform) - len(_huerf) - len(set(DECLARADOS) & _inform)} con origen archivado "
      f"(regenerados o existentes), {len(set(DECLARADOS) & _inform)} declarados, "
      f"{len(_huerf)} sin origen")
for _h in _huerf:
    print(f"  [MAL] sin origen archivado: {_h}")
ok = ok and not _huerf

print("\n" + ("TODAS LAS CIFRAS DEL TEXTO COINCIDEN" if ok else
             "HAY CIFRAS QUE NO COINCIDEN -- revisar las marcadas MAL"))

_dec = len(set(DECLARADOS) & _inform)
_numreg = set()
for _v in REGENERADOS:
    _t = (_v.replace("{+}", "").replace("{-}", "").lstrip("+-")
          .replace("{,}", "").replace(chr(92) + ",", ""))
    if re.fullmatch(r"[0-9]+(?:[.][0-9]+)?", _t):
        _numreg.add(_t)
_K = len(_inform & _numreg)                        # regenerados por chk() y presentes
_E = len(_inform) - len(_huerf) - _dec - _K        # solo existencia en algun JSON
_m = re.search(r"regenerates [$]([0-9]+)[$] reported quantities.*?"
               r"Of the [$]([0-9]+)[$] numeric literals.*?those [$]([0-9]+)[$] are regenerated.*?"
               r"[$]([0-9]+)[$] more coincide.*?and [$]([0-9]+)[$] are declared",
               _norm(tex), re.S)
if _m is None:
    print("  [MAL] la seccion de reproducibilidad no declara sus cuentas")
    ok = False
else:
    _dicho = tuple(int(g) for g in _m.groups())
    _real = (_K, len(_inform), _K, _E, _dec)
    _bien = _dicho == _real
    marca = "OK " if _bien else "MAL"
    print(f"  [{marca}] cuentas declaradas (K,N,K,E,D)={_dicho} contra las medidas {_real}")
    ok = ok and _bien

if os.environ.get("LISTAR"):
    io.open(os.environ["LISTAR"], "w", encoding="utf-8").write(
        json.dumps(sorted(REGENERADOS)))
import sys as _s
_s.exit(0 if ok else 1)
