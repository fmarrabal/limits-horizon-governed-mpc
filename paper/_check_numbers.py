"""Contrasta cada cifra del manuscrito contra su fichero de resultados.

Version para el manuscrito reescrito (encuadre de control): las fuentes son
transport_affine.json, fleet3.json, fleet_decomp.json, alphaN.json,
scale_invariance2.json, reach_law.json, vibration.json, concavidad.json y
transport_viability2.json. Si una cifra deja de coincidir, el script lo dice.
"""
import io
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "code", "results")
tex = io.open("main.tex", encoding="utf-8").read()
ok = True


def J(n):
    return json.load(open(os.path.join(RES, n)))


def chk(nombre, valor):
    global ok
    s = str(valor)
    hit = s in tex
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
mej0 = [r["mejora_pct"] for r in dd["rho_0"]]
chk("mejor reparto rho0", f"{max(mej0):.1f}")
chk("reparto minimo rho0", f"{min(mej0):.1f}")

print("\nHorizonte, lazo unico (alphaN.json)")
al = J("alphaN.json")
fr = al["frontier"]["D_OOD_ambos"]
chk("frontera N=2 divergencias", fr["2"]["divergencias"])
chk("frontera N=16 coste", f'{fr["16"]["coste"]:.0f}')
bank = al["bank"]["D_OOD_ambos"]
rows = [r for r in bank["orden-1-nyq"] if not r.get("divergio")]
chk("computo del gobernador", f'{np.mean([r["computo"] for r in rows]):.0f}')
chk("coste del gobernador", f'{np.mean([r["coste"] for r in rows]):.0f}')

print("\nInvariancia (scale_invariance2.json)")
s2 = J("scale_invariance2.json")
chk("alpha* interior", f'{s2["P1"]["alphas"][0]:.2f}')
chk("N* interior", s2["P1"]["Ns"][0])
chk("dispersion peso", f'{s2["P1"]["dispersion_coste_peso"]:.1e}'
    .replace("e-14", r"\times10^{-14}"))

print("\nLey de alcance (reach_law.json)")
rl = J("reach_law.json")
chk("rho interior", f'{rl["rho_interior"]:.3f}')

print("\nViabilidad (log) y vibracion")
via = io.open(os.path.join(RES, "transport_viability2b.log"), encoding="utf-8").read()
imp = "+4.0" in tex and "+4.03" in via
print(f"  [{'OK ' if imp else 'MAL'}] viabilidad +4.0% presente y respaldada")
ok = ok and imp
v = J("vibration.json")["modulacion"]
todos = all(x["mejora_pct"] < 0 for x in v)
print(f"  [{'OK ' if todos else 'MAL'}] vibracion pierde 8/8: {todos}")
ok = ok and todos

c = J("concavidad.json")
frac = c["no_convexidad"]["no_convexos"] / c["no_convexidad"]["niveles"]
quinto = abs(frac - 0.2) < 0.03 and "one fifth" in tex
print(f"  [{'OK ' if quinto else 'MAL'}] 'one fifth' respaldado ({100*frac:.1f}%)")
ok = ok and quinto

print("\n" + ("TODAS LAS CIFRAS DEL TEXTO COINCIDEN" if ok else
             "HAY CIFRAS QUE NO COINCIDEN -- revisar las marcadas MAL"))
