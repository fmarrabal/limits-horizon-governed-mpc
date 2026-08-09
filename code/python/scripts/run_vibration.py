"""La arena predicha por la regla refinada: demanda oscilatoria de banda estrecha
en un lazo real con sensores ruidosos. Y por que la prediccion falla.

QUE SE PONE A PRUEBA
--------------------
La regla refinada, formulada tras tres arenas, predice que el campo inercial
gana con demanda de BANDA ESTRECHA (oscilatoria) enterrada en ruido. Esta arena
se construyo *ex profeso* para que ganara: estructura resonante poco amortiguada
(zeta = 0.04), perturbacion de batido (dos armonicos proximos -> envolvente que
OSCILA, no de Rayleigh), coste real de fatiga superlineal |u|^4 que hace que el
compromiso optimo dependa de la amplitud de operacion, y un alpha_d DEDUCIDO de
ese coste (no inventado):

    alpha_d(t) = 1 / (1 + kappa E(t)^2),      kappa = 6 r / q * gain^2

Con el NULO DE MECANISMO: misma portadora, misma media y misma varianza de
envolvente, cambiando SOLO su espectro (narrow = tono; broad = ruido filtrado).

EL RESULTADO: LA PREDICCION FALLA, Y NO POR SINTONIA
-----------------------------------------------------
  E1  seguir alpha_d pierde frente al MEJOR PESO CONSTANTE en toda configuracion
  E2  el nulo de mecanismo no separa narrow de broad -> la banda no era el motor
  E3  el horizonte optimo N* NO depende del nivel de envolvente, y los costes
      escalan exactamente con el cuadrado de la amplitud

E3 es la clave y remite a `run_scale_invariance.py`: en regimen lineal el
argmin sobre alpha y sobre N es INVARIANTE a la amplitud. Una envolvente que
module amplitudes -- la forma canonica de fabricar demanda de banda estrecha --
no genera ningun meta-parametro optimo variable en el tiempo. No hay nada que
modular, y por tanto no hay ventaja posible para NINGUN gobernador.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ghi import terminal as gterm
from ghi import vibration as vb
from ghi.mompc import MOMPC, solve
from ghi.plant import Problem
from ghi.suboptimality import HorizonMPC


T_RUN, T_WARM = 600, 100


def run(M, plant, d, alpha_seq, sigma=0.0, seed=0, q=1.0, r=1.0):
    """Lazo cerrado evaluado SIEMPRE con el coste real de fatiga.

    `alpha_seq` es un escalar (peso constante) o una serie temporal (modulado).
    `sigma` es el ruido de medida del sensor que alimenta al MPC y a la
    envolvente: el gobernador NUNCA ve la envolvente limpia.
    """
    rng = np.random.default_rng(seed + 7717)
    x = np.zeros(2)
    tot = 0.0
    a_arr = np.full(T_RUN, float(alpha_seq)) if np.isscalar(alpha_seq) else np.asarray(alpha_seq, float)
    for t in range(T_RUN):
        xm = x + sigma * rng.standard_normal(2)          # medida ruidosa
        s = solve(M, xm, np.array([1.0 - a_arr[t], a_arr[t]]))
        if s is None:
            return np.inf
        u = float(s.U[0, 0])
        if t >= T_WARM:
            tot += vb.true_cost(x, u, q=q, r=r)
        w = np.zeros(2)
        w[1] = d[t]
        x = plant.A @ x + plant.B.ravel() * u + w
        if np.max(np.abs(x)) > 1e6:
            return np.inf
    return tot


def main() -> dict:
    out = {}
    prob = vb.problem_vibration(N=10)
    prob, term = gterm.design(prob)
    plant = prob.plant
    M = MOMPC(prob, term)

    print("=" * 78)
    print("E0  CALIBRACION: el mejor peso CONSTANTE debe caer en el INTERIOR")
    print("=" * 78)
    print("  (si cayera en un extremo, el problema no tendria compromiso que arbitrar)")
    exc = vb.Excitation(E0=2.6, depth=0.75, band="narrow")
    d, E = exc.signal(T_RUN)
    grid = np.round(np.linspace(0.05, 0.95, 19), 3)
    cs = [run(M, plant, d, a) for a in grid]
    j = int(np.argmin(cs))
    a_const = float(grid[j])
    print(f"\n  {'alpha':>7}{'coste real':>15}")
    for a, c in zip(grid, cs):
        mark = "  <== MEJOR CONSTANTE" if a == a_const else ""
        if a in (0.05, 0.15, 0.25, 0.35, 0.5, 0.75, 0.95) or a == a_const:
            print(f"  {a:>7.2f}{c:>15.1f}{mark}")
    interior = 0.05 < a_const < 0.95
    print(f"\n  alpha* constante = {a_const:.2f}  (interior: {interior})")
    out["calibracion"] = {"alpha_const": a_const, "interior": bool(interior),
                          "coste": float(cs[j])}

    print("\n" + "=" * 78)
    print("E1+E2  SEGUIR alpha_d FRENTE AL MEJOR CONSTANTE, CON EL NULO DE MECANISMO")
    print("=" * 78)
    print("  mejora > 0 significa que MODULAR gana. La regla refinada predice que")
    print("  narrow gana y broad no. Si ambos pierden, la banda no era el mecanismo.")
    print(f"\n  {'banda':>8}{'sigma':>7}{'gain':>7}{'constante':>12}{'modulado':>12}{'mejora %':>11}")
    filas = []
    for band in ("narrow", "broad"):
        for sigma in (0.0, 0.20):
            for gain in (0.6, 1.0):
                e = vb.Excitation(E0=2.6, depth=0.75, band=band, seed=1)
                dd, EE = e.signal(T_RUN)
                kappa = vb.kappa_from_costs(gain=gain)
                # el gobernador mide la envolvente con el MISMO ruido del sensor
                rng = np.random.default_rng(4242)
                EE_m = np.maximum(EE + sigma * rng.standard_normal(T_RUN), 0.05)
                a_mod = np.clip(vb.alpha_d_from_envelope(EE_m, kappa), 0.05, 0.95)
                # el constante se re-optimiza PARA CADA condicion (baseline justa)
                cc = [run(M, plant, dd, a, sigma=sigma) for a in grid]
                c_const = float(min(cc))
                c_mod = run(M, plant, dd, a_mod, sigma=sigma)
                mej = 100.0 * (c_const - c_mod) / c_const
                print(f"  {band:>8}{sigma:>7.2f}{gain:>7.1f}{c_const:>12.1f}"
                      f"{c_mod:>12.1f}{mej:>+11.2f}")
                filas.append({"band": band, "sigma": sigma, "gain": gain,
                              "const": c_const, "mod": float(c_mod),
                              "mejora_pct": float(mej)})
    out["modulacion"] = filas
    gana = [f for f in filas if f["mejora_pct"] > 0]
    n_narrow = sum(1 for f in gana if f["band"] == "narrow")
    print(f"\n  configuraciones en que MODULAR gana: {len(gana)}/{len(filas)} "
          f"(de banda estrecha: {n_narrow})")
    print("  => la prediccion de la regla refinada NO se cumple, y el nulo de")
    print("     mecanismo no separa: la banda estrecha no era el motor.")

    print("\n" + "=" * 78)
    print("E3  EL HORIZONTE OPTIMO FRENTE AL NIVEL DE ENVOLVENTE")
    print("=" * 78)
    print("  si N* dependiera del nivel, habria algo que un gobernador podria seguir")
    hm = HorizonMPC(plant, Q=np.diag([1.0, 0.05]), R=np.array([[1.0]]),
                    N_min=2, N_max=24)
    Ns = [2, 4, 8, 12, 16, 24]
    print(f"\n  {'E0':>6}{'N*':>5}{'coste*':>13}{'coste*/E0^2':>14}")
    Ns_star, cn = [], []
    for E0 in (1.0, 1.5, 2.6, 4.0):
        dd, _ = vb.Excitation(E0=E0, depth=0.75, band="narrow", seed=1).signal(T_RUN)
        cs2 = []
        for N in Ns:
            x = np.zeros(2); tot = 0.0
            for t in range(T_RUN):
                u = hm.solve(x, N).u0
                if t >= T_WARM:
                    tot += float(x[0] ** 2 + 0.05 * x[1] ** 2 + u * u)
                w = np.zeros(2); w[1] = dd[t]
                x = plant.A @ x + plant.B.ravel() * u + w
            cs2.append(tot)
        j = int(np.argmin(cs2))
        Ns_star.append(Ns[j]); cn.append(cs2[j] / E0 ** 2)
        print(f"  {E0:>6.1f}{Ns[j]:>5d}{cs2[j]:>13.1f}{cs2[j]/E0**2:>14.3f}")
    disp = float(np.std(cn) / max(np.mean(cn), 1e-12))
    print(f"\n  N* invariante al nivel: {len(set(Ns_star)) == 1}  (valores {sorted(set(Ns_star))})")
    print(f"  coste*/E0^2 constante : dispersion relativa {disp:.2e}")
    out["horizonte"] = {"N_star": Ns_star, "invariante": bool(len(set(Ns_star)) == 1),
                        "dispersion": disp}

    print("\n" + "=" * 78)
    print("VEREDICTO")
    print("=" * 78)
    print("""  La arena se construyo para que el campo ganara y NO gana, pero el motivo
  no es de sintonia: es que en regimen lineal el peso optimo y el horizonte
  optimo NO DEPENDEN de la amplitud de la perturbacion (ver A15 y
  run_scale_invariance.py). Una demanda fabricada modulando AMPLITUDES no
  tiene ningun optimo variable en el tiempo que seguir.

  Para que exista algo que modular hay que cruzar una NO LINEALIDAD; y al
  cruzarla la demanda se vuelve EVENTUAL -- pulsos, banda ANCHA -- que es
  justo el regimen donde alpha_N mostro que gana el integrador con fugas.""")
    return out


if __name__ == "__main__":
    res = main()
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dst = os.path.join(here, "results")
    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(dst, "vibration.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    print(f"\nescrito {os.path.join(dst, 'vibration.json')}")
