"""Banco de invernadero: la demanda deja de ser arbitraria y pasa a costar euros.

QUE ARREGLA
-----------
En el banco sintetico el peso CONGELADO salia el mas barato. No porque la
modulacion no sirva, sino porque alli alpha_d(t) es una senal que el
experimentador inventa: seguirla no vale nada.

Aqui el coste de evaluacion es UNO SOLO y esta en euros,

    coste_real(t) = precio(t) * energia(t) + lambda * desviacion(t)^2

y el peso deseado se DEDUCE de ese coste:  a_d(t) = 1/(1 + kappa*precio(t)).
Seguir la demanda reduce euros de verdad, asi que el eje de coste vuelve a
decir algo -- que es exactamente lo que le faltaba al banco sintetico.

ESCENARIOS
----------
  A  nominal                 : modelo correcto, meteorologia de diseno
  B  invierno frio           : perturbacion mayor de la prevista (mismatch en la
                               amplitud termica exterior)
  C  fuera de distribucion   : desajuste de modelo (aislamiento degradado, k_v
                               real mayor que el del modelo) + ruido
  D  ola de precios          : el mercado se vuelve mas volatil que en diseno,
                               con un tercer pico y mayor amplitud
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ghi import experiment as gx
from ghi import greenhouse as gh
from ghi import regulators as greg
from ghi import stats as gs
from ghi import terminal as gterm
from ghi.mompc import MOMPC


def build_scenarios(kappa: float = 12.0, lam: float = gh.LAMBDA_CALIBRADA, T: int = 96):
    """Cuatro escenarios. La demanda SIEMPRE sale del precio; lo que cambia es
    el mundo en el que el controlador tiene que vivir."""
    mk_nom = gh.Market()
    wx_nom = gh.Weather()

    def make(name, market, weather, mismatch=None, dist_std=0.0, noise_std=0.05):
        price = lambda t: market.price(t)
        return gx.Scenario(
            name=name,
            demand=lambda t: gh.alpha_d_from_price(price(t), kappa),
            mismatch=mismatch,
            dist_std=dist_std,
            noise_std=noise_std,
            x0=(2.0, 1.0),
            T=T,
            warmup=24,                       # se descarta el primer dia
            disturbance=lambda t, n: weather.disturbance(t, n),
            cost=lambda x, u, t, adc: gh.true_cost(x, u, price(t), lam),
        )

    # mercado mas volatil que el de diseno: tercer pico y mayor amplitud
    mk_vol = gh.Market(base=0.10, amp_m=0.16, amp_t=0.22)
    # invierno mas duro de lo previsto
    wx_frio = gh.Weather(dT_out=10.0, q_sol=2.0)
    # aislamiento degradado: la planta real pierde mas calor que el modelo
    p_real = gh.greenhouse_plant(k_v=0.62)
    p_nom = gh.greenhouse_plant()
    mismatch = p_real.A - p_nom.A

    return {
        "A_nominal":      make("A_nominal", mk_nom, wx_nom),
        "B_invierno":     make("B_invierno", mk_nom, wx_frio),
        "C_OOD_modelo":   make("C_OOD_modelo", mk_nom, wx_nom,
                               mismatch=mismatch, dist_std=0.25),
        "D_precio_volatil": make("D_precio_volatil", mk_vol, wx_nom,
                                 mismatch=mismatch, dist_std=0.25),
    }


def main(n_seeds: int = 10, w0: float = 0.55, zeta: float = 0.6,
         theta: float = 0.5, kappa: float = 12.0, grid_points: int = 21) -> dict:
    t0 = time.time()
    prob = gh.problem_greenhouse()
    prob, term = gterm.design(prob)
    M = MOMPC(prob, term)
    tau, g = greg.match_first_order(w0, zeta, 1.0)

    print("=" * 78)
    print("BANCO DE INVERNADERO CON PRECIO HORARIO")
    print("=" * 78)
    print(f"planta      : {prob.name}, N = {prob.N}, muestreo 1 h")
    ev = np.abs(np.linalg.eigvals(prob.plant.A))
    print(f"              constantes de tiempo: {np.round(-1/np.log(ev),1).tolist()} h")
    print(f"terminal    : Kf = {np.round(term.Kf,4).tolist()}, "
          f"Omega con {term.H.shape[0]} filas")
    print(f"igualacion  : w0={w0} zeta={zeta} -> |H| Nyquist = {g:.6f}, tau = {tau:.4f}")
    print(f"demanda     : a_d(t) = 1/(1 + {kappa}*precio(t))   [DEDUCIDA, no inventada]")
    print(f"metrica     : coste real en euros = precio*energia + desviacion^2")
    print(f"semillas    : {n_seeds}   horizonte de simulacion: 96 h (4 dias)\n")

    regs = {
        "orden-0": lambda: greg.Order0(),
        "orden-1": lambda: greg.Order1(tau),
        "orden-2": lambda: greg.Order2(w0, zeta, theta),
        "congelado": lambda: greg.Frozen(0.4),
    }
    scen = build_scenarios(kappa=kappa)
    bank = gx.run_bank(M, regs, scen, range(n_seeds), grid_points=grid_points)

    print("=" * 78)
    print("CONTRASTES PAREADOS POR SEMILLA  (coste = EUROS; delta>0 => gana el primero)")
    table = gs.contrast_table(
        bank, [("orden-2", "orden-1"), ("orden-2", "orden-0"),
               ("orden-2", "congelado"), ("orden-1", "congelado")],
        metrics=("coste", "err_demanda"))
    gs.print_contrasts(table)

    # ahorro relativo frente al peso congelado, que es la comparacion que
    # entiende un ingeniero de planta
    print("\n" + "=" * 78)
    print("AHORRO FRENTE AL PESO CONGELADO (%)  --  positivo = mas barato")
    print(f"  {'escenario':<20}{'orden-0':>10}{'orden-1':>10}{'orden-2':>10}")
    ahorro = {}
    for sname, regs_ in bank.items():
        base = np.mean([r["coste"] for r in regs_["congelado"]])
        row = {}
        for rn in ("orden-0", "orden-1", "orden-2"):
            c = np.mean([r["coste"] for r in regs_[rn]])
            row[rn] = 100.0 * (base - c) / base
        ahorro[sname] = row
        print(f"  {sname:<20}{row['orden-0']:>9.2f}%{row['orden-1']:>9.2f}%{row['orden-2']:>9.2f}%")

    print(f"\ntiempo: {time.time()-t0:.1f}s")
    return {"bank": bank, "ahorro": ahorro,
            "contrasts": {m: [vars(p) for p in fam] for m, fam in table.items()},
            "config": {"w0": w0, "zeta": zeta, "theta": theta, "tau": tau,
                       "kappa": kappa, "n_seeds": n_seeds}}


if __name__ == "__main__":
    res = main()
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dst = os.path.join(here, "results")
    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(dst, "greenhouse.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    print(f"escrito {os.path.join(dst, 'greenhouse.json')}")
