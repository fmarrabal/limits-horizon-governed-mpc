"""El campo del grafo DENTRO del lazo: ¿transporta el compromiso aguas abajo?

MONTAJE
-------
M zonas identicas, cada una con su planta, su MPC y su filtro de seguridad. NO
hay acoplamiento fisico ni comunicacion explicita entre ellas: lo unico que las
une es el campo homeostatico sobre el grafo. La demanda llega SOLO A LA ZONA 1.

TRES CONFIGURACIONES
--------------------
  onda         rama de 2o orden con rigidez espacial c^2 L (transporte)
  difusion     rama de 1er orden con la MISMA K (mismo estacionario)
  independiente c = D = 0: M campos escalares sin acoplamiento. Control nulo.

La comparacion onda/difusion es justa POR CONSTRUCCION: las dos ramas comparten
la misma K, luego el mismo perfil estacionario. La unica diferencia es dinamica.

QUE SE MIDE
-----------
  amplitud relativa : cuanta senal de la demanda sobrevive en la zona i
  retraso           : desfase que maximiza la correlacion con la demanda
  coste aguas abajo : lo unico que le importa a un ingeniero
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
from scipy import stats as sps

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ghi import distributed as gd
from ghi import plant as gplant
from ghi import terminal as gterm
from ghi.field import FieldParams
from ghi.mompc import MOMPC


def main(M: int = 5, n_seeds: int = 8, T: int = 80, period: float = 20.0,
         w0: float = 0.9, zeta: float = 0.5, c: float = 0.75, D: float = 0.1,
         grid_points: int = 21, forcing: str = "source") -> dict:
    t0 = time.time()
    prob = gplant.problem_conflict()
    prob, term = gterm.design(prob)
    mpc = MOMPC(prob, term)

    p = FieldParams(w0=w0, zeta=zeta, c=c, D=D, b=0.0, beta=0.0, gamma=1.0)
    demand = lambda t: 0.5 + 0.35 * np.sin(2 * np.pi * t / period)

    print("=" * 78)
    print("EL CAMPO DEL GRAFO DENTRO DEL LAZO")
    print("=" * 78)
    print(f"zonas: {M} (cadena dirigida)   T = {T}   periodo de la demanda = {period}")
    print(f"campo: w0={w0} zeta={zeta} c={c} D={D}  forzamiento={forcing}")
    print(f"la demanda entra SOLO en la zona 1; las demas estan en reposo\n")

    # forcing="source" es el forzamiento CORREGIDO. El original ("target")
    # tenia ganancia cruzada NULA en continua (ver A18): la parte oscilante de
    # la demanda si viajaba, pero atenuada 2.7x frente al operador correcto, de
    # modo que la primera tanda juzgo al campo con el canal debilitado.
    configs = {
        "onda": lambda: gd.GraphField(M, p, branch="wave", forcing=forcing),
        "difusion": lambda: gd.GraphField(M, p, branch="diffusion", forcing=forcing),
        "independiente": lambda: gd.Independent(M, p, branch="wave", forcing=forcing),
    }

    out = {}
    for cname, make in configs.items():
        amps, lags, corrs, costs, downs = [], [], [], [], []
        for seed in range(n_seeds):
            tr = gd.distributed_loop(mpc, make(), demand, T=T, seed=seed,
                                     grid_points=grid_points)
            if tr is None:
                continue
            m = gd.transport_profile(tr)
            amps.append(m["amplitud_relativa"]); lags.append(m["retraso"])
            corrs.append(m["correlacion"]); costs.append(m["coste_por_zona"])
            downs.append(m["coste_aguas_abajo"])
        A = np.array(amps); L = np.array(lags); C = np.array(corrs)
        out[cname] = {"amplitud_relativa": A.mean(0).tolist(),
                      "retraso": np.nanmean(L, 0).tolist(),
                      "correlacion": C.mean(0).tolist(),
                      "coste_aguas_abajo": downs}
        print(f"### {cname}")
        print(f"  {'zona':>6}" + "".join(f"{i+1:>10}" for i in range(M)))
        print(f"  {'amplitud rel.':>14}"[:6] + "".join(f"{v:>10.4f}" for v in A.mean(0)))
        print(f"  {'retraso':>6}" + "".join(f"{v:>10.1f}" for v in np.nanmean(L, 0)))
        print(f"  {'corr':>6}" + "".join(f"{v:>10.3f}" for v in C.mean(0)))
        print(f"  coste aguas abajo (zonas 2..{M}) = "
              f"{np.mean(downs):.2f} +- {np.std(downs, ddof=1):.2f}\n")

    print("=" * 78)
    print("CONTRASTES PAREADOS POR SEMILLA")
    for a, b in (("onda", "difusion"), ("onda", "independiente"),
                 ("difusion", "independiente")):
        x = np.array(out[a]["coste_aguas_abajo"])
        y = np.array(out[b]["coste_aguas_abajo"])
        n = min(len(x), len(y))
        if n < 2:
            continue
        d = y[:n] - x[:n]
        tt = sps.ttest_rel(y[:n], x[:n])
        star = " *" if tt.pvalue < 0.05 else ""
        print(f"  coste aguas abajo  {a:<14} vs {b:<14} "
              f"delta={d.mean():+8.3f} t={tt.statistic:+7.2f} p={tt.pvalue:.4f}{star}")
    # transporte a la zona mas lejana
    print()
    for cname in configs:
        print(f"  transporte a la zona {M}: amplitud relativa = "
              f"{out[cname]['amplitud_relativa'][-1]:.4f}, "
              f"corr = {out[cname]['correlacion'][-1]:.3f}, "
              f"retraso = {out[cname]['retraso'][-1]:.1f} muestras   [{cname}]")

    print(f"\ntiempo: {time.time()-t0:.1f}s")
    return out


if __name__ == "__main__":
    res = main()
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dst = os.path.join(here, "results")
    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(dst, "distributed.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    print(f"escrito {os.path.join(dst, 'distributed.json')}")
