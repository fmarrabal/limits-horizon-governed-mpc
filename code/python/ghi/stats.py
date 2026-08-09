"""Estadistica del protocolo: contrastes pareados por semilla y correccion de Holm.

DISCIPLINA
----------
El paper del HBP publica un nulo pre-registrado y documenta un fallo que juega
en su contra. Este modulo existe para mantener esa disciplina:

  * TODO se contrasta PAREADO POR SEMILLA -- misma semilla, mismo ruido, misma
    perturbacion, distinto regulador. Sin parear, la varianza entre semillas
    domina y no se ve nada.
  * La familia de contrastes primarios se corrige por HOLM. Un efecto que no
    sobrevive a Holm se reporta como direccionalmente consistente, no como
    demostrado.
  * Los conteos pequenos (bloqueos) NO se contrastan con la t: son procesos de
    conteo. Se usa el intervalo de Poisson sobre la razon, que es la forma
    honesta de decir "esto podria ser ruido".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats as sps


@dataclass
class Paired:
    a_name: str
    b_name: str
    metric: str
    n: int
    delta: float          # media de (b - a); > 0 => gana `a`
    sem: float
    t: float
    p: float
    p_holm: Optional[float] = None

    @property
    def significativo(self) -> bool:
        p = self.p_holm if self.p_holm is not None else self.p
        return bool(p < 0.05)

    def __str__(self) -> str:
        ph = "" if self.p_holm is None else f" p_holm={self.p_holm:.4f}"
        star = " *" if self.significativo else ""
        return (f"{self.a_name} vs {self.b_name:<12} {self.metric:<13}"
                f"delta={self.delta:+9.4f}+-{self.sem:<8.4f} "
                f"t={self.t:+7.2f} p={self.p:.4f}{ph}{star}")


def paired(rows_a: Sequence[dict], rows_b: Sequence[dict], metric: str,
           a_name: str = "A", b_name: str = "B") -> Optional[Paired]:
    """Contraste pareado. delta = media(b) - media(a): positivo si gana `a`
    cuando la metrica es de las que conviene MINIMIZAR (error, coste)."""
    n = min(len(rows_a), len(rows_b))
    if n < 2:
        return None
    a = np.array([r[metric] for r in rows_a[:n]], float)
    b = np.array([r[metric] for r in rows_b[:n]], float)
    d = b - a
    if np.allclose(d.std(ddof=1), 0.0):
        return Paired(a_name, b_name, metric, n, float(d.mean()), 0.0, np.nan, 1.0)
    tt = sps.ttest_rel(b, a)
    return Paired(a_name, b_name, metric, n, float(d.mean()),
                  float(d.std(ddof=1) / np.sqrt(n)),
                  float(tt.statistic), float(tt.pvalue))


def holm(tests: Sequence[Paired]) -> List[Paired]:
    """Correccion de Holm-Bonferroni sobre una familia de contrastes."""
    valid = [t for t in tests if t is not None and np.isfinite(t.p)]
    order = sorted(range(len(valid)), key=lambda i: valid[i].p)
    m = len(valid)
    prev = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (m - rank) * valid[i].p)
        adj = max(adj, prev)          # monotonia de Holm
        valid[i].p_holm = adj
        prev = adj
    return valid


def poisson_ratio_ci(count_a: int, count_b: int, conf: float = 0.95
                     ) -> Tuple[float, float, float]:
    """IC para la razon de dos conteos de Poisson (metodo binomial exacto).

    Si el intervalo cubre 1, la diferencia de conteos es compatible con ruido.
    Es la prueba honesta para 'bloqueos', que son numeros de una cifra.
    """
    tot = count_a + count_b
    if tot == 0:
        return (np.nan, np.nan, np.nan)
    ratio = np.inf if count_b == 0 else count_a / count_b
    lo_p, hi_p = sps.beta.interval(conf, count_a + 0.5, count_b + 0.5) \
        if tot > 0 else (np.nan, np.nan)
    to_ratio = lambda p: np.inf if p >= 1 else p / (1 - p)
    return (float(ratio), float(to_ratio(lo_p)), float(to_ratio(hi_p)))


def contrast_table(bank: Dict[str, Dict[str, List[dict]]],
                   comparisons: Sequence[Tuple[str, str]],
                   metrics: Sequence[str] = ("err_demanda", "coste"),
                   apply_holm: bool = True) -> Dict[str, List[Paired]]:
    """Tabla completa de contrastes, con Holm por metrica (una familia por metrica)."""
    out: Dict[str, List[Paired]] = {}
    for met in metrics:
        fam: List[Paired] = []
        for sname, regs in bank.items():
            for a_name, b_name in comparisons:
                if a_name not in regs or b_name not in regs:
                    continue
                pr = paired(regs[a_name], regs[b_name], met,
                            f"{sname}:{a_name}", b_name)
                if pr is not None:
                    fam.append(pr)
        out[met] = holm(fam) if apply_holm else fam
    return out


def print_contrasts(table: Dict[str, List[Paired]]) -> None:
    for met, fam in table.items():
        print(f"\n--- {met}  (delta > 0 => gana el primero; Holm sobre la familia)")
        for pr in fam:
            print("  " + str(pr))
