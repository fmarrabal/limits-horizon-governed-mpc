"""Gobernador Homeostatico Inercial (GHI) para MPC multiobjetivo.

Implementacion de referencia en Python. Hay un port MATLAB nativo en
`code/matlab/`, validado numericamente contra este (ver `code/README.md`).

MAPA DEL PAQUETE
----------------
    plant       plantas, objetivos y bancos de prueba
    terminal    ley auxiliar, costes terminales por Lyapunov, region invariante
    mompc       el MPC multiobjetivo como QP denso explicito
    weights     conjunto admisible, radio certificado, evidencia de concavidad
    regulators  reguladores de orden 0/1/2, anti-windup, igualacion en discreto
    field       el campo homeostatico sobre el grafo de subsistemas
    experiment  lazo cerrado, escenarios y metricas
    stats       contrastes pareados, Holm, intervalos de Poisson
    audit       la suite de auditoria: cada hallazgo adversarial, ejecutable

USO MINIMO
----------
    from ghi import plant, terminal, mompc, weights
    prob = plant.problem_conflict()
    prob, term = terminal.design(prob)          # ingredientes CORRECTOS
    print(terminal.audit(prob, term)["ok"])     # True
    M = mompc.MOMPC(prob, term)
    sol = mompc.solve(M, [5.0, 5.0], [0.5, 0.5])
"""
from . import plant, terminal, mompc, weights, regulators, field, experiment, stats

__all__ = ["plant", "terminal", "mompc", "weights", "regulators", "field",
           "experiment", "stats"]
__version__ = "1.0.0"
