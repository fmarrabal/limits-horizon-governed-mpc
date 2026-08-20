"""Suite de auditoria: cada hallazgo de la revision adversarial, ejecutable.

La regla de este modulo es simple: **todo defecto que se encontro una vez queda
convertido en una asercion que falla si vuelve**. Ninguna comprobacion se apoya
en la memoria de nadie.

    A1  los ingredientes terminales satisfacen S_i >= 0
    A2  Omega es positivamente invariante bajo Acl
    A3  Omega respeta las restricciones de estado y de entrada terminal
    A4  V*(x, .) es CONCAVA en alpha            (correccion a Bemporad 2009)
    A5  el conjunto admisible puede ser NO convexo
    A6  el radio certificado es valido: todo alpha' a distancia <= r es admisible
    A7  la igualacion 1er/2o orden es exacta EN DISCRETO
    A8  el anti-windup NO se dispara si no hay bloqueo
    A9  el regulador ejecutado tiene la sobreoscilacion teorica
    A10 la colocacion giroscopica es estable en toda la caja de parametros
    A11 el umbral de Merkin es exacto en el caso degenerado y conservador fuera
    A12 el ingrediente terminal HEREDADO DEL TFM falla (regresion documentada)
    A13 SIN terminal: V_N es NO DECRECIENTE en N
    A14 CON terminal correcto: V_N es NO CRECIENTE en N   (dualidad de monotonia)
    A15 en regimen lineal, alpha* y N* son INVARIANTES a la amplitud de d
    A16 la ANTICIPACION solo es utilizable si hay estado de ALMACENAMIENTO
    A17 el campo con beta=0 es EXACTAMENTE un banco de filtros LTI (no crea info)
    A18 forcing='target' tiene ganancia cruzada NULA en continua; 'source' no
    A19 la guarda de causalidad muerde en la sintonia PUBLICADA, no solo
        durante el barrido: los mecanismos publicados la pasan y la
        referencia clarividente NO
"""
from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Callable, List, Optional

import numpy as np

from . import field as gfield
from . import plant as gplant
from . import regulators as greg
from . import terminal as gterm
from . import weights as gw
from .mompc import MOMPC, shifted_sequence, solve


@dataclass
class Check:
    code: str
    title: str
    ok: bool
    detail: str = ""
    data: dict = dc_field(default_factory=dict)

    def __str__(self) -> str:
        mark = "OK  " if self.ok else "FALLA"
        return f"[{mark}] {self.code:<4} {self.title}" + (f"\n         {self.detail}" if self.detail else "")


# --------------------------------------------------------------------------

def a1_a3_terminal(prob: gplant.Problem, term: gterm.Terminal) -> List[Check]:
    rep = gterm.audit(prob, term)
    out = []
    det = "; ".join(
        f"{s['name']}: " + (f"eig_min={s['eig_min']:.3e}" if "eig_min" in s
                            else f"peor={s['peor_violacion']:.3e}")
        for s in rep["terminal"])
    out.append(Check("A1", "desigualdad terminal S_i >= 0 por objetivo",
                     rep["terminal_ok"], det, rep))
    out.append(Check("A2", "Omega positivamente invariante bajo Acl",
                     rep["invariancia"]["ok"],
                     f"peor violacion = {rep['invariancia']['peor_violacion']:.3e}"))
    out.append(Check("A3", "Omega respeta |x|<=xmax y |Kf x|<=umax",
                     rep["restricciones"]["ok"],
                     f"peor violacion = {rep['restricciones']['peor_violacion']:.3e}"))
    return out


def a4_concavity(M: MOMPC, seed: int = 0, n_states: int = 40) -> Check:
    rng = np.random.default_rng(seed)
    states = []
    while len(states) < n_states:
        x = rng.uniform(-6, 6, M.problem.plant.n)
        if solve(M, x, np.array([0.5, 0.5])) is not None:
            states.append(x)
    ev = gw.concavity_evidence(M, states)
    ok = ev["violaciones_concavidad"] == 0 and ev["violaciones_convexidad"] > 0
    det = (f"{ev['tests']} tests | violaciones de CONCAVIDAD = "
           f"{ev['violaciones_concavidad']} | de CONVEXIDAD = "
           f"{ev['violaciones_convexidad']} | d medio = {ev['d_media']:+.4e}")
    return Check("A4", "V*(x,.) es CONCAVA en alpha (correccion a Bemporad 2009)",
                 ok, det, ev)


def a5_nonconvex(M: MOMPC, seed: int = 0, tries: int = 8) -> Check:
    """Demuestra que el conjunto admisible ES no convexo para un rango de J_a.

    Como V*(x,.) es CONCAVA, su maximo cae en el interior del simplex, y todo
    subnivel por debajo de ese maximo es la union de dos intervalos. No es una
    rareza numerica: es la geometria obligada de un subnivel de una concava.
    """
    rng = np.random.default_rng(seed)
    total_niv = total_nc = 0
    con_pico_interior = 0
    n_est = 0
    muestra = None
    for _ in range(tries):
        x = rng.uniform(-6, 6, M.problem.plant.n)
        if solve(M, x, np.array([0.5, 0.5])) is None:
            continue
        ev = gw.nonconvexity_sweep(M, x, grid_points=41, n_levels=25)
        if ev["niveles"] == 0:
            continue
        n_est += 1
        total_niv += ev["niveles"]
        total_nc += ev["no_convexos"]
        con_pico_interior += int(ev["pico_interior"])
        if muestra is None and ev["no_convexos"] > 0:
            muestra = (x, ev)
    ok = total_nc > 0
    frac = total_nc / total_niv if total_niv else 0.0
    det = (f"{n_est} estados, {total_niv} niveles de J_a | NO convexo en "
           f"{total_nc} ({100*frac:.1f}%) | pico de V* en el interior en "
           f"{con_pico_interior}/{n_est} estados")
    if muestra is not None:
        det += f" | ejemplo x={np.round(muestra[0],3).tolist()}"
    return Check("A5", "el conjunto admisible ES no convexo para un rango de J_a",
                 ok, det, {"fraccion": frac, "estados": n_est})


def a6_radius(M: MOMPC, seed: int = 0, n_states: int = 8) -> Check:
    rng = np.random.default_rng(seed)
    peor, n_est, n_tests = -np.inf, 0, 0
    for _ in range(n_states):
        x = rng.uniform(-5, 5, M.problem.plant.n)
        s = solve(M, x, np.array([0.5, 0.5]))
        if s is None:
            continue
        Us = shifted_sequence(M, s)
        Ja = float(np.array([0.5, 0.5]) @ M.problem.costs(x, Us))
        rep = gw.radius_is_valid(M, x, np.array([0.5, 0.5]), Ja, n_dirs=8, seed=seed)
        if rep.get("n_tests", 0) > 0:
            peor = max(peor, rep["peor_exceso"])
            n_tests += rep["n_tests"]
            n_est += 1
    ok = n_est > 0 and peor <= 1e-6
    return Check("A6", "radio certificado valido: ||da||<=r => alpha' admisible",
                 ok, f"{n_est} estados, {n_tests} pruebas | "
                     f"peor exceso sobre J_a = {peor:.3e}")


def a7_matching(w0: float = 0.9, zeta: float = 0.5, dt: float = 1.0) -> Check:
    tau, g2 = greg.match_first_order(w0, zeta, dt)
    g1 = greg.mag_at(*greg.ss_first_order(tau, dt), np.pi / dt, dt)
    err = abs(g1 - g2)
    # y el error que se comete igualando en CONTINUO, que es el fallo original
    g2c = w0 ** 2 / np.sqrt((w0 ** 2 - (np.pi) ** 2) ** 2 + (2 * zeta * w0 * np.pi) ** 2)
    tau_c = (1.0 / g2c + 1.0) / 2.0
    g1c = greg.mag_at(*greg.ss_first_order(tau_c, dt), np.pi / dt, dt)
    return Check("A7", "igualacion 1er/2o orden exacta EN DISCRETO", err < 1e-12,
                 f"tau={tau:.4f} |H1|={g1:.9f} |H2|={g2:.9f} err={err:.2e}  ||  "
                 f"igualando en continuo: tau={tau_c:.4f} -> desajuste "
                 f"{g1c/g2:.4f} (deberia ser 1)",
                 {"tau": tau, "g": g2, "tau_continuo": tau_c, "desajuste_continuo": g1c / g2})


def a8_antiwindup() -> Check:
    regs = {
        "orden-1": greg.Order1(5.0),
        "orden-2 (theta=0.5)": greg.Order2(0.9, 0.5, theta=0.5),
        "orden-2 (theta=0.0)": greg.Order2(0.9, 0.5, theta=0.0),
    }
    rates = {k: greg.sync_fire_rate(v) for k, v in regs.items()}
    ok = all(r == 0.0 for r in rates.values())
    return Check("A8", "el anti-windup NO se dispara si no hay bloqueo", ok,
                 " | ".join(f"{k}: {v:.3f}" for k, v in rates.items()), rates)


def a9_overshoot(w0: float = 0.9, rtol: float = 1e-9) -> List[Check]:
    """La sobreoscilacion EJECUTADA debe coincidir con la del sistema DISCRETO.

    Compararla con la formula continua exp(-pi z/sqrt(1-z^2)) es un error: el
    Verlet a w0*dt = 0.9 amortigua bastante mas que el continuo. La referencia
    correcta es la respuesta al escalon del espacio de estados discreto, que se
    obtiene del MISMO codigo que ejecuta el lazo.

    Este es el test que detecta el fallo original: cuando el anti-windup se
    disparaba en cada muestra, la sobreoscilacion medida caia a 0.41% frente al
    5.66% del Verlet puro.
    """
    out = []
    for zeta in (0.3, 0.5, 0.7):
        med = greg.effective_overshoot(greg.Order2(w0, zeta, theta=0.5))
        Ad, Bd, Cd, Dd = greg.ss_second_order(w0, zeta)
        xs = np.zeros(2); peak = -np.inf
        for _ in range(400):
            xs = Ad @ xs + Bd.ravel() * 1.0      # el regulador devuelve a DESPUES del paso
            peak = max(peak, float((Cd @ xs).item() + float(Dd.item())))
        disc = max(0.0, peak - 1.0)
        cont = greg.theoretical_overshoot(zeta)
        ok = abs(med - disc) <= rtol + 1e-9 * max(1.0, disc)
        out.append(Check("A9", f"sobreoscilacion ejecutada == la del DISCRETO (zeta={zeta})",
                         ok, f"ejecutada={100*med:.4f}%  discreta={100*disc:.4f}%  "
                             f"(continua={100*cont:.2f}%, referencia equivocada)"))
    return out


def a10_a11_field() -> List[Check]:
    out = []
    # (a) caso degenerado de Merkin: umbral EXACTO
    p = gfield.FieldParams(w0=1.0, zeta=0.15, c=0.0, D=0.0, b=0.0)
    r = gfield.collocation_study(M=6, p=p)
    exact = abs(r["ratio_obs_pred"] - 1.0) < 0.02
    out.append(Check("A11", "umbral de Merkin EXACTO con K y C uniformes", exact,
                     f"beta*_pred={r['beta_pred']:.5f} beta*_obs={r['beta_obs']:.5f} "
                     f"ratio={r['ratio_obs_pred']:.4f}", r))
    # (b) con estructura de planta: el umbral debe ser CONSERVADOR (ratio > 1)
    ratios = []
    for c, D in ((0.35, 0.0), (0.0, 0.05), (0.35, 0.05), (0.6, 0.0)):
        pp = gfield.FieldParams(w0=1.0, zeta=0.15, c=c, D=D, b=0.0)
        rr = gfield.collocation_study(M=6, p=pp)
        ratios.append((c, D, rr["ratio_obs_pred"]))
    cons = all(x[2] >= 1.0 - 1e-6 for x in ratios)
    out.append(Check("A11b", "con estructura de planta el umbral es CONSERVADOR", cons,
                     " | ".join(f"c={c},D={D}: {x:.3f}" for c, D, x in ratios),
                     {"ratios": ratios}))
    # (c) colocacion giroscopica estable en toda la caja
    gyro_ok, worst = True, -np.inf
    for c in (0.0, 0.35, 0.6):
        for D in (0.0, 0.05, 0.2):
            for b in (0.0, 0.4):
                pp = gfield.FieldParams(w0=1.0, zeta=0.15, c=c, D=D, b=b)
                rr = gfield.collocation_study(M=6, p=pp, n=201)
                worst = max(worst, rr["gyro_max_re"])
                gyro_ok &= rr["gyro_estable"]
    out.append(Check("A10", "colocacion GIROSCOPICA estable en toda la caja", gyro_ok,
                     f"peor max Re(lambda) = {worst:+.3e} (debe ser < 0)"))
    return out


def a13_a14_monotonia() -> List[Check]:
    """La dualidad de monotonia de V_N, en ambos marcos, como asercion.

    A13: SIN coste terminal y l >= 0 (con factibilidad anidada, trivial con
         solo caja de entrada), V_N es NO DECRECIENTE en N. Tolerancia RELATIVA
         (el error de OSQP escala con V; una tolerancia absoluta daria falsos
         positivos en estados grandes).
    A14: CON ingredientes terminales correctos (P por Lyapunov, S_i = 0),
         V_N es NO CRECIENTE en N, para estados factibles.
    """
    out: List[Check] = []
    from .suboptimality import HorizonMPC
    plant_h = gplant.Plant(A=np.array([[1.0, 1.0], [0.0, 1.0]]),
                           B=np.array([[0.5], [1.0]]),
                           xmax=np.array([1e6, 1e6]), umax=np.array([1.0]))
    hm = HorizonMPC(plant_h, Q=np.diag([1.0, 0.1]), R=np.array([[0.01]]),
                    N_min=2, N_max=12)
    rng = np.random.default_rng(3)
    worst_rel = -np.inf
    for _ in range(15):
        x = rng.uniform(-5, 5, 2)
        Vs = np.array([hm.solve(x, N).V for N in range(2, 13)])
        d = Vs[:-1] - Vs[1:]                      # >0 = viola no-decrecimiento
        worst_rel = max(worst_rel, float(d.max() / max(Vs.max(), 1.0)))
    out.append(Check("A13", "SIN terminal: V_N NO DECRECIENTE en N (rel.)",
                     worst_rel < 1e-7,
                     f"peor violacion relativa = {worst_rel:.3e}"))

    base = gplant.problem_conflict()
    base, term = gterm.design(base)
    mompcs = {}
    for N in (3, 4, 5, 6, 8):
        pN = gplant.Problem(base.plant, base.objectives, N, base.name)
        mompcs[N] = MOMPC(pN, term)
    rng = np.random.default_rng(4)
    worst_rel = -np.inf
    n_ok = 0
    for _ in range(30):
        x = rng.uniform(-4, 4, 2)
        a = rng.uniform(0, 1)
        Vs = []
        for N in (3, 4, 5, 6, 8):
            s = solve(mompcs[N], x, np.array([1 - a, a]))
            Vs.append(np.nan if s is None else s.V)
        Vs = np.array(Vs)
        if np.isnan(Vs).any():
            continue
        n_ok += 1
        d = Vs[1:] - Vs[:-1]                      # >0 = viola no-crecimiento
        worst_rel = max(worst_rel, float(d.max() / max(Vs.max(), 1.0)))
    out.append(Check("A14", "CON terminal correcto: V_N NO CRECIENTE en N (rel.)",
                     n_ok > 0 and worst_rel < 1e-6,
                     f"{n_ok} estados factibles | peor violacion relativa = {worst_rel:.3e}"))
    return out


def a18_dc_cross_gain() -> Check:
    """El forzamiento del campo decide si una alerta SOSTENIDA cruza o no.

    Regresion de un defecto real encontrado por la revision adversarial de la
    arena de transporte: con el forzamiento original f = K(h_d - h), el
    equilibrio es h = h_d EXACTAMENTE, luego H(0) = I y la ganancia cruzada en
    continua es CERO. Un escalon sostenido en un nodo NO llega a ninguno de los
    demas con ninguna sintonia; solo cruzan los flancos. Cualquier experimento
    de transporte que pida al campo entregar una VENTANA sostenida le esta
    pidiendo justo lo que su estructura prohibe.

    Con f = w0^2 h_d la ganancia cruzada es w0^2 K^-1, no nula, y -- esto es lo
    que mantiene honesto el nulo de mecanismo -- IDENTICA en la rama de onda y
    en la difusiva cuando G = 0.
    """
    from .distributed import GraphField

    M = 6
    p = gfield.FieldParams(w0=0.5, zeta=0.5, c=0.6, D=0.0, b=0.0, beta=0.0,
                           gamma=1.0)

    def estacionario(branch, forcing, n=6000):
        f = GraphField(M, p, branch=branch, a0=0.0, forcing=forcing)
        hd = np.zeros(M); hd[0] = 1.0
        for _ in range(n):
            h = f.step(hd, 1.0)
        return h

    cruz_target = np.abs(estacionario("wave", "target")[1:]).max()
    hs_w = estacionario("wave", "source")
    hs_d = estacionario("diffusion", "source")
    teo = p.w0 ** 2 * np.linalg.inv(GraphField(M, p, forcing="source").K)[:, 0]
    ok = (cruz_target < 1e-9                       # el original NO transporta
          and hs_w[1] > 0.05                       # el corregido SI
          and np.abs(hs_w - hs_d).max() < 1e-9     # y el nulo sigue siendo justo
          and np.abs(hs_w - teo).max() < 1e-9)
    return Check("A18", "forcing='target' NO transporta en continua; 'source' si, "
                        "y por igual en ambas ramas", ok,
                 f"target: peor ganancia cruzada {cruz_target:.1e} (cero) | "
                 f"source: d=1,2,3 -> {np.round(hs_w[1:4], 4).tolist()} | "
                 f"|onda - difusion| = {np.abs(hs_w - hs_d).max():.1e}")


def a17_field_is_lti() -> Check:
    """Con b = beta = 0 el campo del grafo es EXACTAMENTE un banco de filtros
    LTI: la respuesta de cada nodo es la convolucion de la fuente con una
    respuesta impulsional fija.

    Es el limite duro del programa de transporte, y es un TEOREMA, no una
    medida: un operador LTI no crea informacion. Todo lo que el campo puede
    aportar sobre la senal que entra por el nodo fuente es una RECONFORMACION
    lineal fija -- retardo mas ensanchamiento --, de modo que cualquier linea
    base que parametrice libremente esa familia lo contiene. El campo solo
    podria ganar por la parte NO lineal (beta != 0) o por informacion que no
    pase por la fuente.
    """
    from .distributed import GraphField
    from .field import FieldParams

    M, T = 4, 160
    p = gfield.FieldParams(w0=0.5, zeta=0.5, c=0.6, D=0.0, b=0.0, beta=0.0,
                           gamma=1.0)

    def resp(src, branch):
        f = GraphField(M, p, branch=branch, a0=0.0)
        H = np.zeros((T, M)); hd = np.zeros(M)
        for t in range(T):
            hd[0] = src[t]
            H[t] = f.step(hd, 1.0)
        return H

    rng = np.random.default_rng(0)
    peor = 0.0
    det = []
    for branch in ("wave", "diffusion"):
        u, v = rng.standard_normal(T), rng.standard_normal(T)
        a, b = 2.3, -0.7
        sup = np.abs(resp(a * u + b * v, branch) - (a * resp(u, branch) + b * resp(v, branch))).max()
        k = 17
        us = np.concatenate([np.zeros(k), u[:-k]])
        ti = np.abs(resp(us, branch)[k:] - resp(u, branch)[:-k]).max()
        imp = np.zeros(T); imp[0] = 1.0
        K, r0 = resp(imp, branch), resp(u, branch)
        cv = max(np.abs(np.convolve(u, K[:, i])[:T] - r0[:, i]).max() for i in range(M))
        peor = max(peor, sup, ti, cv)
        det.append(f"{branch}: superposicion {sup:.1e}, invariancia {ti:.1e}, "
                   f"convolucion {cv:.1e}")
    return Check("A17", "el campo con beta=0 es EXACTAMENTE un banco de filtros LTI",
                 peor < 1e-10, " | ".join(det))


def a16_anticipation_needs_storage() -> Check:
    """La ANTICIPACION es inutilizable por el canal del meta-parametro si el
    punto de operacion optimo en reposo no depende de ese meta-parametro.

    Con planta asintoticamente estable regulada al origen, el equilibrio en
    reposo es x = 0 para TODO peso: no hay nada que pre-posicionar, y un aviso
    anticipado no compra nada por mucha antelacion que tenga. Con un estado de
    ALMACENAMIENTO y objetivos de CONSIGNA DISTINTA, el peso fija el punto de
    operacion y la antelacion si compra.

    La asercion comprueba el CONTRASTE: barrido de antelacion PLANO sin
    almacenamiento, NO plano con almacenamiento. Es la misma clase de argumento
    que A15 y descarta otra familia entera de bancos de prueba.
    """
    from . import transport as gtp

    ALPH = np.round(np.linspace(0.05, 0.95, 10), 4)
    LEADS = (0, 3, 8, 16)

    def barrido(mpc, plant, d, starts, width, runner, kw):
        best = None
        for a_lo in ALPH:
            for a_hi in ALPH:
                if a_hi <= a_lo:
                    continue
                sch = gtp.clairvoyant_schedule(len(d), starts, width, a_lo, a_hi, 0)
                c = runner(mpc, plant, d, sch, **kw)["coste"]
                if best is None or c < best[0]:
                    best = (c, a_lo, a_hi)
        c0, a_lo, a_hi = best
        cs = [runner(mpc, plant, d,
                     gtp.clairvoyant_schedule(len(d), starts, width, a_lo, a_hi, L),
                     **kw)["coste"] for L in LEADS]
        return 100.0 * (c0 - min(cs)) / c0        # ganancia maxima de la antelacion

    # (a) SIN almacenamiento: regulacion al origen -> tiene que salir plano
    tr = gtp.Traffic(seed=0, T=420, sigma_bg=0.045, amp=0.95, width=14,
                     gap_lo=70, gap_hi=100)
    d, st = tr.build()
    p1 = gtp.transport_plant(umax=1.0)
    m1 = gtp.WeightedMPC(p1, np.diag([1.0, 0.02]), np.array([[1.0]]), 10, ALPH)
    g_sin = barrido(m1, p1, d, st, tr.width, lambda m, p, dd, a, **k:
                    gtp.run_schedule(m, p, dd, a), {})

    # (b) CON almacenamiento y consignas distintas -> tiene que NO salir plano
    tr2 = gtp.Traffic(seed=0, T=420, sigma_bg=0.020, amp=1.05, width=16,
                      gap_lo=80, gap_hi=120)
    d2, st2 = tr2.build()
    p2 = gtp.storage_plant(tau_act=3.0, umax=0.6)
    objs = [(np.diag([1.0, 0.05]), np.array([[1.0]]), np.array([0.0, 0.0])),
            (np.diag([1.0, 0.05]), np.array([[1.0]]), np.array([3.0, 0.0]))]
    m2 = gtp.SetpointMPC(p2, objs, 16, ALPH)
    g_con = barrido(m2, p2, d2, st2, tr2.width, gtp.run_storage,
                    dict(b_econ=0.0, b_crit=-2.5, pen=5.0))

    ok = g_sin < 1.0 and g_con > 1.0
    return Check("A16", "la anticipacion solo sirve si hay ALMACENAMIENTO", ok,
                 f"ganancia maxima de la antelacion: SIN almacenamiento "
                 f"{g_sin:+.2f}% (plano, como debe) | CON almacenamiento "
                 f"{g_con:+.2f}% (no plano, como debe)")


def a15_scale_invariance() -> Check:
    """En regimen LINEAL sin saturar, el peso optimo y el horizonte optimo son
    INVARIANTES a la amplitud de la perturbacion.

    Escalar d por lambda escala u* y x* por lambda y el coste por lambda^2 PARA
    CADA (alpha, N) fijo, y un factor comun positivo no mueve el argmin. Luego
    ningun gobernador puede leer nada de la AMPLITUD ABSOLUTA de la perturbacion.

    DOS COSAS QUE ESTA ASERCION APRENDIO A LA FUERZA:

      1. EL ARGMIN TIENE QUE SER INTERIOR. La primera version barria alpha hasta
         0.9 y N hasta 12, y el optimo salia justo en esos extremos en las tres
         escalas. Un argmin pegado al borde es el caso en que MENOS puede
         moverse: comprobar alli que no se mueve no demuestra casi nada. Ahora
         las rejillas llegan mas lejos Y la asercion FALLA si el optimo toca un
         extremo, que es la regla que el propio trabajo predica.
      2. EL COSTE DE EVALUACION TIENE QUE SER CUADRATICO. La proposicion no dice
         nada sobre otros grados: con el cuartico de fatiga de la arena de
         vibracion el argmin SI se mueve, y con razon. Por eso aqui se pondera
         el esfuerzo con r = 40, que ademas es lo que lleva el optimo del peso
         al interior del simplex.
    """
    from . import vibration as gvib
    from .suboptimality import HorizonMPC

    p = gvib.vibration_plant(umax=1e4, xmax=1e5, vmax=1e5)   # nunca satura
    prob = gplant.Problem(p, gvib.problem_vibration().objectives, N=8, name="lin")
    prob, term = gterm.design(prob)
    Mw = MOMPC(prob, term)
    hm = HorizonMPC(p, Q=np.diag([1.0, 0.05]), R=np.array([[1.0]]), N_min=2, N_max=48)

    def loop_w(d, a, T=320, warm=60):
        x = np.zeros(2); tot = 0.0
        for t in range(T):
            s = solve(Mw, x, np.array([1 - a, a]))
            if s is None:
                return np.inf
            u = float(s.U[0, 0])
            if t >= warm:
                tot += float(x[0] ** 2 + 40.0 * u * u)      # coste CUADRATICO
            w = np.zeros(2); w[1] = d[t]
            x = p.A @ x + p.B.ravel() * u + w
        return tot

    def loop_N(d, N, T=320, warm=60):
        x = np.zeros(2); tot = 0.0
        for t in range(T):
            u = hm.solve(x, N).u0
            if t >= warm:
                tot += float(x[0] ** 2 + 0.05 * x[1] ** 2 + u * u)
            w = np.zeros(2); w[1] = d[t]
            x = p.A @ x + p.B.ravel() * u + w
        return tot

    lambdas = [1.0, 3.0, 9.0]
    alphas = np.round(np.linspace(0.05, 0.95, 10), 3)
    Ns = [2, 4, 8, 12, 20, 26, 32, 40, 48]
    # Se compara la CURVA NORMALIZADA entera, no solo su argmin. Es lo que la
    # proposicion afirma (J -> lambda^2 J para CADA valor fijo), y ademas es
    # robusto: cuando el minimo es plano, el argmin salta entre puntos cuyo
    # coste difiere en 1e-16 y compararlo daria un falso fallo.
    CW, CN, a_star, N_star = [], [], [], []
    borde = False
    for lam in lambdas:
        d, _ = gvib.Excitation(E0=lam, depth=0.6).signal(320)
        vw = np.array([loop_w(d, a) for a in alphas]) / lam ** 2
        vn = np.array([loop_N(d, N) for N in Ns]) / lam ** 2
        CW.append(vw); CN.append(vn)
        j, k = int(np.argmin(vw)), int(np.argmin(vn))
        a_star.append(float(alphas[j])); N_star.append(Ns[k])
        borde |= (j in (0, len(alphas) - 1)) or (k in (0, len(Ns) - 1))
    CW, CN = np.array(CW), np.array(CN)
    dw = float(np.abs(CW - CW[0]).max() / max(np.abs(CW[0]).max(), 1e-12))
    dn = float(np.abs(CN - CN[0]).max() / max(np.abs(CN[0]).max(), 1e-12))
    # conjunto de casi-optimos: lo que de verdad debe coincidir entre escalas
    cas = [set(np.flatnonzero(v <= v.min() * (1 + 1e-9)).tolist()) for v in CN]
    mismo = all(c == cas[0] for c in cas)
    ok = (not borde) and dw < 1e-9 and dn < 1e-9 and mismo
    return Check("A15", "regimen lineal: alpha* y N* INVARIANTES a la amplitud",
                 ok, f"curvas J/lambda^2 identicas entre escalas: peso {dw:.1e}, "
                     f"horizonte {dn:.1e} | alpha*={sorted(set(a_star))} "
                     f"N*={sorted(set(N_star))} | argmin de borde: {borde} | "
                     f"mismo conjunto de casi-optimos: {mismo}")


def a12_tfm_regression() -> Check:
    """El ingrediente terminal HEREDADO DEL TFM debe FALLAR. Es una regresion
    documentada: si algun dia pasa, es que alguien cambio los numeros del TFM."""
    prob = gplant.problem_conflict()
    Kf = np.array([[-0.5, -1.4]])          # la ley terminal del TFM
    P_tfm = [np.diag([0.05, 0.05]), np.diag([20.0, 8.0])]
    eigs = []
    for ob, P in zip(prob.objectives, P_tfm):
        S = gterm.terminal_slack(prob.plant, Kf, ob, P)
        eigs.append(float(np.linalg.eigvalsh(S).min()))
    falla = any(e < -1e-9 for e in eigs)
    return Check("A12", "los ingredientes del TFM VIOLAN la desigualdad terminal "
                        "(regresion documentada, debe seguir fallando)", falla,
                 " | ".join(f"eig_min(S_{i}) = {e:+.4f}" for i, e in enumerate(eigs)),
                 {"eig_min": eigs})


# --------------------------------------------------------------------------

def a19_causality_guard() -> Check:
    """A19: la guarda de causalidad, aplicada a la sintonia que se publica.

    POR QUE EXISTE. La guarda se ejecutaba dentro del barrido, descartando
    configuraciones acausales antes de puntuarlas. Eso no deja constancia de
    que el ganador *publicado* la pase: si el JSON se reescribe a mano, o si
    una sintonia se copia de una tanda anterior, nadie se entera. Esta
    asercion vuelve a pasar la sonda por los parametros archivados, y ademas
    exige que la referencia clarividente los suspenda --- una referencia que
    pasara la guarda no seria una referencia acausal, seria un mecanismo, que
    es exactamente el error que este trabajo documenta.
    """
    import json
    import os
    import sys

    aqui = os.path.dirname(os.path.abspath(__file__))
    raiz = os.path.dirname(os.path.dirname(aqui))
    scripts = os.path.join(os.path.dirname(aqui), "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import run_transport_mechanism as base
    import run_transport_causal2 as c2
    import run_transport_affine as af
    from . import transport as tp

    sint = json.load(open(os.path.join(raiz, "results",
                                       "transport_affine.json")))["sintonia"]
    filas, todo_bien = [], True
    for clave, mech, rama in (("afin", af.mech_afin, None),
                              ("lineal", af.mech_lineal, None),
                              ("afin_conformado", af.mech_afin_conformado, None)):
        par = sint[clave]
        pasa = tp.is_causal(c2.gate_of(mech, par, rama))
        todo_bien = todo_bien and pasa
        filas.append(f"{clave}: {'causal' if pasa else 'ACAUSAL'}")

    par_c = sint["clarividente"]
    ref_causal = tp.is_causal(c2.gate_of(af.mech_clarividente, par_c, None))
    todo_bien = todo_bien and not ref_causal
    filas.append("clarividente: " + ("CAUSAL (no deberia)" if ref_causal
                                     else "acausal, como corresponde"))
    filas.append(f"antelacion clarividente = {par_c['lead']} contra "
                 f"{base.TAU_HOP} por salto")
    return Check("A19", "la guarda muerde en la sintonia publicada",
                 todo_bien, "; ".join(filas),
                 {"lead_clarividente": par_c["lead"], "tau_hop": base.TAU_HOP})


def run_all(verbose: bool = True, quick: bool = False) -> List[Check]:
    checks: List[Check] = []

    prob = gplant.problem_conflict()
    prob, term = gterm.design(prob)
    M = MOMPC(prob, term)

    checks += a1_a3_terminal(prob, term)
    checks.append(a4_concavity(M, n_states=8 if quick else 40))
    checks.append(a5_nonconvex(M, tries=4 if quick else 12))
    checks.append(a6_radius(M, n_states=3 if quick else 8))
    checks.append(a7_matching())
    checks.append(a8_antiwindup())
    checks += a9_overshoot()
    if not quick:
        checks += a10_a11_field()
    checks.append(a12_tfm_regression())
    checks += a13_a14_monotonia()
    checks.append(a15_scale_invariance())
    checks.append(a16_anticipation_needs_storage())
    checks.append(a17_field_is_lti())
    checks.append(a18_dc_cross_gain())
    checks.append(a19_causality_guard())

    if verbose:
        print("=" * 78)
        print("SUITE DE AUDITORIA  (cada hallazgo adversarial, ejecutable)")
        print("=" * 78)
        for c in checks:
            print(c)
        n_bad = sum(1 for c in checks if not c.ok)
        print("=" * 78)
        print(f"{len(checks) - n_bad}/{len(checks)} comprobaciones OK"
              + ("" if n_bad == 0 else f"  --  {n_bad} FALLAN"))
    return checks


if __name__ == "__main__":
    import sys
    res = run_all(quick="--quick" in sys.argv)
    sys.exit(0 if all(c.ok for c in res) else 1)
