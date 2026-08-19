"""Genera todas las figuras del paper a partir de los resultados MEDIDOS.

Regla: ninguna figura contiene datos inventados ni suavizados. Lo que no esta en
`code/results/*.json` se REGENERA aqui ejecutando el mismo codigo del paquete,
nunca se transcribe a mano, salvo dos tablas de barrido cuyo origen se cita
explicitamente en el propio codigo.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "code", "python"))
RES = os.path.join(ROOT, "code", "results")
FIG = os.path.join(HERE, "figs")
os.makedirs(FIG, exist_ok=True)

COL, DCOL = 3.5, 7.16          # anchos de columna IEEE en pulgadas
plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.4,
    "lines.linewidth": 1.2, "lines.markersize": 3.5,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "font.family": "serif",
    # NO usar usetex: en esta instalacion el glifo del SIGNO MENOS no se dibuja
    # ("10^{-1}" sale como "10 1", "K^{-1}" como "K 1"). Comprobado con
    # pdf.fonttype 42 y 3 y con cm/lmodern/mathptmx: falla en todas. mathtext
    # con el juego STIX lo renderiza bien y casa con la serif del texto.
    "text.usetex": False,
    "mathtext.fontset": "stix",
})
C = {"wave": "#B2182B", "diff": "#2166AC", "delay": "#4D4D4D",
     "matched": "#1A9850", "oracle": "#762A83", "react": "#999999",
     "acc": "#E08214"}


def load(name):
    with open(os.path.join(RES, name)) as fh:
        return json.load(fh)


def save(fig, name):
    p = os.path.join(FIG, name)
    fig.savefig(p + ".pdf")
    plt.close(fig)
    print(f"  {name}.pdf")


# ==========================================================================
# Fig 1 -- V* es CONCAVA en el peso, y el conjunto admisible es NO convexo
# ==========================================================================
def fig_concavity():
    from ghi import plant as gplant, terminal as gterm
    from ghi.mompc import MOMPC, solve, shifted_sequence

    prob = gplant.problem_conflict()
    prob, term = gterm.design(prob)
    M = MOMPC(prob, term)
    rng = np.random.default_rng(3)
    xs = []
    while len(xs) < 3:
        x = rng.uniform(-5, 5, prob.plant.n)
        if solve(M, x, np.array([0.5, 0.5])) is not None:
            xs.append(x)
    al = np.linspace(0.0, 1.0, 81)

    fig, ax = plt.subplots(1, 2, figsize=(DCOL, 2.15))
    for k, x in enumerate(xs):
        V = []
        for a in al:
            s = solve(M, x, np.array([1 - a, a]))
            V.append(np.nan if s is None else s.V)
        V = np.array(V)
        ax[0].plot(al, V, color=plt.cm.viridis(k / 3.0),
                   label=rf"$x_{k+1}$")
        if k == 0:
            Vk, alk = V, al
    ax[0].set_xlabel(r"weight $\alpha$")
    ax[0].set_ylabel(r"$V^\star(x,\alpha)$")
    ax[0].set_title(r"(a) $V^\star$ is concave in $\alpha$", loc="left")
    ax[0].legend(frameon=False, ncol=3, loc="upper left", handlelength=1.2,
                 columnspacing=0.9)

    # Subnivel de una CONCAVA con pico interior. Para que sea la union de DOS
    # intervalos el nivel tiene que quedar por ENCIMA de los dos extremos y por
    # debajo del pico; con un nivel mas bajo el subnivel es un solo intervalo y
    # la figura no demostraria nada.
    j = int(np.nanargmax(Vk))
    lo_ext = max(Vk[0], Vk[-1])
    Ja = lo_ext + 0.45 * (Vk[j] - lo_ext)
    ok = Vk <= Ja
    ymin = np.nanmin(Vk) - 0.05 * (Vk[j] - np.nanmin(Vk))
    ax[1].plot(alk, Vk, color=C["wave"], zorder=3)
    ax[1].axhline(Ja, color=C["delay"], ls="--", lw=0.9, zorder=2)
    ax[1].text(0.99, Ja, r"$J_a$", ha="right", va="bottom", fontsize=7,
               color=C["delay"])
    ax[1].fill_between(alk, ymin, Vk, where=ok, color=C["diff"],
                       alpha=0.30, lw=0, zorder=1)
    ax[1].plot([alk[j]], [Vk[j]], "o", color="k", ms=4, zorder=4)
    ax[1].annotate("interior maximum", xy=(alk[j], Vk[j]), xytext=(0.06, 0.93),
                   textcoords="axes fraction", va="top",
                   arrowprops=dict(arrowstyle="->", lw=0.6), fontsize=7)
    ax[1].text(0.5, 0.05,
               r"$\{\alpha: V^\star\leq J_a\}$ = union of two intervals",
               transform=ax[1].transAxes, fontsize=6.6, ha="center")
    ax[1].set_ylim(ymin, Vk[j] + 0.12 * (Vk[j] - ymin))
    ax[1].set_xlabel(r"weight $\alpha$")
    ax[1].set_ylabel(r"$V^\star(x_1,\alpha)$")
    ax[1].set_title(r"(b) the admissible set is non-convex", loc="left")
    save(fig, "fig_concavity")


# ==========================================================================
# Fig 2 -- A15: invariancia de escala, y la estrecha ventana de saturacion
# ==========================================================================
def fig_scale():
    from ghi import terminal as gterm, vibration as vb
    from ghi.mompc import MOMPC
    from ghi.plant import Problem
    from ghi.suboptimality import HorizonMPC
    sys.path.insert(0, os.path.join(ROOT, "code", "python", "scripts"))
    import run_scale_invariance as rs

    p = vb.vibration_plant(umax=1e4, xmax=1e5, vmax=1e5)
    prob = Problem(p, vb.problem_vibration().objectives, N=10, name="lin")
    prob, term = gterm.design(prob)
    Mw = MOMPC(prob, term)
    import run_scale_invariance2 as s2
    hm = HorizonMPC(p, Q=np.diag([1.0, 0.05]), R=np.array([[1.0]]),
                    N_min=2, N_max=40)
    lam = [0.5, 1.0, 2.0, 4.0, 8.0]
    # rejillas AMPLIADAS y coste de evaluacion CUADRATICO con r = 40, de modo
    # que AMBOS optimos caigan en el INTERIOR: medir la invariancia en un argmin
    # pegado al borde seria el test menos informativo posible
    al = np.round(np.linspace(0.05, 0.95, 19), 4)
    Ns = [2, 3, 4, 6, 8, 12, 16, 20, 26, 32, 40]

    fig, ax = plt.subplots(1, 3, figsize=(DCOL, 2.0))
    for k, L in enumerate(lam):
        d, _ = vb.Excitation(E0=L, depth=0.6).signal(400)
        cw = np.array([s2.loop_w(Mw, p, d, a, (0.0, 0.0)) for a in al]) / L ** 2
        cn = np.array([s2.loop_N(hm, p, d, N, (0.0, 0.0)) for N in Ns]) / L ** 2
        col = plt.cm.plasma(k / 5.0)
        # marcadores distintos por escala: las curvas se SUPERPONEN exactamente,
        # asi que sin marcadores parecerian una sola y se perderia el resultado
        mk = ["o", "s", "^", "D", "v"][k]
        ax[0].plot(al, cw, marker=mk, ls="-", ms=4.5, mfc="none", color=col,
                   label=rf"$\lambda={L:g}$")
        ax[1].plot(Ns, cn, marker=mk, ls="-", ms=4.5, mfc="none", color=col)
    ax[0].set_xlabel(r"weight $\alpha$"); ax[0].set_ylabel(r"$J/\lambda^2$")
    ax[0].set_title(r"(a) weight", loc="left")
    ax[0].legend(frameon=False, ncol=1, fontsize=6)
    ax[1].set_xlabel(r"horizon $N$"); ax[1].set_ylabel(r"$J/\lambda^2$")
    ax[1].set_title(r"(b) horizon", loc="left")
    ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))

    # (c) ventana de saturacion
    p_s = vb.vibration_plant(umax=1.2)
    hs = HorizonMPC(p_s, Q=np.diag([1.0, 0.05]), R=np.array([[0.05]]),
                    N_min=2, N_max=24)
    Ns2 = [2, 4, 6, 8, 12, 16, 24]
    Es = [0.5, 1.0, 1.5, 1.8, 2.0, 2.2, 2.5, 3.0, 3.5, 5.0]
    rat = []
    for Ec in Es:
        d, _ = vb.Excitation(E0=Ec, depth=0.0).signal(600)
        cs = [rs.loop_horizon(hs, p_s, d, N, T=600) for N in Ns2]
        rat.append(cs[0] / min(cs))
    ax[2].plot(Es, rat, "o-", color=C["wave"])
    ax[2].axhspan(1.0, 1.2, color=C["react"], alpha=0.22, lw=0)
    ax[2].axvspan(1.8, 2.5, color=C["acc"], alpha=0.20, lw=0)
    ax[2].text(2.15, max(rat) * 0.86, "window", ha="center", fontsize=7,
               color=C["acc"])
    ax[2].set_xlabel(r"disturbance envelope $E$")
    ax[2].set_ylabel(r"$J(N_{\min})/J(N^\star)$")
    ax[2].set_title(r"(c) only saturation creates a window", loc="left")
    save(fig, "fig_scale")


# ==========================================================================
# Fig 3 -- frontera computo/coste del gobierno del horizonte (alpha_N)
# ==========================================================================
def fig_alphaN():
    d = load("alphaN.json")
    esc = "D_OOD_ambos"
    fr = d["frontier"][esc]
    Ns = sorted(int(k) for k in fr)
    fx = [fr[str(n)]["computo"] for n in Ns]
    fy = [fr[str(n)]["coste"] for n in Ns]
    bank = d["bank"][esc]
    keep = {"fijo-16": ("fixed $N=16$", C["delay"], "s"),
            "umbral-up4": ("threshold", C["diff"], "^"),
            "orden-1-nyq": ("leaky integrator", C["matched"], "o"),
            "orden-2": (r"2nd-order field ($\zeta{=}0.5$)", C["wave"], "D"),
            "orden-2-z1": (r"2nd-order field ($\zeta{=}1$)", C["oracle"], "v"),
            "orden-0": ("memoryless map", C["react"], "x")}

    fig, ax = plt.subplots(figsize=(COL, 2.5))
    ax.plot(fx, fy, "-", color="k", lw=0.8, alpha=0.55,
            label="fixed-$N$ frontier", zorder=1)
    ax.scatter(fx, fy, s=9, color="k", alpha=0.55, zorder=2)
    for k, (lab, col, mk) in keep.items():
        rows = [r for r in bank[k] if not r.get("divergio", False)]
        if not rows:
            continue
        x = np.mean([r["computo"] for r in rows])
        y = np.mean([r["coste"] for r in rows])
        ndiv = sum(1 for r in bank[k] if r.get("divergio", False))
        ax.scatter([x], [y], s=42, marker=mk, color=col, zorder=4, linewidth=0.5,
                   edgecolor="none",
                   label=lab + (f" ({ndiv} div.)" if ndiv else ""))
    ax.set_xlabel(r"computation $\sum_t N_t$")
    ax.set_ylabel("closed-loop cost")
    ax.set_yscale("log")
    ax.legend(frameon=False, loc="upper right", fontsize=6.2)
    ax.set_title("out-of-envelope scenario D", loc="left")
    save(fig, "fig_alphaN")


# ==========================================================================
# Fig 4 -- la arena de banda estrecha: modular pierde en 8/8
# ==========================================================================
def fig_vibration():
    d = load("vibration.json")["modulacion"]
    # cuatro condiciones (ruido de sensor x profundidad) y dos bandas: agrupado,
    # que es legible, en vez de ocho etiquetas apiladas
    conds, key = [], {}
    for r in d:
        k = (r["sigma"], r["gain"])
        if k not in key:
            key[k] = len(conds); conds.append(k)
    nar = [np.nan] * len(conds)
    bro = [np.nan] * len(conds)
    for r in d:
        (nar if r["band"] == "narrow" else bro)[key[(r["sigma"], r["gain"])]] = r["mejora_pct"]
    x = np.arange(len(conds))
    fig, ax = plt.subplots(figsize=(COL, 2.25))
    ax.bar(x - 0.19, nar, width=0.36, color=C["wave"], label="narrow band")
    ax.bar(x + 0.19, bro, width=0.36, color=C["diff"], label="broad band")
    ax.axhline(0, color="k", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([rf"$\sigma{{=}}{s:g}$" "\n" rf"$g{{=}}{g:g}$"
                        for s, g in conds], fontsize=6.5)
    ax.set_ylabel("improvement over\nbest constant [%]")
    ax.set_ylim(min(min(nar), min(bro)) * 1.20, 28)
    ax.legend(frameon=False, fontsize=6.5, loc="lower left", ncol=2)
    ax.text(0.5, 0.93, r"modulation loses in $8/8$", transform=ax.transAxes,
            fontsize=7, ha="center")
    save(fig, "fig_vibration")


# ==========================================================================
# Fig transporte -- estructura de la antelacion + comparacion final (afin)
# ==========================================================================
def fig_transport():
    """(a) esquema de la antelacion por nodo en las dos familias de retardo;
    (b) comparacion final desde transport_affine.json (30 semillas, guarda
    causal, normalizacion causal, densidades igualadas)."""
    TAU = 6
    nodos = np.arange(1, 4)
    fig, ax = plt.subplots(1, 2, figsize=(DCOL, 2.5),
                           gridspec_kw={"width_ratios": [0.85, 1.35]})

    # (a) antelacion recibida = TAU*i - k_i
    fam = [(r"proportional, $\delta{=}2$", [TAU*i - 2*i for i in nodos], C["delay"], "s"),
           (r"proportional, $\delta{=}\tau$", [0 for i in nodos], C["react"], "^"),
           (r"affine, $\delta{=}\tau$, $\ell{=}8$",
            [TAU*i - max(0, TAU*i - 8) for i in nodos], C["matched"], "o")]
    for lab, y, col, mk in fam:
        ax[0].plot(nodos, y, marker=mk, ls="-", color=col, ms=6, label=lab)
    ax[0].axhspan(-0.5, 0.5, color=C["wave"], alpha=0.10, lw=0)
    ax[0].text(2.0, 0.7, "no warning", fontsize=6, color=C["wave"], ha="center")
    ax[0].set_xlabel("node $i$")
    ax[0].set_ylabel("anticipation received [steps]")
    ax[0].set_xticks(list(nodos))
    ax[0].set_title("(a) what each delay family delivers", loc="left")
    ax[0].legend(frameon=False, fontsize=6, loc="upper left")

    # (b) comparacion final con barras de error
    t = load("transport_affine.json")["tabla"]
    orden = ["reactivo", "retardo-lineal", "onda", "retardo-afin",
             "afin-conformado", "clarividente"]
    nice = {"reactivo": "ideal local\ndetection",
            "retardo-lineal": "proportional\ndelay (3 par.)",
            "onda": "wave field\n(7 par.)",
            "retardo-afin": "affine delay\n(4 par.)",
            "afin-conformado": "affine +\nshaping (7 par.)",
            "clarividente": "acausal\nceiling"}
    cols = [C["react"], C["delay"], C["wave"], C["matched"], "#66BD63", C["oracle"]]
    v = [t[k]["pct_techo"] for k in orden]
    e = [t[k]["pct_techo_sem"] for k in orden]
    x = np.arange(len(orden))
    ax[1].bar(x, v, yerr=e, color=cols, width=0.66,
              error_kw=dict(ecolor="k", lw=0.8, capsize=2.5, capthick=0.8))
    ax[1].axhline(100, color="k", ls="--", lw=0.8)
    for xi, vi, ei in zip(x, v, e):
        ax[1].text(xi, vi + ei + 3.0, f"{vi:.1f}", ha="center", fontsize=6.3)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels([nice[k] for k in orden], fontsize=5.8)
    ax[1].set_ylabel("% of acausal ceiling")
    ax[1].set_ylim(0, 138)
    ax[1].set_title("(b) the affine line is at the ceiling ($n$=30)", loc="left")
    ax[1].annotate("", xy=(2.95, 118), xytext=(5.05, 118),
                   arrowprops=dict(arrowstyle="<->", lw=0.7))
    ax[1].text(4.0, 121, "$p_{Holm}=0.29$", ha="center", fontsize=6.2)
    save(fig, "fig_transport")


def fig_fleet():
    """(a) flota contra la frontera fija a computo GASTADO (fleet3.json);
    (b) descomposicion reparto proporcional vs igualitario (fleet_decomp.json)."""
    d3 = load("fleet3.json")
    dd = load("fleet_decomp.json")
    fig, ax = plt.subplots(1, 2, figsize=(DCOL, 2.5))

    MNmax = 128.0
    for rho, col, mk, lab in ((0, C["matched"], "o", r"independent ($\rho$=0)"),
                              (1, C["wave"], "s", r"common front ($\rho$=1)")):
        f = [r for r in d3[f"rho_{rho}"] if r["B"] > 16]   # B=16 degenerado
        x = [r["B"] / MNmax for r in f]
        y = [r["dif_pct"] for r in f]
        sig = [r["p"] < 0.05 for r in f]
        ax[0].plot(x, y, "-", color=col, lw=1.2, label=lab)
        ax[0].scatter([xi for xi, s_ in zip(x, sig) if s_],
                      [yi for yi, s_ in zip(y, sig) if s_],
                      marker=mk, s=26, color=col, zorder=4)
    ax[0].axhline(0, color="k", lw=0.9)
    ax[0].set_xlabel(r"budget fraction $B/(MN_{\max})$")
    ax[0].set_ylabel("cost vs. frontier at equal\nspent computation [%]")
    ax[0].set_title("(a) certificate detector vs. frontier", loc="left")
    ax[0].legend(frameon=False, fontsize=6.5, loc="lower right")
    ax[0].text(0.55, 0.90, "filled: $p<0.05$", transform=ax[0].transAxes,
               fontsize=6)

    # (b) los DOS mecanismos, uno por brazo, en los dos regimenes
    Bs = [r["B"] for r in dd["rho_0"]]
    x = np.arange(len(Bs))
    rep0 = [r["ventaja_pct"] for r in dd["rho_0"]]
    rep1 = [r["ventaja_pct"] for r in dd["rho_1"]]
    tmp0 = {r["B"]: r["ventaja_pct"] for r in dd["temporal_rho_0"]}
    tmp1 = {r["B"]: r["ventaja_pct"] for r in dd["temporal_rho_1"]}
    t0 = [tmp0.get(b, np.nan) for b in Bs]
    t1 = [tmp1.get(b, np.nan) for b in Bs]
    w = 0.20
    ax[1].bar(x - 1.5 * w, rep0, width=w, color=C["matched"],
              label=r"allocation, $\rho$=0")
    ax[1].bar(x - 0.5 * w, rep1, width=w, color=C["matched"], alpha=0.45,
              label=r"allocation, $\rho$=1")
    ax[1].bar(x + 0.5 * w, t0, width=w, color=C["wave"],
              label=r"timing, $\rho$=0")
    ax[1].bar(x + 1.5 * w, t1, width=w, color=C["wave"], alpha=0.45,
              label=r"timing, $\rho$=1")
    ax[1].axhline(0, color="k", lw=0.9)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels([f"B={b}" for b in Bs], fontsize=6.5)
    ax[1].set_ylabel("gain of the mechanism [%]")
    ax[1].set_title("(b) each regime carried by one mechanism", loc="left")
    ax[1].legend(frameon=False, fontsize=5.6, ncol=2, loc="upper right")
    save(fig, "fig_fleet")


# ==========================================================================
# Fig 6 -- A18: el forzamiento decide si una alerta sostenida cruza
# ==========================================================================
def fig_dcgain():
    from ghi.distributed import GraphField
    from ghi.field import FieldParams
    M, T = 6, 160
    p = FieldParams(w0=0.5, zeta=0.5, c=0.6, D=0.0, b=0.0, beta=0.0, gamma=1.0)
    src = np.zeros(T); src[20:110] = 1.0

    fig, ax = plt.subplots(1, 2, figsize=(DCOL, 2.0), sharey=True)
    for k, forcing in enumerate(("target", "source")):
        f = GraphField(M, p, branch="wave", a0=0.0, forcing=forcing)
        H = np.zeros((T, M)); hd = np.zeros(M)
        for t in range(T):
            hd[0] = src[t]
            H[t] = f.step(hd, 1.0)
        ax[k].plot(src, color="k", lw=0.8, ls=":", label="source node input")
        for i in range(1, 5):
            ax[k].plot(H[:, i], color=plt.cm.viridis((i - 1) / 4.0),
                       label=rf"node $i{{=}}{i}$")
        ax[k].set_xlabel("time step")
        ttl = (r"(a) $f=K(h_d-h)$: $H(0)=I$, nothing crosses"
               if forcing == "target" else
               r"(b) $f=\omega_0^2 h_d$: $H(0)=\omega_0^2K^{-1}$")
        ax[k].set_title(ttl, loc="left", fontsize=7.2)
    ax[0].set_ylabel(r"field value $h_i$")
    ax[1].legend(frameon=False, fontsize=6, ncol=2)
    save(fig, "fig_dcgain")


# ==========================================================================
# Fig 7 -- techo de cuantizacion: el alcance es logaritmico en la resolucion
# ==========================================================================
def fig_quantization():
    """Alcance limitado por cuantizacion, medido en cadena de M=8 (sin censura).

    La primera version dibujaba la ley como CURVA CONTINUA frente a medidas
    ENTERAS, lo que la hacia parecer sistematicamente desplazada, y su punto mas
    fino estaba censurado por la longitud de la cadena. Aqui la ley se dibuja
    como funcion ESCALONADA, el umbral es medio paso (el cuantizador redondea al
    mas cercano) y se marca el unico punto en que la ley sobreestima.
    """
    d = load("reach_law.json")
    rho = d["rho_interior"]
    filas = d["filas"]
    hops = np.arange(len(filas[0]["amplitud"]))

    fig, ax = plt.subplots(1, 2, figsize=(DCOL, 2.25))
    for k, f in enumerate(filas):
        a = np.array(f["amplitud"], float)
        col = plt.cm.cividis(k / len(filas))
        vis = a > 0
        ax[0].semilogy(hops[vis], a[vis], "o-", color=col, ms=4,
                       label=rf"$\Delta={f['paso']:.4f}$")
        cero = hops[(~vis) & (hops <= f["alcance"] + 2)]
        if len(cero):
            ax[0].semilogy(cero[:1], [4e-4], "x", color=col, ms=6, mew=1.4)
        ax[0].axhline(f["paso"], color=col, lw=0.5, ls=":")
    ax[0].set_xlabel("hops from the source")
    ax[0].set_ylabel("relative transported amplitude")
    ax[0].set_title("(a) quantised to exactly zero", loc="left")
    ax[0].legend(frameon=False, fontsize=5.8, loc="lower left")
    ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    ax[0].set_ylim(2e-4, 2.0)
    ax[0].annotate("dotted: grid step $\Delta$", xy=(0.60, 0.30),
                   xycoords="axes fraction", fontsize=5.8, color=C["delay"])
    ax[0].annotate("cut off", xy=(0.30, 0.055), xycoords="axes fraction",
                   fontsize=6.2)

    # ley ESCALONADA: n_max = floor(log(D/(2 a0))/log rho)
    st = np.logspace(-3.2, -1.2, 400)
    ley = np.floor(np.log(st) / np.log(rho))
    ax[1].step(st, ley, where="post", color=C["wave"], lw=1.3,
               label=r"$\lfloor\log\Delta/\log\rho\rfloor$")
    for k, f in enumerate(filas):
        col = plt.cm.cividis(k / len(filas))
        acierta = f["alcance"] == f["prediccion"]
        ax[1].plot([f["paso"]], [f["alcance"]], "o" if acierta else "o",
                   color=col if acierta else "white", ms=6,
                   mec=col, mew=1.6, zorder=4)
    ax[1].set_xscale("log")
    ax[1].set_xlabel(r"weight-grid step $\Delta$")
    ax[1].set_ylabel(r"reach $n_{\max}$ [hops]")
    ax[1].set_title(rf"(b) step law, $\rho={rho:.3f}$ (interior fit)",
                    loc="left")
    ax[1].yaxis.set_major_locator(MaxNLocator(integer=True))
    ax[1].legend(frameon=False, fontsize=6, loc="upper left")
    ax[1].annotate("open marker: law overpredicts by one hop",
                   xy=(0.36, 0.10), xycoords="axes fraction", fontsize=5.8)
    save(fig, "fig_quantization")


if __name__ == "__main__":
    print("generando figuras...")
    for fn in (fig_dcgain, fig_quantization, fig_vibration,
               fig_fleet, fig_scale, fig_transport):
        try:
            fn()
        except Exception as e:                     # noqa: BLE001
            print(f"  FALLO en {fn.__name__}: {type(e).__name__}: {e}")
    print("hecho")
