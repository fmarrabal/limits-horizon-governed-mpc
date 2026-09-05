# When Does Adapting the Meta-Parameter Pay?

**Minimal governors and structural limits for weight- and horizon-governed MPC**

V. Valdivieso, F. G. Montoya (University of Almería, CIAIMBITAL),
A. Campa-Pinto (Universidad de la Costa, Barranquilla) and
F. M. Arrabal-Campos* (University of Almería, CIAIMBITAL; *corresponding).

This repository contains everything needed to reproduce two manuscripts:
the paper in [`paper/`](paper/) and the Comment on Bemporad & Munoz de la
Pena (2009) in [`bemporad-2009/comment/`](bemporad-2009/comment/), whose
sign correction the paper uses and cites rather than restates. It contains
the Python reference
implementation, a native MATLAB port with numerical cross-validation, every
experiment script, the stored results each figure and table is built from, and
an executable audit suite in which every defect ever found in this work is a
regression assertion.

## The two results

**Constant anticipation, not proportional delay.** In a chain where a
disturbance travels physically downstream, delivering the upstream warning
through the meta-parameter channel is worth **12.4 %** of closed-loop cost
against the best possible local detector — but only if every node receives it
with the *same* anticipation. That requires an affine delay,
`k_i = max(0, τ·i − ℓ)`; the proportional family `k_i = δ·i` that per-hop
designs default to is structurally unable to provide it, and a wave field on
the graph or a matched shaped filter add nothing resolvable. The affine line
(4 parameters) sits statistically at a clairvoyant ceiling.

**A certificate detector governing the horizon.** The optimal horizon governor
turns out to be two effective parameters: request the maximum horizon *only*
while the relaxed-dynamic-programming certificate is broken (`α_t ≤ 0`), decay
slowly otherwise. In a fleet of eight thermal loops sharing one embedded
controller under a hard computation budget it beats the best fixed horizon by
**up to 6 %** at equal *spent* computation — at every budget that binds
under independent disturbances, and at four of eight budgets (Holm-adjusted)
under a perfectly correlated front; a decomposition shows the two mechanisms to
be complementary: cross-loop allocation pays only under independence, temporal
adaptation only under a common front.

## The five leaks

Every experimental arena in this project was first decided by a defect, not by
its mechanisms. The five that form a pattern, and the executable guard that
now keeps each out:

| leak | what it inverted | guard |
|---|---|---|
| broken channel (zero DC cross-gain) | the field lost without being able to compete | sustained-step probe (assertion A18) |
| acausal time shift in a tuning grid | both "winners" read the future | single-pulse probe, inside the tuning loop |
| acausal normalization (trace min/max) | gate scale set by the future | truncation probe |
| budget granted vs. computation spent | fleet "won" by overspending 18–26 % | account where it is spent |
| tuning under one metric, evaluating under another | *deflated* a real result to nothing | tune on the evaluation metric |

The fifth is the one to remember: leaks do not consistently favour the pet
mechanism, which is why the guards must be mechanical rather than a matter of
vigilance.

## Repository layout

```
paper/                  the manuscript (IEEEtran) + figure generator + number checker
bemporad-2009/comment/  Comment on Bemporad & Munoz de la Pena (2009):
                        the sign correction, its consequences and an exact
                        repair, with its own number checker
bemporad-2009/evidencia/ scripts and JSON that generate every number in the
                        Comment (independent of the project's own MPC)
code/python/ghi/        reference implementation (plant, terminal ingredients,
                        MOMPC as a dense QP, admissible weight set, governors,
                        graph field, transport & fleet arenas, audit suite)
code/python/scripts/    every experiment: run_*.py
code/matlab/            native MATLAB port (no YALMIP) + cross-validation
code/crossval/          the 60-case reference MATLAB must reproduce
code/results/           stored outputs (*.json, *.log) that figures/tables read
```

## Requirements

Python ≥ 3.11 with `numpy scipy cvxpy osqp clarabel matplotlib`
(tested: numpy 2.4.6, scipy 1.17.1, osqp 1.1.3, matplotlib 3.11.0).
MATLAB R2026a with Optimization, Control System and Statistics toolboxes for
the port. A LaTeX distribution with `IEEEtran` and `elsarticle` for the
documents.

## Reproducing

```bash
# 1. the audit suite: 21 checks over 19 assertions, every past defect encoded
cd code/python
python -m ghi.audit --quick

# 2. the headline experiments (each writes code/results/*.json + .log)
python scripts/run_transport_affine.py     # affine vs proportional vs field  (~1 h)
python scripts/run_fleet3.py               # fleet vs fixed-horizon frontier, tuned under the evaluation metric (~2 h)
python scripts/run_fleet_decomp.py         # allocation vs timing, one arm each
python scripts/holm_fleet3.py              # Holm-adjusted p for both
python scripts/run_alphaN.py               # single-loop horizon governance   (~1 h)

# 3. supporting measurements
python scripts/run_scale_invariance2.py    # Proposition 1, interior optima
python scripts/run_reach_law.py            # quantization-limited reach
python scripts/run_transport_viability2.py # storage viability filters

# 4. figures and the manuscript
cd ../../paper
python make_figures.py                     # all figures from stored results
latexmk -pdf main.tex
python _check_numbers.py                   # every number in the text vs. its JSON
```

Stored results are committed, so steps 2–3 are optional: figures, tables and
the number checker run directly from `code/results/`. Tuning and evaluation
seeds are disjoint in every experiment.

Two environment notes, learned the hard way and documented in the scripts:
matplotlib's `usetex` path drops the minus glyph on some MiKTeX installations
(figures use `mathtext`/STIX instead), and LaTeX-flavoured labels in Python
must be raw strings or `\t`/`\r` become control characters.

## What is deliberately not here

Third-party full-text sources under `bemporad-2009/fuentes/` are excluded for
copyright reasons. The Comment and the scripts and result files that back it
are tracked.
Internal working notes in Spanish and a superseded prototype are excluded as
well — they document the path, not the results.

## License

MIT — see [LICENSE](LICENSE).
