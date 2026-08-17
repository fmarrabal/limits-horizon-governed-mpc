# Paper: *When Does Adapting the Meta-Parameter Pay?*

**Encuadre final (revisado el 17-ago-2026): un solo manuscrito.** El
*Comment* a Automatica NO se enviará; la corrección al Lema 4 de Bemporad &
Muñoz de la Peña (2009) va como **Sección III** de este paper
(`\label{sec:sign}`), y `../bemporad-2009/comment/comment.tex` se conserva
como versión extendida de registro, sin envío previsto a ninguna parte.
El manuscrito: dos gobernadores mínimos que sí pagan, los límites
estructurales, la corrección de signo, y la metodología de las cinco fugas.

```bash
cd paper
python make_figures.py        # regenera las 6 figuras desde code/results/
latexmk -pdf main.tex         # compila -> main.pdf (10 páginas, 0 desbordes)
python _check_numbers.py      # contrasta cada cifra del texto con su JSON
```

## Los dos resultados de titular

**Retardo afín (transporte).** En una cadena con perturbación que viaja, lo que importa es que cada nodo reciba el aviso con la **misma antelación** — y eso exige un término independiente en el retardo (`k_i = max(0, τi − ℓ)`). El afín de 4 parámetros: +12.4 % sobre la mejor detección local, 90.9 % del techo acausal, estadísticamente en el techo. Bate al proporcional (t=+17.87 a igual densidad; t=+5.40 contra su mejor sintonía), al campo de onda (t=+4.64), y conformar no aporta nada resoluble. Fuente: `transport_affine.json` (30 semillas, guarda de causalidad, normalización causal, densidades igualadas, óptimos interiores).

**Detector de certificado (flota).** El gobernador óptimo del horizonte pide N_max **solo cuando α_t ≤ 0** (el certificado roto) y decae con τ≈30. En 8 lazos con presupuesto duro: −5/−6 % contra la mejor frontera fija **a cómputo gastado**, con ambas correlaciones. Descomposición limpia: lo temporal paga siempre; el reparto entre lazos solo con perturbaciones independientes (hasta +8.6 %). Fuentes: `fleet3.json`, `fleet_decomp.json`.

## La sección metodológica: las cinco fugas

| fuga | qué invirtió | guarda |
|---|---|---|
| canal roto (ganancia DC nula) | el campo perdía sin poder competir | sonda de escalón sostenido (A18) |
| desplazamiento acausal | los dos "ganadores" leían el futuro | sonda de un pulso, dentro del barrido |
| normalización acausal | escala fijada con el futuro (0.42 antes del evento) | sonda de truncamiento |
| presupuesto vs gasto | la flota "ganaba" gastando +18/26 % | contabilizar donde se gasta |
| sintonía ≠ métrica de evaluación | **desinfló** el resultado a nada | sintonizar con la métrica de evaluación |

La quinta es la clave de la moraleja: las fugas no favorecen sistemáticamente al favorito — favorecen lo que el desajuste favorezca. Por eso la guarda tiene que ser mecánica.

## Historial y notas de reproducibilidad

- `main_v1_backup.tex` — primera versión (pre-auditorías). `main_v2_backup.tex` — segunda (encuadre antiguo, números pre-3ª ronda).
- `refs.bib` — 31 entradas (23 citadas); 27 verificadas contra fuente primaria o corroboración múltiple, y 4 añadidas el 17-ago (schechter1987, gal1972, miettinen1999, marler2004) con DOI, **pendientes de pasar al ledger** `refs_verificadas.bib` tras verificación contra fuente.
- Las figuras **no** usan `usetex`: el glifo del signo menos no se dibuja en esta MiKTeX por esa ruta (comprobado con fonttype 42/3 y cm/lmodern/mathptmx). Se usa `mathtext` con STIX.
- Ojo con `\t`/`\r` dentro de cadenas no-raw de Python: dos figuras salieron con TAB/CR embebidos por eso. Etiquetas con LaTeX siempre en raw strings.
- Semillas de sintonía y evaluación disjuntas en todos los experimentos; la suite de auditoría (21 comprobaciones sobre 18 aserciones) pasa entera.
