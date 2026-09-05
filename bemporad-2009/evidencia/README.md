# Evidencia del Comment

Dos scripts, independientes del MPC del proyecto, y sus tres ficheros de
resultados. Todo lo que el Comment imprime sale de aquí, y
`../comment/_check_numbers.py` lo comprueba en los dos sentidos.

| fichero | lo genera | qué contiene |
|---|---|---|
| `bemporad_lemma4.json` | `run_bemporad_lemma4.py` | el Lema 4 y sus consecuencias sobre el problema (13): 200 mp-LP aleatorios × 6 cuerdas, pesos **en el simplex** {μ≥0, Σμ≤1}; T1 concavidad en μ, T2 convexidad en x, T3 mínimo-de-piezas, T4/T5 conjunto admisible (l=1) |
| `bemporad_remedy.json` | `run_bemporad_remedy.py` | el remedio exacto por unión de piezas sobre el simplex (l=2): 120 instancias × 4 niveles de contracción tomados como cuantiles 15/35/55/75 de V*; desglose por cuantil |
| `concavidad.json` | `code/python/scripts/run_concavity.py` (artículo principal) | la familia **cuadrática** del artículo principal: 300 cuerdas, no convexidad del conjunto admisible sobre 20 estados × 25 niveles |

## Tres tasas de no convexidad que no son la misma cifra

Un lector que compare los ficheros encontrará tres porcentajes distintos para
"el conjunto admisible es no convexo". No se contradicen: miden familias y
protocolos de niveles distintos, y el Comment cita solo el segundo.

| tasa | fichero | familia | niveles J_a |
|---|---|---|---|
| 175 / 2280 = 7.7 % | `bemporad_lemma4.json`, T4_T5 | mp-LP con **un** peso (l=1), rejilla [0,1] | 38 niveles equiespaciados entre el mínimo y el máximo de V* |
| 232 / 480 = 48.3 % | `bemporad_remedy.json`, T4 | mp-LP con **dos** pesos (l=2), rejilla del simplex paso 1/40 | 4 cuantiles de V* (15/35/55/75) |
| 103 / 500 = 20.6 % | `concavidad.json` | la familia cuadrática del artículo principal | 25 niveles por estado |

La tasa depende de cuántos pesos hay (con uno, el conjunto es una unión de
intervalos y muchos niveles caen en una sola pieza), de dónde se ponen los
niveles (los cuantiles centrales cruzan más piezas que los extremos) y de la
familia. Lo invariante, y lo único que el Comment afirma, es que la tasa **no es
cero**: con un solo nivel no convexo el LP (17) ya no es una reformulación.

## Historia

- 6-ago-2026: primera versión; los pesos se muestreaban en el cubo [0,1]^l y
  el barrido escalar en [0,4], fuera del simplex del artículo.
- 20-ago: el script escribía en tres rutas distintas; se unifica aquí.
- 21-ago: muestreo dentro del simplex (T1 pasa de 791 a 746 estrictas,
  T3 de 5.15 a 1.34; las conclusiones no cambian) y desglose por cuantil en
  el remedio, tras una revisión adversarial que pidió ambas cosas.
