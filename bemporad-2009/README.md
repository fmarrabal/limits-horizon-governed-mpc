# Hilo Bemporad 2009 — el Comment ES el registro primario (20-ago-2026)

**Decisión del 20-ago-2026 (revierte la del 17): el Comment SÍ se envía a
Automatica.** Es el único vehículo que queda enlazado desde la ficha del
artículo de 2009 en ScienceDirect, Scopus y WoS; una sección dentro del
paper de control no aparece por ninguna de esas rutas. En consecuencia
`paper/main.tex` ya NO lleva la corrección como sección: usa el hecho que
necesita y cita el Comment, de modo que no hay contenido duplicado.

El Comment se reescribió el 20-ago para poder enviarse: hipótesis del lema
bien colgadas, claim de seguridad acotado a la rama afín a trozos, el
argumento del Teorema 7 rehecho sobre el hessiano en vez de caracterizar el
alcance de Mangasarian & Rosen, evidencia migrada al problema (13) del
propio artículo comentado, y un **remedio exacto** nuevo (unión de
card(I(x)) LP) con su verificación. `comment/_check_numbers.py` comprueba
cada cifra contra `evidencia/`.

**El correo se envió el 7-ago-2026 (10:36 UTC) y sigue sin respuesta.**
Con el Comment en marcha, la copia del envío deja de ser burocracia: es la
traza de prioridad y la respuesta a la pregunta que el editor hará.

Este directorio es autosuficiente: si vuelves dentro de tres meses sin recordar nada, empieza por aquí y en diez minutos estás donde lo dejaste.

---

## El hallazgo, en cinco líneas

Bemporad y Muñoz de la Peña, *Multiobjective model predictive control*, **Automatica 45(12):2823–2830, 2009**, afirma en su **Lema 4** que la función de valor V\*(μ,x) es *«convex and piecewise affine w.r.t. μ»*. Es **cóncava**. Su problema (13) es

```
    min_z  (c′₀ + μ′C_μ) z      s.a.   G z ≤ b + S x
```

y el factible no depende de μ, así que V\* es el mínimo puntual de funciones afines en μ. Es la asimetría clásica: la función de valor de un LP es **convexa en el lado derecho** y **cóncava en los coeficientes del coste**. En x aciertan; el signo falla solo en μ.

**El error nace en una cláusula de seis palabras** del *companion* del ECC 2009 (`fuentes/ecc2009_companion_LA-PRUEBA.pdf`, p. 2406), donde está la prueba que *Automatica* no incluye:

> *«Convexity and continuity of V\* with respect to x for any given μ follow by the properties of the multiparametric LP (12) [Gal, p. 180], and, **by duality, the same properties hold with respect to μ**.»*

La dualidad **sí** intercambia lado derecho y coste, pero convierte el mínimo primal en un máximo dual: lo que se transfiere es **concavidad**.

---

## Lo que está cerrado, y no hay que volver a hacer

| | estado |
|---|---|
| El Lema 4 dice lo que creemos | **Verificado** sobre el PDF original |
| μ está en el coste y el factible no depende de μ | **Verificado**, palabras suyas |
| El Teorema 6 usa el *máximo* donde va el *mínimo* | **Verificado**, literal |
| Dónde nace el error | **Localizado**: cláusula «by duality» del companion ECC |
| Verificación numérica sobre **su** formulación (13) | **1200 tests en μ: 791 por encima de la cuerda, 0 por debajo.** En x sale convexa, como dicen ellos |
| ¿Existe fe de erratas? | **No.** Crossref devuelve `relation: {}` |
| ¿Lo notó algún citante? | **Nadie**, en ~187 registros. «concave» aparece 1 vez y es irrelevante. **Cero** trabajos cocitan Bemporad 2009 y Mangasarian & Rosen 1964 |
| Laguna de Charitopoulos | **Cerrada en negativo.** Raíz «concav» = 0 veces en ~360 000 caracteres suyos. La cocitación con Gal era un artefacto |

**Se puede escribir «ningún trabajo citante ha señalado el error» sin salvedad.**

## Las dos armas que hacen la nota

1. **Contradicción interna ya impresa.** Tras el Teorema 6: V\* *«may not be a jointly convex function of (μ,x)»*. En la prueba del Teorema 7, tres páginas después: *«is a jointly convex function of (μ,x)»*. Dos frases del mismo artículo. **Abrir con esto**: no hay que afirmar nada, basta preguntar cuál es la buena.
2. **El manual del propio campo.** Pistikopoulos, Diangelakis y Oberdieck, *Multi-parametric Optimization and Control* (Wiley 2020), §1.1.1.2, propiedad (5), **página 3**: *«their pointwise infimum … is a **concave** function on C»*. Y la *Encyclopedia of Optimization* (Springer 2023, mismos autores): la convexidad de z\*(θ) *«no está garantizada por los términos cruzados θᵀHx»*, con la entrada declarada restringida al caso *right-hand side parameterized*.

## Por qué nadie lo vio en 17 años

Toda la comunidad multiparamétrica que ha tocado el multiobjetivo va por **ε-constraint**, o sea con los parámetros en el **lado derecho**. Bemporad y Muñoz de la Peña son **los únicos que usaron la suma ponderada** — la única parametrización donde el peso entra en el **coste** y la única donde la curvatura cambia de signo. Nadie rederivó su Lema 4 porque nadie trabajaba en su parametrización.

Y hay un deslizamiento que lo hace invisible: la literatura reporta que **las regiones críticas son convexas** en el caso de parámetros en el objetivo, y nunca el signo de la curvatura de z. Es fácil confundir «regiones críticas convexas» con «función de valor convexa».

---

## Qué hacer cuando contesten

**Si proponen corrigendum conjunto** → acéptalo sin regatear. Vale más que la nota en solitario y abre la puerta al artículo del GHI.

**Si dicen «tienes razón, publícalo tú»** → sale [`NOTA-TECNICA-concavidad.md`](NOTA-TECNICA-concavidad.md), que ya está escrita. Pide permiso explícito para mencionar el intercambio. Antes de enviar, las dos únicas cosas pendientes:
- releer el artículo **entero**, no solo los tres enunciados, por si hay una hipótesis en el *Assumption 1* o en el cambio de variables α₀ = 1/(1+Σμ) que se nos haya escapado;
- *(opcional, baja probabilidad)* las dos páginas del §9.6 del libro Wiley 2020 sobre mp-MOO, pp. 173–174, que es la única pieza no leída.

**Si refutan** → escúchalo con atención antes de defender nada. Habrás ahorrado un ridículo público a cambio de un correo, que es exactamente el motivo de escribir antes de publicar.

**Si no contestan** → tres o cuatro semanas, recordatorio de dos líneas, y si sigue el silencio publica igual.

---

## Qué hay en cada carpeta

**Raíz**
- `NOTA-TECNICA-concavidad.md` — la nota completa, lista salvo la relectura pendiente
- `BORRADOR-correo-bemporad.md` — el borrador del correo, con el plan para cada rama de respuesta

**`fuentes/`** — todo el material primario, ya extraído
- `ecc2009_companion_LA-PRUEBA.pdf` — **la pieza clave**: aquí está la prueba ausente y la cláusula del error
- `automatica2009_texto-completo.txt` — el artículo de *Automatica* en texto
- `extracto_lema4_teorema6_teorema7.txt` y `extracto_ecuacion13_problema6.txt` — los pasajes relevantes ya aislados
- los tres Charitopoulos leídos para cerrar la laguna

**`evidencia/`**
- `run_bemporad_lemma4.py` — la verificación sobre **su** formulación (13), sin depender de nuestro MPC. Cinco tests
- los `.json` con los resultados

**`dossiers/`** — los informes crudos de los barridos, sin editar

---

## Pendiente de que lo añadas tú

- [ ] **Copia de lo que realmente enviaste**, con fecha y destinatarios. El borrador es el borrador, no el envío. Si esto acaba en corrigendum o en una cuestión de prioridad, esa copia deja de ser burocracia.

---

*Para retomar la conversación: basta decir «recuperemos lo de Bemporad». Está en memoria.*

## `comment/` — el borrador del Comment a Automatica (9 ago 2026)

`comment/comment.tex` (elsarticle, compila con dos pases de pdflatex) es el borrador listo del *Comment on "Multiobjective model predictive control"*. Reencuadrado sobre **su** problema (13), que es donde vive el Lema 4: V* es cóncava **y afín a trozos** en µ (ellos aciertan lo segundo, yerran lo primero). Consecuencias separadas para el Teorema 6 (el LP (17) evalúa V* como máximo de las piezas siendo el mínimo ⇒ aproximación interior convexa: segura, no equivalente) y el Teorema 7 (Mangasarian & Rosen 1964 es para lado derecho ⇒ (21) no es convexo; sin red de seguridad, se propone certificar a posteriori). El contacto previo con los autores queda declarado en el propio texto.
