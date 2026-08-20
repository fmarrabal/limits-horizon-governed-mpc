# Borrador de correo a A. Bemporad y D. Muñoz de la Peña

> **Este borrador NO se ha enviado.** Léelo, edítalo y mándalo tú.
>
> Tres decisiones que he tomado y puedes revertir:
> 1. **Está planteado como una pregunta, no como una corrección.** No porque haya duda sobre la matemática —está verificada por cuatro vías— sino porque (a) podrías estar leyendo mal su notación, (b) invita a responder en lugar de a defenderse, y (c) si tienes razón, es mucho mejor que lo digan ellos.
> 2. **Es corto.** Un correo largo con una acusación larga no se contesta. Este cabe en una pantalla y tiene un adjunto que hace el trabajo.
> 3. **Va en inglés y a los dos.** Si prefieres escribir primero a Muñoz de la Peña en español —está en Sevilla, es la misma comunidad que tus directores— hay una variante corta al final.
>
> **Antes de enviar:** decide si adjuntas el script `run_bemporad_lemma4.py` o solo lo ofreces. Yo lo adjuntaría: hace el argumento sin que tengan que fiarse de nadie.

---

## Datos de contacto (verifícalos antes de enviar)

- **Alberto Bemporad** — IMT School for Advanced Studies Lucca. Página: `cse.lab.imtlucca.it/~bemporad/`
- **David Muñoz de la Peña** — Dpto. de Ingeniería de Sistemas y Automática, Universidad de Sevilla.

En el artículo (2009) las afiliaciones eran Universidad de Siena e Universidad de Sevilla; Bemporad se trasladó después a IMT Lucca.

---

## Asunto

`Question about Lemma 4 in "Multiobjective model predictive control" (Automatica, 2009)`

## Cuerpo

Dear Professor Bemporad, dear Professor Muñoz de la Peña,

I am a researcher at the University of Almería. Back in 2012 I reproduced the algorithm of your paper *Multiobjective model predictive control* (Automatica 45(12):2823–2830) as part of my master's thesis, and I am now building on it again. While re-implementing it from scratch I keep running into something I cannot reconcile with Lemma 4, and I would rather ask you than assume I am misreading your notation.

Lemma 4 states that the value function of problem (13) is *"convex and piecewise affine w.r.t. μ for any given x"*. Problem (13) is

```
    min_z  (c'_0 + μ' C_μ) z      s.t.   G z ≤ b + S x
```

Here the feasible set does not depend on μ, and for every fixed feasible z the objective is affine in μ. So V*(·, x) is a pointwise minimum of affine functions of μ, which would make it **concave** in μ rather than convex — the standard asymmetry that the value function of an LP is convex in the right-hand side and concave in the cost coefficients. Your statement about x (which enters the rhs) matches that; it is only the μ half that I cannot make come out convex.

I went to the ECC 2009 companion paper for the proof, and the step I cannot follow is a single clause on p. 2406: *"Convexity and continuity of V\* with respect to x for any given μ follow by the properties of the multiparametric LP (12) [Gal, p. 180], and, by duality, the same properties hold with respect to μ."* Duality does exchange right-hand-side and cost parameters, but it also turns the primal minimum into a dual maximum, so what transfers from "convex in the rhs" would be "concave in the cost". Everything else that clause carries over — continuity, piecewise affinity, the polyhedral structure of the critical regions — is blind to the sign and seems to me unaffected.

There is also something in the paper itself that made me hesitate before writing to you. The discussion following Theorem 6 states that V*(μ, x) *"may not be a jointly convex function of (μ, x)"*, while the proof of Theorem 7 concludes, via Mangasarian and Rosen (1964), that V*_μ(μ, x) *"is a jointly convex function of (μ, x)"*. I could not reconcile the two, and my reading of the Hessian of ½z′Hz + μ′C_μz with respect to (z, μ), namely [[H, C_μ′], [C_μ, 0]], is that it is positive semidefinite only if C_μ = 0.

I have checked this numerically on randomly generated instances with exactly the structure of (13): over 1200 chord tests in μ, none fell below the chord and 791 fell strictly above it; and V*(μ) matches the minimum of the affine pieces to 7·10⁻¹⁵ while differing from their maximum by O(1). The same tests in x come out convex, as your lemma says.

If this reading is right, I believe the practical consequence is mild for the LP branch: the constraint imposed in problem (17) applies to *all* the affine pieces indexed by I(x), which is stronger than V*(x, μ) ≤ J_a, so (17) is a conservative inner restriction rather than an equivalent reformulation. The stability argument is unaffected — the algorithm never accepts a weight that violates the certificate — but the equivalence with (6), and hence the optimality of α*, would not hold as stated. In some of my random instances the restriction is quite tight; I found one where 128 of 161 sampled weights were admissible for V* ≤ J_a while none satisfied (17).

I may well be missing something, and I would much rather be told so than publish a misunderstanding. I am happy to share the verification script (about a hundred lines of Python, no dependencies beyond SciPy) so you can check it directly.

With my best regards and my thanks for a paper I have found genuinely useful over the years,

Francisco Manuel Arrabal Campos
Universidad de Almería
fmarrabal@ual.es

---

## Variante corta, en español, solo para Muñoz de la Peña

> Úsala si prefieres tantear primero por la vía cercana. Si responde confirmando, el correo a Bemporad se escribe solo — y con un aliado dentro.

Estimado David,

Te escribo desde la Universidad de Almería. En 2012 reproduje vuestro algoritmo de *Multiobjective model predictive control* (Automatica 2009) en mi trabajo fin de máster, dirigido por José Luis Guzmán y José Domingo Álvarez, y ahora estoy construyendo sobre él otra vez.

Al reimplementarlo desde cero me sale sistemáticamente algo que no consigo cuadrar con el Lema 4. En vuestro problema (13),

```
    min_z  (c'_0 + μ' C_μ) z      s.a.   G z ≤ b + S x
```

el factible no depende de μ y, para cada z factible, el objetivo es afín en μ. Eso hace que V*(·,x) sea un mínimo puntual de funciones afines, es decir **cóncava** en μ y no convexa — la asimetría habitual de que la función de valor de un LP es convexa en el lado derecho y cóncava en los coeficientes del coste. En x, que va en el lado derecho, me sale convexa igual que decís vosotros; es solo la mitad de μ la que no me cuadra.

Lo he comprobado numéricamente sobre instancias aleatorias con la estructura exacta de (13): 1200 tests de cuerda en μ, ninguno por debajo de la cuerda y 791 estrictamente por encima.

¿Estoy leyendo mal vuestra notación? La prueba del Lema 4 remite al artículo del ECC 2009, que no he conseguido; si la respuesta está ahí, te agradecería el enlace.

Un saludo cordial, y gracias por un trabajo del que llevo años aprendiendo,

Francisco M. Arrabal Campos
Universidad de Almería

---

## Si responden confirmando

Tres posibilidades, y conviene tenerlas pensadas:

**Proponen un corrigendum conjunto.** Es el mejor resultado y hay que aceptarlo. Un corrigendum en *Automatica* coescrito con Bemporad vale más que una nota tuya en solitario, y te abre la puerta para el artículo del GHI.

**Dicen «sí, tienes razón, publícalo tú».** Entonces sale la nota técnica de `NOTA-TECNICA-concavidad.md`, citándolos con elegancia y agradeciéndoles la confirmación por correo. Pide permiso explícito para mencionar el intercambio.

**No contestan.** Espera tres o cuatro semanas, manda un recordatorio de dos líneas, y si sigue el silencio publica la nota igualmente. Habrás hecho lo correcto y tendrás constancia escrita de que lo intentaste, que es exactamente lo que un editor querrá saber.

## Si responden refutando

Escúchalo con atención antes de defender nada: la posibilidad de que su μ sea otra cosa, o de que el companion del ECC defina (13) de otro modo, sigue viva. Si te refutan, has ahorrado un ridículo público a cambio de un correo. Ese es exactamente el motivo de escribir antes de publicar.
