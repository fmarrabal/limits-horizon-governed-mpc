# Nota técnica (borrador)

**On the curvature of the value function with respect to the weight vector in multiobjective MPC**

*Borrador de trabajo. No enviar sin respuesta de los autores originales.*

> **Actualización tras la verificación.** El companion del ECC 2009 —donde está la prueba ausente— **se ha conseguido**: está alojado en abierto por el propio Bemporad, en `cse.lab.imtlucca.it/~bemporad/publications/papers/ecc09-mompc.pdf` (DOI 10.23919/ECC.2009.7074765, pp. 2402–2407). Allí el Lema 4 es el **Lema 1**, con enunciado literalmente idéntico, y la justificación de la dirección μ son **seis palabras**:
>
> > *«Convexity and continuity of V\* with respect to x for any given μ follow by the properties of the multiparametric LP (12) [Gal, p. 180], and, **by duality, the same properties hold with respect to μ**.»*
>
> Ahí está el desliz, localizado: la dualidad **sí** traslada la estructura del lado derecho a la del coste, pero intercambia mínimo y máximo, de modo que lo que se transfiere es **concavidad**. Todo lo demás que esa cláusula arrastra —continuidad, afinidad a trozos, estructura poliédrica de las regiones críticas— es ciego al signo y sobrevive intacto.
>
> Y explica por qué ha durado diecisiete años: **la prueba no está en la revista que todo el mundo cita.**
>
> **No existe fe de erratas** (Crossref devuelve `relation: {}`; el único *Corrigendum* en la lista de Bemporad corresponde al artículo de 2002, no a éste). Y en ~160 trabajos citantes **nadie lo ha señalado**: la palabra «concave» aparece una sola vez en todo el corpus citante y se refiere al frente de Pareto, no a la función de valor. Cero trabajos cocitan Bemporad 2009 y Mangasarian & Rosen 1964 — el paso del Teorema 7 no lo ha auditado nadie.

---

## Nota previa sobre qué es esto y qué no

Esto **no** es un artículo sobre el Gobernador Homeostático Inercial. Es una nota corta y autocontenida sobre un punto técnico de un artículo muy citado. Sale antes que el trabajo del GHI por tres razones:

- es **verificable en una tarde** por cualquiera, con cien líneas de código;
- si se confirma, **da credibilidad** a todo lo que venga después de la misma mano;
- y la geometría que establece —el conjunto admisible de pesos **no es convexo**— es justamente el argumento técnico que después motiva un generador de pesos con dinámica continua en lugar de una optimización estática por muestra.

Si los autores proponen un corrigendum conjunto, **acéptalo**: vale más que esta nota en solitario.

---

## Abstract (borrador)

Bemporad and Muñoz de la Peña (*Automatica*, 2009) propose a multiobjective MPC scheme in which the scalarization weight is selected online subject to a contractive constraint on the optimal value function, which is what confers closed-loop stability. Their Lemma 4 states that this value function is convex and piecewise affine with respect to the weight for any fixed state. We observe that, since the weight enters the cost of a linear program whose feasible set is independent of it, the value function is a pointwise minimum of affine functions of the weight and is therefore **concave**, not convex. The claim regarding the state, which enters the right-hand side of the constraints, is correct; the asymmetry is the standard one for parametric linear programs. We report the consequences, which differ between the piecewise-affine and the quadratic branches of the paper, and we note that the stability guarantee is unaffected in the former: the linear program actually solved is a conservative inner restriction of the intended feasible set, so the algorithm never accepts a weight violating the certificate, but it is not an equivalent reformulation and the reported optimality of the selected weight does not follow.

---

## 1. La afirmación en cuestión

El esquema selecciona el peso resolviendo

> α*(x, α_d, J_a) = arg min_α f(α − α_d)  s.a.  V*(x, α) ≤ J_a,  Σα_i = 1,  α_i ≥ 0     (6)

con V*(x, α) = α′J(U*(x,α), x). La restricción (6b) es la que hereda la estabilidad en lazo cerrado.

Para el caso de costes convexos afines a trozos, el problema escalarizado se escribe (ec. 13) como

> min_z (c′₀ + μ′C_μ) z    s.a.   G z ≤ b + S x

y sobre él se enuncia:

> **Lemma 4.** *Consider the multiparametric linear problem (13) with parameters μ ∈ ℝˡ in the cost function and x ∈ ℝⁿ in the rhs of the constraints. Then […] the value function V\*: F\* → ℝ is continuous w.r.t. (μ, x), **convex** and piecewise affine w.r.t. μ for any given x and w.r.t. x for any given μ.*

## 1 bis. Cómo abrir la nota: con su propia contradicción, no con la mía

El artículo se contradice **a sí mismo**, y eso es un argumento más fuerte que cualquier análisis externo, porque un árbitro no puede descartarlo como interpretación del autor de la nota. Son dos frases del mismo artículo:

- Tras el **Teorema 6** (rama afín a trozos): *«Problem (17) in general is not jointly convex […] due to the fact that V\*(μ, x) **may not be a jointly convex function** of (μ, x).»*
- En la prueba del **Teorema 7**, tres páginas después (rama cuadrática): *«By the results of Mangasarian and Rosen (1964) […] it follows that V\*_μ(μ, x) **is a jointly convex function** of (μ, x).»*

La nota debe **abrir con eso** y presentar el análisis de la hessiana como la explicación de **cuál de las dos frases es la correcta**, no como la afirmación que hay que defender. Nótese además que la salvedad honesta ya está en el companion del ECC, mientras que la afirmación de convexidad conjunta de la rama cuadrática es **nueva en Automatica**: el congreso solo trata el caso LP, así que ese paso no tiene antecedente ni prueba en ningún sitio.

## 1 ter. El hecho ya está enunciado, correctamente, en el manual de referencia del propio campo

Ésta es la forma más fuerte de presentar la corrección, y no requiere que el autor de la nota afirme nada por su cuenta.

**Pistikopoulos, Diangelakis y Oberdieck, *Multi-parametric Optimization and Control* (Wiley, 2020), §1.1.1.2, propiedad (5), página 3:**

> *«Let f₁(x), …, fₙ(x) be concave functions defined on a convex subset C. If these functions are bounded from below, their pointwise infimum f(x) = min{f₁(x), …, fₙ(x)} is a **concave** function on C.»*

Ése es exactamente el lema que da V\*(μ,x) cóncava en μ, en la **página 3** del libro de referencia de la programación multiparamétrica. No es el autor de esta nota contra Bemporad: es el manual del campo contra Bemporad.

Y para la rama cuadrática, **Kenefake, Pappas, Diangelakis, Avraamidou, Oberdieck y Pistikopoulos, *Encyclopedia of Optimization* (Springer, 2023)**, entrada «Multi-parametric Linear and Quadratic Programming»:

> *«The optimal objective function z\*(θ) for the mpLP program is continuous, convex, and piece-wise affine, **contrasting the mpQP case where the convexity of z\*(θ) is not guaranteed due to the existence of the cross terms introduced by θᵀHx**.»*
>
> *«In the absence of a θᵀHx term in the objective, z\*(θ) is convex.»*

Es la contradicción directa y autorizada de la premisa del Teorema 7: la escuela dice que la convexidad **no** está garantizada precisamente por el término cruzado parámetro–variable, que es exactamente el bloque C_μ que rompe la hessiana. Y esa misma entrada se declara restringida al caso *«right-hand side (RHS) parameterized»*: la escuela **sabe** que fuera del lado derecho el resultado no vale, y por eso se autolimita.

Conviene citar además **Fiacco y Kyparisis, «Convexity and concavity properties of the optimal value function in parametric nonlinear programming», *JOTA* 48(1):95–126, 1986**: es la referencia canónica del hecho y demuestra que es de dominio público desde hace cuarenta años. El fallo de Bemporad es un lapsus aislado, no una confusión del campo — y decirlo así es a la vez exacto y elegante.

## 1 quater. Por qué nadie lo vio en diecisiete años

Hay una explicación estructural, y vale un párrafo de la discusión porque es mucho más informativa que «nadie lo leyó»:

**Toda la comunidad multiparamétrica que ha tocado el multiobjetivo ha ido por el método ε-*constraint*, es decir con los parámetros en el LADO DERECHO.** Oberdieck y Pistikopoulos (CACE 85, 2016) plantean explícitamente las dos escalarizaciones y luego persiguen **solo** la ε-*constraint*; Charitopoulos y Dua (2016) igual; Pappas et al. (*I&EC Res* 60, 2021) lo dicen textualmente: *«Both of the latter approaches utilized the ϵ-constraint approach»*.

**Bemporad y Muñoz de la Peña son los únicos que usaron la suma ponderada** — que es la única parametrización en la que el peso entra en el **coste**, y por tanto la única en la que la curvatura cambia de signo. Nadie rederivó su Lema 4 porque nadie trabajaba en su parametrización. El error quedó en un callejón sin salida bibliográfico.

**Y hay un deslizamiento conceptual que ayuda a que pase inadvertido incluso para especialistas.** Cuando la literatura trata el caso de parámetros en los coeficientes del objetivo, reporta que **las regiones críticas son convexas** (Gass y Saaty, repetido literalmente por Charitopoulos: *«proved that the CRs of such case are convex»*) y **nunca** reporta el signo de la curvatura de z. Un lector rápido confunde «regiones críticas convexas» con «función de valor convexa». Señalarlo es un buen diagnóstico de por qué el error es invisible.

## 2. La observación

En (13) el conjunto factible `{z : Gz ≤ b + Sx}` **no depende de μ**, y para cada z factible el objetivo `(c₀ + C′_μ μ)′z` es una función **afín de μ**. Por tanto

```
V*(μ, x) = min_{z ∈ Z(x)} [ afín en μ ]
```

es el **ínfimo puntual de una familia de funciones afines**, y en consecuencia es **cóncavo** en μ. Es la asimetría clásica de la programación paramétrica lineal: la función de valor es **convexa en el lado derecho** y **cóncava en los coeficientes del coste**. La parte del Lema 4 relativa a x es correcta precisamente porque x entra en el lado derecho; el signo se invierte solo en la mitad de μ.

Dos consecuencias inmediatas del mismo hecho:

**(a) El Teorema 6 usa el máximo donde debería ir el mínimo.** Su demostración concluye que V*(μ,x) *"can be evaluated as the maximum of the affine functions {(φᵢx + γᵢ)′(c′₀ + C′_μ μ)}"*, invocando el resultado de Schechter (1987) para funciones convexas afines a trozos. Para una función cóncava afín a trozos la representación es el **mínimo** de sus piezas.

**(b) La rama cuadrática hereda el problema.** El Teorema 7 parte de (ec. 20)

> V*_μ(μ,x) = min_z ½z′Hz + x′F′z + ½x′Yx + **μ′C_μ z**   s.a.  Gz ≤ b + Sx

y concluye, invocando Mangasarian y Rosen (1964), que V*_μ es conjuntamente convexa en (μ, x). Ese resultado exige convexidad conjunta del objetivo en (z, μ), cuya hessiana es

```
[ H     C′_μ ]
[ C_μ    0   ]
```

que es semidefinida positiva únicamente si C_μ = 0. Un segundo argumento independiente: los propios autores establecen V*(α,x) = V*_μ(μ,x)/(1+Σμ_i), de modo que V*_μ es la **transformada de perspectiva** de V*(·,x); y la perspectiva de una función cóncava es cóncava.

## 3. Verificación numérica

Sobre instancias generadas aleatoriamente con la estructura exacta de (13) (factible acotado, `d`=6, `l`∈{1,2}, `n`=2), resolviendo cada LP con un solver exacto:

| test | resultado |
|---|---|
| Cuerda en **μ** (parámetro del coste), 1200 tests | **791 estrictamente por encima de la cuerda, 0 por debajo.** Desviación media +3.16·10⁻¹ ⇒ cóncava |
| Cuerda en **x** (parámetro del rhs), 1200 tests | **320 por debajo, 0 por encima.** Desviación media −1.85·10⁻² ⇒ convexa, como enuncia el lema |
| V* frente a las piezas afines | `max\|V* − mín de las piezas\| = 7.1·10⁻¹⁵` frente a `max\|V* − máx de las piezas\| = 5.15` |

Y, de forma completamente independiente, sobre un MPC multiobjetivo con ingredientes terminales construidos por ecuación de Lyapunov (de modo que la desigualdad terminal se satisface con igualdad exacta): **300 tests de cuerda, 0 violaciones de concavidad, 300 de convexidad**, replicado en dos implementaciones independientes (Python y MATLAB) validadas entre sí elemento a elemento.

## 4. Consecuencias, que difieren entre las dos ramas

### 4.1 Rama afín a trozos: conservadora pero segura

El problema (17) que efectivamente se resuelve impone

> (φᵢx + γᵢ)′(c′₀ + C′_μ μ) ≤ J_a,  **∀i ∈ I(x)**

es decir, sobre **todas** las piezas afines. Como V* es el **mínimo** de esas piezas, exigirlas todas es **estrictamente más fuerte** que V* ≤ J_a. Por tanto:

- **el conjunto factible de (17) es un subconjunto del conjunto admisible real**;
- **el algoritmo sigue siendo seguro**: nunca acepta un peso que viole el certificado de estabilidad, luego el Teorema 3 se mantiene;
- pero **(17) no es una reformulación equivalente de (6)**, y la optimalidad de α* no se sigue;
- y el conjunto es polihédrico, lo que explica por qué (17) es un LP bien puesto pese a que el conjunto admisible real no sea convexo.

*Verificado:* en 2280 niveles de J_a barridos sobre 60 problemas aleatorios, (17) resultó ser un subconjunto estricto en el **96.6 %** de los casos y **en ninguno** aceptó un μ inadmisible. El grado de conservadurismo puede ser severo: se encontró una instancia con 128 pesos admisibles muestreados de los que (17) no admitía **ninguno**.

> **Consecuencia práctica que conviene señalar con cuidado.** Como α(t−1) satisface V*(x(t), α(t−1)) ≤ J_a(t) por construcción pero no necesariamente todas las piezas afines, **(17) puede resultar infactible en instantes en los que (6) sí lo es**. No es un fallo de seguridad, sino de disponibilidad. En el ejemplo numérico del artículo evidentemente no se manifiesta; que se manifieste o no depende de cuántas piezas estén activas en I(x) y de cuánto se separen.

### 4.2 Rama cuadrática: sin conservadurismo automático

El problema (21) impone `V*_μ(x(t), μ) ≤ J_a(t)(1 + Σμ_i)`, es decir **cóncava ≤ afín**, cuyo conjunto de subnivel no es convexo en general. El artículo lo describe como *"a convex programming problem with l variables, one convex constraint, and l non-negativity constraints"*. Aquí no hay restricción interior que rescate la situación: un algoritmo que suponga convexidad no tiene garantía de caracterizar correctamente el conjunto factible.

La estabilidad, no obstante, sigue asegurada para **cualquier punto factible** que se devuelva, porque el Teorema 3 solo necesita que la restricción se satisfaga, no que el punto sea óptimo.

## 5. Qué se propone

1. Sustituir «convex» por «concave» en el Lema 4 respecto de μ, manteniéndolo respecto de x.
2. Sustituir «maximum» por «minimum» en la demostración del Teorema 6, con la versión cóncava del resultado de Schechter.
3. Reenunciar el Teorema 6 como **restricción interior conservadora** de (6), no como equivalencia, señalando que la estabilidad se preserva.
4. En el Teorema 7, retirar la apelación a Mangasarian y Rosen y reenunciar (21) como problema de optimización sobre un conjunto no convexo, indicando qué garantías sobreviven.

Ninguna de las cuatro afecta al resultado central del artículo —el esquema es estabilizante— y las cuatro son locales.

## 6. Reproducibilidad

Todo el material está en `code/`: `scripts/run_bemporad_lemma4.py` (verificación directa sobre (13), sin dependencias más allá de SciPy) y `scripts/run_concavity.py` (verificación sobre el MPC completo), con implementaciones independientes en Python y MATLAB validadas entre sí a 2.7·10⁻¹² en la construcción y 4.9·10⁻¹⁰ en el valor óptimo de los QP.

---

## 7. Contra qué NO hay que compararse

Conviene delimitar la novedad para que un árbitro no la confunda con literatura vieja:

**Zheng (1997) es ortogonal, verificado.** Sus pesos varían con el **índice de etapa dentro del horizonte**, con perfil fijado *offline* e **idéntico en cada instante de lazo cerrado** — literal de Wang, Grochowski y Brdys (IFAC 2005): *«limited to varying weights only inside the prediction horizon and the weights between each new iterate time step remain the same»*. Quien fija μ offline nunca necesita saber si V\*(μ,x) es convexa o cóncava; quien lo **optimiza online**, sí. Por eso Zheng no se topa con el problema. *(Si hace falta una referencia moderna y auditable de «pesos crecientes por etapa estabilizan sin restricción terminal», la correcta es Alamir, Automatica 87:455–459, 2018, no Zheng: el de Zheng sigue sin ser accesible en texto completo.)*

**Un contraste que da fuerza y es históricamente exacto.** La familia Genceli–Nikolaou–Vuthandam obtiene estabilidad exigiendo la desigualdad sobre **todas** las etapas, que es estructuralmente la misma maniobra del Teorema 6 sobre todas las piezas afines. Allí es legítimo porque se declara como condición **suficiente y conservadora**. El defecto de Bemporad y Muñoz de la Peña en la rama LP no es que el algoritmo sea inseguro: es que presentan como **equivalencia** lo que la literatura previa presentaba honestamente como restricción conservadora.

---

## Lista de comprobación antes de enviar nada

- [x] ~~Leer el companion del ECC 2009~~ — **hecho**: el error nace ahí, en la cláusula «by duality».
- [x] ~~Comprobar que no existe fe de erratas~~ — **no existe**.
- [x] ~~Comprobar si alguno de los ~160 citantes ya lo señaló~~ — **nadie**.
- [x] ~~**Laguna de Charitopoulos**~~ — **CERRADA, y en negativo.** Ver abajo.
- [ ] Esperar la respuesta de los autores.
- [ ] Releer el artículo entero, no solo los tres enunciados: puede haber una hipótesis en el *Assumption 1* o en el cambio de variables que no se haya tenido en cuenta.

### La laguna de Charitopoulos, cerrada

Eran los dos únicos trabajos, de ~187 citantes, que cocitaban Bemporad 2009 y el libro de Gal — los únicos lectores que tuvieron a la vez el artículo y la fuente del hecho correcto. **Identificados, obtenidos y leídos. No dicen nada.**

- **Artículo:** V. M. Charitopoulos y V. Dua, *Applied Energy* **186**:539–548 (fascículo de enero de 2017; en línea mayo de 2016). **CC-BY, abierto** en `discovery.ucl.ac.uk/id/eprint/1508335`.
- **«Libro»:** no es el volumen sino su **capítulo 2**, «Parametric Optimisation: 65 years of developments and status quo», pp. 9–45, DOI 10.1007/978-3-030-38137-0_2. El volumen en sí no tiene referencias propias. La tesis original de UCL (eprint 10061518) está **sin texto completo**, presumiblemente embargada por el contrato con Springer.

**Evidencia:** la raíz «concav» aparece **cero veces** en ~360 000 caracteres de texto completo de Charitopoulos (Applied Energy + AIChE J 2017 + CACE 2018), y **cero** en todo el volumen de Springer según búsqueda de texto completo. La única aparición de «concavity» en el libro es **un título en la bibliografía** — precisamente Fiacco y Kyparisis (1986). Tiene la referencia canónica del hecho y nunca la usa.

**La cocitación era un artefacto bibliométrico**: Gal aparece en el artículo como una de tres referencias genéricas de formulación, y en el capítulo para un ejemplo numérico con incertidumbre en el lado izquierdo y para degeneración. Nunca como fuente de la curvatura. Y su marco usa **ε-*constraint***, con los parámetros multiobjetivo en el lado derecho: por construcción nunca se colocan en el régimen donde el signo cambia.

**Se puede escribir «a la fecha, ningún trabajo citante ha señalado el error» sin salvedad.**

*(Única pieza no leída, y de baja probabilidad: las dos páginas del §9.6 del libro Wiley 2020 sobre mp-MOO, pp. 173–174. El índice analítico del volumen solo indexa «concave function» una vez, en la definición del capítulo 1.)*
