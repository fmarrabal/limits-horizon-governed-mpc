# GHI — Gobernador Homeostático Inercial para MPC multiobjetivo

Implementación completa y auditada, en **Python** y **MATLAB**, con validación cruzada numérica entre ambas.

Código de apoyo a [`../PROPUESTA-MPC-HOMEOSTATICO.md`](../PROPUESTA-MPC-HOMEOSTATICO.md). Sustituye al prototipo de [`../piloto/`](../piloto/), que se conserva solo como registro histórico de cómo se llegó hasta aquí.

---

## Qué hay aquí, y qué se corrigió respecto al prototipo

El prototipo tenía **tres defectos reales** que encontró una ronda de revisión adversarial. Los tres están corregidos, y cada uno tiene ahora una comprobación que falla si vuelve.

| defecto | qué pasaba | dónde está resuelto |
|---|---|---|
| **Ingredientes terminales inválidos** | Los `P_i`, `K_f` y `Ω` heredados del TFM violan la desigualdad terminal (`eig(S₀) = [−11.04, +0.02]`), y la «región terminal» era una **caja**, no un invariante. Sin eso, `V* ≤ J_a` **no certifica decrecimiento de Lyapunov** | `terminal.py` / `designTerminal.m`: `K_f` común por LQR, `P_i` por ecuación de Lyapunov (⇒ `S_i = 0` exacto), `Ω` por Gilbert–Tan. Comprobaciones **A1–A3**, y **A12** conserva el fallo del TFM como regresión |
| **Anti-windup disparándose siempre** | `sync` comparaba el valor aplicado (que cae en una malla) con el estado continuo del campo, así que actuaba en **200/200** pasos con cero bloqueos. El ζ *efectivo* era ≈0.868, no 0.5: el campo analizado no era el ejecutado | `regulators.py` / `Order2.m`: `sync` recibe explícitamente `blocked`. Comprobaciones **A8** y **A9** |
| **Igualación hecha en tiempo continuo** | τ se elegía sobre transferencias continuas, pero los reguladores se integran en discreto y ω₀·dt = 0.9 no es ≪1. Desajuste real: **1.69×** | `match_first_order` / `matchFirstOrder.m`: se iguala sobre la respuesta en z de los integradores **realmente implementados**. Comprobación **A7** |

---

## Estructura

```
code/
  python/ghi/          paquete de referencia
    plant.py           plantas, objetivos y bancos de prueba
    terminal.py        Kf común, P_i por Lyapunov, Ω por Gilbert–Tan, auditoría
    mompc.py           el MO-MPC como QP denso explícito (OSQP + CLARABEL)
    weights.py         conjunto admisible, radio certificado, CONCAVIDAD
    regulators.py      órdenes 0/1/2, anti-windup, igualación en discreto
    field.py           campo homeostático sobre el grafo de subsistemas
    experiment.py      lazo cerrado, escenarios, métricas
    stats.py           contrastes pareados, Holm, intervalos de Poisson
    audit.py           la suite de auditoría
  python/scripts/
    export_crossval.py genera la referencia que MATLAB debe reproducir
    run_concavity.py   la corrección a Bemporad 2009
    run_pilot.py       el banco principal
  matlab/+ghi/         port nativo (quadprog/linprog, SIN YALMIP)
  matlab/scripts/
    run_crossval.m     validación cruzada contra la referencia Python
    run_audit.m        la misma suite de auditoría
    run_pilot.m        el mismo banco
  crossval/            python_reference.json (60 casos canónicos)
  results/             salidas de las ejecuciones
```

## Requisitos

**Python** ≥ 3.11: `numpy scipy cvxpy osqp clarabel`
Probado con numpy 2.4.6, scipy 1.17.1, cvxpy 1.9.2, osqp 1.1.3.

**MATLAB** R2026a con Optimization Toolbox (`quadprog`, `linprog`), Control System Toolbox (`dlqr`, `dlyap`) y Statistics Toolbox (`ttest`). **No hace falta YALMIP ni MPT**: el port es nativo, a diferencia del TFM de 2012.

## Cómo se ejecuta

```bash
# Python
cd code/python
python -m ghi.audit                  # suite de auditoría (15 comprobaciones)
python scripts/run_concavity.py      # la corrección a Bemporad 2009
python scripts/run_pilot.py          # banco principal (~85 s)
python scripts/export_crossval.py    # regenera la referencia para MATLAB
python scripts/run_alphaN.py         # α_N → gobernador → N(t)  (§10)
python scripts/run_vibration.py      # la arena de banda estrecha  (§11)
python scripts/run_scale_invariance.py  # invariancia de escala + ventana (§11)
```

```matlab
% MATLAB
cd code/matlab/scripts
run_audit        % la misma suite (13 comprobaciones)
run_crossval     % validación cruzada contra Python (29 comprobaciones)
run_pilot(10)    % el mismo banco
```

---

## La validación cruzada, y por qué está montada así

No basta con que las dos implementaciones den resultados *parecidos*: eso lo consigue cualquier par de programas que resuelvan problemas distintos pero similares. Lo que se compara es la **formulación**, con tres tolerancias deliberadamente distintas:

| qué se compara | tolerancia | resultado medido |
|---|---|---|
| **Construcción** — ingredientes terminales, Ω, `Sx`, `Su`, `Ain`, `b_const`, `B_x`, `G`, `g`, `bin`, operadores del campo | 1e-10 absoluta | ≤ **2.7e-12** |
| **Solución** — `U*`, `J`, `V*`, secuencia desplazada | 1e-6 **relativa** | ≤ **4.8e-7** |
| **Valor óptimo del QP** | 1e-9 **relativa** | **4.9e-10** |

Las tres filas dicen cosas distintas, y la tercera es la que zanja el asunto:

- La **construcción** debe coincidir a precisión de máquina porque ambas hacen literalmente la misma aritmética. Incluye el **orden de las filas de `Ain`**, no solo su contenido.
- La **solución** solo puede coincidir hasta la tolerancia de los solvers: CLARABEL y `quadprog` son algoritmos distintos. Exigir tolerancia absoluta ahí sería exigir que dos solvers den bit a bit lo mismo.
- El **valor óptimo** es la prueba fuerte: en un óptimo el gradiente proyectado se anula, así que un error `dz` en la solución mueve el objetivo en `O(dz²)`. Medido: la solución concuerda a 4.8e-7 y el valor óptimo a 4.9e-10, **mil veces mejor**. Eso es exactamente el `O(dz²)` predicho, y es la evidencia de que ambos solvers cayeron en el **mismo** mínimo y no en dos distintos.

**Estado: 29/29 comprobaciones OK.**

---

## Resultados que produce

### La corrección a Bemporad & Muñoz de la Peña (2009)

```
V*(x,α) = min_{U ∈ 𝒰(x)} α'J(U,x)
```

Para cada `U` fijo el corchete es **afín** en α, y `𝒰(x)` no depende de α. El ínfimo puntual de una familia de afines es **cóncavo** — el mismo hecho que «la función dual de Lagrange es cóncava».

`run_concavity.py` mide, con los ingredientes terminales **correctos**:

- **300 tests de cuerda: 0 violaciones de concavidad, 300 de convexidad**, desviación media +23.6 por encima de la cuerda.
- El conjunto admisible es **no convexo en el 20.6 %** de los niveles de `J_a` barridos, y el máximo de `V*` cae en el **interior** del símplex en 14 de 20 estados. Que el máximo sea interior es la firma de la concavidad y lo que obliga a que el subnivel sea la unión de dos intervalos: no es una rareza numérica, es geometría forzada.

Consecuencia: el LP de selección de peso de Bemporad no es una reformulación *equivalente* sino una **restricción interior conservadora**. El algoritmo sigue siendo **seguro** — una restricción interior nunca viola el certificado — pero la equivalencia y la optimalidad enunciadas no se sostienen.

> **Antes de citar esto**: verificar a mano sobre el PDF original que su μ es el peso del coste y que su factible no depende de él. Y compararse contra su LP (17), que ya es una convexificación interior y más apretada que un solo hiperplano soporte.

### El banco principal

Con ingredientes terminales correctos, igualación en discreto, anti-windup correcto y corrección de Holm (n=10 semillas, pareado):

| escenario | orden-2 vs orden-1 (error de demanda) | t | p (Holm) |
|---|---|---|---|
| A — cuasi-estático | **−0.0479** (gana el 1.er orden) | −14.36 | <0.0001 |
| B — periódico | **+0.0971** | **+31.59** | <0.0001 |
| C — fuera de distribución | **+0.0936** | **+20.08** | <0.0001 |
| D — fuera de distribución, fuerte | **+0.0619** | **+8.27** | 0.0001 |

Con demanda cuasi-estática el segundo orden **pierde**, y debe perder. Fuera del sobre de diseño gana con claridad. Es la misma firma que el paper del HBP.

Y el detalle que la igualación correcta hace nítido: en los tres escenarios excitados **el filtro de primer orden es peor que el mapa sin memoria** (t = −11.8 / −15.8 / −7.9), porque su retraso de fase le cuesta más de lo que le da el filtrado. El segundo orden es lo único que consigue las dos cosas a la vez.

**El resultado incómodo, que hay que publicar igualmente:** en coste el orden 1 bate al orden 2 y el peso **congelado es el más barato de todos**. En un banco de juguete la demanda oscilante no vale lo que cuesta seguirla. Eso no invalida nada: señala que el banco definitivo necesita que seguir la demanda tenga **valor real** — por ejemplo, precio horario de la energía en un invernadero.

#### Y una comprobación de robustez que sale gratis

El banco MATLAB usa un generador de números aleatorios **completamente distinto** (Mersenne Twister frente al PCG64 de numpy), así que las realizaciones de ruido y perturbación no son las mismas. No se busca que los números coincidan — se busca que la **conclusión** no dependa del flujo de ruido:

| escenario | Python: Δ (t) | MATLAB: Δ (t) |
|---|---|---|
| A — cuasi-estático | −0.0479 (−14.36) | −0.0480 (−14.72) |
| B — periódico | +0.0971 (+31.59) | +0.0958 (+22.39) |
| C — fuera de distribución | +0.0936 (+20.08) | +0.1020 (+28.19) |
| D — fuera de distribución, fuerte | +0.0619 (+8.27) | +0.0657 (+5.53) |

Mismos signos, misma significación y magnitudes dentro del ruido en los cuatro escenarios.

### El campo sobre el grafo

- El umbral `ρ(G) < 2ζω₀²` es **exacto** en el caso degenerado de Merkin (ratio observado/predicho = **1.0044**, y ese 0.4 % es la resolución del barrido).
- Con estructura de planta es **conservador**: tanto el amortiguamiento estructural `D·L` como la rigidez espacial `c²L` aumentan el margen real (ratios 1.23, 1.30, 1.52, 1.94). **Se puede usar como cota segura de diseño**, que es lo que un ingeniero necesita.
- La colocación **giroscópica** no falla en ningún punto de la caja barrida.

---

## La suite de auditoría

Cada defecto que se encontró una vez está convertido en una aserción que falla si vuelve. Ninguna comprobación se apoya en la memoria de nadie.

| | comprobación |
|---|---|
| A1 | desigualdad terminal `S_i ⪰ 0` por objetivo |
| A2 | Ω positivamente invariante bajo `A_cl` |
| A3 | Ω respeta `\|x\| ≤ xmax` y `\|K_f x\| ≤ umax` |
| A4 | `V*(x,·)` es **cóncava** en α |
| A5 | el conjunto admisible **es** no convexo para un rango de `J_a` |
| A6 | el radio certificado es válido: `‖dα‖ ≤ r ⇒ α'` admisible |
| A7 | la igualación 1.er/2.º orden es exacta **en discreto** |
| A8 | el anti-windup **no** se dispara si no hay bloqueo |
| A9 | la sobreoscilación ejecutada **es** la del sistema discreto |
| A10 | la colocación giroscópica es estable en toda la caja |
| A11 | el umbral de Merkin: exacto en el degenerado, conservador fuera |
| A12 | *(regresión)* los ingredientes del TFM **siguen** violando la desigualdad terminal |

**Python: 15/15 OK. MATLAB: 13/13 OK.** (Python desglosa A9 en tres ζ.)

---

## Notas de implementación que conviene conocer

- **`Ain` no depende de `x0`.** Solo el lado derecho es afín: `bin = b_const + B_x·x0`. Eso permite cachear el setup de OSQP y es lo que hace el banco viable (0.4 ms/QP frente a 17 ms reconstruyendo).
- **`J` nunca se lee del epigrafo.** Si `α_i = 0`, las variables de epigrafo del objetivo *i* no aparecen en la función objetivo y el solver las deja en cualquier punto factible por encima del valor verdadero. Se recalcula siempre con `costs`, que es la misma función que produce `J_a`.
- **OSQP con `polish=False`** a propósito: medido contra CLARABEL en 40 instancias, sin pulido el error es 1.4e-6 y con pulido 1.7e-5.
- **El coste de evaluación es neutral**: ponderado por la demanda **limpia**, no por el peso que cada regulador decide. Si no, cada regulador se puntuaría con su propio criterio.
- **`effective_overshoot` resetea a 0**, no al arranque por defecto `a=0.5`. Con 0.5 el escalón tendría amplitud 0.5 y la sobreoscilación relativa saldría justo la mitad.
- **El conjunto admisible se barre, no se proyecta**, porque no es convexo. Es también la razón por la que Bemporad resuelve la selección de peso por enumeración.

## El módulo de suboptimalidad (α_N)

`ghi/suboptimality.py` cierra el lazo **α_N observado → gobernador → horizonte N(t)**, con MPC sin ingredientes terminales (el régimen donde α_N informa), contabilidad honesta del instrumento (todos los QP se cargan, y el solve del instrumento se reutiliza cuando N no cambia), gate **bilateral** de informatividad (un salto de V delata un evento aunque ℓ≈0 — el gate unilateral anulaba justo la muestra del kick), y una batería de gobernadores con los baselines honestos que exigió la revisión adversarial (`umbral-up4`, dos igualaciones de τ para el orden-1, y el campo crítico `orden-2-z1`).

Resultados (`results/alphaN*.log|json`): **H2 y H5 falsadas, H6 confirmada, H3 falsada** — el gobernador correcto para esta señal es un integrador con fugas de constante larga, que alcanza el coste del horizonte conservador con ~25–28 % de su cómputo; el campo subamortiguado timbra y destruye la asignación. Detalle completo en `PROPUESTA-MPC-HOMEOSTATICO.md` §10. Las divergencias se excluyen pareadamente de los t-tests y se reportan como conteos.

La auditoría incluye ahora **A13/A14**: la dualidad de monotonía de V_N (no decreciente sin terminal, no creciente con terminal correcto) como aserciones ejecutables.

## La arena de vibración y la invariancia de escala (§11)

`ghi/vibration.py` construye la arena que la regla refinada predecía como favorable al campo: estructura resonante (ζ_p=0.04), perturbación de **batido** (envolvente que *oscila*; una envolvente de Rayleigh sería paso-bajo, no de banda estrecha — el error que casi cometo), coste real de fatiga `q e² + r|u|⁴`, α_d **deducido** de ese coste, y nulo de mecanismo exacto (narrow vs broad con idéntica varianza de envolvente).

Resultado (`results/vibration.*`): **modular pierde en 0/8 configuraciones** y el nulo no separa. La causa no es de sintonía sino estructural, y está en `scripts/run_scale_invariance.py`:

> En régimen lineal sin restricciones activas, `d → λd ⇒ u*→λu*, J→λ²J` **para cada (α, N) fijo**, luego el peso óptimo y el horizonte óptimo son **invariantes a la amplitud**. Verificado a dispersión relativa 9.15e-14 (peso) y 8.86e-17 (horizonte), y roto *exactamente* al empezar a saturar (0 % de pasos saturados → coste/E₀² idéntico a 6 cifras; 20 % → se desvía un 34 %).

Es **A15** en la suite. Consecuencia: una demanda fabricada modulando amplitudes no genera ningún óptimo variable en el tiempo, así que ningún gobernador puede ganar ahí. Ver `PROPUESTA-MPC-HOMEOSTATICO.md` §11 para la tensión estructural que esto deja al descubierto.

## El transporte (§12): tres filtros antes de dejar competir a nadie

`ghi/transport.py` monta la arena del transporte, y el orden de ejecución importa porque cada paso puede matar la arena antes de gastar nada:

1. `scripts/run_transport_viability.py` — primer intento, **sin almacenamiento**. Falló la condición C3 (el mejor peso constante cayó en el extremo, así que no había nada que programar). Su veredicto no es válido, pero explicar el fallo dio la proposición:

   > **Inutilidad de la anticipación sin almacenamiento.** Con planta asintóticamente estable regulada al origen, el equilibrio en reposo es x=0 **para todo peso y todo horizonte**. No hay nada que pre-posicionar, luego un aviso anticipado es inutilizable por el canal del meta-parámetro por mucha antelación que tenga.

   Es **A16**, y descarta otra familia entera de bancos igual que A15.

2. `scripts/run_transport_viability2.py` — **con almacenamiento** (`storage_plant` + `SetpointMPC`, dos objetivos de *consigna distinta*, de modo que α fija el punto de operación). La calibración se barre exigiendo las tres condiciones necesarias antes de mirar nada. Arena viable: anticipar compra **+4.6 %** sobre el límite reactivo ideal, 9/10 semillas, t=+5.95, p=2.2e-4.

3. `scripts/run_lead_window.py` — de qué depende la antelación útil. `lead* ∈ [2,4]` en todo el rango físico y crece con la lentitud del actuador como predice el mecanismo, **pero la ventana es ancha**: todo lead entre 1 y 6 (hasta 20) queda dentro del 1 % del óptimo.

4. `scripts/run_transport_mechanism.py` — el experimento con todos los mecanismos (onda, difusión, línea de retardo pura sintonizada, mensaje instantáneo, reactivo local ideal, oráculo), cada uno sintonizado en semillas *distintas* de las de evaluación, con contrastes pareados y Holm.

5. **`scripts/run_transport_matched.py` y `run_transport_power.py` — los que deciden.** La primera tanda daba al campo por perdedor, pero era falsa: la revisión adversarial midió que el operador implementado tenía **ganancia cruzada NULA en continua** (`f = K(h_d−h)` ⟹ `h = h_d` exactamente ⟹ `H(0)=I`), de modo que **ninguna alerta sostenida cruzaba entre nodos con ninguna sintonía**. Corregido con `GraphField(..., forcing="source")` (`f = ω₀²h_d`), que sí transporta y mantiene idéntica la ganancia de onda y difusión con G=0 — es **A18**. Con el canal reparado el campo pasa a encabezar la tabla; con un rival **igualado en grados de libertad** (retardo + conformado de 2º orden, 7 parámetros como el campo) y 30 semillas, **deja de separarse** (t=−0.56). Lo que sí es significativo es que *conformar* lleva a la línea de retardo del 70.9 % al 101.3 % del oráculo (t=+5.45, p<10⁻⁴).

> **Aviso de uso.** `forcing="target"` (por defecto, el original) sirve cuando cada nodo tiene consigna propia y el grafo sólo debe acoplar la dinámica. Para **transportar** información desde una fuente hay que usar `forcing="source"`; con el otro no cruza nada en continua. Con `b≠0` el forzamiento `"source"` está prohibido porque G entra sobre la velocidad en la onda y sobre la posición en la difusión, y el nulo de mecanismo quedaría confundido.

**Techo de escalabilidad medido**: la atenuación del campo es geométrica (≈×0.35/salto) y el meta-parámetro está cuantizado, así que el alcance en saltos es `n_max ≈ log Δ / log ρ` — **logarítmico** en la resolución de la rejilla del peso. Con paso 0.05 la señal se cuantiza a cero más allá de tres saltos.

## Caso de aplicación: una flota de lazos con un solo controlador (§V del paper)

`ghi/fleet.py` + `scripts/run_fleet.py`. M=8 lazos de colector solar (fluido rápido acoplado a absorbedor lento) atendidos por un controlador embebido con presupuesto de cálculo **duro** por paso. Lo que hace que el horizonte importe —y sin lo cual el banco estaría vacío— es que **se penaliza el absorbedor**, al que el caudal llega con retardo: penalizando el estado sobre el que actúa la entrada, la razón J(N=2)/J(N=16) es 1.01 y no hay nada que gobernar.

Resultado: con presupuesto escaso (¼ del peor caso) gobernar el horizonte ahorra **9.3 %** frente al mejor horizonte fijo (t=+15.3, p=1.2e-6); con presupuesto holgado **pierde** hasta un 6.6 %; y con un frente de nubes que barre todos los lazos a la vez gana en **0/7** presupuestos. Bajo presupuesto duro el mejor horizonte fijo *es* el reparto a partes iguales, así que "gobernado vs. fijo" y "repartir por demanda vs. a partes iguales" son la misma medida.

## Guarda de causalidad (`tp.is_causal`)

Un desplazamiento temporal libre en una etapa de acondicionamiento es algo de lo más normal en un barrido de sintonía, y si la rejilla admite valores negativos entrega al contendiente información del futuro sin que nadie se entere. Aquí pasó: los dos ganadores del transporte abrían su puerta hasta ocho pasos antes de que la fuente midiera el evento. `tp.is_causal` sondea con un pulso y rechaza cualquier configuración cuya salida se mueva antes que la entrada; corre **dentro** del barrido, junto a `verlet_stable`. `scripts/run_transport_causal.py` rehace el experimento con la guarda puesta.

## Lo que este código todavía no hace

- **No modula `N(t)` ni los pesos de holgura `ρ(t)`.** Ninguno de los teoremas desarrollados lo cubre: cambiar `N` rompe la secuencia desplazada, luego rompe `J_a`, luego rompe el radio certificado. Y no es territorio virgen — hay MPC de horizonte adaptativo con garantías (Krener, arXiv:1602.08619; Sun et al., *IEEE TAC* 2019) — así que hay que citarlo y extenderlo, no reclamarlo.
- **El campo sobre el grafo no está instrumentado en el lazo.** `field.py` lo estudia aislado (colocación, umbral, propagación); el lazo cerrado usa un campo escalar de un nodo.
- **Los objetivos de norma infinito no tienen construcción terminal.** El QP los soporta vía epigrafo y la auditoría comprueba por muestreo si un `P` dado cumple, pero no hay un procedimiento que lo construya. `design` falla explícitamente en vez de inventar un ingrediente.
