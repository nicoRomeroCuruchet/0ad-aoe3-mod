# M2 — Team Gather (4 aldeanos, varios recursos) — Design Spec

**Fecha:** 2026-08-11
**Estado:** Implementado; contrato vigente de anillo, agotamiento y matching conjunto
**Rama:** `feature/polites-pov`
**Precede:** `2026-06-13-rl-gather-loop-design.md` (M0), M1 implementado sobre esa misma arquitectura

## Objetivo

Escalar el agente de recolección de **1 aldeano a 4**, con **6 árboles exclusivos de 100** y
reward por **Δstock colectivo**. El objetivo vigente es entregar los **600** de madera del anillo.
Las posiciones siguen siendo conocidas: M2 no es exploración (eso es M3).

Lo nuevo respecto a M1 es **repartir el trabajo**: qué aldeano va a qué árbol. La mecánica de
recolección en sí ya está resuelta y entrenada en M1, y se reutiliza.

## Contrato vigente — anillo, agotamiento y matching conjunto

La primera versión de M2 era demasiado pequeña: un roble vanilla tiene 200 de madera y admite
hasta ocho recolectores, por lo que cuatro aldeanos podían ir juntos al mismo árbol y aun así
alcanzar el objetivo de `+80`. Ese comportamiento no demuestra reparto de trabajo. El contrato
vigente hace que el reparto sea una restricción real del mundo y de la acción:

- Hay seis `gaia/tree/rl_m2_oak`, cada uno con **100 de madera** y `MaxGatherers = 1`.
  Entregar **`Δstock colectivo >= 600`** exige agotar y llevar la carga de los seis. No existe
  ninguna cuota por aldeano: sólo importa el stock del equipo.
- La cabaña está en el centro. Los seis árboles forman un círculo de radio fijo, separados por
  `2π/6` (60°). En cada reset el entorno inyecta un seed reproducible que rota el anillo completo.
  Radio, formación de aldeanos y separación angular no cambian; lo que se aleatoriza es la
  orientación absoluta. Así no puede memorizarse “el árbol correcto está al norte”.
- Al agotarse, el engine elimina visualmente el roble. El roster conserva su slot estable con
  cantidad cero, de modo que no se reordenan ni las observaciones ni el significado de las
  acciones durante el episodio.

### Acción: una categoría global de 1045 matchings válidos

La acción Gym es `Discrete(1045)`, no cuatro decisiones independientes. Cada índice decodifica a
un vector conceptual de cuatro categorías, una por aldeano: `0 = NO_CLICK` y `j + 1 = TREE_j`.
Los valores no nulos deben ser distintos: un árbol no puede ser reclamado por dos aldeanos en la
misma decisión. Para cuatro aldeanos y seis árboles hay
`Σ(k=0..4) C(4,k) P(6,k) = 1045` matchings parciales posibles, incluidos los que dejan a alguien en
silencio.

La categoría se convierte determinísticamente en los cuatro `[x, z, click]` exactos del slot de
árbol correspondiente y pasa por el mismo intérprete de clicks de mapa que M1. Por eso no es una
acción “gather” que salte la interfaz. El ejecutor local deriva una máscara sólo de la observación:
un árbol que alguien corta queda reservado para ese aldeano y los slots agotados no se pueden
elegir. `NO_CLICK` mientras UnitAI corta o transporta sigue siendo una decisión que PPO debe
aprender —un click alternativo puede interrumpir el ciclo—; sólo se eliminan combinaciones que el
mundo ya rechazaría por capacidad.

PPO usa un único categórico sobre los 1045 matchings. El actor construye puntuaciones por pareja
aldeano/árbol y suma las del matching completo; la normalización global permite representar la
competencia por un árbol, cosa que cuatro categóricos factorizados no podían hacer bajo simetría.
El crítico recibe el estado conjunto. No se asigna un árbol oculto a un aldeano en reset:
`is_current_target` es sólo el historial del último click que eligió la propia política.
La inicialización es neutral sobre los matchings factibles: la tabla impone la física, pero PPO
debe descubrir una primera asignación completa y el segundo reparto cuando se liberan aldeanos.

El muestreo y la pérdida de PPO usan el categórico nativo sin cambios. Sólo su decisión
determinista es jerárquica: primero marginaliza la masa sobre el patrón binario de qué aldeanos
hacen click, y después elige el matching más probable dentro de ese patrón. Eso impide que el único
matching global de silencio gane un `argmax` plano frente a muchos matchings productivos compatibles;
no introduce un árbol asignado ni una acción adicional.

### Observación y criterio

La observación es una matriz de cuatro slices. Cada slice contiene el estado propio del aldeano
(posición, carga, stock común, depósito y ciclo UnitAI) y, para cada árbol estable, su
desplazamiento, distancia, cantidad restante e `is_current_target`. Por lo tanto todos los
árboles y sus posiciones son visibles para la política: M2 sigue siendo un problema de coordinación
de mapa conocido, no exploración con niebla de guerra. La rotación por seed evita la memorización
de orientación, no oculta recursos.

El horizonte vigente es 20 decisiones de 80 turnos. El oracle entrega los 600 en unas 20 decisiones;
el rollout aleatorio no coordinado no alcanza el total dentro de ese margen. El éxito y la parada
dependen exclusivamente de `Δstock colectivo >= 600`; métricas por aldeano son diagnósticos,
nunca requisitos. La corrida PPO se valida después del primer rollout de 512 decisiones sobre 20
orientaciones retenidas y necesita ≥95% de éxito; es una selección por terminación real, no por
reward estocástico de los rollouts.

Los checkpoints, resultados y videos producidos antes de este contrato —incluidos los de objetivo
`+80`, layout fijo o acciones factorizadas— son **legacy**. Su espacio de acción y su benchmark no
son compatibles ni comparables con el M2 vigente; hay que entrenar y renderizar una corrida nueva.

## Archivo: diseño original (no usar como contrato operativo)

> Las secciones restantes preservan el razonamiento previo a la revisión. Referencias a
> `Box(12,)`, `MultiDiscrete`, objetivo `+80`, árboles vanilla, layouts fijos, cuotas implícitas
> o transferencia desde M1 son históricas y quedan sustituidas por el contrato anterior y por los
> TOML `m2_*` versionados.

## Qué cambia y qué no

| | M1 (hecho) | M2 (diseño inicial archivado) |
|---|---|---|
| Aldeanos | 1 | 4 |
| Árboles | 1 | 4 |
| Acción | `Box(3,)` = `[x, z, click]` | `Box(12,)` = `[x, z, click]` × 4 |
| Reward | Δstock del jugador | Δstock del jugador (colectivo) |
| Observación | vector plano de 10 | 4 slices de 31 (núcleo M1 + relacionales + ID) |
| Política | MLP 10→3 | red compartida por aldeano 31→3, aplicada ×4 |
| Crítico | MLP sobre los mismos 10 | centralizado: ve el estado conjunto |
| Problema central | cuándo **no** clicar | **a qué árbol va cada uno** |

## Decisiones cerradas (con evidencia)

### 1. Se mantiene el click bit y su penalización

El roadmap especificaba `Box(8,)` (sólo `[x, z]` por aldeano). Se descarta: **se mantiene el
click bit**, `Box(12,)`.

Motivo medido. Ablación corrida el 2026-08-11 sobre M1, PPO, misma seed, misma config salvo
`click_gather_cycle_penalty`:

| | penalización 1.0 | penalización 0.0 |
|---|---|---|
| Resultado | **resuelto** | tope de seguridad |
| Pasos | 12.800 | 40.448 (tope) |
| Wall clock | 300 s | 949 s |
| Chequeos de éxito | 5 | 16, todos 0% |
| Evaluación final | 100%, reward 30.1 | 0%, reward 1.3 |

Diagnóstico del checkpoint sin penalización, episodio determinista completo:

```
120 × command=walk        (nunca emite un gather)
120 × dstock=+0.0         (el stock jamás se mueve)
click_signal = +1.00      (saturado al máximo en todos los pasos)
```

Oscila hacia el árbol y de vuelta cosechando el shaping de distancia (`+2.21, −0.92, +1.00,
−1.10, …`), ~+1.3 por episodio, sin tocar la madera.

**El mecanismo importa para M2:** la penalización es lo único en el reward que empuja el click
signal hacia abajo. Sin ella el click queda saturado, cada paso emite un comando, y cualquier
`gather` que la exploración encuentre se cancela al paso siguiente — así que la recompensa de
+20 por depositar **nunca se muestrea ni una vez**. No es que aprenda más lento: nunca llega a
experimentar que recolectar paga.

Consecuencia de diseño: el shaping tiene que quedar **modesto**, porque el modo de falla
observado es exactamente "el shaping es más fácil de cosechar que la tarea", y M2 tiene cuatro
veces más superficie para eso.

### 2. Control centralizado, no multi-agente

Los 4 aldeanos se controlan desde un proceso, en un `step`, con observabilidad total y sin
restricciones de comunicación. Es **control centralizado de un sistema multi-unidad**, no un
juego multi-agente. No hace falta framework MARL: un crítico centralizado es "la cabeza de valor
lee el estado global", que son unas líneas en una policy custom de SB3.

### 3. Pesos compartidos por aldeano, no una red por aldeano

Una sola red pequeña, los mismos pesos, aplicada cuatro veces — una por aldeano, cada una viendo
su propio slice. Las cuatro salidas se concatenan en los 12 números de la acción.

Ventajas sobre una red conjunta (27→12): el checkpoint de M1 sirve de inicialización, la simetría
de permutación es gratis (los mismos pesos manejan cualquier slot), y hay 4× menos parámetros que
con redes separadas.

### 4. La información se "cruza" con features relacionales

Con pesos compartidos, si cada aldeano sólo viera su propio entorno, los cuatro convergerían al
mismo árbol. La coordinación necesita que cada slice sepa algo de los demás.

Para M2 se usan **features relacionales** por par (aldeano, árbol) — en particular *"hay otro
aldeano más cerca de este árbol que yo"*, que es la señal directa de reparto. A n=4 esto es más
barato y probablemente mejor que atención, con el presupuesto de muestras que tenemos.

**Largo plazo (M3+):** observación por entidades (`[n_entidades, n_features]` + máscara) y un
encoder con atención, permutation-equivariant y escalable a cualquier n. La decisión consciente
de este spec es **no** pagar esa arquitectura antes de que M2 produzca un solo episodio exitoso.
El costo de diferirla es un refactor del encoder, no del entorno ni de la acción.

### 5. Ruido: el que ya trae PPO, nada extra

La política de PPO ya es gaussiana: durante el entrenamiento las acciones se muestrean con ruido.
No se inyecta ruido adicional, y en particular **no** se usa ruido para romper la simetría: el
criterio de parada evalúa la política de forma **determinista** (toma la media), así que una
divergencia que dependa del azar desaparecería justo donde se mide.

La divergencia se consigue con tres fuentes deterministas: posición propia, features relacionales
y **feature de identidad** (el índice de slot del aldeano, como hace MAPPO en SMAC).

La exploración queda como perilla de config: `policy_kwargs.log_std_init` y `ent_coef`.

## Escenario

`maps/random/rl_gather.js` se extiende manteniendo determinismo (`Seed = 0`, sin RNG):

- 4 × `units/athenai/polites` (player 1), en posiciones distintas y fijas
- 4 × `gaia/tree/oak` (player 0), separados entre sí
- 1 × `structures/athenai/rl_storehouse` (player 1)

El mapa de M0/M1 (`maps/random/rl_gather.js`) **no se toca**: M2 agrega
`maps/random/rl_gather_team.{js,json}` y un `rl/scenarios/team_reset_config.json` propio. Los
replays de `media/replays/` son logs de comandos y se desincronizan si cambia el mapa que los
generó; separar los escenarios los mantiene reproducibles.

## Espacios

### Acción — `Box(12,)`, rango `[-1, 1]`

`[x₀, z₀, click₀, x₁, z₁, click₁, x₂, z₂, click₂, x₃, z₃, click₃]`

Semántica por aldeano, idéntica a M1: apuntar a un árbol y clicar emite `gather`; apuntar al
depósito y clicar emite `returnresource`; clicar en otro lado emite `walk`; no clicar no emite
nada y deja que UnitAI complete el ciclo.

El slot *i* de la acción corresponde al aldeano *i* del roster durante todo el episodio.

### Observación — 4 slices de 31

Cada slice arranca con **los 10 valores de M1, en el mismo orden**, para que el checkpoint de M1
se pueda cargar en el prefijo:

| Índice | Campo |
|---|---|
| 0–1 | `villager_x_norm`, `villager_z_norm` |
| 2–3 | `resource_x_norm`, `resource_z_norm` — del árbol **objetivo actual** de ese aldeano |
| 4 | `distance_norm` a ese árbol |
| 5 | `carried_wood_norm` |
| 6 | `stock_wood_norm` |
| 7–8 | `dropsite_x_norm`, `dropsite_z_norm` |
| 9 | `gather_cycle_active` |
| 10 | `agent_id_norm` = slot / (n − 1) |
| 11–30 | bloque relacional: por cada uno de los 4 árboles → `dx_norm`, `dz_norm`, `dist_norm`, `otro_aldeano_mas_cerca` (0/1), `remaining_norm` |

**Árbol objetivo actual:** el árbol del último `gather` emitido para ese aldeano; en `reset`, el
más cercano. Esto conserva la semántica de M1 ("el recurso que estoy trabajando").

**Orden del bloque relacional:** los 4 árboles siempre en orden de `entity_id` (el mismo del
roster), no por cercanía. Si el orden dependiera de la distancia, el significado de cada índice
cambiaría entre pasos y entre aldeanos, que es la misma clase de bug que el ordenamiento de slots.

**`otro_aldeano_mas_cerca`:** 1.0 si algún otro aldeano está más cerca de ese árbol que yo, 0.0 si
no. Es la señal directa de reparto.

**`remaining_norm`:** madera restante del árbol dividida por su cantidad inicial (`resource_amount_scale`
en la config), recortada a `[0, 1]`.

El actor consume un slice de 31 por vez. El crítico consume los 4 slices concatenados (124) — es
el crítico centralizado.

## Reward

```
reward = Δstock_colectivo
       + (1/n) · Σᵢ [ shaping_distancia(i) + shaping_carga(i) − penalización_click(i) ]
```

- **Δstock colectivo:** la señal real, una sola por paso para todo el equipo.
- **Shaping promediado, no sumado.** Con 4 aldeanos, sumar los términos de shaping los haría
  pesar 4× más frente a un Δstock que no escala igual — exactamente el desbalance que produjo la
  falla de la ablación. Se divide por n para mantener las magnitudes comparables a M1.
- **Escalas heredadas de M1 sin cambios:** `distance_shaping_scale = 0.02`,
  `carried_resource_delta_reward_scale = 0.2`, `click_gather_cycle_penalty = 1.0`.

## Criterio de parada

Idéntico en forma a M0/M1: **completar la tarea**, no un número de pasos.

- **Éxito de un episodio:** el stock de madera del jugador sube **≥ 80** (cuatro cargas de 20).
  Elegido para que un solo aldeano eficiente no pueda satisfacerlo mientras los otros tres pasean:
  es lo que convierte a M2 en tarea de equipo y no en M1 con tres espectadores.
- **Resuelto:** ≥ 80% de éxito en 20 episodios deterministas con seeds fijas.
- `total_steps` es sólo tope de seguridad: 200.000. Chequeo cada 5.000 pasos, mínimo 5.000.
- Horizonte 120 pasos × 80 turnos de simulación.

## Transferencia desde M1

El checkpoint de M1 inicializa la red compartida por aldeano:

- Los índices 0–9 del slice son exactamente la observación de M1 → los pesos de la primera capa se
  copian tal cual.
- Los índices 10–30 son nuevos → sus pesos de entrada se inicializan **en cero**. Al arrancar, esos
  inputs no aportan nada y la política se comporta exactamente como M1 entrenado: ir al árbol,
  recolectar, quedarse quieta. El entrenamiento aprende después a usarlos para repartirse.
- La cabeza de acción (3 salidas) se copia igual.

Es opcional y va por config (`resume_from` apuntando a un checkpoint de M1); M2 tiene que poder
entrenarse desde cero también, aunque salga más caro.

## Refactor previo (enfoque A)

`rl/gather/env.py` tiene 950 líneas y hoy mezcla ciclo de vida del backend, reintentos, modos de
reward, máquina de estados del gather y armado de observación. Añadir multi-unidad ahí lo empeora
justo donde viven los bugs sutiles. Antes de M2:

| Módulo nuevo | Responsabilidad |
|---|---|
| `rl/gather/roster.py` | orden estable de aldeanos y árboles por `entity_id`; mapa slot↔entidad congelado en `reset` |
| `rl/gather/observation.py` | armado de slices (núcleo M1 + relacionales + ID) y sus labels |
| `rl/gather/reward.py` | composición de términos de reward como funciones puras |
| `rl/gather/env.py` | queda con: API Gym, backend, reintentos, orquestación del step |

`core.py` no cambia: sigue siendo la matemática de bajo nivel.

Después del refactor, M2 entra como **configuración** (`villager_count`, `resource_count`), con
M0/M1 como el caso n=1.

### Round-trips al engine

Hoy `_resource_snapshot` hace un `game.evaluate` por paso para un aldeano. Con 4, la versión
ingenua cuadruplica el costo de RPC en un milestone que ya cuesta ~18 min de entrenamiento en M1.

M2 usa **una sola expresión JS por paso** que devuelve `{stock, carried: {id: cantidad}, trees:
{id: restante}}` para todas las entidades. Es parte del refactor, no una optimización posterior.

## Ordenamiento estable (clase de bug a evitar)

`state.units(...)` no garantiza orden entre pasos. Si el orden cambia, el slot 2 de la acción pasa
a manejar otro aldeano y la observación se mezcla, sin ningún error visible.

Regla: en `reset` se ordenan las entidades por `entity_id`, se congela el mapa slot↔id, y en cada
paso se resuelven las entidades **por id**, no por posición en la lista. Si aparece un conteo
distinto al configurado, se falla fuerte (no hay combate en M2: un aldeano que falta es un bug).

## Configs

`m2_oracle.toml`, `m2_random.toml`, `m2_sb3_ppo.toml`, `m2_sb3_sac.toml`, con las mismas claves de
parada que M0/M1 para que las corridas sean comparables.

PPO primero: en M1 resolvió en 300 s contra 1067 s de SAC con la misma cantidad de muestras, y M2
tiene un espacio de acción más grande, lo que favorece on-policy.

## Baselines antes de entrenar

- **Oracle:** asignación greedy — cada aldeano al árbol libre más cercano, apuntar al depósito
  cuando lleva carga. Tiene que llegar a 80 de madera de forma consistente. Es lo que valida el
  entorno (orden, batching, reward) **sin gastar entrenamiento**, y es donde va a aparecer el bug
  de ordenamiento si existe.
- **Random:** piso de comparación.

Ningún entrenamiento arranca antes de que el oracle pase.

## Estrategia de verificación

Todo offline, con backends falsos; ningún test necesita el juego:

| Qué | Cómo |
|---|---|
| Orden estable | `state.units` devuelve la lista barajada entre pasos → slots no se mezclan |
| Conteo inesperado | 3 aldeanos con `villager_count = 4` → falla fuerte |
| Ruteo de acción | 12 números → 4 comandos, cada uno a su entidad |
| Features relacionales | "otro más cerca" correcto en casos construidos a mano |
| Composición de reward | shaping promediado, no sumado; penalización por aldeano |
| Batching | una sola llamada `evaluate` por paso, con la forma esperada |
| Prefijo M1 | los índices 0–9 del slice coinciden con la observación de M1 |
| Transferencia | cargar un checkpoint M1 en la red compartida deja los inputs nuevos en cero |
| Contrato de mapa | `test_map_contract.py` extendido: 4 polites, 4 robles, 1 storehouse |
| Contrato de config | `test_tracked_configs.py` extendido a los `m2_*` |

## Orden de implementación

El spec cubre más de un PR. El plan lo va a detallar, pero el orden es parte del diseño porque
cada etapa deja algo verificable antes de la siguiente:

1. **Refactor sin cambio de comportamiento.** Extraer `roster.py`, `observation.py`, `reward.py`
   de `env.py`. Los 326 tests actuales tienen que seguir pasando sin tocarlos: es la red de
   seguridad de que M0/M1 no se movieron.
2. **Multi-unidad en el entorno**, con `villager_count` / `resource_count` y M0/M1 como n=1.
   Incluye el `evaluate` batcheado y el ordenamiento estable. Verificable con backends falsos.
3. **Mapa y escenario de equipo** + contrato de mapa.
4. **Baselines:** oracle greedy y random sobre el mapa nuevo. **Gate:** el oracle llega a 80 de
   madera de forma consistente. Nada de entrenamiento antes de esto.
5. **Política compartida por aldeano** (policy custom de SB3: actor por slice, crítico
   centralizado), con la carga del prefijo M1 y zero-init.
6. **Configs y corridas:** PPO primero, SAC de comparación.

Las etapas 1–4 no necesitan ninguna decisión de RL: son entorno y verificación. Si algo del diseño
de política resulta equivocado, se descarta la etapa 5 sin perder 1–4.

## Riesgos

1. **No se reparten.** Los cuatro van al mismo árbol. Mitigación: features relacionales + ID; si
   igual pasa, la señal está en el oracle vs la política y el siguiente paso es atención.
2. **Cosecha de shaping.** El modo de falla ya observado en la ablación. Mitigación: shaping
   promediado y modesto; el diagnóstico es el desglose `parts=[...]` de `--verbose`.
3. **Wall clock.** 4 unidades por paso y 20 episodios por chequeo. M1 costó 300 s con PPO; M2 va a
   costar bastante más. Mitigación: chequeos cada 5.000 pasos, tope 200.000, PPO primero.
4. **Transferencia parcial.** Si el prefijo M1 no ayuda, se entrena desde cero; es una perilla, no
   un supuesto del diseño.

## Fuera de alcance (YAGNI para M2)

- Atención / observación por entidades (es la evolución natural en M3, no ahora).
- Múltiples **tipos** de recurso (madera + comida). M2 es varios árboles del mismo tipo.
- Niebla de guerra y exploración (M3).
- Frameworks MARL: con control centralizado no hacen falta.
- Curriculum 1-árbol → 4-árboles: se agrega sólo si el entrenamiento se estanca.
