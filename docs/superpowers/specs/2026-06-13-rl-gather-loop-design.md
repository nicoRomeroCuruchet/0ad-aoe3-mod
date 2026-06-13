# RL Gather Loop — Design Spec

**Fecha:** 2026-06-13
**Estado:** Aprobado (diseño) — pendiente plan de implementación
**Rama:** `rl-gather-experiment`

## Objetivo

Primer proyecto de Reinforcement Learning sobre el mod Athenai en 0 A.D.: entrenar un agente
que controle aldeanos (Polites) para juntar recursos. El alcance inmediato es **deliberadamente
mínimo** — un loop de debug con 1 aldeano y 1 recurso — para validar que toda la cañería
(Gymnasium ↔ Stable-Baselines3 ↔ interfaz RL nativa de 0 A.D.) funciona antes de complicar la
tarea.

### Por qué es viable y simple
0 A.D. ya trae **interfaz de RL nativa** (`pyrogenesis --rl-interface=127.0.0.1:6000`) y un
cliente Python oficial `zero_ad` (`source/tools/rlclient/python/`) con semántica tipo Gym:
`reset(config)`, `step(actions)`, `state.units(owner, type)` con `.position()`, y
`game.evaluate(js)` para leer estado de simulación arbitrario. No hay que construir el puente.

### Por qué SAC / off-policy
Cada `step()` avanza la simulación de un RTS → es **caro por muestra**. Un algoritmo off-policy
con replay (SAC) exprime cada interacción, a diferencia de PPO on-policy. Restricción: SAC en
SB3 es **solo acciones continuas**, así que toda la formulación es continua.

## Escalera de milestones

- **M0 (este spec):** 1 aldeano, navegar a 1 recurso. Reward = acercamiento. *Valida el loop.*
- **M1:** 1 aldeano, reward = Δstock real (emitir `gather`), confirmar recolección.
- **M2:** 4 aldeanos + varios recursos, acción `Box(8,)`, reward = Δstock colectivo. *Proyecto "real".*

M1 y M2 quedan fuera de este spec; se especifican por separado cuando M0 esté verde.

## Arquitectura (M0)

```
┌─────────────────────┐    HTTP localhost:6000    ┌──────────────────────────┐
│ Python (entrenador) │ ◀───────────────────────▶ │ pyrogenesis --rl-interface│
│  SB3 SAC            │   zero_ad (reset/step)    │  (headless, 1 proceso)    │
│   └─ ZeroADGatherEnv│                           │  mapa-scenario fijo       │
│       (gymnasium)   │                           │   1 Polites + 1 recurso   │
└─────────────────────┘                           └──────────────────────────┘
```

### Componentes

1. **Mapa-scenario fijo** (`scenario`, no aleatorio) — 1 aldeano Polites (player 1) y 1 recurso
   gaia (árbol o arbusto de bayas) en posiciones fijas, sin enemigos, sin niebla, terreno plano.
   Fijo ⇒ tarea estacionaria ⇒ aprendizaje rápido y verificable.
   - *Interfaz:* un archivo de scenario + un JSON de config que `game.reset()` consume (igual que
     `samples/arcadia.json` del cliente).
   - *Dependencia / riesgo:* autoría del mapa (ver "Riesgos abiertos").

2. **`ZeroADGatherEnv(gymnasium.Env)`** — wrapper fino sobre `zero_ad.ZeroAD`.
   - *Qué hace:* traduce acción continua → comando de 0 A.D.; avanza la sim; calcula reward y
     observación.
   - *Cómo se usa:* `env = ZeroADGatherEnv(uri, scenario_config); SAC("MlpPolicy", env).learn(...)`.
   - *De qué depende:* un proceso de 0 A.D. corriendo con `--rl-interface`; el paquete `zero_ad`.

3. **Script de entrenamiento** — instancia el env, corre `SAC("MlpPolicy", env).learn(...)`,
   loguea reward por episodio, guarda el modelo.

### Espacios y dinámica (M0)

| Elemento | Definición |
|---|---|
| **Observación** | `Box(float32)`: `[pos_aldeano_x, pos_aldeano_z, pos_recurso_x, pos_recurso_z, dist]` normalizados a los límites del mapa. ~5 dims. (Posición del recurso es fija pero se incluye para que el env sea reutilizable en M2.) |
| **Acción** | `Box(-1, 1, (2,))` = punto-objetivo (x,z) normalizado. Se desnormaliza a coords del mapa y se emite `zero_ad.actions.walk(aldeano, x, z)`. |
| **Recompensa** | `dist_prev − dist_actual` (acercamiento al recurso). Densa, centrada en cero. Opcional bonus terminal al alcanzar. |
| **terminated** | `dist < umbral` (alcanzó el recurso). |
| **truncated** | horizonte fijo (~50 steps). |
| **reset** | recarga el scenario vía `game.reset(config)`; devuelve observación inicial. |
| **sim-turns por step** | a determinar (ver riesgos); arrancar con un valor que haga que el aldeano se mueva de forma visible entre observaciones. |

### Flujo de un step

1. Recibir acción `a ∈ [-1,1]²`.
2. Desnormalizar a coords de mapa `(x,z)`.
3. `game.step([walk(aldeano, x, z)])` (avanzando K turnos de sim).
4. Leer nuevo estado: `state.units(owner=1)[0].position()` y la posición del recurso.
5. `dist_actual = ‖pos_aldeano − pos_recurso‖`; `reward = dist_prev − dist_actual`.
6. `terminated = dist_actual < umbral`; `truncated = paso ≥ horizonte`.
7. Devolver `(obs, reward, terminated, truncated, info)`.

## Estrategia de verificación

M0 está **verde** cuando:
1. El loop corre sin excepciones: `SAC(...).learn(total_timesteps=N)` completa.
2. La **curva de reward por episodio sube** durante el entrenamiento (vs. una política random).
3. La política entrenada hace que el aldeano **llegue al recurso** en pocos steps de forma
   consistente (distancia final < umbral en la mayoría de los episodios de evaluación).

Tests/chequeos:
- Smoke test del env: `reset()` y N `step()` con acciones random devuelven shapes/tipos correctos.
- Sanity de reward: acercarse a mano (acción = posición del recurso) da reward acumulada positiva.
- Curva de aprendizaje guardada (CSV/tensorboard) para inspección.

## Riesgos abiertos (a despejar al inicio de la implementación)

1. **Autoría del mapa-scenario.** Crear el scenario con 1 Polites + 1 recurso. Opciones:
   (a) Atlas (editor ya compilado), (b) mini-script de mapa que coloque exactamente esas dos
   entidades. Decidir en el plan; (a) suele ser lo más directo para un scenario fijo.
2. **Tiempo de sim por `step()` y wall-clock.** Medir cuánto avanza la sim y cuánto tarda cada
   step headless; ajustar K turnos/step y horizonte para que M0 entrene en una tarde.
3. **Formato del config de `reset()`.** Confirmar el JSON que espera `game.reset()` (basarse en
   `source/tools/rlclient/python/samples/arcadia.json`).

## Fuera de alcance (YAGNI para M0)

- Múltiples aldeanos / múltiples recursos (M2).
- `gather` real / reward por Δstock (M1).
- Vectorización / múltiples procesos de juego en paralelo.
- Combate, enemigos, mecánicas custom del mod (Falange, Stamina) en la recompensa.
- Tuning serio de hiperparámetros (se usan los defaults de SB3 SAC en M0).

## Dependencias

- `pip install .` del paquete `zero_ad` (en `~/Documents/0ad/source/tools/rlclient/python/`).
- `pip install gymnasium stable-baselines3`.
- Binario `pyrogenesis` compilado con soporte `--rl-interface` (ya verificado en este build).
