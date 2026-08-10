# RL Gather — Milestones 0 y 1

Primer experimento de Reinforcement Learning sobre el mod: **SAC entrena 1 aldeano (Polites)
para ir a 1 recurso**, usando la interfaz RL nativa de 0 A.D. (`--rl-interface`) + el cliente
`zero_ad`, envuelto en un `gymnasium.Env`.

**Estado: M0 cumplido y M1 implementado.** M0 valida navegación con reward por acercamiento.
M1 reutiliza la misma arquitectura modular y la misma vista de política, pero cambia el entorno
a reward por **Δstock real** usando acción `[target_x, target_z, click_signal]`.
Si `click_signal` está activo y el target cae sobre el árbol, el env emite un comando `gather`.
Si `click_signal` está inactivo, el env avanza la simulación sin comando para que UnitAI de
0 A.D. pueda completar el ciclo de juntar/volver/depositar. Si la política clickea durante ese
ciclo, el click se envía igual y puede interrumpirlo: aprender a no clicar también es parte de M1.
El entorno lee el stock de madera con `game.evaluate(...)` y recompensa la madera
efectivamente depositada. M1 suma shaping opcional por acercarse, entrar en rango de
recolección, empezar a cargar madera, no clicar mientras UnitAI completa el ciclo útil, y
penalizar clicks que interrumpen ese ciclo.

## Cómo correr todo (paso a paso)

> **Reglas de oro:**
> 1. Todos los comandos se corren **desde la raíz del repo** (donde está la carpeta `rl/`),
>    si no, los `import rl.*` fallan.
> 2. Usá los comandos `make`: el repo fija Python 3.11 y todas las dependencias en `uv.lock`.
> 3. Mantené abierto el proceso de 0 A.D. mientras entrenás o evaluás.

### Paso 0 — Instalar dependencias (una sola vez)

```bash
cd /ruta/al/0ad-aoe3-mod
# Si `uv` no existe en tu terminal Ubuntu:
sudo apt install pipx
pipx install uv
export PATH="$HOME/.local/bin:$PATH"

make setup
```

Si usaste el instalador standalone de `uv` desde una terminal de VS Code Snap y
lo instaló bajo `~/snap/code/.../.local/bin`, corré el `source .../env` que
imprime el instalador, o simplemente repetí `make setup`; el Makefile también
busca esa ruta de Snap.

Eso crea `.venv/` con Python 3.11, SAC/SB3, PyTorch CPU y el cliente `zero_ad` fijado al
commit oficial de 0 A.D. Release 28. `pyproject.toml` declara las dependencias y `uv.lock`
fija las versiones y hashes exactos. Esos dos archivos son la única fuente de dependencias.
`make setup` es un wrapper corto de `uv sync --locked`; corré `make help` para listar todos
los comandos disponibles.

### Paso 1 — Construir y arrancar el server de 0 A.D. (Terminal 1)

```bash
make engine-observer  # sólo la primera vez; compila 0 A.D. Release 28
make server           # headless, recomendado para entrenar
# o, para usar --agent-view:
make server-view
```
Esperá hasta que imprima **`RL interface listening on 127.0.0.1:6000`**. `make server` no abre
una ventana; `make server-view` sí. Dejá esta terminal abierta. `Ctrl+C` detiene el server.

### Paso 2 — Correr la política / evaluar (Terminal 2)

```bash
make oracle EPISODES=1 ARGS="--mode deterministic --delay 0.5 --agent-view --verbose"
```
Esto ejecuta el **oracle** (baseline que apunta directamente a las coordenadas del recurso): el
aldeano camina hacia el árbol en la ventana de 0 A.D. Para correr M1 con stock real:

```bash
make m1-oracle EPISODES=1 ARGS="--mode deterministic --delay 0.5 --agent-view --verbose"
```

En M1, `--verbose` también imprime `stock` y `dstock` cuando la madera entra al stock del
jugador. Para evaluar un modelo aprendido, indicá su experimento y checkpoint:

```bash
make eval MODEL=rl/runs/REEMPLAZAR_CON_LA_CORRIDA/model \
  TRUST_MODEL=1 ARGS="--mode both"
# o para un checkpoint entrenado con M1:
make m1-eval MODEL=rl/runs/REEMPLAZAR_CON_LA_CORRIDA/model \
  TRUST_MODEL=1 ARGS="--mode both"
```

Reemplazá `REEMPLAZAR_CON_LA_CORRIDA` por el directorio exacto que imprime `rl.train`;
no copies los caracteres `<` y `>` en un comando de shell.

> Los checkpoints de SB3 pueden contener objetos Python serializados. Cargá sólo modelos que
> generaste vos o cuya fuente confiás; `--trust-model` hace explícita esa decisión.

`--agent-view` abre una segunda ventana centrada en el Polites. La imagen viene directamente del
renderer de 0 A.D.: muestra el terreno, los modelos, las animaciones y la niebla de guerra que ve
el Player 1, usando el rango vivo del componente `Vision` del aldeano. No mueve la cámara principal.
Debajo de la imagen se conserva el input normalizado que recibe la política. M0 usa cinco
valores geométricos; M1 agrega `carried_wood_norm` y `stock_wood_norm` para que la política pueda
distinguir cuándo conviene dejar correr UnitAI. La advertencia explícita sigue siendo válida:
M0/M1 todavía entregan las coordenadas globales del árbol aunque esté fuera de la visión física.
Esos píxeles son sólo para depuración; la política no los consume.

La primera compilación con `make engine-observer` tarda, ocupa aproximadamente 7 GB y queda en
`.runtime/0ad-observer/`. En Ubuntu, instalá primero:

```bash
sudo apt install build-essential cmake curl libboost-dev libboost-filesystem-dev \
  libcurl4-gnutls-dev libenet-dev libfmt-dev libfreetype-dev libicu-dev \
  libpng-dev libsdl2-dev libsodium-dev libx11-dev libxml2-dev llvm m4 \
  patch pkg-config python3 uuid-dev xvfb zlib1g-dev
```

Este build de visualización omite audio, lobby y Atlas. El observer usa el backend OpenGL de Release 28;
si no hay un display X11, `make server-view` intenta usar `xvfb-run` automáticamente. Mantené visible
la ventana del juego y con un tamaño de por lo menos 512×512 mientras uses `--agent-view`; esa
ventana de debug necesita una sesion grafica real.

Opciones de `rl.eval`:

| Flag | Para qué |
|------|----------|
| `--mode configured\|stochastic\|deterministic\|both` | por defecto respeta `evaluation.deterministic`; `both` compara ambos modos |
| `--delay 0.5` | pausa (seg) entre decisiones; con `--agent-view`, pausa antes de ejecutar la acción |
| `--agent-view` | abre el render real del LOS del Polites y audita aparte el input omnisciente de la política |
| `--verbose` | imprime paso a paso (observación, target, distancia, stock si existe, reward) |
| `--episodes N` | cuántos episodios correr (la config fija usa 1 para oracle/SAC y 10 para random) |
| `--replay` | guarda un replay por episodio (verlo después en 0 A.D. → menú **Replays**) |
| `--record-agent-view DIR` | guarda frames PNG del observer y un `index.html` reproducible sin Tk |
| `--record-agent-view-video FILE.mp4` | codifica esos frames renderizados a MP4 con `ffmpeg` |
| `--allow-schematic-recording` | fallback explícito a frames esquemáticos si el renderer no captura |
| `--experiment rl/configs/...toml` | entorno, agente, seeds e hiperparámetros |
| `--model rl/runs/.../model` | checkpoint; sólo hace falta para agentes aprendidos |
| `--trust-model` | confirma que el checkpoint es confiable antes de deserializarlo |
| `--allow-remote-server` | habilita explícitamente un server no local; sólo si confiás en él |

### Reentrenar (opcional)

```bash
make train STEPS=2000
# M1: reward por madera real depositada
make m1-train STEPS=10000
```

Para continuar M1 desde un checkpoint propio, usá `MODEL` como origen de reanudación y
confirmá que confiás en ese archivo:

```bash
make m1-train MODEL=rl/runs/REEMPLAZAR_CON_LA_CORRIDA/best_model \
  TRUST_MODEL=1 STEPS=100000
```

Cada corrida crea una carpeta ignorada por git en `rl/runs/` con el modelo, la config resuelta,
las métricas por episodio y metadata. El comando imprime la ruta exacta al terminar.
La config y `status=running` se escriben al crear la corrida; un fallo previo al checkpoint queda
marcado como `failed` o `interrupted` en vez de dejar un directorio vacío.
El modelo se guarda **antes** de la evaluación final: si el server se corta durante esa etapa,
el entrenamiento largo no se pierde. `--out` no pisa un checkpoint existente salvo que agregues
`--force`.
Cada checkpoint SAC incluye un archivo hermano `*.replay_buffer.pkl`; conservá ambos para
reanudar sin descartar la experiencia acumulada. `TRUST_MODEL=1` cubre los dos archivos
serializados. Los checkpoints antiguos que sólo tienen el modelo siguen funcionando, pero SAC
vuelve a llenar un buffer inicial antes de actualizar la red.
Mientras entrena SAC, también guarda `best_model` cuando mejora el reward medio del siguiente
bloque de 10 episodios terminados. Esto evita elegir un outlier y reduce escrituras síncronas;
usá ese checkpoint para inspeccionar la mejor política encontrada aunque el run siga abierto.

El entrenamiento imprime la carpeta de la corrida **antes** de empezar y SB3 escribe métricas en
`training/progress.csv` y `training/progress.json` dentro de esa carpeta. Para seguirlo en vivo:

```bash
tail -f rl/runs/ULTIMA_CORRIDA/training/progress.csv
```

`training.log_interval = 1` en los TOML hace que SB3 vuelque métricas cada episodio terminado;
podés cambiarlo por corrida con `make m1-train ARGS="--log-interval 2"`.

`make train` y `make m1-train` no capturan imágenes durante SAC. Para depurar visualmente una
corrida, arrancá el engine con `make server-view` y agregá la vista de forma explícita, por
ejemplo `make train ARGS="--agent-view --delay 0.5"`. La vista se actualiza con cada decisión y
sigue activa durante la evaluación final.

### Correr los tests (no necesitan el juego)

```bash
make test
# Tests, lint, imports, dependencias y lockfile:
make verify
```

### Si algo se traba

```bash
pgrep -af 'pyrogenesis|0ad'      # buscá el PID exacto del server colgado
kill 12345                       # reemplazá 12345 por ese PID; probá sin -KILL primero
```
Si no responde después de unos segundos, usá `kill -KILL 12345` con el mismo PID y relanzá el
Paso 1. (Detalle en "Lecciones del server headless" más abajo.)

## Qué vas a ver

Con `--mode both --verbose`:
- **Estocástico:** el agente llega al recurso (recompensa alta, episodios cortos) — la política
  aprendida.
- **Determinista:** la acción *media* lleva al aldeano de ~160 m a ~20 m del árbol. Aprendió a
  navegar hacia el recurso, pero con solo 2000 steps la media no converge fina (el aldeano
  físicamente frena ~9.5 m antes del árbol porque es un obstáculo sólido; `reach_threshold=12`).
  Un run más largo (5000+) afina la política determinista.

## Componentes

| Archivo | Qué es |
|---------|--------|
| `agents/base.py` | Contratos mínimos `Policy`/`Trainer` y resultados inmutables |
| `agents/registry.py` | Registro explícito de implementaciones y capacidades |
| `agents/baselines.py` | Políticas random y oracle para validar/comparar |
| `agents/sb3.py` | Adaptador de SAC de Stable-Baselines3; import opcional y tardío |
| `agents/custom/` | Lugar de las implementaciones de los alumnos |
| `experiments/` | Config TOML, construcción, entrenamiento, evaluación y artefactos comunes |
| `configs/` | Experimentos versionados y comparables (`m0_*.toml`, `m1_*.toml`) |
| `gather/core.py` | Funciones puras (geometría, normalización, observación, rewards) — con tests |
| `gather/env.py` | `ZeroADGatherEnv(gymnasium.Env)` sobre `zero_ad`, con backend inyectable |
| `gather/agent_view.py` | Ventana debug opcional con el frame real producido por el engine |
| `gather/engine_observer.py` | Cliente y validación del endpoint de frames PPM del engine |
| `train.py`, `eval.py` | CLIs finas: parsean opciones y delegan a los módulos anteriores |
| `../run_game.sh`, `run_server.sh` | Lanzadores RL para AppImage y build desde source |
| `../engine/` | Patch versionado de Release 28 y builder del observer renderizado |
| `reset_config.json` | Config de la partida (mapa `random/rl_gather`, civ athenai, 1 jugador) |
| `tests/` | Contratos unitarios/integración offline; no necesitan el juego ni `zero_ad` |

El mapa determinista (1 Polites + 1 árbol + 1 storehouse) está en
`maps/random/rl_gather.{js,json}` (parte del mod).

## Lecciones del server headless (encapsuladas en `run_server.sh`)

- El proceso se llama **`main`** en `comm`, no `pyrogenesis` → `pkill -x pyrogenesis` NO lo mata;
  hay que matar por cmdline (`pkill -f`). Si no, se acumulan servidores zombis que traban el puerto.
- Hay que **desactivar el splashscreen** (`gui.splashscreen.enable=false`): en headless bloquea el arranque.
- El config necesita `settings.mapName` o cada `reset` tira un error de l10n que desestabiliza el server.
- **No abrir conexiones TCP de prueba** al puerto del RL interface (lo desestabiliza); esperar la
  línea `RL interface listening` en el log.

## Roadmap

- **M1** — Acción = `[target_x, target_z, click_signal]`; click cerca del árbol emite `gather`,
  no-click deja correr UnitAI; reward = Δstock real en vez de acercamiento. ✓ El env puede
  reintentar/reconectar y trunca de forma limpia un episodio interrumpido.
- **M2** — 4 aldeanos + varios recursos. Acción `Box(8,)` (un punto por aldeano), reward = Δstock
  colectivo. Sigue con posiciones **conocidas** (observación incluye los recursos).
- **M3+ (idea futura) — Exploración con niebla de guerra.** Un salto de dificultad: el agente
  **no** conoce dónde están los recursos y tiene que **explorar para encontrarlos**.
  Implica:
  - Observabilidad **parcial**: la observación NO incluye las posiciones de los recursos
    (solo lo revelado por la niebla de guerra) → quizás obs basada en un mapa/grilla de visibilidad.
  - Reward por **encontrar/recolectar** de verdad (no por acercarse a un punto conocido).
  - El agente debe **aprender a explorar** (mucho más caro de entrenar; probablemente requiera
    recurrencia/memoria o curiosidad/intrinsic reward).
  - Es un problema cualitativamente distinto a M0–M2 (búsqueda, no navegación a objetivo conocido).

## Arquitectura modular para alumnos

La dependencia central va en una sola dirección:

```text
config TOML -> registry -> Trainer.fit(TrainRequest) -> Policy
                                                   |
entorno Gym ---------------------------------------+-> evaluator -> métricas
```

El evaluador sólo llama `Policy.act()`: no sabe si la política viene de SAC, una red propia,
una tabla, random o el oracle. Cada `Trainer` es dueño de su loop de actualización. Por eso se
puede cambiar un algoritmo sin tocar el entorno y comparar todos con exactamente los mismos
episodios y seeds.

Para comprobar primero el pipeline y los dos extremos de referencia:

```bash
make random ARGS="--mode deterministic"
make oracle ARGS="--mode deterministic"
make m1-oracle ARGS="--mode deterministic --verbose"
```

Para inspeccionar un rollout sin ventana Tk, grabá el observer renderizado a PNG + HTML + MP4:

```bash
make m1-oracle EPISODES=1 ARGS="--mode deterministic --verbose --record-agent-view rl/runs/m1-oracle-rollout --record-agent-view-video rl/runs/m1-oracle-rollout.mp4"
make m1-eval MODEL=rl/runs/REEMPLAZAR_CON_LA_CORRIDA/best_model TRUST_MODEL=1 EPISODES=1 \
  ARGS="--mode deterministic --verbose --record-agent-view rl/runs/m1-model-rollout --record-agent-view-video rl/runs/m1-model-rollout.mp4"
```

El oracle **no aprende**: usa la posición del recurso presente en la observación y marca un techo
de navegación para detectar errores del entorno/reward. Random marca el piso. El algoritmo del
alumno debería compararse con ambos.

Para sumar una implementación propia, seguí `agents/custom/README.md`: implementá un `Policy`,
un `Trainer`, agregá una entrada explícita al registro y un TOML. Los tests del agente van en
`tests/agents/` y usan entornos falsos; el test contra 0 A.D. queda como integración separada.

## Por dónde empezar

**La parte difícil ya está hecha: la infraestructura para entrenar.** Conectar el motor de 0 A.D.,
un entorno tipo Gym estable, lanzar el server headless de forma confiable, el mapa y la config —
todo eso (lo tedioso) está resuelto y es **reutilizable**. Ustedes se concentran en el RL.

### Lo que heredan ya armado

| Ya hecho (no lo toquen, reúsenlo) | Dónde |
|---|---|
| Lanzar 0 A.D. con la interfaz RL | `make engine-observer` una vez; después `make server` (headless) o `make server-view` (debug) |
| Entorno Gym (`reset`/`step`/obs/acción) | `gather/env.py` (lo **extienden**, no lo reescriben) |
| Orquestación de entrenamiento + evaluación | `experiments/training.py`, `experiments/evaluation.py` |
| Conexión al motor y acciones (`walk`/`gather`/…) | cliente `zero_ad` |
| Mapa/escenario parametrizable | `maps/random/rl_gather.js` |
| Funciones puras testeadas (geometría, obs, reward) | `gather/core.py` + `tests/` |

### Orden sugerido (rampa de dificultad)

1. **Arrancá corriendo lo que ya está** (sección "Cómo correr todo"): mirá al agente de M0 ir al recurso.
2. **M1** — corré `make m1-oracle` y `make m1-train`: ahora el reward es Δstock real.
3. **M2** — escalá a 4 aldeanos + varios recursos (acción `Box(8,)`). Sigue con posiciones conocidas.
4. **M3+ (capstone)** — exploración con niebla de guerra. Ambicioso; encaralo al final.

### Dónde meter mano (enganches concretos)

- **Reward:** `reward_mode` en los TOML; `gather_reward()` / `stock_delta_reward()` en
  `gather/core.py`; lectura de stock con `game.evaluate(...)` desde `env.py`.
- **Observación:** `build_observation()` en `gather/core.py` + `_positions()` en `env.py`.
  En M1, `resource_state_observation` agrega madera cargada y stock actual normalizados.
- **Acción / nº de aldeanos:** `action_space` y `step()` en `gather/env.py`; en M1
  `[target_x, target_z, click_signal]` separa apuntar de clicar.
- **Escenario (recursos, tamaño, niebla):** `maps/random/rl_gather.js` + `reset_config.json`.
- **Algoritmo:** un `Policy` + `Trainer` en `agents/custom/`, conectado en `agents/registry.py`.
- **Hiperparámetros:** un TOML propio en `configs/`; no se hardcodean en `train.py`.

### Dos advertencias honestas

1. **Para corridas largas, preferí M1 con retries activos** (`backend_retries` en el TOML) y
   relanzá el server si el proceso externo quedó colgado.
2. **La exploración (M3+) es mucho más difícil de entrenar** que M0–M2 (reward esparso, hay que
   aprender a explorar; quizás política con memoria/recurrencia, ej. `RecurrentPPO` de `sb3-contrib`).
   No la encaren como primer proyecto: hagan M1/M2 de calentamiento.
