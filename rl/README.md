# RL Gather — Milestone 0

Primer experimento de Reinforcement Learning sobre el mod: **SAC entrena 1 aldeano (Polites)
para ir a 1 recurso**, usando la interfaz RL nativa de 0 A.D. (`--rl-interface`) + el cliente
`zero_ad`, envuelto en un `gymnasium.Env`.

**Estado: M0 cumplido.** El loop completo (Gymnasium ↔ Stable-Baselines3 ↔ 0 A.D.) funciona y
el agente **aprende la política óptima**: en un run de 2000 steps el reward medio subió de
~74 a **~151** (máximo ~150) y los episodios se acortaron de 50 a ~12. Los checkpoints son
regenerables, no se versionan, y ahora quedan aislados por corrida en `rl/runs/`.

## Cómo correr todo (paso a paso)

> **Reglas de oro:**
> 1. Todos los comandos se corren **desde la raíz del repo** (donde está la carpeta `rl/`),
>    si no, los `import rl.*` fallan.
> 2. Usá `uv run ...`: el repo fija Python 3.11 y todas las dependencias en `uv.lock`.
> 3. Mantené abierto el proceso de 0 A.D. mientras entrenás o evaluás.

### Paso 0 — Instalar dependencias (una sola vez)

```bash
cd /ruta/al/0ad-aoe3-mod
uv sync --locked
```

Eso crea `.venv/` con Python 3.11, SAC/SB3, PyTorch CPU y el cliente `zero_ad` fijado al
commit oficial de 0 A.D. Release 28. `pyproject.toml` declara las dependencias y `uv.lock`
fija las versiones y hashes exactos. Esos dos archivos son la única fuente de dependencias.

### Paso 1 — Arrancar el server de 0 A.D. (Terminal 1)

```bash
./run_game.sh --rl-interface=127.0.0.1:6000
```
Esperá hasta que imprima **`RL interface listening on 127.0.0.1:6000`** (se abre la ventana
del juego). Dejá esta terminal abierta. `Ctrl+C` para detener el server.

### Paso 2 — Correr la política / evaluar (Terminal 2)

```bash
uv run python -m rl.eval --experiment rl/configs/m0_oracle.toml \
  --mode deterministic --delay 0.4 --verbose
```
Esto ejecuta el **oracle** (baseline que apunta directamente a las coordenadas del recurso): el
aldeano camina hacia el árbol en la ventana de 0 A.D. Para evaluar un modelo aprendido, indicá
su experimento y checkpoint:

```bash
uv run python -m rl.eval --experiment rl/configs/m0_sb3_sac.toml \
  --model rl/runs/REEMPLAZAR_CON_LA_CORRIDA/model --trust-model --mode both
```

Reemplazá `REEMPLAZAR_CON_LA_CORRIDA` por el directorio exacto que imprime `rl.train`;
no copies los caracteres `<` y `>` en un comando de shell.

> Los checkpoints de SB3 pueden contener objetos Python serializados. Cargá sólo modelos que
> generaste vos o cuya fuente confiás; `--trust-model` hace explícita esa decisión.

Opciones de `rl.eval`:

| Flag | Para qué |
|------|----------|
| `--mode configured\|stochastic\|deterministic\|both` | por defecto respeta `evaluation.deterministic`; `both` compara ambos modos |
| `--delay 0.4` | pausa (seg) entre pasos, para **seguir la partida con el ojo** |
| `--verbose` | imprime paso a paso (target, distancia, reward) |
| `--episodes N` | cuántos episodios correr (default 10) |
| `--replay` | guarda un replay por episodio (verlo después en 0 A.D. → menú **Replays**) |
| `--experiment rl/configs/...toml` | entorno, agente, seeds e hiperparámetros |
| `--model rl/runs/.../model` | checkpoint; sólo hace falta para agentes aprendidos |
| `--trust-model` | confirma que el checkpoint es confiable antes de deserializarlo |
| `--allow-remote-server` | habilita explícitamente un server no local; sólo si confiás en él |

### Reentrenar (opcional)

```bash
uv run python -m rl.train --experiment rl/configs/m0_sb3_sac.toml --timesteps 2000
```

Cada corrida crea una carpeta ignorada por git en `rl/runs/` con el modelo, la config resuelta,
las métricas por episodio y metadata. El comando imprime la ruta exacta al terminar.
El modelo se guarda **antes** de la evaluación final: si el server se corta durante esa etapa,
el entrenamiento largo no se pierde. `--out` no pisa un checkpoint existente salvo que agregues
`--force`.

### Correr los tests (no necesitan el juego)

```bash
uv run pytest rl/tests/ -v
# Con el umbral de cobertura del repo (mínimo 80%):
uv run pytest --cov
```

### Si algo se traba

```bash
pkill -9 -f pyrogenesis          # matar servers colgados (OJO: el proceso se llama 'main')
```
y relanzá el Paso 1. (Detalle en "Lecciones del server headless" más abajo.)

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
| `configs/` | Experimentos versionados y comparables |
| `gather/core.py` | Funciones puras (geometría, normalización, observación, reward) — con tests |
| `gather/env.py` | `ZeroADGatherEnv(gymnasium.Env)` sobre `zero_ad`, con backend inyectable |
| `train.py`, `eval.py` | CLIs finas: parsean opciones y delegan a los módulos anteriores |
| `../run_game.sh`, `run_server.sh` | Lanzadores RL para AppImage y build desde source |
| `reset_config.json` | Config de la partida (mapa `random/rl_gather`, civ athenai, 1 jugador) |
| `tests/` | Contratos unitarios/integración offline; no necesitan el juego ni `zero_ad` |

El mapa determinista (1 Polites + 1 árbol) está en `maps/random/rl_gather.{js,json}` (parte del mod).

## Lecciones del server headless (encapsuladas en `run_server.sh`)

- El proceso se llama **`main`** en `comm`, no `pyrogenesis` → `pkill -x pyrogenesis` NO lo mata;
  hay que matar por cmdline (`pkill -f`). Si no, se acumulan servidores zombis que traban el puerto.
- Hay que **desactivar el splashscreen** (`gui.splashscreen.enable=false`): en headless bloquea el arranque.
- El config necesita `settings.mapName` o cada `reset` tira un error de l10n que desestabiliza el server.
- **No abrir conexiones TCP de prueba** al puerto del RL interface (lo desestabiliza); esperar la
  línea `RL interface listening` en el log.

## Roadmap

- **M1** — Reward = Δstock real (emitir `gather`) en vez de acercamiento. *Antes*, hacer el `env`
  **resiliente** (reintentar/relanzar el server ante desconexiones) para entrenamientos largos confiables.
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
uv run python -m rl.eval --experiment rl/configs/m0_random.toml --mode deterministic
uv run python -m rl.eval --experiment rl/configs/m0_oracle.toml --mode deterministic
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
| Lanzar 0 A.D. con la interfaz RL | `../run_game.sh --rl-interface=127.0.0.1:6000` |
| Entorno Gym (`reset`/`step`/obs/acción) | `gather/env.py` (lo **extienden**, no lo reescriben) |
| Orquestación de entrenamiento + evaluación | `experiments/training.py`, `experiments/evaluation.py` |
| Conexión al motor y acciones (`walk`/`gather`/…) | cliente `zero_ad` |
| Mapa/escenario parametrizable | `maps/random/rl_gather.js` |
| Funciones puras testeadas (geometría, obs, reward) | `gather/core.py` + `tests/` |

### Orden sugerido (rampa de dificultad)

1. **Arrancá corriendo lo que ya está** (sección "Cómo correr todo"): mirá al agente de M0 ir al recurso.
2. **M1** — cambiá el reward a Δstock real (que junte, no que se acerque). Tractable, reusa todo.
3. **M2** — escalá a 4 aldeanos + varios recursos (acción `Box(8,)`). Sigue con posiciones conocidas.
4. **M3+ (capstone)** — exploración con niebla de guerra. Ambicioso; encaralo al final.

### Dónde meter mano (enganches concretos)

- **Reward:** `gather_reward()` en `gather/core.py` (y leer el stock con `game.evaluate(...)` desde `env.py`).
- **Observación:** `build_observation()` en `gather/core.py` + `_positions()` en `env.py`.
- **Acción / nº de aldeanos:** `action_space` y `step()` en `gather/env.py`.
- **Escenario (recursos, tamaño, niebla):** `maps/random/rl_gather.js` + `reset_config.json`.
- **Algoritmo:** un `Policy` + `Trainer` en `agents/custom/`, conectado en `agents/registry.py`.
- **Hiperparámetros:** un TOML propio en `configs/`; no se hardcodean en `train.py`.

### Dos advertencias honestas

1. **Cierren primero la resiliencia del `env`** (la tarea-prep de M1: reintentar/relanzar el server
   ante desconexiones). Hoy el server se cae solo y los entrenamientos largos se cortan — y RL
   necesita MUCHAS muestras. Es la única pieza de infra que falta y la van a agradecer.
2. **La exploración (M3+) es mucho más difícil de entrenar** que M0–M2 (reward esparso, hay que
   aprender a explorar; quizás política con memoria/recurrencia, ej. `RecurrentPPO` de `sb3-contrib`).
   No la encaren como primer proyecto: hagan M1/M2 de calentamiento.
