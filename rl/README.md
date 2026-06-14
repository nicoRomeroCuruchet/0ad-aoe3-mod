# RL Gather — Milestone 0

Primer experimento de Reinforcement Learning sobre el mod: **SAC entrena 1 aldeano (Polites)
para ir a 1 recurso**, usando la interfaz RL nativa de 0 A.D. (`--rl-interface`) + el cliente
`zero_ad`, envuelto en un `gymnasium.Env`.

**Estado: M0 cumplido.** El loop completo (Gymnasium ↔ Stable-Baselines3 ↔ 0 A.D.) funciona y
el agente **aprende la política óptima**: en un run de 2000 steps el reward medio sube de
~74 a **~151** (máximo ~150) y los episodios se acortan de 50 a ~12 (llega al recurso cada vez
más rápido). Modelo entrenado: `rl/sac_gather.zip`.

## Cómo correr todo (paso a paso)

> **Reglas de oro:**
> 1. Todos los comandos se corren **desde la raíz del repo** (donde está la carpeta `rl/`),
>    si no, los `import rl.*` fallan: `cd ~/dev/research/0ad-aoe3-mod`
> 2. Siempre usá el **Python 3.11** standalone (no el del sistema):
>    `~/Documents/0ad/.toolchain/python/bin/python3`
> 3. **Un server = un cliente.** El server se cierra cuando el cliente (eval/train) se
>    desconecta → relanzá `run_server.sh` para cada corrida.

### Paso 0 — Instalar dependencias (una sola vez)

```bash
cd ~/dev/research/0ad-aoe3-mod
PY=~/Documents/0ad/.toolchain/python/bin/python3
$PY -m pip install -r rl/requirements.txt
$PY -m pip install ~/Documents/0ad/source/tools/rlclient/python   # cliente zero_ad
```

> Si `~/Documents/0ad/.toolchain` no existe, recrealo según `MANUAL.md` (Python 3.11 efímero),
> o usá cualquier otro Python 3.11 con esas deps.

### Paso 1 — Arrancar el server de 0 A.D. (Terminal 1)

```bash
cd ~/dev/research/0ad-aoe3-mod
bash rl/run_server.sh
```
Esperá hasta que imprima **`Server RL listo en 127.0.0.1:6000`** (se abre la ventana del juego).
Dejá esta terminal abierta. `Ctrl+C` para detener el server.

### Paso 2 — Correr la política / evaluar (Terminal 2)

```bash
cd ~/dev/research/0ad-aoe3-mod
PY=~/Documents/0ad/.toolchain/python/bin/python3
$PY -m rl.eval --mode stochastic --delay 0.4 --verbose
```
Esto carga el modelo `rl/sac_gather.zip` y **ejecuta la política**: el aldeano camina hacia el
árbol en la ventana de 0 A.D.

Opciones de `rl.eval`:

| Flag | Para qué |
|------|----------|
| `--mode stochastic\|deterministic\|both` | `stochastic` llega al árbol; `deterministic` usa la acción media |
| `--delay 0.4` | pausa (seg) entre pasos, para **seguir la partida con el ojo** |
| `--verbose` | imprime paso a paso (target, distancia, reward) |
| `--episodes N` | cuántos episodios correr (default 10) |
| `--replay` | guarda un replay por episodio (verlo después en 0 A.D. → menú **Replays**) |
| `--model rl/sac_gather` | qué modelo cargar |

### Reentrenar (opcional)

```bash
cd ~/dev/research/0ad-aoe3-mod          # con el server del Paso 1 corriendo
PY=~/Documents/0ad/.toolchain/python/bin/python3
$PY -m rl.train --timesteps 2000        # entrena SAC y guarda rl/sac_gather.zip
```

### Correr los tests (no necesitan el juego)

```bash
cd ~/dev/research/0ad-aoe3-mod
~/Documents/0ad/.toolchain/python/bin/python3 -m pytest rl/tests/ -v
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
| `gather/core.py` | Funciones puras (geometría, normalización, observación, reward) — con tests |
| `gather/env.py` | `ZeroADGatherEnv(gymnasium.Env)` sobre `zero_ad` |
| `train.py` | Entrena SAC + eval rápida al final |
| `eval.py` | Corre/evalúa un modelo guardado (`--mode`, `--delay`, `--verbose`, `--replay`, `--episodes`) |
| `run_server.sh` | Lanza 0 A.D. headless con la interfaz RL de forma confiable |
| `reset_config.json` | Config de la partida (mapa `random/rl_gather`, civ athenai, 1 jugador) |
| `tests/` | `pytest rl/tests/` (lógica pura, no necesita el juego) |

El mapa determinista (1 Polites + 1 árbol) está en `maps/random/rl_gather.{js,json}` (parte del mod).

## Lecciones del server headless (encapsuladas en `run_server.sh`)

- El proceso se llama **`main`** en `comm`, no `pyrogenesis` → `pkill -x pyrogenesis` NO lo mata;
  hay que matar por cmdline (`pkill -f`). Si no, se acumulan servidores zombis que traban el puerto.
- Hay que **desactivar el splashscreen** (`gui.splashscreen.enable=false`): en headless bloquea el arranque.
- El config necesita `settings.mapName` o cada `reset` tira un error de l10n que desestabiliza el server.
- **No abrir conexiones TCP de prueba** al puerto del RL interface (lo desestabiliza); esperar la
  línea `RL interface listening` en el log.

## Próximo (M1)

Reward = Δstock real (emitir `gather`) en vez de acercamiento. Antes, hacer el `env` **resiliente**
(reintentar/relanzar el server ante desconexiones) para entrenamientos largos confiables.
