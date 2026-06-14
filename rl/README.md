# RL Gather — Milestone 0

Primer experimento de Reinforcement Learning sobre el mod: **SAC entrena 1 aldeano (Polites)
para ir a 1 recurso**, usando la interfaz RL nativa de 0 A.D. (`--rl-interface`) + el cliente
`zero_ad`, envuelto en un `gymnasium.Env`.

**Estado: M0 cumplido.** El loop completo (Gymnasium ↔ Stable-Baselines3 ↔ 0 A.D.) funciona y
el agente **aprende la política óptima**: en un run de 2000 steps el reward medio sube de
~74 a **~151** (máximo ~150) y los episodios se acortan de 50 a ~12 (llega al recurso cada vez
más rápido). Modelo entrenado: `rl/sac_gather.zip`.

## Setup (una vez)

```bash
PY=~/Documents/0ad/.toolchain/python/bin/python3   # Python 3.11 (ver MANUAL.md)
$PY -m pip install -r rl/requirements.txt
$PY -m pip install ~/Documents/0ad/source/tools/rlclient/python   # cliente zero_ad
```

## Probarlo / evaluar el modelo

Necesitás **dos terminales**.

**Terminal 1 — arrancar el server RL** (robusto; Ctrl+C para detener):
```bash
bash rl/run_server.sh
```

**Terminal 2 — evaluar el agente entrenado** (mirá cómo se acerca al recurso):
```bash
PY=~/Documents/0ad/.toolchain/python/bin/python3
$PY -m rl.eval --verbose                 # traza paso a paso (target, distancia, reward)
# opciones: --episodes N  --mode deterministic|stochastic|both  --model rl/sac_gather
```

**Reentrenar** (en vez de evaluar):
```bash
$PY -m rl.train --timesteps 2000         # entrena SAC y guarda rl/sac_gather.zip
```

> **Importante:** un server = un cliente. El server se cierra cuando el cliente (eval/train)
> se desconecta, así que volvé a correr `run_server.sh` para cada sesión.

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
| `eval.py` | Evalúa un modelo guardado (determinista/estocástico, `--verbose`) |
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
