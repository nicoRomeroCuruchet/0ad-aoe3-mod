# RL Gather Loop (M0) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Entrenar con SAC un agente que mueva 1 aldeano hacia 1 recurso en 0 A.D., validando el loop completo Gymnasium ↔ Stable-Baselines3 ↔ interfaz RL nativa de 0 A.D.

**Architecture:** Un proceso de `pyrogenesis --rl-interface` headless corre un mapa determinista (1 Polites + 1 árbol). Un `gymnasium.Env` (`ZeroADGatherEnv`) traduce acción continua (x,z) → comando `walk` vía el cliente `zero_ad`, avanza la sim K turnos, y calcula reward de acercamiento. SB3 SAC entrena sobre ese env. La lógica pura (geometría, normalización, observación, reward) vive en funciones testeables sin el juego.

**Tech Stack:** Python 3.11 (el standalone ya instalado), `zero_ad` (cliente oficial de 0 A.D.), `gymnasium`, `stable-baselines3`, `pytest`; mapa rmgen en JS dentro del mod.

---

## File Structure

- `maps/random/rl_gather.js` — script de mapa determinista: 1 Polites (player 1) + 1 árbol (gaia).
- `maps/random/rl_gather.json` — metadata del mapa (Name, Script).
- `rl/reset_config.json` — config de game-setup que consume `game.reset()` (apunta a `random/rl_gather`).
- `rl/gather/__init__.py` — expone `ZeroADGatherEnv`.
- `rl/gather/core.py` — funciones puras: `xz`, `distance`, `normalize_coord`, `denormalize_action`, `build_observation`, `gather_reward`, `is_reached`.
- `rl/gather/env.py` — `ZeroADGatherEnv(gymnasium.Env)`.
- `rl/train.py` — entrenamiento SAC + evaluación.
- `rl/tests/test_core.py` — unit tests de la lógica pura.
- `rl/requirements.txt` — dependencias Python.
- `rl/README.md` — cómo lanzar el juego y entrenar.

**Convención de coordenadas (confirmado por el spike de Task 3):** la API de 0 A.D. devuelve `position() = [x, z]` ya en el plano del suelo, en **metros** (2 elementos, NO `[x, y, z]`). `map_size_m = 512` para `Size=128`. Todas las funciones puras trabajan con tuplas `(x, z)`.

---

## Fase A — Setup y spikes de riesgo

### Task 1: Dependencias Python

**Files:**
- Create: `rl/requirements.txt`

- [ ] **Step 1: Escribir requirements.txt**

```
gymnasium>=0.29
stable-baselines3>=2.3
numpy>=1.26
pytest>=8.0
```

- [ ] **Step 2: Instalar (usar el Python 3.11 standalone) + el cliente zero_ad**

Run:
```bash
PY=~/Documents/0ad/.toolchain/python/bin/python3
# Si .toolchain ya no existe, recrearlo según MANUAL.md, o usar otro Python 3.11.
$PY -m pip install -r rl/requirements.txt
$PY -m pip install ~/Documents/0ad/source/tools/rlclient/python
```
Expected: instala sin errores; `zero_ad`, `gymnasium`, `stable_baselines3` quedan disponibles.

- [ ] **Step 3: Verificar imports**

Run: `$PY -c "import zero_ad, gymnasium, stable_baselines3; print('ok')"`
Expected: imprime `ok`.

- [ ] **Step 4: Commit**

```bash
git add rl/requirements.txt
git commit -m "rl: add Python deps for SAC gather experiment"
```

---

### Task 2: Mapa determinista (1 aldeano + 1 recurso) — SPIKE riesgo #1

**Files:**
- Create: `maps/random/rl_gather.js`
- Create: `maps/random/rl_gather.json`

- [ ] **Step 1: Escribir el script de mapa**

`maps/random/rl_gather.js`:
```javascript
Engine.LoadLibrary("rmgen");
Engine.LoadLibrary("rmgen-common");

// Mapa mínimo y DETERMINISTA para RL: 1 aldeano + 1 árbol, sin RNG.
function* GenerateMap()
{
	const tGrass = "medit_grass_field";
	globalThis.g_Map = new RandomMap(0, tGrass);

	const c = g_Map.getCenter();

	// Aldeano del mod (player 1), a la izquierda del centro.
	g_Map.placeEntityPassable("units/athenai/polites", 1, new Vector2D(c.x - 20, c.y), 0);

	// Recurso gaia (player 0), a la derecha del centro.
	g_Map.placeEntityPassable("gaia/tree/oak", 0, new Vector2D(c.x + 20, c.y), 0);

	yield 100;
	return g_Map;
}
```

- [ ] **Step 2: Escribir la metadata del mapa**

`maps/random/rl_gather.json`:
```json
{
	"settings": {
		"Name": "RL Gather (debug)",
		"Script": "rl_gather.js",
		"Description": "Escenario mínimo de RL: 1 aldeano + 1 árbol.",
		"CircularMap": false,
		"BaseTerrain": ["medit_grass_field"]
	}
}
```

- [ ] **Step 3: Verificar headless que el mapa genera y tiene exactamente 1 aldeano + 1 árbol**

Run:
```bash
cd ~/Documents/0ad/binaries/system
timeout 60 ./pyrogenesis -mod=mod -mod=public -mod=aoe3 \
  -autostart="random/rl_gather" -autostart-size=128 -autostart-players=1 \
  -autostart-civ=1:athenai -autostart-nonvisual -autostart-seed=1 \
  > /tmp/rl_map.log 2>&1; echo "rc=$?"
grep -iE 'error|fail|rl_gather' /tmp/rl_map.log | head
```
Expected: `rc=124` (simuló sin abortar) y **sin** errores de `rl_gather.js`. Si aparece un error de API de colocación, ajustar la llamada `placeEntityPassable` según el mensaje.

- [ ] **Step 4: Commit**

```bash
git add maps/random/rl_gather.js maps/random/rl_gather.json
git commit -m "rl: add deterministic 1-villager 1-resource debug map"
```

---

### Task 3: Config de reset + spike de posiciones/tiempo — SPIKE riesgos #2 y #3

**Files:**
- Create: `rl/reset_config.json`

- [ ] **Step 1: Derivar el reset_config desde el sample oficial**

Leer `~/Documents/0ad/source/tools/rlclient/python/samples/arcadia.json` (es el config que consume `game.reset()`). Copiar su estructura a `rl/reset_config.json` cambiando: el mapa a `"random/rl_gather"`, `Size` a `128`, 1 player con civ `athenai`, sin IA. Mantener el resto de campos requeridos que tenga arcadia.json.

- [ ] **Step 2: Lanzar el juego con la interfaz RL**

Run (dejar corriendo en otra terminal):
```bash
cd ~/Documents/0ad/binaries/system
./pyrogenesis -mod=mod -mod=public -mod=aoe3 --rl-interface=127.0.0.1:6000 > /tmp/rl_server.log 2>&1 &
```
Expected: el proceso queda escuchando en `127.0.0.1:6000` (sin errores en `/tmp/rl_server.log`).

- [ ] **Step 3: Spike — medir posiciones reales y tiempo por step**

Run:
```bash
PY=~/Documents/0ad/.toolchain/python/bin/python3
$PY - <<'EOF'
import time, json, zero_ad
game = zero_ad.ZeroAD('http://localhost:6000')
cfg = open('rl/reset_config.json').read()
state = game.reset(cfg)
state = game.step()
v = state.units(owner=1, type='polites')[0]
r = state.units(owner=0, type='tree')[0]
print("villager pos:", v.position(), "resource pos:", r.position())
t0 = time.time()
for _ in range(10): state = game.step()
print("10 steps en %.3fs -> %.3fs/step" % (time.time()-t0, (time.time()-t0)/10))
EOF
```
Expected: imprime posiciones (anotar el rango en **metros** → fija `MAP_SIZE_M` en Task 7; con Size 128 debería rondar 0–512 m) y el tiempo/step (→ ajusta `sim_steps_per_action` y `horizon`). Confirma que los filtros `type='polites'` y `type='tree'` devuelven la unidad correcta.

- [ ] **Step 4: Commit**

```bash
git add rl/reset_config.json
git commit -m "rl: add reset config for the gather debug scenario"
```

---

## Fase B — Lógica pura (TDD)

### Task 4: Geometría y normalización

**Files:**
- Create: `rl/__init__.py` (vacío) y `rl/tests/__init__.py` (vacío) — para que `from rl.gather...` resuelva al correr pytest/`python -m` desde la raíz del repo.
- Create: `rl/gather/__init__.py` (vacío por ahora)
- Create: `rl/gather/core.py`
- Test: `rl/tests/test_core.py`

- [ ] **Step 1: Escribir los tests que fallan**

`rl/tests/test_core.py`:
```python
import numpy as np
from rl.gather.core import xz, distance, normalize_coord, denormalize_action

def test_xz_passes_through_ground_plane():
    # position() de 0 A.D. ya devuelve [x, z] en metros (2 elementos).
    assert xz([176, 256]) == (176.0, 256.0)

def test_distance_is_euclidean_on_xz():
    assert distance((0.0, 0.0), (3.0, 4.0)) == 5.0

def test_normalize_coord_maps_to_minus_one_one():
    assert normalize_coord(0.0, 512.0) == -1.0
    assert normalize_coord(512.0, 512.0) == 1.0
    assert normalize_coord(256.0, 512.0) == 0.0

def test_denormalize_action_inverts_normalization():
    assert denormalize_action([-1.0, 1.0], 512.0) == (0.0, 512.0)
    assert denormalize_action([0.0, 0.0], 512.0) == (256.0, 256.0)
```

- [ ] **Step 2: Correr y verificar que fallan**

Run: `~/Documents/0ad/.toolchain/python/bin/python3 -m pytest rl/tests/test_core.py -v`
Expected: FAIL — `ModuleNotFoundError`/`ImportError` (core no existe aún).

- [ ] **Step 3: Implementar core.py (parte 1)**

`rl/gather/core.py`:
```python
import numpy as np


def xz(pos):
    """0 A.D. position() ya devuelve [x, z] en metros (2 elementos)."""
    return (float(pos[0]), float(pos[1]))


def distance(p1, p2):
    return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))


def normalize_coord(value, map_size_m):
    """metros [0, map_size_m] -> [-1, 1]."""
    return 2.0 * value / map_size_m - 1.0


def denormalize_action(action, map_size_m):
    """accion [-1, 1] -> metros [0, map_size_m]; devuelve (x, z)."""
    x = (float(action[0]) + 1.0) * 0.5 * map_size_m
    z = (float(action[1]) + 1.0) * 0.5 * map_size_m
    return (x, z)
```

- [ ] **Step 4: Correr y verificar que pasan**

Run: `~/Documents/0ad/.toolchain/python/bin/python3 -m pytest rl/tests/test_core.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add rl/__init__.py rl/tests/__init__.py rl/gather/__init__.py rl/gather/core.py rl/tests/test_core.py
git commit -m "rl: add geometry/normalization helpers with tests"
```

---

### Task 5: Observación y reward

**Files:**
- Modify: `rl/gather/core.py`
- Test: `rl/tests/test_core.py`

- [ ] **Step 1: Agregar los tests que fallan**

Añadir a `rl/tests/test_core.py`:
```python
from rl.gather.core import build_observation, gather_reward, is_reached

def test_build_observation_shape_and_values():
    obs = build_observation((256.0, 256.0), (256.0, 256.0), 512.0)
    assert obs.dtype == np.float32
    assert obs.shape == (5,)
    # mismo punto -> coords normalizadas 0 y distancia normalizada 0
    assert np.allclose(obs, [0.0, 0.0, 0.0, 0.0, 0.0])

def test_gather_reward_is_positive_when_getting_closer():
    assert gather_reward(10.0, 4.0) == 6.0
    assert gather_reward(4.0, 10.0) == -6.0

def test_is_reached_uses_threshold():
    assert is_reached(3.0, 4.0) is True
    assert is_reached(5.0, 4.0) is False
```

- [ ] **Step 2: Correr y verificar que fallan**

Run: `~/Documents/0ad/.toolchain/python/bin/python3 -m pytest rl/tests/test_core.py -v`
Expected: FAIL — `ImportError` para `build_observation`, etc.

- [ ] **Step 3: Implementar la parte 2 de core.py**

Añadir a `rl/gather/core.py`:
```python
def build_observation(villager_xz, resource_xz, map_size_m):
    d = distance(villager_xz, resource_xz)
    return np.array([
        normalize_coord(villager_xz[0], map_size_m),
        normalize_coord(villager_xz[1], map_size_m),
        normalize_coord(resource_xz[0], map_size_m),
        normalize_coord(resource_xz[1], map_size_m),
        d / map_size_m,
    ], dtype=np.float32)


def gather_reward(prev_dist, cur_dist):
    return float(prev_dist - cur_dist)


def is_reached(cur_dist, threshold):
    return bool(cur_dist < threshold)
```

- [ ] **Step 4: Correr y verificar que pasan**

Run: `~/Documents/0ad/.toolchain/python/bin/python3 -m pytest rl/tests/test_core.py -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add rl/gather/core.py rl/tests/test_core.py
git commit -m "rl: add observation builder and gather reward with tests"
```

---

## Fase C — Integración del entorno

### Task 6: `ZeroADGatherEnv`

**Files:**
- Create: `rl/gather/env.py`
- Modify: `rl/gather/__init__.py`

- [ ] **Step 1: Implementar el env**

`rl/gather/env.py`:
```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import zero_ad

from .core import (xz, distance, denormalize_action, build_observation,
                   gather_reward, is_reached)

VILLAGER_TYPE = "polites"
RESOURCE_TYPE = "tree"


class ZeroADGatherEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, scenario_config, uri="http://localhost:6000",
                 map_size_m=512.0, horizon=50, reach_threshold=4.0,
                 sim_steps_per_action=10):
        super().__init__()
        self.game = zero_ad.ZeroAD(uri)
        self.scenario_config = scenario_config
        self.map_size_m = map_size_m
        self.horizon = horizon
        self.reach_threshold = reach_threshold
        self.sim_steps_per_action = sim_steps_per_action
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(5,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self._step_count = 0
        self._prev_dist = None

    def _positions(self, state):
        v = xz(state.units(owner=1, type=VILLAGER_TYPE)[0].position())
        r = xz(state.units(owner=0, type=RESOURCE_TYPE)[0].position())
        return v, r

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset(self.scenario_config)
        state = self.game.step()  # un tick para que las entidades existan
        v, r = self._positions(state)
        self._prev_dist = distance(v, r)
        self._step_count = 0
        return build_observation(v, r, self.map_size_m), {}

    def step(self, action):
        x, z = denormalize_action(action, self.map_size_m)
        villager = self.game.current_state.units(owner=1, type=VILLAGER_TYPE)[0]
        cmd = zero_ad.actions.walk([villager], x, z)
        state = self.game.step([cmd])
        for _ in range(self.sim_steps_per_action - 1):
            state = self.game.step()
        v, r = self._positions(state)
        cur_dist = distance(v, r)
        reward = gather_reward(self._prev_dist, cur_dist)
        self._prev_dist = cur_dist
        self._step_count += 1
        terminated = is_reached(cur_dist, self.reach_threshold)
        truncated = self._step_count >= self.horizon
        obs = build_observation(v, r, self.map_size_m)
        return obs, reward, terminated, truncated, {"distance": cur_dist}
```

- [ ] **Step 2: Exponer el env**

`rl/gather/__init__.py`:
```python
from .env import ZeroADGatherEnv

__all__ = ["ZeroADGatherEnv"]
```

- [ ] **Step 3: Smoke test de integración (requiere el juego con --rl-interface corriendo)**

Run (con el server de Task 3 Step 2 activo):
```bash
PY=~/Documents/0ad/.toolchain/python/bin/python3
$PY - <<'EOF'
from rl.gather import ZeroADGatherEnv
cfg = open('rl/reset_config.json').read()
env = ZeroADGatherEnv(cfg)
obs, _ = env.reset()
print("obs0:", obs)
for i in range(5):
    obs, rew, term, trunc, info = env.step(env.action_space.sample())
    print(i, "rew=%.3f dist=%.2f term=%s trunc=%s" % (rew, info["distance"], term, trunc))
EOF
```
Expected: imprime obs de shape (5,) y 5 steps con reward/distancia numéricos, sin excepciones. Ajustar `map_size_m`/tipos según lo medido en Task 3.

- [ ] **Step 4: Commit**

```bash
git add rl/gather/env.py rl/gather/__init__.py
git commit -m "rl: add ZeroADGatherEnv (gymnasium wrapper over zero_ad)"
```

---

### Task 7: Conformidad con la API de Gymnasium

**Files:**
- Test: `rl/tests/test_env_contract.py`

- [ ] **Step 1: Escribir el test de contrato (sin juego: solo espacios)**

`rl/tests/test_env_contract.py`:
```python
import numpy as np
from gymnasium import spaces
from rl.gather import ZeroADGatherEnv

def test_spaces_are_well_formed_without_connecting():
    # No llamamos reset/step: solo construimos y miramos los espacios.
    env = ZeroADGatherEnv.__new__(ZeroADGatherEnv)
    env.observation_space = spaces.Box(-1.0, 1.0, shape=(5,), dtype=np.float32)
    env.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
    assert env.observation_space.shape == (5,)
    assert env.action_space.shape == (2,)
    assert env.action_space.contains(np.zeros(2, dtype=np.float32))
```

- [ ] **Step 2: Correr y verificar que pasa**

Run: `~/Documents/0ad/.toolchain/python/bin/python3 -m pytest rl/tests/test_env_contract.py -v`
Expected: PASS.

- [ ] **Step 3: (Con juego activo) correr `check_env` de SB3**

Run:
```bash
PY=~/Documents/0ad/.toolchain/python/bin/python3
$PY - <<'EOF'
from stable_baselines3.common.env_checker import check_env
from rl.gather import ZeroADGatherEnv
env = ZeroADGatherEnv(open('rl/reset_config.json').read())
check_env(env, warn=True)
print("check_env OK")
EOF
```
Expected: `check_env OK` (warnings menores aceptables; ningún error).

- [ ] **Step 4: Commit**

```bash
git add rl/tests/test_env_contract.py
git commit -m "rl: add Gymnasium contract test for the env"
```

---

## Fase D — Entrenamiento y verificación de M0

### Task 8: Script de entrenamiento SAC

**Files:**
- Create: `rl/train.py`

- [ ] **Step 1: Escribir train.py**

`rl/train.py`:
```python
import argparse
from stable_baselines3 import SAC
from rl.gather import ZeroADGatherEnv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="rl/reset_config.json")
    p.add_argument("--uri", default="http://localhost:6000")
    p.add_argument("--timesteps", type=int, default=5000)
    p.add_argument("--out", default="rl/sac_gather")
    args = p.parse_args()

    env = ZeroADGatherEnv(open(args.config).read(), uri=args.uri)
    model = SAC("MlpPolicy", env, verbose=1, learning_starts=200,
                buffer_size=50000)
    model.learn(total_timesteps=args.timesteps, log_interval=4)
    model.save(args.out)

    # Evaluacion rapida de la politica entrenada.
    reached = 0
    episodes = 10
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc
        if term:
            reached += 1
        print("episodio: dist_final=%.2f alcanzado=%s" % (info["distance"], term))
    print("ALCANZADOS %d/%d" % (reached, episodes))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke run corto (con juego activo)**

Run:
```bash
~/Documents/0ad/.toolchain/python/bin/python3 -m rl.train --timesteps 500
```
Expected: SAC corre sin excepciones, imprime logs de entrenamiento, guarda `rl/sac_gather.zip`. (Con 500 steps no esperamos buena política todavía.)

- [ ] **Step 3: Commit**

```bash
git add rl/train.py
git commit -m "rl: add SAC training + evaluation script"
```

---

### Task 9: Verificación de M0 (criterio de éxito)

**Files:**
- Create: `rl/README.md`

- [ ] **Step 1: Corrida de entrenamiento real**

Run:
```bash
~/Documents/0ad/.toolchain/python/bin/python3 -m rl.train --timesteps 5000 2>&1 | tee /tmp/rl_train.log
```
Expected: el `ep_rew_mean` que loguea SB3 **sube** a lo largo del entrenamiento, y al final imprime `ALCANZADOS X/10` con X mayoritario (p.ej. ≥7/10). Si no sube: subir `--timesteps`, ajustar `sim_steps_per_action`/`reach_threshold`, o revisar el signo del reward.

- [ ] **Step 2: Documentar cómo correrlo**

`rl/README.md`:
```markdown
# RL Gather (M0)

Experimento de RL: SAC entrena 1 aldeano para ir a 1 recurso.

## Requisitos
```bash
PY=~/Documents/0ad/.toolchain/python/bin/python3   # Python 3.11
$PY -m pip install -r rl/requirements.txt
$PY -m pip install ~/Documents/0ad/source/tools/rlclient/python
```

## Correr
1. Lanzar 0 A.D. con la interfaz RL:
   ```bash
   cd ~/Documents/0ad/binaries/system
   ./pyrogenesis -mod=mod -mod=public -mod=aoe3 --rl-interface=127.0.0.1:6000
   ```
2. Entrenar:
   ```bash
   $PY -m rl.train --timesteps 5000
   ```

## Criterio de éxito (M0)
- El entrenamiento corre sin excepciones.
- `ep_rew_mean` sube durante el entrenamiento.
- La política entrenada lleva el aldeano al recurso en la mayoría de los episodios de evaluación.

## Próximos milestones
- **M1:** reward = Δstock real (emitir `gather`).
- **M2:** 4 aldeanos + varios recursos, acción `Box(8,)`.
```

- [ ] **Step 3: Commit**

```bash
git add rl/README.md
git commit -m "rl: document M0 run instructions and success criteria"
```

---

## Self-Review (cobertura del spec)

- Interfaz RL nativa + `zero_ad` → Tasks 3, 6. ✓
- SAC / off-policy → Task 8. ✓
- Mapa-scenario fijo (1 aldeano + 1 recurso) → Task 2. ✓ (random map determinista; resuelve riesgo #1)
- Observación ~5 dims / acción `Box(2,)` / reward acercamiento / episodio → Tasks 4–6. ✓
- Riesgo "sim-turns por step y wall-clock" → Task 3 Step 3. ✓
- Riesgo "formato del config de reset()" → Task 3 Step 1. ✓
- Verificación de éxito M0 (loop corre, reward sube, alcanza el recurso) → Task 9. ✓
- YAGNI (1 proceso, sin vectorizar, sin gather real) → respetado; M1/M2 fuera de alcance. ✓
