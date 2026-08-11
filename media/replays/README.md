# Replays de las políticas entrenadas

Cada carpeta es un replay de 0 A.D. de **un episodio determinista** de la política
correspondiente, grabado con `rl.eval --replay`. Son logs de comandos de simulación
(no video): 0 A.D. vuelve a simular la partida y la reproduce con la cámara normal,
a framerate completo.

| Carpeta | Milestone / algoritmo | Resultado del episodio | Turnos de simulación |
|---------|----------------------|------------------------|----------------------|
| `m0_sb3_sac` | M0 &mdash; SAC | reward 157.4, llega al árbol a 2.6 m | 90 (~18 s) |
| `m0_sb3_ppo` | M0 &mdash; PPO | reward 156.5, llega al árbol a 3.5 m | 90 (~18 s) |
| `m1_sb3_sac` | M1 &mdash; SAC | reward 28.3, **deposita 20 de madera** | 480 (~96 s) |
| `m1_sb3_ppo` | M1 &mdash; PPO | reward 30.1, **deposita 20 de madera** | 480 (~96 s) |

Los cuatro corresponden a las corridas comparadas en `rl/sac_ppo_comparison.csv`.
Escenario `rl_gather` con `Seed = 0` y sin RNG, así que el episodio es reproducible.

## Cómo verlos

0 A.D. sólo lista replays que estén en su directorio de perfil, así que estas
carpetas están enlazadas desde ahí:

```bash
ls -la ~/.local/share/0ad/replays/0.28.0/
# 2026-08-10_0001 -> .../media/replays/m0_sb3_sac
# ...
```

Con los symlinks en su lugar:

```bash
./run_game.sh          # lanzar el juego sin la interfaz RL
```

y después **menú principal → Replays**. Los cuatro aparecen con fecha 2026-08-10,
ordenados igual que la tabla de arriba (0001 a 0004).

Si clonás el repo en otra máquina, volvé a crear los enlaces:

```bash
PROFILE="$HOME/.local/share/0ad/replays/0.28.0"
mkdir -p "$PROFILE"
i=1
for name in m0_sb3_sac m0_sb3_ppo m1_sb3_sac m1_sb3_ppo; do
  ln -sfnT "$PWD/media/replays/$name" "$PROFILE/$(date +%Y-%m-%d)_000$i"
  i=$((i + 1))
done
```

Copiar las carpetas en vez de enlazarlas también funciona.

## Regenerarlos

```bash
make server   # en otra terminal
.venv/bin/python -m rl.eval --experiment rl/configs/m1_sb3_ppo.toml \
  --model rl/runs/LA_CORRIDA/model --trust-model \
  --episodes 1 --mode deterministic --replay --verbose
```

0 A.D. escribe `metadata.json` recién cuando termina la sesión, así que el último
replay de una tanda queda incompleto hasta que otra partida haga `reset`. Corré
cualquier evaluación más (por ejemplo `make m1-oracle`) para cerrarlo.
