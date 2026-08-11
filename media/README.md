# media/

Material grabado de las políticas entrenadas. Dos formatos, distintos usos:

| Carpeta | Qué es | Para qué |
|---------|--------|----------|
| [`videos/`](videos/) | MP4 (H.264), vista renderizada por el motor centrada en el Polites | mirar/compartir sin abrir 0 A.D. |
| [`replays/`](replays/) | replays nativos de 0 A.D. (log de comandos) | verlos dentro del juego, cámara normal, framerate completo |

`media/videos/` está en `.gitignore`: son binarios pesados y se regeneran con el comando de
más abajo. Los replays sí se versionan (88 KB en total).

Ambos vienen de las mismas cuatro corridas comparadas en `rl/sac_ppo_comparison.csv`.

## videos/

| Archivo | Política | Duración | Qué se ve |
|---------|----------|----------|-----------|
| `m0_sac.mp4` | M0 &mdash; SAC | 4.5 s (90 turnos) | camina hasta el árbol y frena a 2.6 m |
| `m0_ppo.mp4` | M0 &mdash; PPO | 4.5 s (90 turnos) | lo mismo, frena a 3.5 m |
| `m1_sac.mp4` | M1 &mdash; SAC | 24 s (480 turnos) | ciclo completo: ir, juntar, volver, depositar 20 de madera |
| `m1_ppo.mp4` | M1 &mdash; PPO | 24 s (480 turnos) | ídem, con el mejor reward de evaluación (30.1) |
| `m2_ppo.mp4` | M2 &mdash; PPO | 80 s (1600 turnos) | 4 aldeanos, 4 árboles: 80 de madera depositada |

En M2 la cámara sigue **un** aldeano (`observer_villager_slot`, por defecto el slot 0): su radio
de visión es 32 m, así que los otros tres suelen quedar fuera de cuadro. Para ver el reparto de
trabajo hay que grabar los cuatro slots y ponerlos en mosaico.

512&times;512, H.264 (`libx264`, crf 20, `yuv420p`, `+faststart`), 20 fps &mdash; un cuadro por turno de simulación, que en 0 A.D. dura
200 ms, así que la reproducción va a velocidad real. La cámara es la vista del observador
del engine parcheado: mira desde arriba al Polites y abarca su radio de visión (32 m), por
eso el aldeano se ve chico y el árbol aparece recién cuando entra en ese radio. La niebla de
guerra del borde es real, no un viñeteado.

### Regenerarlos

Necesita el engine con renderer (`make server-view`, no el headless) porque los cuadros
salen del endpoint del observador:

```bash
make server-view      # terminal 1

# terminal 2
.venv/bin/python -m rl.eval --experiment rl/configs/m1_sb3_ppo.toml \
  --model rl/runs/LA_CORRIDA/model --trust-model \
  --episodes 1 --mode deterministic \
  --record-agent-view /tmp/rollout_m1_ppo --record-sim-turns \
  --record-agent-view-video media/videos/m1_ppo.webm
```

`--record-sim-turns` es lo que hace que el video sea fluido: sin ese flag se graba **un
cuadro por decisión** (9 en M0, 6 en M1), que alcanza para inspeccionar la política pero no
para mirarla. Con el flag se captura un cuadro por turno de simulación.

La codificación usa `ffmpeg` si está en el `PATH` (sale MP4) y si no cae a `gst-launch-1.0`
con VP8 (sale WebM). Esta máquina no tiene `ffmpeg` del sistema ni un encoder H.264 en
GStreamer, así que estos MP4 se generaron con el binario estático que trae el wheel
`imageio-ffmpeg`, en un venv aparte para no tocar `uv.lock`:

```bash
python3 -m venv /tmp/ffenv && /tmp/ffenv/bin/pip install imageio-ffmpeg
FF=$(/tmp/ffenv/bin/python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())")
"$FF" -y -framerate 20 -i /tmp/rollout_m1_ppo/deterministic/turn_frames/turn%05d.png \
  -c:v libx264 -preset slow -crf 20 -pix_fmt yuv420p -movflags +faststart \
  media/videos/m1_ppo.mp4
```

Con `sudo apt install ffmpeg` no hace falta nada de esto: `--record-agent-view-video` escribe
el MP4 directamente.

## replays/

Ver [`replays/README.md`](replays/README.md). Son logs de comandos, no video: 0 A.D. vuelve a
simular la partida. Dependen del mod y del mapa &mdash; si cambiás
`maps/random/rl_gather.js` los replays viejos se desincronizan. Los videos no: son píxeles.
