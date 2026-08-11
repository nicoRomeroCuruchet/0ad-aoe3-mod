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
| `m2_ppo.mp4` | M2 **legacy** &mdash; PPO | 80 s (1600 turnos) | benchmark anterior: 4 aldeanos, 4 árboles y +80 de madera |
| `m2_ppo_assignment_team.mp4` | M2 asignación **legacy** &mdash; PPO | 24 s (480 turnos) | vista de equipo del benchmark anterior: +80 en 6 decisiones |
| `m2_ppo_joint_ring_team.mp4` | M2 PPO vigente | 72 s (1440 turnos) | 4 aldeanos agotan y entregan los 6 robles del anillo (+600) en vista de equipo |

Los dos videos M2 existentes son **archivales**. Se grabaron antes del diseño vigente: seis
robles de 100, uno por recolector, anillo equiangular rotado por seed, objetivo colectivo de 600
y acción conjunta sin colisiones `Discrete(1045)`. No prueban ni deben usarse para comparar ese
benchmark nuevo. El primero además se grabó antes del observer de equipo y por eso todavía sigue
un solo aldeano.

La grabación M2 vigente usa `observer_view = "team"`: calcula una caja alrededor de los
cuatro aldeanos, los seis árboles y el depósito, y ajusta el centro y rango de una única
cámara. Así muestran el reparto completo alrededor del anillo sin cuatro POVs ni mosaico. La
niebla de guerra sigue siendo la del Player 1, por lo que una zona del encuadre puede verse negra
aunque esté dentro de cámara.

512&times;512, H.264 (`libx264`, crf 20, `yuv420p`, `+faststart`), 20 fps &mdash; un cuadro por turno de simulación, que en 0 A.D. dura
200 ms, así que la reproducción va a velocidad real. La cámara sale del observer del engine
parcheado. M0/M1 miran desde arriba al Polites y abarcan su radio de visión (32 m); M2 usa el
encuadre dinámico del equipo. La niebla de guerra del borde es real, no un viñeteado.

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

# M2 vigente: matching conjunto de 1045 opciones, anillo de 6×100 y equipo completo.
# Usar una corrida nueva; los modelos previos a este espacio de acción son incompatibles.
.venv/bin/python -m rl.eval --experiment rl/configs/m2_sb3_ppo.toml \
  --model rl/runs/LA_CORRIDA/model --trust-model \
  --episodes 1 --mode deterministic \
  --record-agent-view /tmp/rollout_m2_ppo --record-sim-turns \
  --record-agent-view-video media/videos/m2_ppo_joint_ring_team.mp4
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
