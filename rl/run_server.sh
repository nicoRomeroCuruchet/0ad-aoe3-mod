#!/usr/bin/env bash
# Lanza 0 A.D. con la interfaz RL de forma robusta (headless) para entrenar/evaluar.
#
# Uso:
#   bash rl/run_server.sh          # arranca el server y bloquea (Ctrl+C para detener)
# En otra terminal:
#   <python> -m rl.eval --verbose  # o: <python> -m rl.train --timesteps 2000
#
# Encapsula las lecciones aprendidas para que el arranque sea confiable:
#  - mata servidores previos por CMDLINE (el proceso se llama 'main' en comm,
#    asi que `pkill -x pyrogenesis` NO los mata);
#  - desactiva el splashscreen (en headless bloquea el arranque);
#  - espera la senal "RL interface listening" SIN abrir sockets de prueba
#    (una conexion TCP cruda al RL interface lo desestabiliza);
#  - un solo cliente por server: el server se cierra cuando el cliente se
#    desconecta, asi que relanza este script para cada sesion de eval/train.
set -u

OAD="${OAD:-$HOME/Documents/0ad}"
PORT="${PORT:-6000}"
LOG="${LOG:-/tmp/rl_server.log}"
CFG="$HOME/.config/0ad/config/user.cfg"

cleanup() { pkill -9 -f 'pyrogenesi[s]' 2>/dev/null; }
trap cleanup EXIT INT TERM

# 1. Matar servidores previos (por cmdline)
pkill -9 -f 'pyrogenesi[s]' 2>/dev/null
sleep 1

# 2. Splashscreen off
if grep -q 'gui.splashscreen.enable' "$CFG" 2>/dev/null; then
	sed -i 's/gui.splashscreen.enable = "true"/gui.splashscreen.enable = "false"/' "$CFG"
else
	mkdir -p "$(dirname "$CFG")"
	echo 'gui.splashscreen.enable = "false"' >> "$CFG"
fi

# 3. Lanzar el server
cd "$OAD/binaries/system" || { echo "No existe $OAD/binaries/system"; exit 1; }
: > "$LOG"
./pyrogenesis -mod=mod -mod=public -mod=aoe3 --rl-interface="127.0.0.1:$PORT" > "$LOG" 2>&1 &
SRV=$!

# 4. Esperar a que el RL interface escuche (sin tocar el socket)
for _ in $(seq 1 60); do
	grep -q 'RL interface listening' "$LOG" 2>/dev/null && break
	sleep 1
done
if ! grep -q 'RL interface listening' "$LOG" 2>/dev/null; then
	echo "ERROR: el server no llego a 'RL interface listening'. Ver $LOG"
	tail -5 "$LOG"; exit 1
fi
sleep 2  # warmup

echo "================================================================"
echo " Server RL listo en 127.0.0.1:$PORT  (PID $SRV, log: $LOG)"
echo " En otra terminal, con el Python 3.11:"
echo "   ~/Documents/0ad/.toolchain/python/bin/python3 -m rl.eval --verbose"
echo " Ctrl+C aca para detener el server."
echo "================================================================"
wait "$SRV"
