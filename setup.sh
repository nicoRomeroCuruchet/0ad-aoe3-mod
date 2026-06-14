#!/usr/bin/env bash
# ============================================================================
# setup.sh — Compila 0 A.D. + el mod Athenai desde cero en Ubuntu 24.04 / 26.04.
#
#   Uso (desde la raiz de ESTE repo):   bash setup.sh
#
# Es IDEMPOTENTE: podes re-correrlo; detecta lo ya hecho y retoma.
# Verificado end-to-end en Ubuntu 26.04. Disenado tambien para 24.04 (mismo
# enfoque: Python 3.11 efimero + parches inocuos); en 24.04 conviene una
# primera corrida de prueba.
#
# Lo que hace, en orden (ver MANUAL.md para el detalle de cada paso):
#   1. deps apt (sudo)   2. Rust via rustup   3. cbindgen
#   4. clona 0 A.D.+LFS  5. Python 3.11 efimero   6. parche Boost.System
#   7. flags CMake       8. libs+makefiles    9. compila    10. instala el mod
# ============================================================================
set -euo pipefail

OAD="${OAD:-$HOME/Documents/0ad}"                      # destino del codigo de 0 A.D.
MOD_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"  # este repo (el mod)
JOBS="-j$(nproc)"
PY311_TAG="20260610"
PY311_VER="3.11.15"
PY_URL="https://github.com/astral-sh/python-build-standalone/releases/download/${PY311_TAG}/cpython-${PY311_VER}+${PY311_TAG}-x86_64-unknown-linux-gnu-install_only.tar.gz"

log(){ printf '\n\033[1;36m=== %s ===\033[0m\n' "$*"; }

# ---- 0. Chequeo de OS -------------------------------------------------------
if [ -r /etc/os-release ]; then . /etc/os-release; echo "Sistema: ${PRETTY_NAME:-?}"; fi
case "${VERSION_ID:-}" in
	24.04|26.04) ;;
	*) echo "AVISO: verificado en Ubuntu 26.04 (pensado tambien para 24.04). Tu version: ${VERSION_ID:-desconocida}. Sigo igual." ;;
esac
if [[ "$MOD_REPO" == *" "* ]]; then echo "ERROR: la ruta del repo no puede tener espacios."; exit 1; fi
if [[ "$OAD" == *" "* ]]; then echo "ERROR: \$OAD no puede tener espacios (rompe SpiderMonkey)."; exit 1; fi

# ---- 1. Dependencias del sistema (sudo) ------------------------------------
log "1/10  Dependencias apt (pide sudo)"
sudo apt-get update
sudo apt-get install -y build-essential cmake curl git git-lfs subversion \
	libboost-dev libboost-system-dev libboost-filesystem-dev \
	libcurl4-gnutls-dev libenet-dev libfmt-dev libfreetype-dev \
	libgloox-dev libicu-dev libminiupnpc-dev libnvtt-dev \
	libogg-dev libopenal-dev libpng-dev libsdl2-dev libsodium-dev \
	libvorbis-dev libwxgtk3.2-dev libxml2-dev llvm m4 python3 \
	rustc cargo zlib1g-dev xz-utils patch

# ---- 2. Rust 1.76+ via rustup (la apt de 24.04 trae 1.75) ------------------
log "2/10  Rust (rustup)"
[ -f "$HOME/.cargo/env" ] && source "$HOME/.cargo/env"
need_rust=1
if command -v rustc >/dev/null 2>&1; then
	minor="$(rustc --version | awk '{print $2}' | cut -d. -f2)"
	[ "${minor:-0}" -ge 76 ] && need_rust=0
fi
if [ "$need_rust" -eq 1 ]; then
	curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
	source "$HOME/.cargo/env"
fi
rustc --version

# ---- 3. cbindgen -----------------------------------------------------------
log "3/10  cbindgen 0.29.0"
command -v cbindgen >/dev/null 2>&1 || cargo install --locked cbindgen@0.29.0

# ---- 4. Clonar 0 A.D. + assets LFS -----------------------------------------
log "4/10  Codigo de 0 A.D. (varios GB; si ya existe, lo reusa)"
if [ ! -d "$OAD/.git" ]; then
	git clone https://github.com/0ad/0ad.git "$OAD"
fi
cd "$OAD"
git lfs install
git lfs pull

# ---- 5. Python 3.11 efimero (solo para el build) ---------------------------
log "5/10  Python 3.11 standalone (efimero, borrable al terminar)"
if [ ! -x "$OAD/.toolchain/python/bin/python3" ]; then
	mkdir -p "$OAD/.toolchain"
	curl -fL -o "$OAD/.toolchain/py311.tar.gz" "$PY_URL"
	tar xzf "$OAD/.toolchain/py311.tar.gz" -C "$OAD/.toolchain"
fi
export PATH="$OAD/.toolchain/python/bin:$PATH"
python3 --version

# ---- 6. Parche Boost.System (header-only desde Boost 1.69; quita -lboost_system) ----
log "6/10  Parche Boost.System en build/premake/extern_libs5.lua"
sed -i 's#unix_names = { os.findlib("boost_filesystem-mt") and "boost_filesystem-mt" or "boost_filesystem", os.findlib("boost_system-mt") and "boost_system-mt" or "boost_system" },#unix_names = { os.findlib("boost_filesystem-mt") and "boost_filesystem-mt" or "boost_filesystem" },#' build/premake/extern_libs5.lua

# ---- 7. Flags de compat CMake (NVTT con CMake 4.x en 26.04; inocuo en 24.04) ----
export CMAKE_FLAGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5"
export CMAKE_POLICY_VERSION_MINIMUM=3.5

# ---- 8. Build libs de terceros + makefiles ---------------------------------
log "8/10  Librerias de terceros + makefiles (~20-40 min)"
JOBS="$JOBS" ./build/workspaces/update-workspaces.sh

# ---- 9. Compilar el juego --------------------------------------------------
log "9/10  Compilando 0 A.D. (~10-20 min)"
make -C build/workspaces/gcc/ "$JOBS" config=release

# ---- 10. Instalar el mod (symlink) + habilitarlo + splashscreen off --------
log "10/10  Instalando y habilitando el mod"
ln -sfn "$MOD_REPO" "$OAD/binaries/data/mods/aoe3"
CFG="$HOME/.config/0ad/config/user.cfg"
mkdir -p "$(dirname "$CFG")"; touch "$CFG"
grep -q 'mod.enabledmods' "$CFG" || echo 'mod.enabledmods = "mod public aoe3"' >> "$CFG"
if grep -q 'gui.splashscreen.enable' "$CFG"; then
	sed -i 's/gui.splashscreen.enable = "true"/gui.splashscreen.enable = "false"/' "$CFG"
else
	echo 'gui.splashscreen.enable = "false"' >> "$CFG"
fi

log "LISTO"
cat <<EOF
0 A.D. + mod Athenai compilado.

  Jugar:   cd $OAD/binaries/system && ./pyrogenesis -mod=mod -mod=public -mod=aoe3
           (en el menu: civilizacion "Athenai (AOE3 Mod)", mapa Aleatorio)

  El Python 3.11 efimero ($OAD/.toolchain) ya no hace falta para jugar;
  podes borrarlo con:  rm -rf "$OAD/.toolchain"
  (solo se re-necesita si recompilas el motor C++)
EOF
