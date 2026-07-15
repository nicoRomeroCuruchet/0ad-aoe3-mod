#!/usr/bin/env bash
# Launch the repo-local 0 A.D. AppImage with this checkout registered as aoe3.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OAD_APPIMAGE="${OAD_APPIMAGE:-$REPO_ROOT/.runtime/0ad/0ad.AppImage}"

if [ ! -x "$OAD_APPIMAGE" ]; then
	printf '0 A.D. AppImage not found or not executable: %s\n' "$OAD_APPIMAGE" >&2
	exit 1
fi

# Snap-launched terminals can leak a Snap-specific XDG_DATA_HOME. Pin the
# normal user data directory so 0 A.D. finds the same mod on every launch.
export XDG_DATA_HOME="${OAD_XDG_DATA_HOME:-$HOME/.local/share}"
# The Release 28 AppImage can render an invisible cursor through native
# Wayland. XWayland keeps SDL mouse rendering reliable on GNOME Wayland.
export SDL_VIDEODRIVER="${OAD_SDL_VIDEODRIVER:-x11}"
MODS_DIR="$XDG_DATA_HOME/0ad/mods"
MOD_LINK="$MODS_DIR/aoe3"
mkdir -p "$MODS_DIR"
if [ -e "$MOD_LINK" ] && [ ! -L "$MOD_LINK" ]; then
	printf 'Refusing to replace existing mod directory: %s\n' "$MOD_LINK" >&2
	exit 1
fi
ln -sfnT "$REPO_ROOT" "$MOD_LINK"

exec "$OAD_APPIMAGE" -mod=mod -mod=public -mod=aoe3 "$@"
