#!/usr/bin/env bash
# Launch the repo-local 0 A.D. AppImage with this checkout registered as aoe3.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OAD_APPIMAGE="${OAD_APPIMAGE:-$REPO_ROOT/.runtime/0ad/0ad.AppImage}"
OAD_OBSERVER_BINARY="${OAD_OBSERVER_BINARY:-$REPO_ROOT/.runtime/0ad-observer/source/0ad-0.28.0/binaries/system/pyrogenesis}"

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

require_observer=0
forwarded_args=()
for arg in "$@"; do
	if [[ "$arg" == "--require-rl-observer" ]]; then
		require_observer=1
	else
		forwarded_args+=("$arg")
	fi
done

if [[ "$require_observer" -eq 0 ]]; then
	exec "$OAD_APPIMAGE" -mod=mod -mod=public -mod=aoe3 "${forwarded_args[@]}"
fi
if [[ ! -x "$OAD_OBSERVER_BINARY" ]]; then
	printf '%s\n' \
		'Patched 0 A.D. observer engine not found. Run `make engine-observer` first.' >&2
	exit 1
fi

binary_root="$(cd "$(dirname "$OAD_OBSERVER_BINARY")/.." && pwd)"
if ! command -v flock >/dev/null 2>&1; then
	printf '%s\n' 'The observer launcher requires `flock` from util-linux.' >&2
	exit 1
fi
exec {observer_lock_fd}>"$binary_root/.agent-observer.lock"
if ! flock --nonblock "$observer_lock_fd"; then
	printf '%s\n' \
		'Another patched observer engine is already running from this build.' >&2
	exit 1
fi

# The custom binary is small, but the official art/data inside the AppImage is
# several GB. Mount it read-only and point the source build at those exact
# Release 28 assets instead of duplicating them on disk.
mount_log="$(mktemp)"
mount_pid=""
cleanup_mount() {
	if [[ -n "$mount_pid" ]]; then
		kill "$mount_pid" 2>/dev/null || true
		wait "$mount_pid" 2>/dev/null || true
	fi
	rm -f "$mount_log"
}
trap cleanup_mount EXIT

"$OAD_APPIMAGE" --appimage-mount >"$mount_log" 2>&1 &
mount_pid=$!
appdir=""
for _attempt in {1..100}; do
	while IFS= read -r candidate; do
		if [[ -d "$candidate/usr/data" ]]; then
			appdir="$candidate"
			break
		fi
	done < "$mount_log"
	[[ -n "$appdir" ]] && break
	if ! kill -0 "$mount_pid" 2>/dev/null; then
		break
	fi
	sleep 0.05
done
if [[ -z "$appdir" ]]; then
	printf 'Could not mount 0 A.D. AppImage data. Details:\n' >&2
	sed -n '1,20p' "$mount_log" >&2
	exit 1
fi

mkdir -p "$binary_root/data/mods"
for relative in config mods/mod mods/public; do
	target="$binary_root/data/$relative"
	if [[ -e "$target" && ! -L "$target" ]]; then
		printf 'Refusing to replace observer data path: %s\n' "$target" >&2
		exit 1
	fi
	ln -sfnT "$appdir/usr/data/$relative" "$target"
done

export LD_LIBRARY_PATH="$binary_root/system${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
set +e
"$OAD_OBSERVER_BINARY" -mod=mod -mod=public -mod=aoe3 \
	"${forwarded_args[@]}" -conf=rendererbackend:gl
status=$?
set -e
exit "$status"
