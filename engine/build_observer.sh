#!/usr/bin/env bash
# Build the exact 0 A.D. Release 28 source with the RL observer patch.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_ROOT="${OAD_OBSERVER_RUNTIME:-$REPO_ROOT/.runtime/0ad-observer}"
ARCHIVE_NAME="0ad-0.28.0-unix-build.tar.xz"
ARCHIVE="$RUNTIME_ROOT/downloads/$ARCHIVE_NAME"
SOURCE_ROOT="$RUNTIME_ROOT/source"
SOURCE_DIR="$SOURCE_ROOT/0ad-0.28.0"
PATCH_FILE="$REPO_ROOT/engine/patches/0ad-v0.28.0-agent-observer.patch"
SOURCE_URL="https://releases.wildfiregames.com/$ARCHIVE_NAME"
SOURCE_SHA256="27e217755ef76a922fe58dbf593d96e54b6ed2375d23f548c35619aa6bd5a42a"
RUSTUP_VERSION="1.28.2"
RUST_TOOLCHAIN="1.85.1"
RUSTUP_INIT="$RUNTIME_ROOT/downloads/rustup-init-$RUSTUP_VERSION-x86_64"
RUSTUP_SHA256="20a06e644b0d9bd2fbdbfd52d42540bdde820ea7df86e92e533c073da0cdd43c"
JOBS="${JOBS:--j$(nproc)}"

if [[ ! "$JOBS" =~ ^-j[1-9][0-9]*$ ]]; then
	printf 'JOBS must have the form -jN with N positive: %s\n' "$JOBS" >&2
	exit 2
fi

mkdir -p "$RUNTIME_ROOT"
if ! command -v flock >/dev/null 2>&1; then
	printf '%s\n' 'The observer builder requires `flock` from util-linux.' >&2
	exit 1
fi
exec {builder_lock_fd}>"$RUNTIME_ROOT/.observer-build.lock"
if ! flock --nonblock "$builder_lock_fd"; then
	printf 'Another observer build is already using %s\n' "$RUNTIME_ROOT" >&2
	exit 1
fi

download_verified() {
	local url="$1"
	local destination="$2"
	local checksum="$3"
	local description="$4"
	local temporary

	mkdir -p "$(dirname "$destination")"
	if [[ -f "$destination" ]] && \
		printf '%s  %s\n' "$checksum" "$destination" | sha256sum --check --status; then
		return
	fi
	if [[ -f "$destination" ]]; then
		printf 'Cached %s is incomplete or corrupt; downloading it again.\n' "$description"
	fi
	temporary="$(mktemp "${destination}.partial.XXXXXX")"
	if ! curl --proto '=https' --tlsv1.2 --fail --location --retry 2 \
		--output "$temporary" "$url"; then
		rm -f "$temporary"
		return 1
	fi
	if ! printf '%s  %s\n' "$checksum" "$temporary" | sha256sum --check --status; then
		printf '%s checksum mismatch after download.\n' "$description" >&2
		rm -f "$temporary"
		return 1
	fi
	mv -f "$temporary" "$destination"
}

mkdir -p "$SOURCE_ROOT"
download_verified "$SOURCE_URL" "$ARCHIVE" "$SOURCE_SHA256" \
	"official 0 A.D. Release 28 build source"

SOURCE_MARKER="$SOURCE_DIR/.observer-source-ready"
if [[ -d "$SOURCE_DIR" && ! -f "$SOURCE_MARKER" ]]; then
	source_complete=1
	for required in \
		libraries/build-source-libs.sh \
		libraries/source/spidermonkey/build.sh \
		build/workspaces/update-workspaces.sh \
		source/renderer/Renderer.cpp \
		source/rlinterface/RLInterface.cpp; do
		if [[ ! -f "$SOURCE_DIR/$required" ]]; then
			source_complete=0
			break
		fi
	done
	if [[ "$source_complete" -eq 1 ]]; then
		touch "$SOURCE_MARKER"
	else
		printf 'Removing an incomplete observer source extraction.\n'
		rm -rf "$SOURCE_DIR"
	fi
fi
if [[ ! -f "$SOURCE_MARKER" ]]; then
	printf 'Extracting %s...\n' "$ARCHIVE_NAME"
	extraction_root="$(mktemp -d "$SOURCE_ROOT/.extract.XXXXXX")"
	if ! tar -xJf "$ARCHIVE" -C "$extraction_root"; then
		rm -rf "$extraction_root"
		exit 1
	fi
	mv "$extraction_root/0ad-0.28.0" "$SOURCE_DIR"
	rmdir "$extraction_root"
	touch "$SOURCE_MARKER"
fi

if patch --batch --forward --directory "$SOURCE_DIR" --strip 1 --dry-run --silent < "$PATCH_FILE"; then
	patch --batch --forward --directory "$SOURCE_DIR" --strip 1 < "$PATCH_FILE"
elif patch --batch --forward --directory "$SOURCE_DIR" --strip 1 --reverse --dry-run --silent < "$PATCH_FILE"; then
	printf 'Engine observer patch is already applied.\n'
else
	printf 'Observer patch does not apply cleanly to %s\n' "$SOURCE_DIR" >&2
	exit 1
fi

# Release 28 needs Python 3.11 for its bundled SpiderMonkey build. Reuse the
# repository runtime installed by setup.sh when available.
if [[ -x "$REPO_ROOT/.runtime/python/bin/python3" ]]; then
	export PATH="$REPO_ROOT/.runtime/python/bin:$PATH"
fi
export CMAKE_FLAGS="${CMAKE_FLAGS:--DCMAKE_POLICY_VERSION_MINIMUM=3.5}"
export CMAKE_POLICY_VERSION_MINIMUM="${CMAKE_POLICY_VERSION_MINIMUM:-3.5}"
export BUILD_RELEASE_ONLY=1

# Ubuntu installs LLVM tools with a version suffix. Mozilla's configure accepts
# the explicit path, so use the first packaged llvm-objdump when no unversioned
# command is available.
if [[ -n "${LLVM_OBJDUMP:-}" ]]; then
	if [[ ! -x "$LLVM_OBJDUMP" ]]; then
		printf 'LLVM_OBJDUMP is not executable: %s\n' "$LLVM_OBJDUMP" >&2
		exit 1
	fi
elif ! command -v llvm-objdump >/dev/null 2>&1; then
	for candidate in /usr/bin/llvm-objdump-[0-9]*; do
		if [[ -x "$candidate" ]]; then
			export LLVM_OBJDUMP="$candidate"
			break
		fi
	done
	if [[ -z "${LLVM_OBJDUMP:-}" ]]; then
		printf 'SpiderMonkey needs llvm-objdump; install the Ubuntu llvm package.\n' >&2
		exit 1
	fi
fi

# Keep the modern Rust toolchain required by SpiderMonkey inside the ignored
# observer runtime. This does not modify the user's shell profile or system
# packages, and the rustup bootstrap itself is versioned and checksummed.
if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
	printf 'The repo-local observer toolchain currently supports Linux x86_64 only.\n' >&2
	exit 1
fi
export RUSTUP_HOME="$RUNTIME_ROOT/toolchain/rustup"
export CARGO_HOME="$RUNTIME_ROOT/toolchain/cargo"
export PATH="$CARGO_HOME/bin:$PATH"
if [[ ! -x "$CARGO_HOME/bin/rustup" ]]; then
	download_verified \
		"https://static.rust-lang.org/rustup/archive/$RUSTUP_VERSION/x86_64-unknown-linux-gnu/rustup-init" \
		"$RUSTUP_INIT" "$RUSTUP_SHA256" "rustup-init $RUSTUP_VERSION"
	chmod +x "$RUSTUP_INIT"
	"$RUSTUP_INIT" -y --no-modify-path --profile minimal \
		--default-toolchain "$RUST_TOOLCHAIN"
fi
rustc_version="$($CARGO_HOME/bin/rustc --version 2>/dev/null || true)"
if [[ "$rustc_version" != "rustc $RUST_TOOLCHAIN "* ]]; then
	"$CARGO_HOME/bin/rustup" toolchain install "$RUST_TOOLCHAIN" --profile minimal
	"$CARGO_HOME/bin/rustup" default "$RUST_TOOLCHAIN"
	rustc_version="$($CARGO_HOME/bin/rustc --version)"
fi
if [[ "$rustc_version" != "rustc $RUST_TOOLCHAIN "* ]]; then
	printf 'Expected Rust %s, found: %s\n' "$RUST_TOOLCHAIN" "$rustc_version" >&2
	exit 1
fi
cbindgen_version="$($CARGO_HOME/bin/cbindgen --version 2>/dev/null || true)"
if [[ "$cbindgen_version" != "cbindgen 0.29.0" ]]; then
	"$CARGO_HOME/bin/cargo" install --force --locked cbindgen@0.29.0
fi

# Boost.System has been header-only for years; this keeps the Release 28
# workspace compatible with current Ubuntu packages, matching setup.sh.
sed -i 's#unix_names = { os.findlib("boost_filesystem-mt") and "boost_filesystem-mt" or "boost_filesystem", os.findlib("boost_system-mt") and "boost_system-mt" or "boost_system" },#unix_names = { os.findlib("boost_filesystem-mt") and "boost_filesystem-mt" or "boost_filesystem" },#' \
	"$SOURCE_DIR/build/premake/extern_libs5.lua"

if ! command -v pkg-config >/dev/null 2>&1 || ! pkg-config --exists libenet; then
	printf '%s\n' \
		'Missing 0 A.D. build prerequisite. Install it with: sudo apt install libenet-dev' >&2
	exit 1
fi

printf 'Building bundled 0 A.D. dependencies (%s)...\n' "$JOBS"
"$SOURCE_DIR/libraries/build-source-libs.sh" "$JOBS"
"$SOURCE_DIR/build/workspaces/update-workspaces.sh" \
	--without-atlas \
	--without-audio \
	--without-dap-interface \
	--without-lobby \
	--without-miniupnpc \
	--without-tests
make --directory "$SOURCE_DIR/build/workspaces/gcc" "$JOBS" config=release

BINARY="$SOURCE_DIR/binaries/system/pyrogenesis"
if [[ ! -x "$BINARY" ]]; then
	printf 'Build finished without producing %s\n' "$BINARY" >&2
	exit 1
fi
printf '\nPatched observer engine ready:\n  %s\n' "$BINARY"
