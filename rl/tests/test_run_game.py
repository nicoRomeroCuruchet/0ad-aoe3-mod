import os
import subprocess
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_GAME = REPO_ROOT / "run_game.sh"


def test_launcher_ignores_snap_xdg_data_home_and_registers_mod(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    fake_appimage = tmp_path / "0ad.AppImage"
    captured_environment = tmp_path / "environment.txt"
    fake_appimage.write_text(
        "#!/usr/bin/env bash\n"
        'printf "%s\\n" "$XDG_DATA_HOME" "$SDL_VIDEODRIVER" "$@" '
        '> "$CAPTURED_ENVIRONMENT"\n',
        encoding="utf-8",
    )
    fake_appimage.chmod(0o755)

    environment = {
        **os.environ,
        "HOME": str(home),
        "XDG_DATA_HOME": str(home / "snap/code/current/.local/share"),
        "OAD_APPIMAGE": str(fake_appimage),
        "OAD_OBSERVER_BINARY": str(tmp_path / "missing-pyrogenesis"),
        "CAPTURED_ENVIRONMENT": str(captured_environment),
    }
    environment.pop("OAD_XDG_DATA_HOME", None)
    environment.pop("OAD_SDL_VIDEODRIVER", None)
    environment.pop("SDL_VIDEODRIVER", None)
    subprocess.run(
        [RUN_GAME, "-quickstart"],
        check=True,
        env=environment,
    )

    assert captured_environment.read_text(encoding="utf-8").splitlines() == [
        str(home / ".local/share"),
        "x11",
        "-mod=mod",
        "-mod=public",
        "-mod=aoe3",
        "-quickstart",
    ]
    assert (home / ".local/share/0ad/mods/aoe3").resolve() == REPO_ROOT


def test_launcher_refuses_to_overwrite_an_existing_mod_directory(tmp_path):
    home = tmp_path / "home"
    existing_mod = home / ".local/share/0ad/mods/aoe3"
    existing_mod.mkdir(parents=True)
    marker = existing_mod / "keep-me.txt"
    marker.write_text("user data", encoding="utf-8")
    fake_appimage = tmp_path / "0ad.AppImage"
    fake_appimage.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    fake_appimage.chmod(0o755)

    result = subprocess.run(
        [RUN_GAME],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "HOME": str(home),
            "OAD_APPIMAGE": str(fake_appimage),
            "OAD_XDG_DATA_HOME": str(home / ".local/share"),
        },
    )

    assert result.returncode != 0
    assert "Refusing to replace existing mod directory" in result.stderr
    assert marker.read_text(encoding="utf-8") == "user data"


def test_launcher_allows_native_wayland_override(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    fake_appimage = tmp_path / "0ad.AppImage"
    captured_driver = tmp_path / "driver.txt"
    fake_appimage.write_text(
        '#!/usr/bin/env bash\nprintf "%s\\n" "$SDL_VIDEODRIVER" > "$CAPTURED_DRIVER"\n',
        encoding="utf-8",
    )
    fake_appimage.chmod(0o755)

    subprocess.run(
        [RUN_GAME],
        check=True,
        env={
            **os.environ,
            "HOME": str(home),
            "OAD_APPIMAGE": str(fake_appimage),
            "OAD_OBSERVER_BINARY": str(tmp_path / "missing-pyrogenesis"),
            "OAD_SDL_VIDEODRIVER": "wayland",
            "CAPTURED_DRIVER": str(captured_driver),
        },
    )

    assert captured_driver.read_text(encoding="utf-8").strip() == "wayland"


def test_launcher_uses_full_appimage_unless_observer_is_requested(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    selected_binary = tmp_path / "selected-binary.txt"
    fake_appimage = tmp_path / "0ad.AppImage"
    fake_appimage.write_text(
        '#!/usr/bin/env bash\nprintf "appimage\\n" > "$SELECTED_BINARY"\n',
        encoding="utf-8",
    )
    fake_appimage.chmod(0o755)
    fake_observer = tmp_path / "pyrogenesis"
    fake_observer.write_text(
        '#!/usr/bin/env bash\nprintf "observer\\n" > "$SELECTED_BINARY"\n',
        encoding="utf-8",
    )
    fake_observer.chmod(0o755)

    subprocess.run(
        [RUN_GAME],
        check=True,
        env={
            **os.environ,
            "HOME": str(home),
            "OAD_APPIMAGE": str(fake_appimage),
            "OAD_OBSERVER_BINARY": str(fake_observer),
            "SELECTED_BINARY": str(selected_binary),
        },
    )

    assert selected_binary.read_text(encoding="utf-8").strip() == "appimage"


def test_launcher_requires_the_patched_binary_when_requested(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    fake_appimage = tmp_path / "0ad.AppImage"
    fake_appimage.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    fake_appimage.chmod(0o755)

    result = subprocess.run(
        [RUN_GAME, "--require-rl-observer"],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "HOME": str(home),
            "OAD_APPIMAGE": str(fake_appimage),
            "OAD_OBSERVER_BINARY": str(tmp_path / "missing-pyrogenesis"),
        },
    )

    assert result.returncode != 0
    assert "make engine-observer" in result.stderr


def test_launcher_mounts_appimage_data_for_the_patched_binary(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    appdir = tmp_path / "appdir"
    for relative in ("usr/data/config", "usr/data/mods/mod", "usr/data/mods/public"):
        (appdir / relative).mkdir(parents=True)

    fake_appimage = tmp_path / "0ad.AppImage"
    fake_appimage.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "${1:-}" == "--appimage-mount" ]]; then\n'
        '  printf "%s\\n" "$FAKE_APPDIR"\n'
        "  sleep 30\n"
        "  exit 0\n"
        "fi\n"
        "exit 9\n",
        encoding="utf-8",
    )
    fake_appimage.chmod(0o755)

    engine_root = tmp_path / "engine/binaries"
    engine_binary = engine_root / "system/pyrogenesis"
    engine_binary.parent.mkdir(parents=True)
    captured_arguments = tmp_path / "engine-arguments.txt"
    engine_binary.write_text(
        '#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$CAPTURED_ARGUMENTS"\n',
        encoding="utf-8",
    )
    engine_binary.chmod(0o755)

    subprocess.run(
        [
            RUN_GAME,
            "--require-rl-observer",
            "--rl-interface=127.0.0.1:6000",
        ],
        check=True,
        env={
            **os.environ,
            "DISPLAY": ":99",
            "HOME": str(home),
            "OAD_APPIMAGE": str(fake_appimage),
            "OAD_OBSERVER_BINARY": str(engine_binary),
            "FAKE_APPDIR": str(appdir),
            "CAPTURED_ARGUMENTS": str(captured_arguments),
        },
    )

    assert captured_arguments.read_text(encoding="utf-8").splitlines() == [
        "-mod=mod",
        "-mod=public",
        "-mod=aoe3",
        "--rl-interface=127.0.0.1:6000",
        "-xres=1024",
        "-yres=768",
        "-conf=windowed:true",
        "-conf=rendererbackend:gl",
    ]
    assert (engine_root / "data/config").resolve() == appdir / "usr/data/config"
    assert (engine_root / "data/mods/mod").resolve() == appdir / "usr/data/mods/mod"
    assert (
        engine_root / "data/mods/public"
    ).resolve() == appdir / "usr/data/mods/public"


def test_nonvisual_observer_does_not_require_a_display(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    appdir = tmp_path / "appdir"
    for relative in ("usr/data/config", "usr/data/mods/mod", "usr/data/mods/public"):
        (appdir / relative).mkdir(parents=True)

    fake_appimage = tmp_path / "0ad.AppImage"
    fake_appimage.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "${1:-}" == "--appimage-mount" ]]; then\n'
        '  printf "%s\\n" "$FAKE_APPDIR"\n'
        "  sleep 30\n"
        "fi\n",
        encoding="utf-8",
    )
    fake_appimage.chmod(0o755)

    engine_binary = tmp_path / "engine/binaries/system/pyrogenesis"
    engine_binary.parent.mkdir(parents=True)
    captured_arguments = tmp_path / "engine-arguments.txt"
    engine_binary.write_text(
        '#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$CAPTURED_ARGUMENTS"\n',
        encoding="utf-8",
    )
    engine_binary.chmod(0o755)

    environment = {
        **os.environ,
        "HOME": str(home),
        "OAD_APPIMAGE": str(fake_appimage),
        "OAD_OBSERVER_BINARY": str(engine_binary),
        "OAD_OBSERVER_XVFB": "0",
        "FAKE_APPDIR": str(appdir),
        "CAPTURED_ARGUMENTS": str(captured_arguments),
    }
    environment.pop("DISPLAY", None)

    subprocess.run(
        [
            RUN_GAME,
            "--require-rl-observer",
            "-autostart-nonvisual",
            "--rl-interface=127.0.0.1:6000",
        ],
        check=True,
        env=environment,
    )

    assert captured_arguments.read_text(encoding="utf-8").splitlines() == [
        "-mod=mod",
        "-mod=public",
        "-mod=aoe3",
        "-autostart-nonvisual",
        "--rl-interface=127.0.0.1:6000",
    ]


def test_launcher_wraps_observer_in_xvfb_when_x11_display_is_missing(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    appdir = tmp_path / "appdir"
    for relative in ("usr/data/config", "usr/data/mods/mod", "usr/data/mods/public"):
        (appdir / relative).mkdir(parents=True)

    fake_appimage = tmp_path / "0ad.AppImage"
    fake_appimage.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "${1:-}" == "--appimage-mount" ]]; then\n'
        '  printf "%s\\n" "$FAKE_APPDIR"\n'
        "  sleep 30\n"
        "  exit 0\n"
        "fi\n"
        "exit 9\n",
        encoding="utf-8",
    )
    fake_appimage.chmod(0o755)

    engine_binary = tmp_path / "engine/binaries/system/pyrogenesis"
    engine_binary.parent.mkdir(parents=True)
    captured_arguments = tmp_path / "engine-arguments.txt"
    engine_binary.write_text(
        '#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$CAPTURED_ARGUMENTS"\n',
        encoding="utf-8",
    )
    engine_binary.chmod(0o755)

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    captured_xvfb = tmp_path / "xvfb-arguments.txt"
    fake_xvfb = fake_bin / "xvfb-run"
    fake_xvfb.write_text(
        "#!/usr/bin/env bash\n"
        'printf "%s\\n" "$@" > "$CAPTURED_XVFB"\n'
        'while [[ "$#" -gt 0 ]]; do\n'
        '  case "$1" in\n'
        "    -a) shift ;;\n"
        "    -s) shift 2 ;;\n"
        "    --) shift; break ;;\n"
        "    -*) shift ;;\n"
        "    *) break ;;\n"
        "  esac\n"
        "done\n"
        'exec "$@"\n',
        encoding="utf-8",
    )
    fake_xvfb.chmod(0o755)

    environment = {
        **os.environ,
        "HOME": str(home),
        "OAD_APPIMAGE": str(fake_appimage),
        "OAD_OBSERVER_BINARY": str(engine_binary),
        "FAKE_APPDIR": str(appdir),
        "CAPTURED_ARGUMENTS": str(captured_arguments),
        "CAPTURED_XVFB": str(captured_xvfb),
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
    }
    environment.pop("DISPLAY", None)
    environment.pop("OAD_SDL_VIDEODRIVER", None)

    subprocess.run(
        [RUN_GAME, "--require-rl-observer", "--rl-interface=127.0.0.1:6000"],
        check=True,
        env=environment,
    )

    assert captured_xvfb.read_text(encoding="utf-8").splitlines() == [
        "-a",
        "-s",
        "-screen 0 1024x768x24",
        str(engine_binary),
        "-mod=mod",
        "-mod=public",
        "-mod=aoe3",
        "--rl-interface=127.0.0.1:6000",
        "-xres=1024",
        "-yres=768",
        "-conf=windowed:true",
        "-conf=rendererbackend:gl",
    ]
    assert captured_arguments.read_text(encoding="utf-8").splitlines() == [
        "-mod=mod",
        "-mod=public",
        "-mod=aoe3",
        "--rl-interface=127.0.0.1:6000",
        "-xres=1024",
        "-yres=768",
        "-conf=windowed:true",
        "-conf=rendererbackend:gl",
    ]


def test_launcher_explains_missing_display_when_xvfb_is_disabled(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    fake_appimage = tmp_path / "0ad.AppImage"
    fake_appimage.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    fake_appimage.chmod(0o755)
    engine_binary = tmp_path / "engine/binaries/system/pyrogenesis"
    engine_binary.parent.mkdir(parents=True)
    engine_binary.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    engine_binary.chmod(0o755)

    environment = {
        **os.environ,
        "HOME": str(home),
        "OAD_APPIMAGE": str(fake_appimage),
        "OAD_OBSERVER_BINARY": str(engine_binary),
        "OAD_OBSERVER_XVFB": "0",
    }
    environment.pop("DISPLAY", None)

    result = subprocess.run(
        [RUN_GAME, "--require-rl-observer"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode != 0
    assert "No X11 display is available" in result.stderr
    assert "xvfb" in result.stderr


def test_launcher_rejects_a_second_observer_using_the_same_binary(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    appdir = tmp_path / "appdir"
    for relative in ("usr/data/config", "usr/data/mods/mod", "usr/data/mods/public"):
        (appdir / relative).mkdir(parents=True)

    fake_appimage = tmp_path / "0ad.AppImage"
    fake_appimage.write_text(
        '#!/usr/bin/env bash\nprintf "%s\\n" "$FAKE_APPDIR"\nsleep 30\n',
        encoding="utf-8",
    )
    fake_appimage.chmod(0o755)

    engine_binary = tmp_path / "engine/binaries/system/pyrogenesis"
    engine_binary.parent.mkdir(parents=True)
    engine_binary.write_text(
        "#!/usr/bin/env bash\n"
        'if mkdir "$ENGINE_GATE" 2>/dev/null; then\n'
        '  touch "$ENGINE_STARTED"\n'
        '  while [[ ! -e "$RELEASE_ENGINE" ]]; do sleep 0.02; done\n'
        '  rmdir "$ENGINE_GATE"\n'
        "fi\n",
        encoding="utf-8",
    )
    engine_binary.chmod(0o755)

    engine_started = tmp_path / "engine-started"
    release_engine = tmp_path / "release-engine"
    environment = {
        **os.environ,
        "DISPLAY": ":99",
        "HOME": str(home),
        "OAD_APPIMAGE": str(fake_appimage),
        "OAD_OBSERVER_BINARY": str(engine_binary),
        "FAKE_APPDIR": str(appdir),
        "ENGINE_GATE": str(tmp_path / "engine-gate"),
        "ENGINE_STARTED": str(engine_started),
        "RELEASE_ENGINE": str(release_engine),
    }
    first = subprocess.Popen(
        [RUN_GAME, "--require-rl-observer"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
    )
    try:
        deadline = time.monotonic() + 3.0
        while not engine_started.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert engine_started.exists()

        second = subprocess.run(
            [RUN_GAME, "--require-rl-observer"],
            check=False,
            capture_output=True,
            text=True,
            timeout=3.0,
            env=environment,
        )

        assert second.returncode != 0
        assert "already running" in second.stderr
    finally:
        release_engine.touch()
        first.communicate(timeout=3.0)
