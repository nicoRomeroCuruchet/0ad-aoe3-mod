import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_launcher_ignores_snap_xdg_data_home_and_registers_mod(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    fake_appimage = tmp_path / "0ad.AppImage"
    captured_environment = tmp_path / "environment.txt"
    fake_appimage.write_text(
        "#!/usr/bin/env bash\n"
        'printf "%s\\n" "$XDG_DATA_HOME" "$@" > "$CAPTURED_ENVIRONMENT"\n',
        encoding="utf-8",
    )
    fake_appimage.chmod(0o755)

    environment = {
        **os.environ,
        "HOME": str(home),
        "XDG_DATA_HOME": str(home / "snap/code/current/.local/share"),
        "OAD_APPIMAGE": str(fake_appimage),
        "CAPTURED_ENVIRONMENT": str(captured_environment),
    }
    environment.pop("OAD_XDG_DATA_HOME", None)
    subprocess.run(
        [REPO_ROOT / "run_game.sh", "-quickstart"],
        check=True,
        env=environment,
    )

    assert captured_environment.read_text(encoding="utf-8").splitlines() == [
        str(home / ".local/share"),
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
        [REPO_ROOT / "run_game.sh"],
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
