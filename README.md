# 0 A.D. — AoE3 Mod

A mod for [0 A.D.](https://play0ad.com/) (Pyrogenesis engine) that brings **Age of Empires III** gameplay to the ancient world: custom civilizations, age progression, and unique unit mechanics inspired by historical warfare.

---

## Status

**In development.** Currently featuring:

- **Athenians (Classical Greece, 490–323 BC)** — playable civilization
- **3-age progression** — Golden Age → Peloponnesian Age → Macedonian Age
- **Custom unit mechanics** — Phalanx physics, area suppression, armor cracking
- **Custom components** — Stamina (sprint/exhaust), HoplitePhalanx (Othismos, flank vulnerability, rotational inertia)

More civilizations and eras planned.

---

## Quick Install

You need 0 A.D. already installed (from package manager or source).

```bash
# Clone this mod into the 0 A.D. mods directory
git clone <repo-url> /path/to/0ad/binaries/data/mods/aoe3

# Run with the mod enabled
cd /path/to/0ad/binaries/system
./pyrogenesis -mod=mod -mod=public -mod=aoe3
```

Or enable it from the in-game menu: **Settings → Mod Selection → aoe3**.

Then start a match and select **Athenians** as your civilization.

### Repo-local AppImage

For a downloaded Linux AppImage, keep the large binary untracked at
`.runtime/0ad/0ad.AppImage` and launch it through the repo wrapper:

```bash
./run_game.sh
```

The wrapper registers this checkout as the `aoe3` user mod and normalizes the
0 A.D. data directory. This avoids Snap-launched terminals making the game look
for mods under a Snap-specific `XDG_DATA_HOME`. On GNOME Wayland it also uses
XWayland by default to avoid an invisible cursor in the Release 28 AppImage;
set `OAD_SDL_VIDEODRIVER=wayland` to opt back into native Wayland.

For full build-from-source instructions see [MANUAL.md](MANUAL.md).

### Standard RL environment

The reinforcement-learning code uses a reproducible repo-level Python environment. With
[`uv`](https://docs.astral.sh/uv/) and `make` installed, one command creates `.venv/`, installs
Python 3.11 when needed, and syncs the exact versions from `uv.lock`:

```bash
# If `uv` is not installed yet on Ubuntu:
sudo apt install pipx
pipx install uv
export PATH="$HOME/.local/bin:$PATH"

make setup
```

If you used the standalone `uv` installer from a VS Code Snap terminal and it
installed under `~/snap/code/.../.local/bin`, run the `source .../env` command it
prints, or just rerun `make setup`; the Makefile also searches that Snap path.

Use the short, versioned project commands without activating the environment manually:

```bash
make help
make test
make oracle
```

The optional RL debug window uses a small, versioned engine patch so it can show the real
0 A.D. terrain, models, animation, and Player 1 line of sight instead of a schematic. Build it
once, then start the RL server:

```bash
make engine-observer
make server
```

The builder downloads and verifies the official 0 A.D. Release 28 build source under the ignored
`.runtime/` directory. The launcher reuses the art data from `.runtime/0ad/0ad.AppImage`, so it
does not create another full copy of the public mod assets. On Ubuntu, install the native build
helpers first with `sudo apt install build-essential cmake curl libboost-dev libboost-filesystem-dev libcurl4-gnutls-dev libenet-dev libfmt-dev libfreetype-dev libicu-dev libpng-dev libsdl2-dev libsodium-dev libx11-dev libxml2-dev llvm m4 patch pkg-config python3 uuid-dev xvfb zlib1g-dev`.
Allow roughly 7 GB for the source, local Rust toolchain, and build products. This observer build
omits audio, the lobby, and Atlas because it is intended only for local RL visualization.
If `make server` runs from a terminal without an X11 display, the launcher uses `xvfb-run`
automatically when `xvfb` is installed; otherwise install it with `sudo apt install xvfb`.

The Makefile is only a thin interface: `make setup` runs `uv sync --locked`, and the other
targets run their Python tools through `uv run`. For an interactive terminal, you can instead
run `source .venv/bin/activate` once and then use `python`, `pytest`, and `ruff` directly.

See [rl/README.md](rl/README.md) for the live 0 A.D. server, training, evaluation, and custom
agent workflow.

---

## Features

### Civilization: Athenians

| Element | Detail |
|---------|--------|
| Civ bonus | **Civic Duty** — villagers gather 10% faster |
| Hero | **Pericles** — 8 dual-ring auras (heal, gather, stamina, combat) |
| Theme | Classical Greek polis with citizen-soldier mechanics |

### Age progression

| Age | Greek Name | Requirement | Cost |
|-----|-----------|-------------|------|
| Golden Age | Chrysoun Aion | (start) | free |
| Peloponnesian Age | Peloponnesiakon Polemon | 5 Village buildings | 500 F + 500 W |
| Macedonian Age | Makedonikon Aion | 3 Town buildings | 750 S + 750 M |

### Unit roster (Golden Age)

| Unit | Greek | Role | Custom mechanic |
|------|-------|------|-----------------|
| Citizen | Polites | Worker | Stamina (sprint/exhaust) |
| Hoplite | Hoplites | Heavy melee | **Phalanx** (Othismos bonus, flank vulnerability, inertia) |
| Peltast | Peltastes | Skirmisher | — |
| Archer | Toxotes | Area suppression | **Suppressed** status (-15% speed, -10% attack) |
| Slinger | Sfendonetes | Armor cracker | **Cracked** status (-30% Hack/Pierce resistance) |
| Hero | Pericles | Support hero | 8 dual-ring auras |

### Tactical synergy

```
Slingers crack armor (-30% resistance)
   ↓
Archers suppress movement (-15% speed)
   ↓
Hoplites charge in phalanx (Othismos x2-3 damage)
```

---

## Project structure

```
aoe3/
├── README.md              # This file
├── MANUAL.md              # Detailed setup + mechanics reference
├── engine/                # Versioned Release 28 observer patch + build helper
├── mod.json               # Mod metadata
├── simulation/            # Game logic
│   ├── components/        # Custom JS components (HoplitePhalanx, Stamina)
│   ├── data/              # Civs, auras, technologies, status effects
│   └── templates/         # Unit and structure XML definitions
├── gui/                   # GUI overrides (session.js null-check patch)
└── art/                   # Visual actors and textures
```

---

## Development

The mod is fully self-contained — JS/XML changes are live (no recompile needed). Only C++ engine changes require rebuilding 0 A.D. from source.

```bash
# Edit any file under aoe3/, then relaunch the game
./pyrogenesis -mod=mod -mod=public -mod=aoe3
```

See [MANUAL.md](MANUAL.md) for:
- Full build-from-source instructions (Ubuntu/Debian)
- Mod structure walkthrough
- Custom component documentation
- Debugging tips

---

## Roadmap

- [x] Athenians (Golden Age unit roster)
- [x] Phalanx mechanics (Othismos, flank, inertia)
- [x] Status effects (Suppressed, Cracked)
- [x] Age progression system
- [ ] Peloponnesian Age units (cavalry, fortifications)
- [ ] Macedonian Age units (pike phalanx, siege)
- [ ] Additional civilizations (Sparta, Macedon, Persia)
- [ ] Technology tree per civilization
- [ ] Naval units

---

## Credits

- **Engine:** [0 A.D. (Pyrogenesis)](https://play0ad.com/) by Wildfire Games
- **Inspiration:** Age of Empires III by Ensemble Studios / Microsoft
- **Mod author:** [@nicoRomeroCuruchet](https://github.com/nicoRomeroCuruchet)

---

## License

This mod follows the 0 A.D. licensing model: code under GPLv2+, art/assets under CC-BY-SA 3.0.
