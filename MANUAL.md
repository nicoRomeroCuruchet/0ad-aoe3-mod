# AOE3 Mod — Athenai Civilization

A mod for 0 A.D. that adds the **Athenians** (Classical Greece, 490–323 BC) as a playable civilization with custom mechanics inspired by Age of Empires III.

System verified: **Ubuntu 24.04 / Debian 12**

---

## Quick Start (if 0 A.D. is already installed)

```bash
# Clone this mod into the 0 A.D. mods directory
git clone <repo-url> ~/Documents/0ad/binaries/data/mods/aoe3

# Run the game
cd ~/Documents/0ad/binaries/system
./pyrogenesis -mod=mod -mod=public -mod=aoe3
```

In the match setup screen, select **Athenians** as your civilization.

---

## Full Installation (from source)

### 1. System dependencies

```bash
sudo apt install build-essential cmake curl git git-lfs subversion \
  libboost-dev libboost-system-dev libboost-filesystem-dev \
  libcurl4-gnutls-dev libenet-dev libfmt-dev libfreetype-dev \
  libgloox-dev libicu-dev libminiupnpc-dev libnvtt-dev \
  libogg-dev libopenal-dev libpng-dev libsdl2-dev libsodium-dev \
  libvorbis-dev libwxgtk3.2-dev libxml2-dev llvm m4 python3 \
  rustc cargo zlib1g-dev xz-utils patch
```

### 2. Rust 1.76+ (rustup)

The apt `rustc` on Ubuntu 24.04 is 1.75 — SpiderMonkey needs 1.76+:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
rustc --version   # must be 1.76+
```

### 3. cbindgen

```bash
cargo install --locked cbindgen@0.29.0
```

### 4. Clone 0 A.D. source

```bash
git clone https://github.com/0ad/0ad.git ~/Documents/0ad
cd ~/Documents/0ad
```

> The path **cannot contain spaces** — breaks SpiderMonkey compilation.

### 5. Git LFS assets

```bash
git lfs install
git lfs pull
```

### 6. Build third-party libraries (~20-40 min)

```bash
./libraries/build-source-libs.sh -j$(nproc)
```

> If interrupted, just rerun — it detects what's already built.
> Requires `svn.wildfiregames.com` to be reachable (for FCollada).

### 7. Generate makefiles

```bash
./build/workspaces/update-workspaces.sh
```

### 8. Compile the game (~10-20 min)

```bash
make -C build/workspaces/gcc/ -j$(nproc) config=release
```

### 9. Install this mod

```bash
git clone <repo-url> ~/Documents/0ad/binaries/data/mods/aoe3
```

### 10. Run

```bash
cd ~/Documents/0ad/binaries/system
./pyrogenesis -mod=mod -mod=public -mod=aoe3
```

The mod is also activatable from **Settings → Mod Selection** in the game menu.

---

## Mod Structure

```
aoe3/
├── mod.json                          # Mod metadata
├── MANUAL.md                         # This file
├── simulation/
│   ├── data/
│   │   ├── civs/athenai.json         # Civilization definition
│   │   ├── auras/units/heroes/       # Pericles auras (8 files)
│   │   ├── status_effects/           # Suppressed, Cracked
│   │   └── technologies/             # Phase definitions (Golden Age, etc.)
│   ├── templates/
│   │   ├── civ/athenai.xml
│   │   ├── special/                  # Player + rallypoint entities
│   │   ├── structures/athenai/       # Agora, Stratopedion, Oikos, Apotheke
│   │   └── units/athenai/            # Polites, Hoplite, Peltast, Archer, Slinger, Pericles
│   └── components/
│       ├── HoplitePhalanx.js         # Custom: Othismos, flank vulnerability, inertia
│       ├── Stamina.js                # Custom: sprint/exhaust system
│       └── interfaces/               # Component interface registrations
├── gui/session/session.js            # Override: null-check patch for missing templates
└── art/
    ├── actors/units/athenai/         # Visual actor definitions
    └── textures/ui/session/icons/    # UI assets
```

---

## Mechanics Overview

### Civilization
- **Code:** `athenai`
- **Starting age:** Golden Age (auto-researched `phase_village`)
- **Civ bonus:** "Civic Duty" — Villagers gather 10% faster
- **Hero:** Pericles (Athenian Strategos)

### Age progression

| Age | Engine phase | Requirement | Cost |
|-----|-------------|-------------|------|
| Golden Age | `phase_village` | (start) | free |
| Peloponnesian Age | `phase_town` | 5 Village buildings | 500 food + 500 wood |
| Macedonian Age | `phase_city` | 3 Town buildings | 750 stone + 750 metal |

### Unit roster (Golden Age)

| Unit | Greek | Role |
|------|-------|------|
| Polites | Citizen | Worker with Stamina system |
| Hoplite | Hoplites | Heavy infantry + Phalanx mechanics |
| Peltast | Peltastes | Light javelin skirmisher |
| Archer | Toxotes | Area suppression (slow + attack speed debuff) |
| Slinger | Sfendonetes | Armor cracker (-30% Hack/Pierce resistance) |
| Hero | Pericles | 8-aura support hero |

### Custom components

| Component | Effect |
|-----------|--------|
| `HoplitePhalanx` | Othismos proximity bonus, directional vulnerability, rotational inertia |
| `Stamina` | Sprint/exhaust speed system for villagers |

---

## Developer Notes

- **JS/XML changes** are live — no recompile needed.
- **C++ engine changes** require `make -C build/workspaces/gcc/ config=release` from the 0 A.D. root.
- **Debug build:** use `config=debug` (slower, with symbols).
- The mod overrides `gui/session/session.js` with a null-check patch — this is intentional and required for the engine not to crash on certain template lookups.

---

## Restoring on a Fresh Machine

### Simple case: just want the mod

1. Install 0 A.D. (follow steps 1-8 above, or `sudo apt install 0ad`)
2. Clone this mod into the 0 A.D. mods directory:
   ```bash
   git clone git@github.com:nicoRomeroCuruchet/0ad-aoe3-mod.git \
     ~/Documents/0ad/binaries/data/mods/aoe3
   ```
3. Launch with `-mod=mod -mod=public -mod=aoe3` or activate in the menu.

No other files need to be modified outside the mod — everything is self-contained.

---

## Full OS Migration

When changing operating systems or moving to a new machine, the only things that aren't already in git are:

- **Claude Code memory** (~/.claude/projects/-home-nromero-Documents-0ad/) — project context and notes
- **Any loose files** in the 0 A.D. repo root that you want to keep (screenshots, etc.)

### Before migration: create the backup

```bash
# Make sure the mod is pushed to GitHub first
cd ~/Documents/0ad/binaries/data/mods/aoe3
git push

# Create migration backup with Claude memory + any loose files
mkdir -p /tmp/0ad-backup
cp -r ~/.claude/projects/-home-nromero-Documents-0ad /tmp/0ad-backup/claude-project
cp ~/Documents/0ad/image_2.jpg /tmp/0ad-backup/ 2>/dev/null || true
cd /tmp && tar czf ~/0ad-migration-backup.tar.gz 0ad-backup/
rm -rf /tmp/0ad-backup

# Save the tarball somewhere safe (USB, cloud, another machine)
ls -lh ~/0ad-migration-backup.tar.gz
```

### After migration: restore on the new machine

**Step 1 — Install 0 A.D. from source:**
```bash
git clone https://github.com/0ad/0ad.git ~/Documents/0ad
# Then follow sections 1-8 of this MANUAL
```

**Step 2 — Clone the mod:**
```bash
git clone git@github.com:nicoRomeroCuruchet/0ad-aoe3-mod.git \
  ~/Documents/0ad/binaries/data/mods/aoe3
```

**Step 3 — Restore the backup:**
```bash
cd ~ && tar xzf 0ad-migration-backup.tar.gz

# Restore Claude Code memory
mkdir -p ~/.claude/projects/
mv ~/0ad-backup/claude-project ~/.claude/projects/-home-nromero-Documents-0ad

# Restore loose files (if any)
mv ~/0ad-backup/image_2.jpg ~/Documents/0ad/ 2>/dev/null || true

# Cleanup
rm -rf ~/0ad-backup
```

**Step 4 — Compile and run:**
```bash
cd ~/Documents/0ad
./libraries/build-source-libs.sh -j$(nproc)
./build/workspaces/update-workspaces.sh
make -C build/workspaces/gcc/ -j$(nproc) config=release
./binaries/system/pyrogenesis -mod=mod -mod=public -mod=aoe3
```

### What's preserved by git (no backup needed)

| Asset | Source |
|-------|--------|
| 0 A.D. game source | `github.com/0ad/0ad` |
| This mod (all code, civs, mechanics) | `github.com/nicoRomeroCuruchet/0ad-aoe3-mod` |
| Setup instructions | This file (inside the mod) |
| Game binaries | Recompiled from source |
| Vanilla fonts | Git LFS (`git lfs pull` re-downloads them) |
