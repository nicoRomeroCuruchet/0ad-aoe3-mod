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

For full build-from-source instructions see [MANUAL.md](MANUAL.md).

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
