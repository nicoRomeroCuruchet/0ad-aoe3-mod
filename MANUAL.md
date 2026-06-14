# AOE3 Mod — Athenai Civilization

A mod for 0 A.D. that adds the **Athenians** (Classical Greece, 490–323 BC) as a playable civilization with custom mechanics inspired by Age of Empires III.

System verified: **Ubuntu 24.04 / Debian 12**

---

## Quick Start (if 0 A.D. is already installed)

```bash
# Clone this mod into the 0 A.D. mods directory
git clone <repo-url> ~/Documents/0ad/binaries/data/mods/aoe3

# Habilitar el mod de forma persistente (si no, la civ no aparece)
echo 'mod.enabledmods = "mod public aoe3"' >> ~/.config/0ad/config/user.cfg

# Run the game
cd ~/Documents/0ad/binaries/system
./pyrogenesis -mod=mod -mod=public -mod=aoe3
```

Para jugar con el mod: **Un jugador → Combate → Tipo de mapa: _Aleatorio_** (importante,
NO Escaramuza) → civilización **"Athenai (AOE3 Mod)"**. Empezás con Pericles y tu ejército.

> Si compilás 0 A.D. en Ubuntu 26.04 / Python moderno, leé primero la sección
> **"Build Notes & Known Issues"** más abajo: hay 3 parches obligatorios de compilación
> y la guía completa de habilitación del mod.

---

## Compilación automática (recomendado) — `setup.sh`

Para **Ubuntu 24.04 y 26.04** hay un script que hace TODO de punta a punta (deps, clona 0 A.D.,
parches, compila, instala y habilita el mod). Es idempotente (se puede re-correr):

```bash
# 1. Clonar este repo
git clone git@github.com:nicoRomeroCuruchet/0ad-aoe3-mod.git ~/dev/0ad-aoe3-mod
cd ~/dev/0ad-aoe3-mod

# 2. Un comando (pide sudo para las deps; tarda ~1 h: descarga + compila)
bash setup.sh

# 3. Jugar
cd ~/Documents/0ad/binaries/system
./pyrogenesis -mod=mod -mod=public -mod=aoe3
```

> Verificado end-to-end en **Ubuntu 26.04**. Diseñado también para **24.04** (mismo enfoque:
> Python 3.11 efímero + parches inocuos); en 24.04 conviene una primera corrida de prueba.
> El destino del código de 0 A.D. es `~/Documents/0ad` (cambialo con `OAD=/otra/ruta bash setup.sh`).

La sección de abajo es el **paso a paso manual** (lo que el script automatiza), útil para entender
o depurar. Los problemas concretos y sus parches están en **"Build Notes & Known Issues"**.

---

## Full Installation (from source) — paso a paso manual

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

## Build Notes & Known Issues (per environment)

> Registro de pasos que fallaron al compilar desde cero, para que el equipo no
> tropiece con lo mismo. La verificación original fue en Ubuntu 24.04 / Debian 12;
> estas notas agregan lo encontrado en sistemas más nuevos.

### Ubuntu 26.04 — 2026-06-13 (build + mod jugable, VERIFICADO end-to-end)

**1. `libboost-system-dev` ya no existe como paquete (menor, se resuelve solo).**
En Ubuntu 26.04 (Boost 1.90) Boost.System pasó a ser header-only. `libboost-system-dev`
es ahora un paquete *virtual* provisto por `libboost-dev`. `apt install` lo resuelve
automáticamente, así que el comando de la sección 1 funciona sin cambios.

**2. `./libraries/build-source-libs.sh` NO existe en el mirror de GitHub.**
El clon de `github.com/0ad/0ad` (commit `61a3b95`, ago-2024) no trae ese script
unificado. En esta versión, **`update-workspaces.sh` compila las librerías de terceros
(FCollada, SpiderMonkey, NVTT) Y genera los makefiles en un solo paso.** Entonces las
secciones 6 y 7 se fusionan:

```bash
# Reemplaza los pasos 6 y 7 en esta versión del código:
JOBS="-j$(nproc)" ./build/workspaces/update-workspaces.sh
```

(Cada librería igual tiene su `build.sh` individual en `libraries/source/<lib>/`.)

**3. ⛔ BLOQUEANTE — SpiderMonkey no compila con Python 3.14 (Ubuntu 26.04).**
`mozjs-91.13.1` (incluido en 0 A.D., de 2021) está pensado para Python 3.6–3.9. Con el
Python 3.14 de Ubuntu 26.04 el build falla:

```
ModuleNotFoundError: No module named 'six.moves'
ERROR: SpiderMonkey build failed
```

El `build.sh` de SpiderMonkey fuerza `PYTHONNOUSERSITE=true`, así que instalar `six`
con `pip --user` NO sirve. FCollada sí compila bien; el bloqueo es solo SpiderMonkey.

**Solución (verificada):** usar un **Python 3.11 standalone efímero**, ya que Python solo
se necesita en tiempo de *build* (no es dependencia de runtime del juego). Se descarga un
CPython 3.11 portable, se pone en el `PATH` solo para el build, y se borra al terminar —
sin tocar el Python del sistema (downgrade rompería Ubuntu 26.04) ni dejar nada permanente.

```bash
cd ~/Documents/0ad && mkdir -p .toolchain && cd .toolchain
# Buscar el tag/fecha del último release en:
#   https://github.com/astral-sh/python-build-standalone/releases
# (verificado con tag 20260610 → CPython 3.11.15)
curl -fL -o py311.tar.gz \
  "https://github.com/astral-sh/python-build-standalone/releases/download/20260610/cpython-3.11.15%2B20260610-x86_64-unknown-linux-gnu-install_only.tar.gz"
tar xzf py311.tar.gz                       # -> ./python/
export PATH="$PWD/python/bin:$PATH"
python3 --version                          # 3.11.x (solo en esta terminal)
```

Dejá esta terminal abierta con el `PATH` exportado para correr los pasos 6-8. Al terminar
de compilar, podés borrar `~/Documents/0ad/.toolchain` (solo se vuelve a necesitar si
recompilás el motor C++ — los cambios JS/XML del mod NO lo requieren).

**4. ⛔ BLOQUEANTE — NVTT no configura con CMake 4.x (Ubuntu 26.04).**
Ubuntu 26.04 trae CMake 4.x, que eliminó la compatibilidad con `cmake_minimum_required`
< 3.5. NVTT (viejo) la declara, así que su `cmake` aborta:

```
CMake Error at CMakeLists.txt:1 (CMAKE_MINIMUM_REQUIRED):
  Compatibility with CMake < 3.5 has been removed from CMake.
ERROR: NVTT build failed
```

**Solución (verificada):** exportar el flag de compatibilidad antes de
`update-workspaces.sh` (el `build.sh` de NVTT inyecta `$CMAKE_FLAGS` a su llamada a cmake):

```bash
export CMAKE_FLAGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5"
export CMAKE_POLICY_VERSION_MINIMUM=3.5
```

**5. ⛔ BLOQUEANTE — el enlace de `pyrogenesis` falla con `cannot find -lboost_system`.**
Otra consecuencia de Boost 1.90: como Boost.System es header-only, **no existe**
`libboost_system.so`, pero el build de 0 A.D. todavía intenta enlazarlo. El `make`
(paso 8) aborta al linkear:

```
/usr/bin/x86_64-linux-gnu-ld.bfd: cannot find -lboost_system: No such file or directory
collect2: error: ld returned 1 exit status
make: *** [pyrogenesis] Error 2
```

**Solución (verificada):** editar `build/premake/extern_libs5.lua` (dentro del repo de
0 A.D., **no** en el mod) y quitar `boost_system` de la lista `unix_names`, dejando solo
`boost_filesystem` (que sí tiene `.so` y sigue haciendo falta). En la definición de boost,
la línea `unix_names = { ... "boost_system" }` queda:

```lua
unix_names = { os.findlib("boost_filesystem-mt") and "boost_filesystem-mt" or "boost_filesystem" },
```

> ⚠️ Este parche es sobre el **árbol de 0 A.D.**, no sobre el mod, así que se pierde si
> reclonás 0 A.D. desde cero. Hay que reaplicarlo en cada checkout nuevo del engine.

Tras editarlo, **regenerar los makefiles** (`update-workspaces.sh`) y recién ahí `make`,
porque el `-lboost_system` queda horneado en los `.make` ya generados.

### Resumen: secuencia completa que funciona en Ubuntu 26.04

```bash
cd ~/Documents/0ad

# (a) Python 3.11 efímero
mkdir -p .toolchain && cd .toolchain
curl -fL -o py311.tar.gz "https://github.com/astral-sh/python-build-standalone/releases/download/20260610/cpython-3.11.15%2B20260610-x86_64-unknown-linux-gnu-install_only.tar.gz"
tar xzf py311.tar.gz && cd ..
export PATH="$PWD/.toolchain/python/bin:$PATH"

# (b) Flags de compat CMake (para NVTT)
export CMAKE_FLAGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5"
export CMAKE_POLICY_VERSION_MINIMUM=3.5

# (b2) Parche Boost.System: quitar boost_system del enlace (header-only en Boost 1.90)
sed -i 's#unix_names = { os.findlib("boost_filesystem-mt") and "boost_filesystem-mt" or "boost_filesystem", os.findlib("boost_system-mt") and "boost_system-mt" or "boost_system" },#unix_names = { os.findlib("boost_filesystem-mt") and "boost_filesystem-mt" or "boost_filesystem" },#' build/premake/extern_libs5.lua

# (c) Build libs de terceros + makefiles (reemplaza pasos 6 y 7)
JOBS="-j$(nproc)" ./build/workspaces/update-workspaces.sh

# (d) Compilar el juego
make -C build/workspaces/gcc/ -j$(nproc) config=release

# (e) Limpieza opcional del Python efímero
rm -rf ~/Documents/0ad/.toolchain
```

---

## Habilitar el mod y problemas de compatibilidad (post-compilación)

> Tras compilar, hubo que resolver varios problemas para que el **mod** funcionara
> contra 0 A.D. trunk (0.0.27). El mod fue escrito contra una versión anterior y varias
> APIs/convenciones cambiaron. Todo lo de abajo ya está **arreglado en este repo** —
> se documenta para que el equipo entienda y para reaplicar si reclonan 0 A.D.

### 0. Instalar el mod y habilitarlo (¡el paso que más confunde!)

Symlinkear (o clonar) el mod en la carpeta de mods de 0 A.D.:

```bash
ln -sfn ~/dev/research/0ad-aoe3-mod ~/Documents/0ad/binaries/data/mods/aoe3
```

**CRÍTICO:** además de pasar `-mod=aoe3` por línea de comando, hay que **habilitar el mod
de forma persistente**, o al abrir el juego normalmente arranca sin él y la civ no aparece.
Editar `~/.config/0ad/config/user.cfg` y agregar:

```
mod.enabledmods = "mod public aoe3"
```

(O activarlo desde el juego: **Configuración → Selección de Mods → aoe3 → Guardar**.)

### 1. La civ no aparecía / aparecía duplicada como "Athenians"

En 0.0.27 el **nombre de la civilización ya NO sale del JSON** (`civs/athenai.json`), sino
del template de jugador `simulation/templates/special/players/athenai.xml` →
`Identity/GenericName`. El mod tenía ahí `Athenians`, **idéntico a la civ vanilla**, así que
en el menú había dos "Athenians" y se elegía la equivocada.
**Fix:** `GenericName` = `Athenai (AOE3 Mod)` (nombre distintivo).

### 2. Crash de la GUI de partida (`CinemaOverlay is not defined`)

El mod sobrescribía `gui/session/session.js` con una copia vieja de la vanilla, incompatible
con el resto de la GUI del engine actual → la sesión no inicializaba.
**Fix:** se eliminó el override (se renombró a `session.js.stale-override-disabled`); el
engine usa su propia `session.js`. (La nota del MANUAL sobre un "null-check patch requerido"
quedó obsoleta: el diff era puro drift de versión, sin lógica propia del mod.)

### 3. Plantillas de unidades que no cargaban

- `units/aldeano` heredaba de `template_unit_support_civilian` (renombrado en trunk).
  **Fix:** parent → `template_unit_support_female_citizen`.
- Faltaban 4 plantillas del roster estándar que el civic-centre vanilla espera
  (`units/athenai/{support_female_citizen,infantry_spearman_b,cavalry_javelineer_b}`,
  `structures/athenai/defense_tower`). **Fix:** stubs que heredan de la civ vanilla `athen`.
- Actores visuales rotos por typo: `units/athenians/citizen_female.xml` →
  `units/athenians/female_citizen.xml` (en `aldeano.xml` y `polites.xml`).
- Ícono de retrato inexistente `units/athen/support_civilian.png` →
  `units/athen/support_female_citizen.png`.

### 4. El mapa aleatorio del mod (`mapa_simple`) abortaba el juego

Dos cambios de API en los scripts de mapa (RMS) de 0.0.27, en `maps/random/mapa_simple.js`:

- Línea 4: `export function* generateMap(mapSettings)` → **`function* GenerateMap()`**
  (sin `export`, G mayúscula, sin parámetros).
- Línea 121: `playerPlacementCircle()` devuelve un **array**, no un objeto:
  `const { playerIDs, playerPosition } = ...` → `const [playerIDs, playerPosition] = ...`.

### 5. ⚠️ Importante para jugar: usar mapa **Aleatorio**, no Escaramuza

Las unidades custom (Pericles, Polites y el ejército inicial) están en las `StartEntities`
de la civ, que **solo se aplican en mapas Aleatorios**. En mapas de **Escaramuza** las
unidades iniciales las coloca el mapa (roster estándar → se ven como las vanilla).

**Para ver el mod:** Un jugador → Combate → **Tipo de mapa: Aleatorio** → mapa
`Mapa Simple AOE3` (o cualquier aleatorio) → civilización **Athenai (AOE3 Mod)** → Empezar.

### Lanzamiento rápido verificado (headless o visual)

```bash
cd ~/Documents/0ad/binaries/system
# Visual, partida directa como Atenienses en el mapa del mod:
./pyrogenesis -mod=mod -mod=public -mod=aoe3 \
  -autostart="random/mapa_simple" -autostart-size=192 -autostart-players=2 \
  -autostart-civ=1:athenai -autostart-civ=2:athen
```

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
