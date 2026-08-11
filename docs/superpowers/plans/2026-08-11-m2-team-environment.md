# M2 Team Environment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the 4-villager / 4-tree gather environment and prove it works with a greedy oracle, without training anything.

**Architecture:** Extract stable-ordering, observation assembly, and reward composition out of the 950-line `rl/gather/env.py` into three focused pure modules, then generalize the environment over unit counts so M0/M1 become the `n=1` case. The environment reads all engine state through one batched `game.evaluate` call per step. A greedy oracle baseline is the acceptance gate.

**Tech Stack:** Python 3.11, Gymnasium, NumPy, pytest, uv (locked). No new dependencies. Stable-Baselines3 is untouched by this plan.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-11-m2-team-gather-design.md`. Where this plan and the spec disagree, the spec wins.
- **No behavior change for M0/M1.** The 326 existing tests must keep passing untouched at every commit. They are the safety net for the refactor.
- Slice layout is fixed at **31 values**: indices 0–9 are M1's observation in M1's exact order, index 10 is the agent ID, indices 11–30 are the relational block (4 trees × 5 values).
- Relational block is ordered by `entity_id`, never by distance.
- Shaping terms are **averaged** over villagers (divided by `n`), never summed.
- Every test in this plan runs offline with fake backends. No test may require 0 A.D.
- Run tests with `.venv/bin/python -m pytest` (this machine's `uv` is older than the lockfile requires; `make test` fails with a version error).
- Lint with `.venv/bin/python -m ruff check rl` before every commit.
- Python style follows the existing modules: `from __future__ import annotations`, frozen slotted dataclasses for value objects, explicit `ValueError` subclasses for contract violations.

---

## File Structure

**Create:**
- `rl/gather/roster.py` — stable slot↔entity ordering and count validation. No engine knowledge beyond `.id()` / `.position()`.
- `rl/gather/observation.py` — team observation slices and their labels. Pure; takes a snapshot value object, returns an array.
- `rl/gather/reward.py` — reward term composition. Pure; takes scalars, returns a term breakdown.
- `rl/tests/gather/test_roster.py`, `rl/tests/gather/test_observation.py`, `rl/tests/gather/test_reward.py`
- `maps/random/rl_gather_team.js`, `maps/random/rl_gather_team.json`
- `rl/scenarios/team_reset_config.json`
- `rl/configs/m2_oracle.toml`, `rl/configs/m2_random.toml`

**Modify:**
- `rl/gather/env.py` — consume the three new modules; add `villager_count` / `resource_count`; batched snapshot.
- `rl/gather/core.py` — add `nearest_index` helper.
- `rl/agents/baselines.py` — add `TeamGatherOraclePolicy`.
- `rl/agents/registry.py` — register `team_oracle`.
- `rl/experiments/environments.py:24-57` — allow the new environment parameters.
- `rl/tests/test_map_contract.py`, `rl/tests/experiments/test_tracked_configs.py`, `rl/tests/gather/test_env.py`

---

### Task 1: Stable roster

The bug this prevents: `state.units(...)` does not promise a stable order between steps. If the order changes, action slot 2 silently starts driving a different villager.

**Files:**
- Create: `rl/gather/roster.py`
- Test: `rl/tests/gather/test_roster.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Roster` (frozen dataclass with `.villagers: tuple`, `.resources: tuple`, `.dropsite`), `build_roster(state, *, villager_count, resource_count, villager_type="polites", resource_type="tree", dropsite_type="storehouse") -> Roster`, and `RosterError(ValueError)`.

- [ ] **Step 1: Write the failing test**

```python
# rl/tests/gather/test_roster.py
from dataclasses import dataclass

import pytest

from rl.gather.roster import Roster, RosterError, build_roster


@dataclass(frozen=True)
class FakeUnit:
    coordinates: tuple[float, float]
    entity_id: int

    def position(self):
        return self.coordinates

    def id(self):
        return self.entity_id


class FakeState:
    def __init__(self, villagers, resources, dropsites):
        self._units = {
            (1, "polites"): list(villagers),
            (0, "tree"): list(resources),
            (1, "storehouse"): list(dropsites),
        }

    def units(self, *, owner, entity_type):
        return self._units[(owner, entity_type)]


def _state(villager_ids, resource_ids):
    villagers = [FakeUnit((float(i), 0.0), i) for i in villager_ids]
    resources = [FakeUnit((0.0, float(i)), i) for i in resource_ids]
    return FakeState(villagers, resources, [FakeUnit((5.0, 5.0), 99)])


def test_roster_orders_entities_by_id():
    roster = build_roster(_state([30, 10, 20], [7, 5]), villager_count=3, resource_count=2)

    assert isinstance(roster, Roster)
    assert [unit.id() for unit in roster.villagers] == [10, 20, 30]
    assert [unit.id() for unit in roster.resources] == [5, 7]
    assert roster.dropsite.id() == 99


def test_roster_slot_order_survives_a_shuffled_engine_listing():
    first = build_roster(_state([30, 10, 20], [7, 5]), villager_count=3, resource_count=2)
    second = build_roster(_state([20, 30, 10], [5, 7]), villager_count=3, resource_count=2)

    assert [unit.id() for unit in first.villagers] == [
        unit.id() for unit in second.villagers
    ]
    assert [unit.id() for unit in first.resources] == [
        unit.id() for unit in second.resources
    ]


def test_roster_rejects_an_unexpected_villager_count():
    with pytest.raises(RosterError, match="expected 4 villagers, found 3"):
        build_roster(_state([1, 2, 3], [4, 5]), villager_count=4, resource_count=2)


def test_roster_rejects_an_unexpected_resource_count():
    with pytest.raises(RosterError, match="expected 3 resources, found 2"):
        build_roster(_state([1, 2], [4, 5]), villager_count=2, resource_count=3)


def test_roster_allows_a_missing_dropsite():
    state = FakeState([FakeUnit((0.0, 0.0), 1)], [FakeUnit((1.0, 1.0), 2)], [])

    roster = build_roster(state, villager_count=1, resource_count=1)

    assert roster.dropsite is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest rl/tests/gather/test_roster.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rl.gather.roster'`

- [ ] **Step 3: Write minimal implementation**

```python
# rl/gather/roster.py
"""Stable slot assignment between action indices and engine entities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class RosterError(ValueError):
    """Raised when the scenario does not match the configured unit counts."""


@dataclass(frozen=True, slots=True)
class Roster:
    """Entities addressed by slot for the whole episode."""

    villagers: tuple[Any, ...]
    resources: tuple[Any, ...]
    dropsite: Any | None


def _sorted_by_id(units: list[Any]) -> tuple[Any, ...]:
    return tuple(sorted(units, key=lambda unit: int(unit.id())))


def _require_count(units: tuple[Any, ...], expected: int, noun: str) -> None:
    if len(units) != expected:
        raise RosterError(f"expected {expected} {noun}, found {len(units)}")


def build_roster(
    state: Any,
    *,
    villager_count: int,
    resource_count: int,
    villager_type: str = "polites",
    resource_type: str = "tree",
    dropsite_type: str = "storehouse",
) -> Roster:
    """Order entities by engine id so action slots stay put across steps.

    The engine does not promise a stable ordering from `state.units(...)`, and a
    reordering would silently repoint an action slot at a different villager.
    Sorting by entity id is the slot assignment.
    """

    villagers = _sorted_by_id(state.units(owner=1, entity_type=villager_type))
    resources = _sorted_by_id(state.units(owner=0, entity_type=resource_type))
    _require_count(villagers, villager_count, "villagers")
    _require_count(resources, resource_count, "resources")
    dropsites = _sorted_by_id(state.units(owner=1, entity_type=dropsite_type))
    return Roster(
        villagers=villagers,
        resources=resources,
        dropsite=dropsites[0] if dropsites else None,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest rl/tests/gather/test_roster.py -v`
Expected: PASS, 5 tests

- [ ] **Step 5: Verify nothing else broke, then commit**

Run: `.venv/bin/python -m pytest -q && .venv/bin/python -m ruff check rl`
Expected: 331 passed, "All checks passed!"

```bash
git add rl/gather/roster.py rl/tests/gather/test_roster.py
git commit -m "feat(rl): add stable entity roster for multi-unit slots"
```

---

### Task 2: Nearest-index helper

**Files:**
- Modify: `rl/gather/core.py` (append at end of file)
- Test: `rl/tests/test_core.py` (append at end of file)

**Interfaces:**
- Consumes: `distance` from `rl.gather.core`.
- Produces: `nearest_index(origin_xz, candidates_xz) -> int`.

- [ ] **Step 1: Write the failing test**

```python
# append to rl/tests/test_core.py
def test_nearest_index_picks_the_closest_candidate():
    from rl.gather.core import nearest_index

    candidates = [(10.0, 0.0), (1.0, 1.0), (5.0, 5.0)]

    assert nearest_index((0.0, 0.0), candidates) == 1


def test_nearest_index_breaks_ties_by_the_lowest_index():
    from rl.gather.core import nearest_index

    candidates = [(3.0, 0.0), (0.0, 3.0)]

    assert nearest_index((0.0, 0.0), candidates) == 0


def test_nearest_index_rejects_an_empty_candidate_list():
    from rl.gather.core import nearest_index
    import pytest

    with pytest.raises(ValueError, match="at least one candidate"):
        nearest_index((0.0, 0.0), [])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest rl/tests/test_core.py -k nearest_index -v`
Expected: FAIL with `ImportError: cannot import name 'nearest_index'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to rl/gather/core.py
def nearest_index(origin_xz, candidates_xz):
    """Index of the closest candidate; ties resolve to the lowest index."""

    candidates = list(candidates_xz)
    if not candidates:
        raise ValueError("nearest_index requires at least one candidate")
    best_index = 0
    best_distance = distance(origin_xz, candidates[0])
    for index in range(1, len(candidates)):
        candidate_distance = distance(origin_xz, candidates[index])
        if candidate_distance < best_distance:
            best_index = index
            best_distance = candidate_distance
    return best_index
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest rl/tests/test_core.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add rl/gather/core.py rl/tests/test_core.py
git commit -m "feat(rl): add nearest_index geometry helper"
```

---

### Task 3: Team observation

Builds the `(villager_count, 31)` array. Indices 0–9 must equal M1's observation exactly so an M1 checkpoint can initialize the shared network's input prefix later.

**Files:**
- Create: `rl/gather/observation.py`
- Test: `rl/tests/gather/test_observation.py`

**Interfaces:**
- Consumes: `normalize_coord`, `normalize_non_negative`, `distance` from `rl.gather.core`.
- Produces: `TeamSnapshot` (frozen dataclass), `TeamObservationScales` (frozen dataclass), `build_team_observation(snapshot, scales) -> np.ndarray` of shape `(n_villagers, 11 + 5 * n_resources)`, `team_observation_labels(villager_count, resource_count) -> tuple[str, ...]`, and `TEAM_CORE_LABELS`.

- [ ] **Step 1: Write the failing test**

```python
# rl/tests/gather/test_observation.py
import numpy as np
import pytest

from rl.gather.core import GATHER_LIFECYCLE_OBSERVATION_LABELS
from rl.gather.observation import (
    TEAM_CORE_LABELS,
    TeamObservationScales,
    TeamSnapshot,
    build_team_observation,
    team_observation_labels,
)


SCALES = TeamObservationScales(
    map_size_m=512.0,
    carried_resource_scale=20.0,
    stock_scale=1000.0,
    resource_amount_scale=200.0,
)


def _snapshot():
    return TeamSnapshot(
        villager_xz=((128.0, 256.0), (384.0, 256.0)),
        resource_xz=((256.0, 256.0), (400.0, 256.0)),
        resource_remaining=(200.0, 100.0),
        carried=(0.0, 20.0),
        target_index=(0, 1),
        gather_cycle_active=(False, True),
        dropsite_xz=(64.0, 320.0),
        stock=300.0,
    )


def test_core_prefix_matches_the_m1_observation_layout():
    assert TEAM_CORE_LABELS == GATHER_LIFECYCLE_OBSERVATION_LABELS


def test_slice_shape_and_labels_follow_the_configured_counts():
    observation = build_team_observation(_snapshot(), SCALES)
    labels = team_observation_labels(2, 2)

    assert observation.shape == (2, 21)
    assert observation.dtype == np.float32
    assert len(labels) == 21
    assert labels[:10] == TEAM_CORE_LABELS
    assert labels[10] == "agent_id_norm"
    assert labels[11:16] == (
        "tree0_dx_norm",
        "tree0_dz_norm",
        "tree0_dist_norm",
        "tree0_other_villager_closer",
        "tree0_remaining_norm",
    )


def test_core_values_reproduce_the_m1_fields():
    observation = build_team_observation(_snapshot(), SCALES)

    # villager 0 at (128, 256), its target tree 0 at (256, 256), map 512 m.
    assert observation[0][0] == pytest.approx(-0.5)
    assert observation[0][1] == pytest.approx(0.0)
    assert observation[0][2] == pytest.approx(0.0)
    assert observation[0][3] == pytest.approx(0.0)
    assert observation[0][4] == pytest.approx(128.0 / 512.0)
    assert observation[0][5] == pytest.approx(0.0)
    assert observation[0][6] == pytest.approx(0.3)
    assert observation[0][7] == pytest.approx(-0.75)
    assert observation[0][8] == pytest.approx(0.25)
    assert observation[0][9] == pytest.approx(0.0)
    # villager 1 carries a full load and its cycle is running.
    assert observation[1][5] == pytest.approx(1.0)
    assert observation[1][9] == pytest.approx(1.0)


def test_agent_id_is_normalized_over_the_team():
    observation = build_team_observation(_snapshot(), SCALES)

    assert observation[0][10] == pytest.approx(0.0)
    assert observation[1][10] == pytest.approx(1.0)


def test_single_villager_gets_agent_id_zero():
    snapshot = TeamSnapshot(
        villager_xz=((128.0, 256.0),),
        resource_xz=((256.0, 256.0),),
        resource_remaining=(200.0,),
        carried=(0.0,),
        target_index=(0,),
        gather_cycle_active=(False,),
        dropsite_xz=(64.0, 320.0),
        stock=0.0,
    )

    observation = build_team_observation(snapshot, SCALES)

    assert observation.shape == (1, 16)
    assert observation[0][10] == pytest.approx(0.0)


def test_relational_block_flags_the_villager_that_is_not_closest():
    observation = build_team_observation(_snapshot(), SCALES)

    # Tree 0 sits at (256, 256): villager 0 is 128 m away, villager 1 is 128 m away,
    # so the tie resolves to neither being strictly closer than the other.
    assert observation[0][14] == pytest.approx(0.0)
    assert observation[1][14] == pytest.approx(0.0)
    # Tree 1 sits at (400, 256): villager 1 is 16 m away, villager 0 is 272 m away.
    assert observation[0][19] == pytest.approx(1.0)
    assert observation[1][19] == pytest.approx(0.0)


def test_relational_block_is_ordered_by_roster_not_by_distance():
    observation = build_team_observation(_snapshot(), SCALES)

    # Villager 1 is nearest tree 1, but tree 0 still occupies the first block.
    assert observation[1][13] == pytest.approx(128.0 / 512.0)
    assert observation[1][18] == pytest.approx(16.0 / 512.0)


def test_remaining_is_normalized_and_clipped():
    observation = build_team_observation(_snapshot(), SCALES)

    assert observation[0][15] == pytest.approx(1.0)
    assert observation[0][20] == pytest.approx(0.5)


def test_snapshot_rejects_inconsistent_lengths():
    with pytest.raises(ValueError, match="per-villager fields must have equal length"):
        TeamSnapshot(
            villager_xz=((0.0, 0.0), (1.0, 1.0)),
            resource_xz=((2.0, 2.0),),
            resource_remaining=(10.0,),
            carried=(0.0,),
            target_index=(0, 0),
            gather_cycle_active=(False, False),
            dropsite_xz=(0.0, 0.0),
            stock=0.0,
        )


def test_snapshot_rejects_a_target_index_outside_the_roster():
    with pytest.raises(ValueError, match="target_index out of range"):
        TeamSnapshot(
            villager_xz=((0.0, 0.0),),
            resource_xz=((2.0, 2.0),),
            resource_remaining=(10.0,),
            carried=(0.0,),
            target_index=(3,),
            gather_cycle_active=(False,),
            dropsite_xz=(0.0, 0.0),
            stock=0.0,
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest rl/tests/gather/test_observation.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rl.gather.observation'`

- [ ] **Step 3: Write minimal implementation**

```python
# rl/gather/observation.py
"""Team observation assembly for multi-villager gather environments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import (
    GATHER_LIFECYCLE_OBSERVATION_LABELS,
    distance,
    normalize_coord,
    normalize_non_negative,
)


# Indices 0-9 of every slice are M1's observation, in M1's order, so an M1
# checkpoint can initialize the shared network's input prefix unchanged.
TEAM_CORE_LABELS = GATHER_LIFECYCLE_OBSERVATION_LABELS

RELATIONAL_FIELDS = (
    "dx_norm",
    "dz_norm",
    "dist_norm",
    "other_villager_closer",
    "remaining_norm",
)


@dataclass(frozen=True, slots=True)
class TeamObservationScales:
    """Normalization constants shared by every slice."""

    map_size_m: float
    carried_resource_scale: float
    stock_scale: float
    resource_amount_scale: float


@dataclass(frozen=True, slots=True)
class TeamSnapshot:
    """Everything one observation needs, already read from the engine."""

    villager_xz: tuple[tuple[float, float], ...]
    resource_xz: tuple[tuple[float, float], ...]
    resource_remaining: tuple[float, ...]
    carried: tuple[float, ...]
    target_index: tuple[int, ...]
    gather_cycle_active: tuple[bool, ...]
    dropsite_xz: tuple[float, float]
    stock: float

    def __post_init__(self) -> None:
        per_villager = {
            len(self.villager_xz),
            len(self.carried),
            len(self.target_index),
            len(self.gather_cycle_active),
        }
        if len(per_villager) != 1:
            raise ValueError("per-villager fields must have equal length")
        if len(self.resource_xz) != len(self.resource_remaining):
            raise ValueError("per-resource fields must have equal length")
        if not self.resource_xz:
            raise ValueError("a team snapshot needs at least one resource")
        if any(
            index < 0 or index >= len(self.resource_xz) for index in self.target_index
        ):
            raise ValueError("target_index out of range")


def team_observation_labels(
    villager_count: int,
    resource_count: int,
) -> tuple[str, ...]:
    """Names for one villager's slice, in slice order."""

    del villager_count
    relational = tuple(
        f"tree{index}_{field}"
        for index in range(resource_count)
        for field in RELATIONAL_FIELDS
    )
    return (*TEAM_CORE_LABELS, "agent_id_norm", *relational)


def build_team_observation(
    snapshot: TeamSnapshot,
    scales: TeamObservationScales,
) -> np.ndarray:
    """Return one `(villager_count, slice_dim)` float32 observation."""

    villager_count = len(snapshot.villager_xz)
    stock_norm = normalize_non_negative(snapshot.stock, scales.stock_scale)
    dropsite = (
        normalize_coord(snapshot.dropsite_xz[0], scales.map_size_m),
        normalize_coord(snapshot.dropsite_xz[1], scales.map_size_m),
    )
    distances = [
        [distance(villager, resource) for resource in snapshot.resource_xz]
        for villager in snapshot.villager_xz
    ]

    slices = []
    for villager in range(villager_count):
        villager_xz = snapshot.villager_xz[villager]
        target = snapshot.resource_xz[snapshot.target_index[villager]]
        values = [
            normalize_coord(villager_xz[0], scales.map_size_m),
            normalize_coord(villager_xz[1], scales.map_size_m),
            normalize_coord(target[0], scales.map_size_m),
            normalize_coord(target[1], scales.map_size_m),
            distance(villager_xz, target) / scales.map_size_m,
            normalize_non_negative(
                snapshot.carried[villager],
                scales.carried_resource_scale,
            ),
            stock_norm,
            dropsite[0],
            dropsite[1],
            float(snapshot.gather_cycle_active[villager]),
            0.0 if villager_count == 1 else villager / (villager_count - 1),
        ]
        for resource, resource_xz in enumerate(snapshot.resource_xz):
            own_distance = distances[villager][resource]
            others_closer = any(
                distances[other][resource] < own_distance
                for other in range(villager_count)
                if other != villager
            )
            values.extend(
                [
                    (resource_xz[0] - villager_xz[0]) / scales.map_size_m,
                    (resource_xz[1] - villager_xz[1]) / scales.map_size_m,
                    own_distance / scales.map_size_m,
                    float(others_closer),
                    normalize_non_negative(
                        snapshot.resource_remaining[resource],
                        scales.resource_amount_scale,
                    ),
                ]
            )
        slices.append(values)
    return np.array(slices, dtype=np.float32)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest rl/tests/gather/test_observation.py -v`
Expected: PASS, 10 tests

- [ ] **Step 5: Verify the whole suite, then commit**

Run: `.venv/bin/python -m pytest -q && .venv/bin/python -m ruff check rl`
Expected: all pass

```bash
git add rl/gather/observation.py rl/tests/gather/test_observation.py
git commit -m "feat(rl): add team observation assembly with relational features"
```

---

### Task 4: Team reward composition

The M1 ablation failed because shaping was easier to harvest than the task. With 4 villagers, summing per-villager shaping would weight it 4× against a team Δstock that does not scale the same way, so shaping is averaged.

**Files:**
- Create: `rl/gather/reward.py`
- Test: `rl/tests/gather/test_reward.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `TeamRewardScales` (frozen dataclass), `VillagerRewardInputs` (frozen dataclass), `TeamRewardTerms` (frozen dataclass with `.total()`), `compose_team_reward(stock_delta, villagers, scales) -> TeamRewardTerms`.

- [ ] **Step 1: Write the failing test**

```python
# rl/tests/gather/test_reward.py
import pytest

from rl.gather.reward import (
    TeamRewardScales,
    TeamRewardTerms,
    VillagerRewardInputs,
    compose_team_reward,
)


SCALES = TeamRewardScales(
    distance_shaping_scale=0.02,
    carried_resource_delta_reward_scale=0.2,
    click_gather_cycle_penalty=1.0,
)


def _villager(closed=0.0, carried_delta=0.0, interrupted=False):
    return VillagerRewardInputs(
        distance_closed_m=closed,
        carried_resource_delta=carried_delta,
        interrupted_gather_cycle=interrupted,
    )


def test_stock_delta_passes_through_unscaled():
    terms = compose_team_reward(20.0, (_villager(), _villager()), SCALES)

    assert terms.stock_delta == pytest.approx(20.0)
    assert terms.total() == pytest.approx(20.0)


def test_shaping_is_averaged_over_villagers_not_summed():
    villagers = (_villager(closed=100.0), _villager(closed=100.0))

    terms = compose_team_reward(0.0, villagers, SCALES)

    # 0.02 * 100 = 2.0 each; averaged, not summed.
    assert terms.distance_shaping == pytest.approx(2.0)
    assert terms.total() == pytest.approx(2.0)


def test_carried_delta_is_averaged_and_never_negative():
    villagers = (_villager(carried_delta=20.0), _villager(carried_delta=-5.0))

    terms = compose_team_reward(0.0, villagers, SCALES)

    # 0.2 * 20 = 4.0 for the first, 0.0 for the second, averaged over two.
    assert terms.carried_delta == pytest.approx(2.0)


def test_interruption_penalty_is_averaged_and_subtracted():
    villagers = (_villager(interrupted=True), _villager(), _villager(), _villager())

    terms = compose_team_reward(0.0, villagers, SCALES)

    assert terms.click_penalty == pytest.approx(0.25)
    assert terms.total() == pytest.approx(-0.25)


def test_terms_sum_into_the_total():
    villagers = (_villager(closed=50.0, carried_delta=20.0, interrupted=True),)

    terms = compose_team_reward(20.0, villagers, SCALES)

    assert isinstance(terms, TeamRewardTerms)
    assert terms.total() == pytest.approx(20.0 + 1.0 + 4.0 - 1.0)


def test_composition_requires_at_least_one_villager():
    with pytest.raises(ValueError, match="at least one villager"):
        compose_team_reward(0.0, (), SCALES)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest rl/tests/gather/test_reward.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rl.gather.reward'`

- [ ] **Step 3: Write minimal implementation**

```python
# rl/gather/reward.py
"""Reward term composition for team gather environments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TeamRewardScales:
    """Config-owned weights for every shaping term."""

    distance_shaping_scale: float
    carried_resource_delta_reward_scale: float
    click_gather_cycle_penalty: float


@dataclass(frozen=True, slots=True)
class VillagerRewardInputs:
    """One villager's contribution to this step's shaping."""

    distance_closed_m: float = 0.0
    carried_resource_delta: float = 0.0
    interrupted_gather_cycle: bool = False


@dataclass(frozen=True, slots=True)
class TeamRewardTerms:
    """The reward broken into terms, for logging and for `--verbose`."""

    stock_delta: float
    distance_shaping: float
    carried_delta: float
    click_penalty: float

    def total(self) -> float:
        """Sum the terms the environment actually returns."""

        return (
            self.stock_delta
            + self.distance_shaping
            + self.carried_delta
            - self.click_penalty
        )


def compose_team_reward(
    stock_delta: float,
    villagers: tuple[VillagerRewardInputs, ...],
    scales: TeamRewardScales,
) -> TeamRewardTerms:
    """Combine the team stock signal with averaged per-villager shaping.

    Shaping is averaged rather than summed: the team stock delta does not grow
    with the squad size, so summing would let shaping dominate as villagers are
    added. The M1 ablation showed a policy farming shaping instead of gathering
    when shaping outweighed the real signal.
    """

    if not villagers:
        raise ValueError("team reward needs at least one villager")
    count = len(villagers)
    distance_shaping = (
        sum(villager.distance_closed_m for villager in villagers)
        * scales.distance_shaping_scale
        / count
    )
    carried_delta = (
        sum(max(0.0, villager.carried_resource_delta) for villager in villagers)
        * scales.carried_resource_delta_reward_scale
        / count
    )
    click_penalty = (
        sum(1.0 for villager in villagers if villager.interrupted_gather_cycle)
        * scales.click_gather_cycle_penalty
        / count
    )
    return TeamRewardTerms(
        stock_delta=float(stock_delta),
        distance_shaping=float(distance_shaping),
        carried_delta=float(carried_delta),
        click_penalty=float(click_penalty),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest rl/tests/gather/test_reward.py -v`
Expected: PASS, 6 tests

- [ ] **Step 5: Verify the whole suite, then commit**

Run: `.venv/bin/python -m pytest -q && .venv/bin/python -m ruff check rl`
Expected: all pass

```bash
git add rl/gather/reward.py rl/tests/gather/test_reward.py
git commit -m "feat(rl): add averaged team reward composition"
```

---

### Task 5: Batched engine snapshot

Today `_resource_snapshot` issues one `game.evaluate` per step for one villager. Four villagers naively means four round-trips per step on a milestone that already costs ~18 minutes to train.

**Files:**
- Modify: `rl/gather/env.py` (add beside `resource_snapshot_expression`, around line 72)
- Test: `rl/tests/gather/test_env.py` (append)

**Interfaces:**
- Consumes: nothing.
- Produces: `team_snapshot_expression(player_id, villager_ids, resource_ids, resource) -> str` and `parse_team_snapshot(value, resource, villager_ids, resource_ids) -> tuple[float, tuple[float, ...], tuple[float, ...]]` returning `(stock, carried_per_villager, remaining_per_resource)`.

- [ ] **Step 1: Write the failing test**

```python
# append to rl/tests/gather/test_env.py
def test_team_snapshot_expression_names_every_entity_once():
    from rl.gather.env import team_snapshot_expression

    expression = team_snapshot_expression(1, (11, 12), (21, 22), "wood")

    assert "GetResourceCounts" in expression
    assert "[11,12]" in expression.replace(" ", "")
    assert "[21,22]" in expression.replace(" ", "")
    assert '"wood"' in expression


def test_parse_team_snapshot_orders_values_by_entity_id():
    from rl.gather.env import parse_team_snapshot

    payload = {
        "stock": {"wood": 300.0},
        "carried": {"12": 20.0, "11": 0.0},
        "remaining": {"22": 50.0, "21": 200.0},
    }

    stock, carried, remaining = parse_team_snapshot(payload, "wood", (11, 12), (21, 22))

    assert stock == 300.0
    assert carried == (0.0, 20.0)
    assert remaining == (200.0, 50.0)


def test_parse_team_snapshot_defaults_missing_entities_to_zero():
    from rl.gather.env import parse_team_snapshot

    payload = {"stock": {"wood": 10.0}, "carried": {}, "remaining": {}}

    stock, carried, remaining = parse_team_snapshot(payload, "wood", (11,), (21,))

    assert stock == 10.0
    assert carried == (0.0,)
    assert remaining == (0.0,)


def test_parse_team_snapshot_rejects_a_malformed_payload():
    from rl.gather.env import parse_team_snapshot
    import pytest

    with pytest.raises(TypeError, match="team snapshot"):
        parse_team_snapshot(["not", "a", "mapping"], "wood", (11,), (21,))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest rl/tests/gather/test_env.py -k team_snapshot -v`
Expected: FAIL with `ImportError: cannot import name 'team_snapshot_expression'`

- [ ] **Step 3: Write minimal implementation**

```python
# rl/gather/env.py, after resource_snapshot_expression
def team_snapshot_expression(
    player_id: int,
    villager_ids: tuple[int, ...],
    resource_ids: tuple[int, ...],
    resource: str,
) -> str:
    """One JS round-trip covering stock, every carried load, and every tree."""

    resource_literal = json.dumps(resource)
    villagers_literal = json.dumps([int(entity) for entity in villager_ids])
    resources_literal = json.dumps([int(entity) for entity in resource_ids])
    return (
        "(() => { "
        "const cmpPlayerManager = Engine.QueryInterface(SYSTEM_ENTITY, "
        "IID_PlayerManager); "
        f"const playerEntity = cmpPlayerManager.GetPlayerByID({int(player_id)}); "
        "const cmpPlayer = Engine.QueryInterface(playerEntity, IID_Player); "
        "const stock = cmpPlayer.GetResourceCounts(); "
        "const carried = {}; "
        f"for (const id of {villagers_literal}) {{ "
        "const cmpGatherer = Engine.QueryInterface(id, IID_ResourceGatherer); "
        "let total = 0; "
        "if (cmpGatherer) for (const item of cmpGatherer.GetCarryingStatus()) { "
        "const itemType = item.type || item.generic || item.resource || ''; "
        f"if (itemType === {resource_literal} || "
        f"itemType.split('.')[0] === {resource_literal}) "
        "total += +(item.amount || 0); "
        "} "
        "carried[id] = total; "
        "} "
        "const remaining = {}; "
        f"for (const id of {resources_literal}) {{ "
        "const cmpSupply = Engine.QueryInterface(id, IID_ResourceSupply); "
        "remaining[id] = cmpSupply ? +cmpSupply.GetCurrentAmount() : 0; "
        "} "
        "return { stock, carried, remaining }; "
        "})()"
    )


def _entity_values(
    values: object,
    entity_ids: tuple[int, ...],
    label: str,
) -> tuple[float, ...]:
    if not isinstance(values, Mapping):
        raise TypeError(f"team snapshot {label} must be a mapping")
    ordered = []
    for entity in entity_ids:
        value = values.get(str(int(entity)), values.get(int(entity), 0.0))
        ordered.append(float(value))
    return tuple(ordered)


def parse_team_snapshot(
    value: object,
    resource: str,
    villager_ids: tuple[int, ...],
    resource_ids: tuple[int, ...],
) -> tuple[float, tuple[float, ...], tuple[float, ...]]:
    """Split one batched evaluation into stock, carried loads, and supplies."""

    if not isinstance(value, Mapping):
        raise TypeError("team snapshot must be a mapping")
    for key in ("stock", "carried", "remaining"):
        if key not in value:
            raise TypeError(f"team snapshot is missing '{key}'")
    return (
        _extract_stock(value["stock"], resource),
        _entity_values(value["carried"], villager_ids, "carried"),
        _entity_values(value["remaining"], resource_ids, "remaining"),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest rl/tests/gather/test_env.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add rl/gather/env.py rl/tests/gather/test_env.py
git commit -m "feat(rl): batch team state into one engine evaluation"
```

---

### Task 6: Team environment

A separate `gym.Env` that composes the modules from Tasks 1–5. It is a sibling of `ZeroADGatherEnv`, not a rewrite of it: M0/M1 keep their proven code path untouched, and this file stays small enough to reason about.

**Files:**
- Create: `rl/gather/team_env.py`
- Test: `rl/tests/gather/test_team_env.py`

**Interfaces:**
- Consumes: `build_roster`, `Roster`, `RosterError`; `TeamSnapshot`, `TeamObservationScales`, `build_team_observation`, `team_observation_labels`; `TeamRewardScales`, `VillagerRewardInputs`, `compose_team_reward`; `team_snapshot_expression`, `parse_team_snapshot`; `xz`, `distance`, `is_reached`, `denormalize_action`, `nearest_index`.
- Produces: `ZeroADTeamGatherEnv(scenario_config, *, villager_count=4, resource_count=4, ...)` with `observation_space = Box(-1, 1, (villager_count, 11 + 5 * resource_count))` and `action_space = Box(-1, 1, (3 * villager_count,))`.

- [ ] **Step 1: Write the failing test**

```python
# rl/tests/gather/test_team_env.py
from dataclasses import dataclass

import numpy as np
import pytest

from rl.gather.team_env import ZeroADTeamGatherEnv


@dataclass(frozen=True)
class FakeUnit:
    coordinates: tuple[float, float]
    entity_id: int

    def position(self):
        return self.coordinates

    def id(self):
        return self.entity_id


class FakeState:
    def __init__(self, villagers, resources, dropsite):
        self.villagers = villagers
        self.resources = resources
        self.dropsite = dropsite

    def units(self, *, owner, entity_type):
        return {
            (1, "polites"): list(self.villagers),
            (0, "tree"): list(self.resources),
            (1, "storehouse"): [self.dropsite],
        }[(owner, entity_type)]


class FakeActions:
    def __init__(self):
        self.calls = []

    def walk(self, units, x, z):
        self.calls.append(("walk", units[0].id(), x, z))
        return ("walk", units[0].id())

    def gather(self, units, resource):
        self.calls.append(("gather", units[0].id(), resource.id()))
        return ("gather", units[0].id())

    def returnresource(self, units, dropsite):
        self.calls.append(("returnresource", units[0].id(), dropsite.id()))
        return ("returnresource", units[0].id())


class FakeGame:
    def __init__(self, state, snapshots):
        self.current_state = state
        self.snapshots = list(snapshots)
        self.evaluate_calls = []
        self.step_calls = []

    def reset(self, scenario_config, *, save_replay=False):
        return self.current_state

    def step(self, commands=None):
        self.step_calls.append(commands)
        return self.current_state

    def step_many(self, commands=None, *, turns):
        self.step_calls.append((commands, turns))
        return self.current_state

    def evaluate(self, expression):
        self.evaluate_calls.append(expression)
        return self.snapshots.pop(0) if self.snapshots else self.snapshots_default()

    def snapshots_default(self):
        return {"stock": {"wood": 0.0}, "carried": {}, "remaining": {}}


def _backend(stock=300.0):
    villagers = [FakeUnit((100.0, 100.0), 11), FakeUnit((200.0, 100.0), 12)]
    resources = [FakeUnit((300.0, 100.0), 21), FakeUnit((400.0, 100.0), 22)]
    state = FakeState(villagers, resources, FakeUnit((50.0, 150.0), 31))
    snapshot = {
        "stock": {"wood": stock},
        "carried": {"11": 0.0, "12": 0.0},
        "remaining": {"21": 200.0, "22": 200.0},
    }
    return FakeGame(state, [snapshot] * 12), FakeActions()


def _env(**overrides):
    game, actions = _backend()
    parameters = {
        "villager_count": 2,
        "resource_count": 2,
        "map_size_m": 512.0,
        "horizon": 3,
        "sim_steps_per_action": 2,
        "gather_command_distance": 12.0,
        "stock_success_threshold": 40.0,
        "game": game,
        "actions": actions,
    }
    parameters.update(overrides)
    return ZeroADTeamGatherEnv("scenario", **parameters), game, actions


def test_spaces_follow_the_configured_counts():
    env, _game, _actions = _env()

    assert env.observation_space.shape == (2, 21)
    assert env.action_space.shape == (6,)


def test_reset_returns_one_slice_per_villager():
    env, _game, _actions = _env()

    observation, info = env.reset()

    assert observation.shape == (2, 21)
    assert info["resource_stock"] == pytest.approx(300.0)


def test_each_action_slot_drives_its_own_villager():
    env, _game, actions = _env()
    env.reset()
    # Villager 0 -> tree 21 at (300, 100); villager 1 -> dropsite at (50, 150).
    action = np.array([0.171875, -0.609375, 1.0, -0.804688, -0.414062, 1.0], dtype=np.float32)

    env.step(action)

    assert ("gather", 11, 21) in actions.calls
    assert ("returnresource", 12, 31) in actions.calls


def test_no_click_emits_no_command_for_that_villager():
    env, _game, actions = _env()
    env.reset()
    action = np.array([0.171875, -0.609375, 1.0, 0.0, 0.0, -1.0], dtype=np.float32)

    env.step(action)

    driven = {call[1] for call in actions.calls}
    assert 11 in driven
    assert 12 not in driven


def test_one_batched_evaluation_per_step():
    env, game, _actions = _env()
    env.reset()
    before = len(game.evaluate_calls)

    env.step(np.zeros(6, dtype=np.float32))

    assert len(game.evaluate_calls) - before == 1


def test_episode_terminates_when_collective_stock_reaches_the_threshold():
    game, actions = _backend()
    game.snapshots = [
        {"stock": {"wood": 300.0}, "carried": {}, "remaining": {"21": 200.0, "22": 200.0}},
        {"stock": {"wood": 345.0}, "carried": {}, "remaining": {"21": 200.0, "22": 200.0}},
    ]
    env = ZeroADTeamGatherEnv(
        "scenario",
        villager_count=2,
        resource_count=2,
        map_size_m=512.0,
        horizon=5,
        sim_steps_per_action=1,
        stock_success_threshold=40.0,
        game=game,
        actions=actions,
    )
    env.reset()

    _observation, reward, terminated, truncated, info = env.step(np.zeros(6, dtype=np.float32))

    assert terminated is True
    assert truncated is False
    assert reward == pytest.approx(45.0)
    assert info["episode_resource_stock_delta"] == pytest.approx(45.0)


def test_horizon_truncates_without_termination():
    env, _game, _actions = _env()
    env.reset()

    for _ in range(2):
        _, _, terminated, truncated, _ = env.step(np.zeros(6, dtype=np.float32))
        assert terminated is False
        assert truncated is False

    _, _, terminated, truncated, _ = env.step(np.zeros(6, dtype=np.float32))
    assert terminated is False
    assert truncated is True


def test_unexpected_unit_count_fails_loudly():
    from rl.gather.roster import RosterError

    game, actions = _backend()
    env = ZeroADTeamGatherEnv(
        "scenario",
        villager_count=4,
        resource_count=2,
        map_size_m=512.0,
        horizon=3,
        sim_steps_per_action=1,
        game=game,
        actions=actions,
    )

    with pytest.raises(RosterError, match="expected 4 villagers, found 2"):
        env.reset()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest rl/tests/gather/test_team_env.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rl.gather.team_env'`

- [ ] **Step 3: Write the implementation**

```python
# rl/gather/team_env.py
"""Gymnasium environment for a team of villagers gathering one resource."""

from __future__ import annotations

from typing import Any, Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .core import denormalize_action, distance, is_reached, nearest_index, xz
from .env import (
    DROPSITE_TYPE,
    RESOURCE_TYPE,
    VILLAGER_TYPE,
    _resolve_backend,
    parse_team_snapshot,
    team_snapshot_expression,
)
from .observation import (
    TeamObservationScales,
    TeamSnapshot,
    build_team_observation,
    team_observation_labels,
)
from .reward import TeamRewardScales, VillagerRewardInputs, compose_team_reward
from .roster import Roster, build_roster


class ZeroADTeamGatherEnv(gym.Env):
    """Drive `villager_count` villagers with one action vector."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario_config,
        *,
        villager_count: int = 4,
        resource_count: int = 4,
        uri: str = "http://localhost:6000",
        map_size_m: float = 512.0,
        horizon: int = 120,
        sim_steps_per_action: int = 80,
        gather_command_distance: float = 12.0,
        click_action_threshold: float = 0.0,
        stock_resource: str = "wood",
        stock_player: int = 1,
        stock_success_threshold: float = 80.0,
        carried_resource_observation_scale: float = 20.0,
        stock_observation_scale: float = 1000.0,
        resource_amount_scale: float = 200.0,
        distance_shaping_scale: float = 0.02,
        carried_resource_delta_reward_scale: float = 0.2,
        click_gather_cycle_penalty: float = 1.0,
        save_replay: bool = False,
        game: Any = None,
        actions: Any = None,
        backend_factory: Callable[[str], tuple[Any, Any]] | None = None,
    ) -> None:
        super().__init__()
        if villager_count <= 0 or resource_count <= 0:
            raise ValueError("villager_count and resource_count must be positive")
        self.game, self.actions = _resolve_backend(uri, game, actions, backend_factory)
        self.scenario_config = scenario_config
        self.save_replay = save_replay
        self.villager_count = villager_count
        self.resource_count = resource_count
        self.map_size_m = map_size_m
        self.horizon = horizon
        self.sim_steps_per_action = sim_steps_per_action
        self.gather_command_distance = gather_command_distance
        self.click_action_threshold = click_action_threshold
        self.stock_resource = stock_resource
        self.stock_player = stock_player
        self.stock_success_threshold = stock_success_threshold
        self.observation_scales = TeamObservationScales(
            map_size_m=map_size_m,
            carried_resource_scale=carried_resource_observation_scale,
            stock_scale=stock_observation_scale,
            resource_amount_scale=resource_amount_scale,
        )
        self.reward_scales = TeamRewardScales(
            distance_shaping_scale=distance_shaping_scale,
            carried_resource_delta_reward_scale=carried_resource_delta_reward_scale,
            click_gather_cycle_penalty=click_gather_cycle_penalty,
        )
        self.observation_labels = team_observation_labels(
            villager_count,
            resource_count,
        )
        self.observation_space = spaces.Box(
            -1.0,
            1.0,
            shape=(villager_count, len(self.observation_labels)),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            -1.0,
            1.0,
            shape=(3 * villager_count,),
            dtype=np.float32,
        )
        self._roster: Roster | None = None
        self._step_count = 0
        self._initial_stock = 0.0
        self._previous_stock = 0.0
        self._previous_carried: tuple[float, ...] = ()
        self._previous_distance: tuple[float, ...] = ()
        self._target_index: tuple[int, ...] = ()
        self._cycle_active: tuple[bool, ...] = ()

    def _roster_for(self, state: Any) -> Roster:
        return build_roster(
            state,
            villager_count=self.villager_count,
            resource_count=self.resource_count,
            villager_type=VILLAGER_TYPE,
            resource_type=RESOURCE_TYPE,
            dropsite_type=DROPSITE_TYPE,
        )

    def _read_engine(self, roster: Roster) -> tuple[float, tuple[float, ...], tuple[float, ...]]:
        villager_ids = tuple(int(unit.id()) for unit in roster.villagers)
        resource_ids = tuple(int(unit.id()) for unit in roster.resources)
        payload = self.game.evaluate(
            team_snapshot_expression(
                self.stock_player,
                villager_ids,
                resource_ids,
                self.stock_resource,
            )
        )
        return parse_team_snapshot(
            payload,
            self.stock_resource,
            villager_ids,
            resource_ids,
        )

    def _snapshot(
        self,
        roster: Roster,
        stock: float,
        carried: tuple[float, ...],
        remaining: tuple[float, ...],
    ) -> TeamSnapshot:
        if roster.dropsite is None:
            raise RuntimeError("the team environment requires a storehouse")
        return TeamSnapshot(
            villager_xz=tuple(xz(unit.position()) for unit in roster.villagers),
            resource_xz=tuple(xz(unit.position()) for unit in roster.resources),
            resource_remaining=remaining,
            carried=carried,
            target_index=self._target_index,
            gather_cycle_active=self._cycle_active,
            dropsite_xz=xz(roster.dropsite.position()),
            stock=stock,
        )

    def reset(self, *, seed=None, options=None):
        del options
        super().reset(seed=seed)
        state = self.game.reset(self.scenario_config, save_replay=self.save_replay)
        if state is None:
            state = self.game.step()
        roster = self._roster_for(state)
        self._roster = roster
        villager_xz = tuple(xz(unit.position()) for unit in roster.villagers)
        resource_xz = tuple(xz(unit.position()) for unit in roster.resources)
        self._target_index = tuple(
            nearest_index(villager, resource_xz) for villager in villager_xz
        )
        self._cycle_active = tuple(False for _ in roster.villagers)
        stock, carried, remaining = self._read_engine(roster)
        self._initial_stock = stock
        self._previous_stock = stock
        self._previous_carried = carried
        self._previous_distance = tuple(
            distance(villager_xz[index], resource_xz[self._target_index[index]])
            for index in range(self.villager_count)
        )
        self._step_count = 0
        observation = build_team_observation(
            self._snapshot(roster, stock, carried, remaining),
            self.observation_scales,
        )
        return observation, {
            "resource_stock": stock,
            "episode_resource_stock_delta": 0.0,
        }

    def _commands(self, action: np.ndarray, roster: Roster) -> tuple[list[Any], list[bool]]:
        commands: list[Any] = []
        interrupted: list[bool] = []
        resource_xz = [xz(unit.position()) for unit in roster.resources]
        dropsite_xz = xz(roster.dropsite.position())
        targets = list(self._target_index)
        cycles = list(self._cycle_active)
        for villager in range(self.villager_count):
            base = 3 * villager
            x, z = denormalize_action(action[base : base + 2], self.map_size_m)
            clicked = float(action[base + 2]) > self.click_action_threshold
            interrupted.append(False)
            if not clicked:
                continue
            unit = roster.villagers[villager]
            hit_resource = nearest_index((x, z), resource_xz)
            if is_reached(
                distance((x, z), resource_xz[hit_resource]),
                self.gather_command_distance,
            ):
                if cycles[villager]:
                    interrupted[villager] = True
                targets[villager] = hit_resource
                cycles[villager] = True
                commands.append(
                    self.actions.gather([unit], roster.resources[hit_resource])
                )
            elif is_reached(
                distance((x, z), dropsite_xz),
                self.gather_command_distance,
            ):
                commands.append(self.actions.returnresource([unit], roster.dropsite))
            else:
                if cycles[villager]:
                    interrupted[villager] = True
                cycles[villager] = False
                commands.append(self.actions.walk([unit], x, z))
        self._target_index = tuple(targets)
        self._cycle_active = tuple(cycles)
        return commands, interrupted

    def step(self, action):
        if self._roster is None:
            raise RuntimeError("the environment must be reset before stepping")
        values = np.asarray(action, dtype=np.float32).reshape(-1)
        if values.size != 3 * self.villager_count:
            raise ValueError("action must hold three values per villager")
        roster = self._roster
        commands, interrupted = self._commands(values, roster)
        payload = commands or None
        step_many = getattr(self.game, "step_many", None)
        if callable(step_many):
            state = step_many(payload, turns=self.sim_steps_per_action)
        else:
            state = self.game.step(payload)
            for _ in range(self.sim_steps_per_action - 1):
                state = self.game.step()

        roster = self._roster_for(state)
        self._roster = roster
        stock, carried, remaining = self._read_engine(roster)
        villager_xz = tuple(xz(unit.position()) for unit in roster.villagers)
        resource_xz = tuple(xz(unit.position()) for unit in roster.resources)
        dropsite_xz = xz(roster.dropsite.position())

        villagers = []
        current_distance = []
        for index in range(self.villager_count):
            carrying = self._previous_carried[index] > 0.0
            target_xz = dropsite_xz if carrying else resource_xz[self._target_index[index]]
            now = distance(villager_xz[index], target_xz)
            current_distance.append(now)
            villagers.append(
                VillagerRewardInputs(
                    distance_closed_m=self._previous_distance[index] - now,
                    carried_resource_delta=carried[index] - self._previous_carried[index],
                    interrupted_gather_cycle=interrupted[index],
                )
            )
        terms = compose_team_reward(
            stock - self._previous_stock,
            tuple(villagers),
            self.reward_scales,
        )

        # A deposit ends whichever cycles were running, mirroring M1's rule that
        # the cycle finishes when wood actually lands in the player's stock.
        if stock > self._previous_stock:
            self._cycle_active = tuple(False for _ in self._cycle_active)
        self._previous_stock = stock
        self._previous_carried = carried
        self._previous_distance = tuple(current_distance)
        self._step_count += 1

        episode_delta = stock - self._initial_stock
        terminated = episode_delta >= self.stock_success_threshold
        truncated = not terminated and self._step_count >= self.horizon
        observation = build_team_observation(
            self._snapshot(roster, stock, carried, remaining),
            self.observation_scales,
        )
        info = {
            "resource_stock": stock,
            "resource_stock_delta": terms.stock_delta,
            "episode_resource_stock_delta": episode_delta,
            "distance_shaping_reward": terms.distance_shaping,
            "carried_resource_delta_reward": terms.carried_delta,
            "click_gather_cycle_penalty": terms.click_penalty,
            "commands": len(commands),
        }
        return observation, terms.total(), terminated, truncated, info

    def close(self):
        close_backend = getattr(self.game, "close", None)
        if callable(close_backend):
            close_backend()
        super().close()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest rl/tests/gather/test_team_env.py -v`
Expected: PASS, 8 tests

- [ ] **Step 5: Verify the whole suite, then commit**

Run: `.venv/bin/python -m pytest -q && .venv/bin/python -m ruff check rl`
Expected: all pass

```bash
git add rl/gather/team_env.py rl/tests/gather/test_team_env.py
git commit -m "feat(rl): add multi-villager team gather environment"
```

---

### Task 7: Team map and scenario

**Files:**
- Create: `maps/random/rl_gather_team.js`, `maps/random/rl_gather_team.json`, `rl/scenarios/team_reset_config.json`
- Test: `rl/tests/test_map_contract.py` (append)

**Interfaces:**
- Consumes: nothing.
- Produces: a deterministic 4-villager / 4-tree scenario reachable as `rl/scenarios/team_reset_config.json`.

- [ ] **Step 1: Write the failing test**

```python
# append to rl/tests/test_map_contract.py
def test_rl_gather_team_places_four_villagers_and_four_trees():
    source = (REPO_ROOT / "maps/random/rl_gather_team.js").read_text(encoding="utf-8")

    assert "export function* generateMap(" in source
    assert source.count('"units/athenai/polites"') == 4
    assert source.count('"gaia/tree/oak"') == 4
    assert '"structures/athenai/rl_storehouse"' in source


def test_rl_gather_team_scenario_is_deterministic():
    import json

    scenario = json.loads(
        (REPO_ROOT / "rl/scenarios/team_reset_config.json").read_text(encoding="utf-8")
    )

    assert scenario["script"] == "rl_gather_team.js"
    assert scenario["settings"]["Seed"] == 0
    assert scenario["settings"]["AISeed"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest rl/tests/test_map_contract.py -v`
Expected: FAIL with `FileNotFoundError` for `maps/random/rl_gather_team.js`

- [ ] **Step 3: Write the map, its metadata, and the scenario**

```javascript
// maps/random/rl_gather_team.js
Engine.LoadLibrary("rmgen");
Engine.LoadLibrary("rmgen-common");

// Mapa DETERMINISTA para M2: 4 aldeanos + 4 arboles + 1 deposito, sin RNG.
export function* generateMap()
{
	const tGrass = "medit_grass_field";
	globalThis.g_Map = new RandomMap(0, tGrass);

	const c = g_Map.getCenter();

	// Cuatro aldeanos separados, a la izquierda del centro.
	g_Map.placeEntityPassable("units/athenai/polites", 1, new Vector2D(c.x - 20, c.y - 12), 0);
	g_Map.placeEntityPassable("units/athenai/polites", 1, new Vector2D(c.x - 20, c.y - 4), 0);
	g_Map.placeEntityPassable("units/athenai/polites", 1, new Vector2D(c.x - 20, c.y + 4), 0);
	g_Map.placeEntityPassable("units/athenai/polites", 1, new Vector2D(c.x - 20, c.y + 12), 0);

	// Un unico deposito: obliga a volver al mismo punto con la carga.
	g_Map.placeEntityPassable("structures/athenai/rl_storehouse", 1, new Vector2D(c.x - 28, c.y + 10), 0);

	// Cuatro arboles separados entre si: hay una asignacion perfecta posible.
	g_Map.placeEntityPassable("gaia/tree/oak", 0, new Vector2D(c.x + 20, c.y - 18), 0);
	g_Map.placeEntityPassable("gaia/tree/oak", 0, new Vector2D(c.x + 20, c.y - 6), 0);
	g_Map.placeEntityPassable("gaia/tree/oak", 0, new Vector2D(c.x + 20, c.y + 6), 0);
	g_Map.placeEntityPassable("gaia/tree/oak", 0, new Vector2D(c.x + 20, c.y + 18), 0);

	yield 100;
	return g_Map;
}
```

```json
{
	"settings": {
		"Name": "RL Gather Team (debug)",
		"Script": "rl_gather_team.js",
		"Description": "Escenario RL de M2: 4 aldeanos + 4 arboles.",
		"CircularMap": false,
		"BaseTerrain": ["medit_grass_field"]
	}
}
```

```json
{
    "mapType": "random",
    "script": "rl_gather_team.js",
    "gameSpeed": 1,
    "settings": {
        "Name": "RL Gather Team",
        "mapName": "RL Gather Team",
        "mapType": "random",
        "Script": "rl_gather_team.js",
        "Seed": 0,
        "AISeed": 0,
        "Size": 128,
        "CheatsEnabled": true,
        "VictoryConditions": [],
        "PlayerData": [
            {
                "Name": "Agent",
                "Civ": "athenai",
                "Color": { "r": 20, "g": 70, "b": 150 },
                "AI": "",
                "Team": -1
            }
        ]
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest rl/tests/test_map_contract.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add maps/random/rl_gather_team.js maps/random/rl_gather_team.json rl/scenarios/team_reset_config.json rl/tests/test_map_contract.py
git commit -m "feat(maps): add deterministic 4-villager gather scenario"
```

---

### Task 8: Team oracle and random baselines

The acceptance gate for the whole plan. The oracle is what proves slot ordering, batching, and reward are wired correctly — without spending a single training step.

**Files:**
- Modify: `rl/agents/baselines.py` (append), `rl/agents/registry.py:75-90`
- Test: `rl/tests/agents/test_baselines.py` (append), `rl/tests/agents/test_registry.py` (append)

**Interfaces:**
- Consumes: `team_observation_labels` layout — slice index 5 is `carried_wood_norm`, 7–8 are the dropsite, and each relational block starts at index 11.
- Produces: `TeamGatherOraclePolicy(villager_count, resource_count)` returning a `(3 * villager_count,)` action, registered as `team_oracle`.

- [ ] **Step 1: Write the failing test**

```python
# append to rl/tests/agents/test_baselines.py
import numpy as np
import pytest

from rl.agents.baselines import TeamGatherOraclePolicy
from rl.gather.observation import (
    TeamObservationScales,
    TeamSnapshot,
    build_team_observation,
)


def _observation(carried):
    snapshot = TeamSnapshot(
        villager_xz=((100.0, 100.0), (200.0, 100.0)),
        resource_xz=((300.0, 100.0), (400.0, 100.0)),
        resource_remaining=(200.0, 200.0),
        carried=carried,
        target_index=(0, 1),
        gather_cycle_active=(False, False),
        dropsite_xz=(50.0, 150.0),
        stock=0.0,
    )
    return build_team_observation(
        snapshot,
        TeamObservationScales(
            map_size_m=512.0,
            carried_resource_scale=20.0,
            stock_scale=1000.0,
            resource_amount_scale=200.0,
        ),
    )


def test_team_oracle_sends_empty_villagers_to_their_nearest_free_tree():
    policy = TeamGatherOraclePolicy(villager_count=2, resource_count=2)

    action = policy.act(_observation((0.0, 0.0)), deterministic=True)

    assert action.shape == (6,)
    # Villager 0 -> tree 0 at (300, 100) -> normalized 2*300/512 - 1.
    assert action[0] == pytest.approx(2.0 * 300.0 / 512.0 - 1.0, abs=1e-3)
    assert action[1] == pytest.approx(2.0 * 100.0 / 512.0 - 1.0, abs=1e-3)
    assert action[2] == pytest.approx(1.0)
    # Villager 1 -> tree 1 at (400, 100).
    assert action[3] == pytest.approx(2.0 * 400.0 / 512.0 - 1.0, abs=1e-3)


def test_team_oracle_sends_loaded_villagers_to_the_dropsite():
    policy = TeamGatherOraclePolicy(villager_count=2, resource_count=2)

    action = policy.act(_observation((20.0, 0.0)), deterministic=True)

    # Villager 0 carries a load, so it targets the dropsite at (50, 150).
    assert action[0] == pytest.approx(2.0 * 50.0 / 512.0 - 1.0, abs=1e-3)
    assert action[1] == pytest.approx(2.0 * 150.0 / 512.0 - 1.0, abs=1e-3)
    assert action[2] == pytest.approx(1.0)


def test_team_oracle_gives_each_villager_a_distinct_tree():
    policy = TeamGatherOraclePolicy(villager_count=2, resource_count=2)

    action = policy.act(_observation((0.0, 0.0)), deterministic=True)

    assert not np.allclose(action[0:2], action[3:5])


def test_team_oracle_rejects_a_mismatched_observation():
    policy = TeamGatherOraclePolicy(villager_count=4, resource_count=4)

    with pytest.raises(ValueError, match="observation shape"):
        policy.act(_observation((0.0, 0.0)), deterministic=True)
```

```python
# append to rl/tests/agents/test_registry.py
def test_registry_builds_the_team_oracle():
    from rl.agents.baselines import TeamGatherOraclePolicy

    class TeamEnv:
        action_space = type("Space", (), {"shape": (6,)})()
        villager_count = 2
        resource_count = 2

    policy = build_policy(AgentSpec("team_oracle"), TeamEnv(), seed=0)

    assert isinstance(policy, TeamGatherOraclePolicy)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest rl/tests/agents -k team -v`
Expected: FAIL with `ImportError: cannot import name 'TeamGatherOraclePolicy'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to rl/agents/baselines.py
class TeamGatherOraclePolicy:
    """Greedy assignment ceiling for the M2 team gather environment.

    Empty villagers claim their nearest unclaimed tree; loaded villagers head
    for the dropsite. It does not learn: it exists to prove the environment's
    slot ordering, batching, and reward are wired correctly before training.
    """

    CORE_WIDTH = 11
    RELATIONAL_WIDTH = 5
    CARRIED_INDEX = 5
    DROPSITE_X_INDEX = 7
    DROPSITE_Z_INDEX = 8

    def __init__(self, villager_count: int = 4, resource_count: int = 4) -> None:
        if villager_count <= 0 or resource_count <= 0:
            raise ValueError("team oracle needs positive counts")
        self._villager_count = villager_count
        self._resource_count = resource_count

    def _tree_offsets(self, slice_values: np.ndarray) -> list[tuple[float, float, float]]:
        offsets = []
        for resource in range(self._resource_count):
            base = self.CORE_WIDTH + resource * self.RELATIONAL_WIDTH
            offsets.append(
                (
                    float(slice_values[base]),
                    float(slice_values[base + 1]),
                    float(slice_values[base + 2]),
                )
            )
        return offsets

    def act(self, observation: np.ndarray, *, deterministic: bool) -> np.ndarray:
        del deterministic
        values = np.asarray(observation, dtype=np.float32)
        expected = (
            self._villager_count,
            self.CORE_WIDTH + self.RELATIONAL_WIDTH * self._resource_count,
        )
        if values.shape != expected:
            raise ValueError(f"observation shape {values.shape} != {expected}")

        action = np.zeros(3 * self._villager_count, dtype=np.float32)
        claimed: set[int] = set()
        for villager in range(self._villager_count):
            slice_values = values[villager]
            base = 3 * villager
            action[base + 2] = 1.0
            if slice_values[self.CARRIED_INDEX] > 0.0:
                action[base] = slice_values[self.DROPSITE_X_INDEX]
                action[base + 1] = slice_values[self.DROPSITE_Z_INDEX]
                continue
            offsets = self._tree_offsets(slice_values)
            order = sorted(
                range(self._resource_count),
                key=lambda resource: offsets[resource][2],
            )
            choice = next(
                (resource for resource in order if resource not in claimed),
                order[0],
            )
            claimed.add(choice)
            dx, dz, _dist = offsets[choice]
            action[base] = np.clip(slice_values[0] + 2.0 * dx, -1.0, 1.0)
            action[base + 1] = np.clip(slice_values[1] + 2.0 * dz, -1.0, 1.0)
        return action
```

```python
# rl/agents/registry.py — add beside _build_oracle
def _build_team_oracle(agent: AgentSpec, env: Any, seed: int) -> Policy:
    del seed
    _require_no_parameters(agent)
    villager_count = int(getattr(env, "villager_count", 4))
    resource_count = int(getattr(env, "resource_count", 4))
    return TeamGatherOraclePolicy(
        villager_count=villager_count,
        resource_count=resource_count,
    )
```

Add `TeamGatherOraclePolicy` to the `from .baselines import ...` line, and register:

```python
        "team_oracle": AgentRegistration(policy_factory=_build_team_oracle),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest rl/tests/agents -v`
Expected: PASS

- [ ] **Step 5: Verify the whole suite, then commit**

Run: `.venv/bin/python -m pytest -q && .venv/bin/python -m ruff check rl`
Expected: all pass

```bash
git add rl/agents/baselines.py rl/agents/registry.py rl/tests/agents/test_baselines.py rl/tests/agents/test_registry.py
git commit -m "feat(rl): add greedy team gather oracle baseline"
```

---

### Task 9: Environment registration and configs

**Files:**
- Modify: `rl/experiments/environments.py:24-57` (parameter allowlist and dispatch)
- Create: `rl/configs/m2_oracle.toml`, `rl/configs/m2_random.toml`
- Test: `rl/tests/experiments/test_environments.py` (append), `rl/tests/experiments/test_tracked_configs.py` (append)

**Interfaces:**
- Consumes: `ZeroADTeamGatherEnv` from Task 6.
- Produces: environment name `zero_ad_team_gather` usable from tracked configs.

- [ ] **Step 1: Write the failing test**

`build_environment` takes no `backend_factory`, so the test asserts the dispatch and
parameter forwarding by monkeypatching the factory rather than opening a socket.

```python
# append to rl/tests/experiments/test_environments.py
def test_team_gather_config_dispatches_to_the_team_factory(monkeypatch):
    from rl.experiments.config import EnvironmentConfig
    from rl.experiments.environments import build_environment

    captured = {}

    def fake_make_team_gather_env(scenario, **parameters):
        captured["scenario"] = scenario
        captured["parameters"] = parameters
        return "team-env"

    monkeypatch.setattr(
        "rl.experiments.environments.make_team_gather_env",
        fake_make_team_gather_env,
    )
    config = EnvironmentConfig(
        name="zero_ad_team_gather",
        parameters={
            "scenario": "rl/scenarios/team_reset_config.json",
            "uri": "http://localhost:6000",
            "villager_count": 4,
            "resource_count": 4,
            "map_size_m": 512.0,
            "horizon": 120,
            "sim_steps_per_action": 80,
            "stock_success_threshold": 80.0,
        },
    )

    env = build_environment(config)

    assert env == "team-env"
    assert captured["scenario"] == "rl/scenarios/team_reset_config.json"
    assert captured["parameters"]["villager_count"] == 4
    assert captured["parameters"]["resource_count"] == 4
    assert "scenario" not in captured["parameters"]


def test_unknown_environment_lists_both_supported_names():
    from rl.experiments.config import EnvironmentConfig
    from rl.experiments.environments import UnknownEnvironmentError, build_environment
    import pytest

    config = EnvironmentConfig(name="typo", parameters={})

    with pytest.raises(UnknownEnvironmentError) as error:
        build_environment(config)

    assert "zero_ad_gather" in str(error.value)
    assert "zero_ad_team_gather" in str(error.value)


def test_team_gather_rejects_an_unknown_parameter():
    from rl.experiments.config import EnvironmentConfig
    from rl.experiments.environments import EnvironmentConfigError, build_environment
    import pytest

    config = EnvironmentConfig(
        name="zero_ad_team_gather",
        parameters={
            "scenario": "rl/scenarios/team_reset_config.json",
            "typo_parameter": 1,
        },
    )

    with pytest.raises(EnvironmentConfigError, match="typo_parameter"):
        build_environment(config)
```

```python
# append to rl/tests/experiments/test_tracked_configs.py
def test_m2_configs_share_the_team_scenario_and_threshold():
    paths = sorted(CONFIG_DIRECTORY.glob("m2_*.toml"))
    configs = {path.name: load_experiment_config(path) for path in paths}

    assert {"m2_oracle.toml", "m2_random.toml"}.issubset(configs)
    for config in configs.values():
        assert config.environment.name == "zero_ad_team_gather"
        assert config.environment.parameters["villager_count"] == 4
        assert config.environment.parameters["resource_count"] == 4
        assert config.environment.parameters["stock_success_threshold"] == 80.0
        assert (
            config.environment.parameters["scenario"]
            == "rl/scenarios/team_reset_config.json"
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest rl/tests/experiments -k team -v` and `.venv/bin/python -m pytest rl/tests/experiments/test_tracked_configs.py -v`
Expected: FAIL with `UnknownEnvironmentError` and a missing-config assertion

- [ ] **Step 3: Write minimal implementation**

Add to `rl/experiments/environments.py`:

```python
_TEAM_GATHER_PARAMETERS = frozenset(
    {
        "scenario",
        "uri",
        "villager_count",
        "resource_count",
        "map_size_m",
        "horizon",
        "sim_steps_per_action",
        "gather_command_distance",
        "click_action_threshold",
        "stock_resource",
        "stock_player",
        "stock_success_threshold",
        "carried_resource_observation_scale",
        "stock_observation_scale",
        "resource_amount_scale",
        "distance_shaping_scale",
        "carried_resource_delta_reward_scale",
        "click_gather_cycle_penalty",
        "save_replay",
    }
)
```

Replace the single-name guard at `rl/experiments/environments.py:359-372` with a dispatch. Note the
existing code raises for *any* name other than `zero_ad_gather`, so the error message must change too:

```python
def build_environment(
    config: EnvironmentConfig,
    *,
    allow_remote: bool = False,
) -> Any:
    """Construct the environment described by a validated config."""

    if config.name not in {"zero_ad_gather", "zero_ad_team_gather"}:
        raise UnknownEnvironmentError(
            f"unknown environment '{config.name}'; available environments: "
            "zero_ad_gather, zero_ad_team_gather",
        )

    parameters = _thaw(config.parameters)
    if config.name == "zero_ad_team_gather":
        _validate_team_gather_parameters(parameters, allow_remote=allow_remote)
    else:
        _validate_gather_parameters(parameters, allow_remote=allow_remote)
    scenario = parameters.pop("scenario", None)
    if not isinstance(scenario, str) or not scenario.strip():
        raise EnvironmentConfigError(
            f"{config.name} requires 'scenario' as a non-empty path",
        )
    if config.name == "zero_ad_team_gather":
        return make_team_gather_env(scenario, **parameters)
    return make_gather_env(scenario, **parameters)
```

`_validate_team_gather_parameters` mirrors `_validate_gather_parameters`: reject any key outside
`_TEAM_GATHER_PARAMETERS`, run the existing URI check when `uri` is present, and require
`villager_count` and `resource_count` to be positive integers no greater than 64 via the existing
`_validate_positive_integer` helper.

Add to `rl/gather/factory.py`:

```python
def make_team_gather_env(
    scenario_path: str | PathLike[str], **env_parameters: Any
) -> ZeroADTeamGatherEnv:
    """Read a repo-owned scenario config and forward team parameters."""

    validated_path = _validated_scenario_path(scenario_path)
    scenario_config = validated_path.read_bytes().decode("utf-8")
    json.loads(scenario_config)
    return ZeroADTeamGatherEnv(scenario_config, **env_parameters)
```

```toml
# rl/configs/m2_oracle.toml
[environment]
name = "zero_ad_team_gather"
scenario = "rl/scenarios/team_reset_config.json"
uri = "http://localhost:6000"
villager_count = 4
resource_count = 4
map_size_m = 512.0
horizon = 120
sim_steps_per_action = 80
gather_command_distance = 12.0
click_action_threshold = 0.0
stock_resource = "wood"
stock_player = 1
stock_success_threshold = 80.0
carried_resource_observation_scale = 20.0
stock_observation_scale = 1000.0
resource_amount_scale = 200.0
distance_shaping_scale = 0.02
carried_resource_delta_reward_scale = 0.2
click_gather_cycle_penalty = 1.0

[agent]
name = "team_oracle"

[training]
total_steps = 1
seed = 0

[evaluation]
episodes = 1
deterministic = true
seed = 1000
```

`rl/configs/m2_random.toml` is identical except `[agent] name = "random"` and `[evaluation] episodes = 10`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest rl/tests/experiments -v`
Expected: PASS

- [ ] **Step 5: Verify the whole suite, then commit**

Run: `.venv/bin/python -m pytest -q && .venv/bin/python -m ruff check rl`
Expected: all pass

```bash
git add rl/experiments/environments.py rl/gather/factory.py rl/configs/m2_oracle.toml rl/configs/m2_random.toml rl/tests/experiments
git commit -m "feat(rl): register the team gather environment and M2 baseline configs"
```

---

### Task 10: Live acceptance gate

The first and only step in this plan that needs 0 A.D. running. Nothing in the M2 policy work starts until this passes.

**Files:**
- Modify: `Makefile` (add `m2-oracle` and `m2-random` targets, mirroring `m1-oracle`)
- Test: `rl/tests/test_makefile.py` (append)

- [ ] **Step 1: Write the failing test**

```python
# append to rl/tests/test_makefile.py
def test_make_m2_targets_use_the_team_configs(tmp_path):
    m2_oracle = _run_with_fake_uv(tmp_path, "m2-oracle")
    m2_random = _run_with_fake_uv(tmp_path, "m2-random")

    assert "rl/configs/m2_oracle.toml" in m2_oracle
    assert "rl/configs/m2_random.toml" in m2_random
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest rl/tests/test_makefile.py -k m2 -v`
Expected: FAIL — make reports no rule to make target `m2-oracle`

- [ ] **Step 3: Add the targets**

Copy the `m1-oracle` and `m1-random` recipes verbatim, renaming the targets to `m2-oracle` / `m2-random` and pointing `--experiment` at `rl/configs/m2_oracle.toml` / `rl/configs/m2_random.toml`. Add both names to the `.PHONY` line and a line each to the `help` block.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest rl/tests/test_makefile.py -v`
Expected: PASS

- [ ] **Step 5: Run the live gate**

Terminal 1:
```bash
make server
```
Wait for `RL interface listening on 127.0.0.1:6000`.

Terminal 2:
```bash
make m2-oracle EPISODES=3 ARGS="--verbose"
```

**Gate criteria — all four must hold:**
1. Every episode ends `alcanzado=True` with `stock_delta >= 80`.
2. Each of the four villagers deposits at least once (the `--verbose` output shows `dstock=+20.0` on four separate occasions).
3. No `RosterError` and no backend retry warnings.
4. `make m2-random` runs to completion without crashing and, as expected, does *not* reach 80.

If criterion 1 or 2 fails, the environment is wrong, not the oracle — debug before proceeding. The most likely culprits, in order: slot ordering (villagers driven by the wrong action indices), the relational block ordering, and the `gather_command_distance` being too small for the tree spacing chosen in Task 7.

- [ ] **Step 6: Commit**

```bash
git add Makefile rl/tests/test_makefile.py
git commit -m "feat(make): add M2 oracle and random baseline targets"
```

---

## What this plan deliberately excludes

Everything in spec sections "Transferencia desde M1" and the shared-per-villager policy. Those need `ZeroADTeamGatherEnv` to exist and the oracle gate to pass first, and they carry their own risk (a custom SB3 `ActorCriticPolicy`, zero-init prefix loading, centralized critic). They get a second plan once this one is green.

Also excluded, per the spec's YAGNI list: attention encoders, multiple resource types, fog of war, MARL frameworks, and the 1-tree → 4-tree curriculum.
