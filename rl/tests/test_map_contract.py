from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_rl_gather_exports_release_28_map_generator():
    source = (REPO_ROOT / "maps/random/rl_gather.js").read_text(encoding="utf-8")

    assert "export function* generateMap(" in source
    assert "function* GenerateMap(" not in source
    assert '"structures/athenai/rl_storehouse"' in source


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
