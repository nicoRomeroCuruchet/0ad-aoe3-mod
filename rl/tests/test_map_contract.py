from pathlib import Path
import xml.etree.ElementTree as ET


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_rl_gather_exports_release_28_map_generator():
    source = (REPO_ROOT / "maps/random/rl_gather.js").read_text(encoding="utf-8")

    assert "export function* generateMap(" in source
    assert "function* GenerateMap(" not in source
    assert '"structures/athenai/rl_storehouse"' in source


def test_rl_gather_team_places_four_villagers_and_six_trees_on_a_hut_centred_circle():
    source = (REPO_ROOT / "maps/random/rl_gather_team.js").read_text(encoding="utf-8")

    assert "export function* generateMap(" in source
    assert source.count('"units/athenai/polites"') == 4
    assert source.count('"gaia/tree/rl_m2_oak"') == 1
    assert '"structures/athenai/rl_storehouse"' in source
    assert "const hutPosition = g_Map.getCenter();" in source
    assert "const treeCount = 6;" in source
    assert "const treeRadius = 12;" in source
    assert "const treeRotation = randFloat(0, 2 * Math.PI);" in source
    assert "const treeAngle = treeRotation + index * (2 * Math.PI / treeCount);" in source
    assert "Vector2D.add(hutPosition, new Vector2D(treeRadius, 0).rotate(treeAngle))" in source


def test_m2_tree_template_makes_assignment_a_real_resource_constraint():
    template = ET.parse(
        REPO_ROOT / "simulation/templates/gaia/tree/rl_m2_oak.xml"
    ).getroot()

    assert template.attrib["parent"] == "gaia/tree/oak"
    supply = template.find("ResourceSupply")
    assert supply is not None
    assert supply.findtext("Max") == "100"
    assert supply.findtext("MaxGatherers") == "1"


def test_rl_gather_team_scenario_is_deterministic():
    import json

    scenario = json.loads(
        (REPO_ROOT / "rl/scenarios/team_reset_config.json").read_text(encoding="utf-8")
    )

    assert scenario["script"] == "rl_gather_team.js"
    assert scenario["settings"]["Seed"] == 0
    assert scenario["settings"]["AISeed"] == 0
