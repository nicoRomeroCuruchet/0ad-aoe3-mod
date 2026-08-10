from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_rl_gather_exports_release_28_map_generator():
    source = (REPO_ROOT / "maps/random/rl_gather.js").read_text(encoding="utf-8")

    assert "export function* generateMap(" in source
    assert "function* GenerateMap(" not in source
    assert '"structures/athenai/rl_storehouse"' in source
