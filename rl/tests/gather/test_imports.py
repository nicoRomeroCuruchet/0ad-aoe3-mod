import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def run_python(source):
    return subprocess.run(
        [sys.executable, "-c", source],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_core_import_does_not_require_zero_ad_or_import_the_environment():
    result = run_python(
        "import sys\n"
        "sys.modules['zero_ad'] = None\n"
        "from rl.gather.core import distance\n"
        "assert distance((0, 0), (3, 4)) == 5.0\n"
        "assert 'rl.gather.env' not in sys.modules\n"
    )

    assert result.returncode == 0, result.stderr


def test_environment_class_can_be_imported_without_zero_ad_installed():
    result = run_python(
        "import sys\n"
        "sys.modules['zero_ad'] = None\n"
        "from rl.gather import ZeroADGatherEnv\n"
        "assert ZeroADGatherEnv.__name__ == 'ZeroADGatherEnv'\n"
    )

    assert result.returncode == 0, result.stderr
