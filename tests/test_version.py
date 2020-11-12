from pathlib import Path
import toml

from chesscog import __version__


def test_version():
    pyproject_toml = Path(__file__).parent.parent / "pyproject.toml"
    assert pyproject_toml.exists()
    with pyproject_toml.open("r") as f:
        parsed_toml = toml.load(f)
    assert parsed_toml["tool"]["poetry"]["version"] == __version__
