"""Script to be run immediately before GitHub release in order to bump up the version number."""

import toml
from pathlib import Path
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Change the version number")
    parser.add_argument(
        "--version", help="the new version number (by default will increment the patch by one)", type=str, default=None)
    args = parser.parse_args()

    pyproject_toml = Path(__file__).parent.parent / "pyproject.toml"
    version_file = Path(__file__).parent.parent / "chesscog" / "__version__.py"
    with pyproject_toml.open("r") as f:
        parsed_toml = toml.load(f)

    old_version = parsed_toml["tool"]["poetry"]["version"]
    print("Old version:", old_version)
    if args.version is None:
        major, minor, patch = map(int, old_version.split("."))
        patch += 1
        new_version = ".".join(map(str, (major, minor, patch)))
    else:
        new_version = args.version
    print("New version:", new_version)
    parsed_toml["tool"]["poetry"]["version"] = new_version
    with pyproject_toml.open("w") as f:
        toml.dump(parsed_toml, f)
    with version_file.open("w") as f:
        f.write(f"__version__ = \"{new_version}\"")
    print("Done.")
