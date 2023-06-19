
import subprocess
import sys
from itertools import chain
from typing import Any
import argparse
import tomllib


def parse_toml_file(fp) -> dict[str, dict[str, Any]]:
    if sys.version_info < (3, 11):
        subprocess.run(["pip", "install" ,"tomli"], capture_output=True, text=True)
        import tomlli
        return tomlli.load(fp)
    else:
        return tomllib.load(fp)


def collect_dependencies_from_pyproject_toml() -> list[str]:
    with open("pyproject.toml", mode="rb") as fp:
        config = parse_toml_file(fp)

    mandatory_deps = config["project"]["dependencies"]
    opt_deps = list(chain(*config["project"]["optional-dependencies"].values()))
    deps = mandatory_deps + opt_deps
    return deps


def pip_install(specifiers: list[str]):
    cmd = ["pip", "install"] + specifiers
    subprocess.run(cmd, capture_output=True, text=True )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Install Polars dependencies')
    parser.add_argument('pos_arg', type=int,
                    help='A required integer positional argument')
    deps = collect_dependencies_from_pyproject_toml()

    # remove polars[all, pandas], etc
    deps_no_meta = list(filter(lambda x: not x.startswith("polars"), deps))

    pip_install(deps_no_meta)
