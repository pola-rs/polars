"""
Install Polars dependencies as defined in pyproject.toml without building Polars.

In our dev workflow, we often want to either not build (Rust) Polars at all, or in
develop mode. Unfortunately, `pip install .` or `pip install . -e` will always call
`maturin build` to build Polars. This is very slow compared to `maturin develop`.
We work around this by calling `maturin develop`, and for the Python dependencies,
this script collects the relevant tags from pyproject.toml and pass to `pip install`.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from typing import Any


def parse_toml_file(fp) -> dict[str, dict[str, Any]]:
    if sys.version_info < (3, 11):
        subprocess.run(["pip", "install", "tomli"], capture_output=False, text=True)
        import tomli

        return tomli.load(fp)
    else:
        import tomllib

        return tomllib.load(fp)


def collect_dependencies_from_pyproject_toml(tags: str | None = None) -> list[str]:
    """
    Collects all dependencies, mandatory and optional, from pyproject.toml.

    Parameters
    ----------
    tags
        Select dependencies under these tags defined in optional-dependencies.
        Multiple tags can be provided by passing in a comma delimited string, for
        example "dev-test,dev-lint". If `None` is passed in, no optional dependencies
        are returned.

    """
    with open("pyproject.toml", mode="rb") as fp:
        config = parse_toml_file(fp)

    deps: list[str] = config["project"]["dependencies"]

    if tags:
        if "," in tags:
            # multiple tags are passed in
            for t in tags.split(","):
                deps += config["project"]["optional-dependencies"][t.strip()]
        else:
            # a single tag is passed in
            deps += config["project"]["optional-dependencies"][tags]

    # resolve polars[] tags
    for d in deps:
        if d.startswith("polars["):
            tags_as_comma_delimited_string = d.split("[")[1].replace("]", "")
            for pt in tags_as_comma_delimited_string.split(","):
                deps += config["project"]["optional-dependencies"][pt]

    # drop the polars[] tags from the list
    deps = [d for d in deps if not d.startswith("polars[")]

    return deps


def install_dependencies(tags: str | None = None) -> None:
    deps = collect_dependencies_from_pyproject_toml(tags)
    cmd = ["pip", "install"] + deps
    subprocess.run(cmd, capture_output=False, text=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Install Polars dependencies from pyproject.toml without building Polars"
    )
    parser.add_argument(
        "tags",
        nargs="?",  # allows no arguments to be passed in
        default=None,
        type=str,
        help="Optional-dependency tag(s) in pyproject.toml. Provide multiple by separating with commas.",
    )
    args = parser.parse_args()
    install_dependencies(args.tags)
