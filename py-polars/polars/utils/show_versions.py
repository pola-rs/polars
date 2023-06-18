from __future__ import annotations

import sys

from polars.utils.meta import get_index_type
from polars.utils.polars_version import get_polars_version


def show_versions() -> None:
    """
    Print out version of Polars and dependencies to stdout.

    Examples
    --------
    >>> pl.show_versions()  # doctest: +SKIP
    --------Version info---------
    Polars:      0.17.11
    Index type:  UInt32
    Platform:    Linux-5.15.90.1-microsoft-standard-WSL2-x86_64-with-glibc2.35
    Python:      3.11.3 (main, Apr 15 2023, 14:44:51) [GCC 11.3.0]
    \b
    ----Optional dependencies----
    numpy:       1.24.2
    pandas:      2.0.0
    pyarrow:     11.0.0
    connectorx:  <not installed>
    deltalake:   0.8.1
    fsspec:      2023.4.0
    matplotlib:  3.7.1
    xlsx2csv:    0.8.1
    xlsxwriter:  3.1.0
    """
    # note: we import 'platform' here as a micro-optimisation for initial import
    import platform

    # optional dependencies
    deps = _get_dependency_info()

    # determine key length for alignment
    keylen = (
        max(
            len(x) for x in [*deps.keys(), "Polars", "Index type", "Platform", "Python"]
        )
        + 1
    )

    print("--------Version info---------")
    print(f"{'Polars:':{keylen}s} {get_polars_version()}")
    print(f"{'Index type:':{keylen}s} {get_index_type()}")
    print(f"{'Platform:':{keylen}s} {platform.platform()}")
    print(f"{'Python:':{keylen}s} {sys.version}")

    print("\n----Optional dependencies----")
    for name, v in deps.items():
        print(f"{name:{keylen}s} {v}")


def _get_dependency_info() -> dict[str, str]:
    # see the list of dependencies in pyproject.toml
    opt_deps = [
        "numpy",
        "pandas",
        "pyarrow",
        "connectorx",
        "deltalake",
        "fsspec",
        "matplotlib",
        "xlsx2csv",
        "xlsxwriter",
    ]
    return {f"{name}:": _get_dependency_version(name) for name in opt_deps}


def _get_dependency_version(dep_name: str) -> str:
    # note: we import 'importlib' here as a significiant optimisation for initial import
    import importlib

    if sys.version_info >= (3, 8):
        # importlib.metadata was introduced in Python 3.8;
        # metadata submodule must be imported explicitly
        import importlib.metadata
    try:
        module = importlib.import_module(dep_name)
    except ImportError:
        return "<not installed>"

    if hasattr(module, "__version__"):
        module_version = module.__version__
    elif sys.version_info >= (3, 8):
        module_version = importlib.metadata.version(dep_name)
    else:
        module_version = "<version not detected>"

    return module_version
