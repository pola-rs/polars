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
    ---Version info---
    Polars: 0.16.13
    Index type: UInt32
    Platform: macOS-13.2.1-arm64-arm-64bit
    Python: 3.11.2 (main, Feb 16 2023, 02:55:59) [Clang 14.0.0 (clang-1400.0.29.202)]
    ---Optional dependencies---
    numpy: 1.24.2
    pandas: 1.5.3
    pyarrow: 11.0.0
    connectorx: 0.3.2_alpha.2
    deltalake: <version not detected>
    fsspec: <not installed>
    matplotlib: <not installed>
    xlsx2csv: 0.8.1
    xlsxwriter: 3.0.8

    """
    # note: we import 'platform' here as a micro-optimisation for initial import
    import platform

    print("---Version info---")
    print(f"Polars: {get_polars_version()}")
    print(f"Index type: {get_index_type()}")
    print(f"Platform: {platform.platform()}")
    print(f"Python: {sys.version}")

    print("---Optional dependencies---")
    deps = _get_dependency_info()
    for name, v in deps.items():
        print(f"{name}: {v}")


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
    return {name: _get_dependency_version(name) for name in opt_deps}


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
