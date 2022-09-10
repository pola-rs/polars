from __future__ import annotations

import importlib
import platform
import sys

try:
    from polars.polars import get_idx_type as _get_idx_type
    from polars.polars import version

    _DOCUMENTING = False
except ImportError:
    _DOCUMENTING = True


def show_versions() -> None:
    """
    Print out version of Polars and dependencies to stdout.

    Examples
    --------
    >>> pl.show_versions()  # doctest: +SKIP
    ---Version info---
    Polars: 0.14.0
    Index type: UInt32
    Platform: Linux-5.10.16.3-microsoft-standard-WSL2-x86_64-with-glibc2.31
    Python: 3.10.5 (main, Jul  8 2022, 14:32:56) [GCC 10.2.1 20210110]
    ---Optional dependencies---
    pyarrow: 8.0.0
    pandas: 1.4.3
    numpy: 1.23.0
    fsspec: <not installed>
    connectorx: <not installed>
    xlsx2csv: <not installed>

    """
    print("---Version info---")
    print(f"Polars: {version()}")
    print(f"Index type: {_get_idx_type().__name__}")
    print(f"Platform: {platform.platform()}")
    print(f"Python: {sys.version}")

    print("---Optional dependencies---")
    deps = _get_dependency_info()
    for name, v in deps.items():
        print(f"{name}: {v}")


def _get_dependency_info() -> dict[str, str]:
    # see the list of dependencies in pyproject.toml
    opt_deps = [
        "pyarrow",
        "pandas",
        "numpy",
        "fsspec",
        "connectorx",
        "xlsx2csv",
    ]
    return {name: _get_dep_version(name) for name in opt_deps}


def _get_dep_version(dep_name: str) -> str:
    try:
        module = importlib.import_module(dep_name)
    except ImportError:
        return "<not installed>"

    # all our dependencies as of 2022-08-11 implement __version__
    return getattr(module, "__version__", "<version not detected>")
