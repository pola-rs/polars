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
    Polars:              0.18.7
    Index type:          UInt32
    Platform:            Linux-6.3.0-1-amd64-x86_64-with-glibc2.36
    Python:              3.11.3 (main, May  7 2023, 13:19:39) [GCC 12.2.0]
    \b
    ----Optional dependencies----
    adbc_driver_sqlite:  0.5.1
    connectorx:          <not installed>
    deltalake:           0.10.0
    fsspec:              2023.6.0
    hypothesis:          6.80.0
    matplotlib:          <not installed>
    numpy:               1.25.0
    pandas:              2.0.3
    pyarrow:             12.0.1
    pydantic:            2.0
    sqlalchemy:          <not installed>
    xlsx2csv:            0.8.1
    xlsxwriter:          3.1.2
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


def _is_stdlib_module(module: str) -> bool:
    if sys.version_info >= (3, 10):  # not available in older versions
        return module in sys.stdlib_module_names

    import importlib.util

    if spec := importlib.util.find_spec(module):
        if origin := spec.origin:
            return origin.startswith(sys.base_prefix)

    return False


def _get_dependency_info() -> dict[str, str]:
    import polars.dependencies

    third_party_lazy = [
        s
        for s in polars.dependencies.__all__
        if not s.startswith("_") and not _is_stdlib_module(s)
    ]

    third_party_eager = [
        "adbc_driver_sqlite",
        "connectorx",
        "matplotlib",
        "sqlalchemy",
        "xlsx2csv",
        "xlsxwriter",
    ]

    third_party_deps = sorted(third_party_lazy + third_party_eager)
    return {f"{name}:": _get_dependency_version(name) for name in third_party_deps}


def _get_dependency_version(dep_name: str) -> str:
    # note: we import 'importlib' here as a significant optimisation for initial import
    import importlib
    import importlib.metadata

    try:
        module = importlib.import_module(dep_name)
    except ImportError:
        return "<not installed>"

    if hasattr(module, "__version__"):
        module_version = module.__version__
    else:
        module_version = importlib.metadata.version(dep_name)

    return module_version
