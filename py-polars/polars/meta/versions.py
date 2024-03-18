from __future__ import annotations

import sys

from polars._utils.polars_version import get_polars_version
from polars.meta.index_type import get_index_type


def show_versions() -> None:
    """
    Print out the version of Polars and its optional dependencies.

    Examples
    --------
    >>> pl.show_versions()  # doctest: +SKIP
    --------Version info---------
    Polars:               0.20.14
    Index type:           UInt32
    Platform:             macOS-14.3.1-arm64-arm-64bit
    Python:               3.11.8 (main, Feb  6 2024, 21:21:21) [Clang 15.0.0 (clang-1500.1.0.2.5)]
    ----Optional dependencies----
    adbc_driver_manager:  0.10.0
    cloudpickle:          3.0.0
    connectorx:           0.3.2
    deltalake:            0.16.0
    fastexcel:            0.9.1
    fsspec:               2023.12.2
    gevent:               24.2.1
    hvplot:               0.9.2
    matplotlib:           3.8.3
    numpy:                1.26.4
    openpyxl:             3.1.2
    pandas:               2.2.1
    pyarrow:              15.0.0
    pydantic:             2.6.3
    pyiceberg:            0.6.0
    pyxlsb:               1.0.10
    sqlalchemy:           2.0.28
    xlsx2csv:             0.8.2
    xlsxwriter:           3.2.0
    """  # noqa: W505
    # Note: we import 'platform' here (rather than at the top of the
    # module) as a micro-optimization for polars' initial import
    import platform

    deps = _get_dependency_info()
    core_properties = ("Polars", "Index type", "Platform", "Python")
    keylen = max(len(x) for x in [*core_properties, *deps.keys()]) + 1

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
        "adbc_driver_manager",
        "cloudpickle",
        "connectorx",
        "deltalake",
        "fastexcel",
        "fsspec",
        "gevent",
        "hvplot",
        "matplotlib",
        "numpy",
        "openpyxl",
        "pandas",
        "pyarrow",
        "pydantic",
        "pyiceberg",
        "pyxlsb",
        "sqlalchemy",
        "xlsx2csv",
        "xlsxwriter",
    ]
    return {f"{name}:": _get_dependency_version(name) for name in opt_deps}


def _get_dependency_version(dep_name: str) -> str:
    # note: we import 'importlib' here as a significiant optimisation for initial import
    import importlib
    import importlib.metadata

    try:
        module = importlib.import_module(dep_name)
    except ImportError:
        return "<not installed>"

    if hasattr(module, "__version__"):
        module_version = module.__version__
    else:
        module_version = importlib.metadata.version(dep_name)  # pragma: no cover

    return module_version
