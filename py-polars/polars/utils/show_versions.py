from __future__ import annotations

import sys

from polars.utils.meta import get_index_type
from polars.utils.polars_version import get_polars_version


def show_versions() -> None:
    r"""
    Print out version of Polars and dependencies to stdout.

    Examples
    --------
    >>> pl.show_versions()  # doctest: +SKIP
    --------Version info---------
    Polars:               0.19.3
    Index type:           UInt32
    Platform:             macOS-13.5.2-arm64-arm-64bit
    Python:               3.11.5 (main, Aug 24 2023, 15:09:45) [Clang 14.0.3 (clang-1403.0.22.14.1)]
    ----Optional dependencies----
    adbc_driver_sqlite:   0.6.0
    cloudpickle:          2.2.1
    connectorx:           0.3.2
    deltalake:            0.10.1
    fsspec:               2023.9.1
    gevent:               23.9.1
    matplotlib:           3.8.0
    numpy:                1.26.0
    openpyxl:             3.1.2
    pandas:               2.1.0
    pyarrow:              13.0.0
    pydantic:             2.3.0
    pyiceberg:            0.5.1
    pyxlsb:               <not installed>
    sqlalchemy:           2.0.21
    xlsx2csv:             0.8.1
    xlsxwriter:           3.1.4

    """  # noqa: W505
    # note: we import 'platform' here as a micro-optimisation for initial import
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
        "adbc_driver_sqlite",
        "cloudpickle",
        "connectorx",
        "deltalake",
        "fsspec",
        "gevent",
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
