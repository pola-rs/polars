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
    Polars:               0.19.16
    Index type:           UInt32
    Platform:             macOS-14.1.1-arm64-arm-64bit
    Python:               3.11.6 (main, Oct  2 2023, 13:45:54) [Clang 15.0.0 (clang-1500.0.40.1)]
    ----Optional dependencies----
    adbc_driver_manager:  0.8.0
    cloudpickle:          3.0.0
    connectorx:           0.3.2
    deltalake:            0.13.0
    fsspec:               2023.10.0
    gevent:               23.9.1
    matplotlib:           3.8.2
    numpy:                1.26.2
    openpyxl:             3.1.2
    pandas:               2.1.3
    pyarrow:              14.0.1
    pydantic:             2.5.2
    pyiceberg:            0.5.1
    pyxlsb:               1.0.10
    sqlalchemy:           2.0.23
    xlsx2csv:             0.8.1
    xlsxwriter:           3.1.9

    """  # noqa: W505
    # note: we import 'platform' here (rather than at the top of the
    # module) as a micro-optimisation for polars' initial import
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
