import importlib.util
import sys
from types import ModuleType
from typing import Any, Callable


def numpy_mod() -> ModuleType:
    import numpy

    return numpy


def pandas_mod() -> ModuleType:
    import pandas

    return pandas


def pyarrow_mod() -> ModuleType:
    import pyarrow

    return pyarrow


def zoneinfo_mod() -> ModuleType:
    if sys.version_info >= (3, 9):
        import zoneinfo
    else:
        from backports import zoneinfo
    return zoneinfo


def pkg_is_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def lazy_isinstance(value: Any, module_bound: str, types: Callable[[], Any]) -> bool:
    if module_bound in str(type(value)):
        check = types()
        return isinstance(value, check)
    return False


_FSSPEC_AVAILABLE = pkg_is_available("fsspec")
_NUMPY_AVAILABLE = pkg_is_available("numpy")
_PANDAS_AVAILABLE = pkg_is_available("pandas")
_PYARROW_AVAILABLE = pkg_is_available("pyarrow")
_ZONEINFO_AVAILABLE = (
    pkg_is_available("zoneinfo")
    if sys.version_info >= (3, 9)
    else pkg_is_available("backports.zoneinfo")
)
