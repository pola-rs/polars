import importlib.util
import sys
from types import ModuleType
from typing import TYPE_CHECKING, Any


def lazy_import(module_name: str) -> ModuleType:
    # check if module is already loaded
    if module_name in sys.modules:
        return sys.modules[module_name]

    # check if the module is available (if not, user needs to install)
    try:
        spec = importlib.util.find_spec(module_name)
        if spec is None or spec.loader is None:
            return None  # type: ignore[return-value]
    except ModuleNotFoundError:
        return None  # type: ignore[return-value]

    # module is available, but not yet imported into the environment; create
    # a lazy loader that proxies (then replaces) the module in sys.modules
    loader = importlib.util.LazyLoader(spec.loader)
    spec.loader = loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    loader.exec_module(module)
    return module


if TYPE_CHECKING:
    import fsspec
    import numpy
    import pandas
    import pyarrow

    if sys.version_info >= (3, 9):
        import zoneinfo
    else:
        from backports import zoneinfo
else:
    fsspec = lazy_import("fsspec")
    numpy = lazy_import("numpy")
    pandas = lazy_import("pandas")
    pyarrow = lazy_import("pyarrow")
    if sys.version_info >= (3, 9):
        zoneinfo = lazy_import("zoneinfo")
    else:
        zoneinfo = lazy_import("backports.zoneinfo")


_FSSPEC_AVAILABLE = fsspec is not None
_NUMPY_AVAILABLE = numpy is not None
_PANDAS_AVAILABLE = pandas is not None
_PYARROW_AVAILABLE = pyarrow is not None
_ZONEINFO_AVAILABLE = zoneinfo is not None


def _NUMPY_TYPE(obj: Any) -> bool:
    return _NUMPY_AVAILABLE and "numpy" in str(type(obj))


def _PANDAS_TYPE(obj: Any) -> bool:
    return _PANDAS_AVAILABLE and "pandas" in str(type(obj))


def _PYARROW_TYPE(obj: Any) -> bool:
    return _PYARROW_AVAILABLE and "pyarrow" in str(type(obj))


__all__ = [
    "fsspec",
    "numpy",
    "pandas",
    "pyarrow",
    "zoneinfo",
    "_FSSPEC_AVAILABLE",
    "_NUMPY_AVAILABLE",
    "_NUMPY_TYPE",
    "_PANDAS_AVAILABLE",
    "_PANDAS_TYPE",
    "_PYARROW_AVAILABLE",
    "_PYARROW_TYPE",
    "_ZONEINFO_AVAILABLE",
]
