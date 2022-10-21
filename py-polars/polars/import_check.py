import importlib.util
from typing import Any, Callable


def pandas_mod() -> Any:
    import pandas as pd

    return pd


def pyarrow_mod() -> Any:
    import pyarrow as pa

    return pa


def pkg_is_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def lazy_isinstance(value: Any, module_bound: str, types: Callable[[], Any]) -> bool:
    if module_bound in str(type(value)):
        check = types()
        return isinstance(value, check)
    return False


_PANDAS_AVAILABLE = pkg_is_available("pandas")
_PYARROW_AVAILABLE = pkg_is_available("pyarrow")
_NUMPY_AVAILABLE = pkg_is_available("numpy")
