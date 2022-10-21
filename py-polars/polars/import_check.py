import importlib.util
from typing import Any, Callable


def pandas_mod() -> Any:
    import pandas as pd

    return pd


def pkg_is_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


_PANDAS_AVAILABLE = pkg_is_available("pandas")


def lazy_isinstance(value: Any, module_bound: str, types: Callable[[], Any]) -> bool:
    if module_bound in str(type(value)):
        check = types()
        return isinstance(value, check)
    return False
