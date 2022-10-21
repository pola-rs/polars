import importlib.util
from typing import Any


def pandas_mod() -> Any:
    import pandas as pd

    return pd


def pkg_is_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


_PANDAS_AVAILABLE = pkg_is_available("pandas")
