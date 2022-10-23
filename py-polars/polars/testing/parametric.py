from typing import Any

from polars.dependencies import _HYPOTHESIS_AVAILABLE

if _HYPOTHESIS_AVAILABLE:
    from polars.testing._parametric import (
        column,
        columns,
        dataframes,
        series,
        strategy_dtypes,
    )
else:

    def __getattr__(*args: Any, **kwargs: Any) -> Any:
        raise ModuleNotFoundError(
            f"polars.testing.parametric.{args[0]} requires the 'hypothesis' module"
        ) from None


__all__ = [
    "column",
    "columns",
    "dataframes",
    "series",
    "strategy_dtypes",
]
