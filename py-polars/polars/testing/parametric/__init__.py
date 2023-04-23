from typing import Any

from polars.dependencies import _HYPOTHESIS_AVAILABLE

if _HYPOTHESIS_AVAILABLE:
    from polars.testing.parametric.primitives import (
        column,
        columns,
        dataframes,
        series,
    )
    from polars.testing.parametric.profiles import load_profile, set_profile
    from polars.testing.parametric.strategies import (
        create_list_strategy,
        scalar_strategies,
    )
else:

    def __getattr__(*args: Any, **kwargs: Any) -> Any:
        raise ModuleNotFoundError(
            f"polars.testing.parametric.{args[0]} requires the 'hypothesis' module"
        ) from None


__all__ = [
    "column",
    "columns",
    "create_list_strategy",
    "dataframes",
    "load_profile",
    "scalar_strategies",
    "series",
    "set_profile",
]
