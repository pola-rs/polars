import os
from typing import Any

from hypothesis import settings

from polars.dependencies import _HYPOTHESIS_AVAILABLE

if _HYPOTHESIS_AVAILABLE:
    # Default profile (eg: running locally)
    common_settings = {"print_blob": True, "deadline": None}
    settings.register_profile(
        name="polars.default",
        max_examples=100,
        **common_settings,  # type: ignore[arg-type]
    )
    # CI 'max' profile (10x the number of iterations).
    # this is more expensive (about 4x slower), and not actually
    # enabled in our usual CI pipeline; requires explicit opt-in.
    settings.register_profile(
        name="polars.ci",
        max_examples=1000,
        **common_settings,  # type: ignore[arg-type]
    )
    if os.getenv("CI_MAX"):
        settings.load_profile("polars.ci")
    else:
        settings.load_profile("polars.default")

    from polars.testing.parametric.primitives import (
        column,
        columns,
        dataframes,
        series,
    )
    from polars.testing.parametric.strategies import (
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
    "dataframes",
    "series",
    "scalar_strategies",
]
