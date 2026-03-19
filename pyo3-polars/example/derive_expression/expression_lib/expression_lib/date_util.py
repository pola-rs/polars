from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from polars.plugins import register_plugin_function

from expression_lib._utils import LIB

if TYPE_CHECKING:
    from expression_lib._typing import IntoExprColumn


def is_leap_year(expr: IntoExprColumn) -> pl.Expr:
    return register_plugin_function(
        plugin_path=LIB,
        args=[expr],
        function_name="is_leap_year",
        is_elementwise=True,
    )


# Note that this already exists in Polars. It is just for explanatory
# purposes.
def change_time_zone(expr: IntoExprColumn, tz: str = "Europe/Amsterdam") -> pl.Expr:
    return register_plugin_function(
        plugin_path=LIB,
        args=[expr],
        function_name="change_time_zone",
        is_elementwise=True,
        kwargs={"tz": tz},
    )
