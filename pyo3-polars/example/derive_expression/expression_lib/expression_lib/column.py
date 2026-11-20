from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from polars.plugins import register_plugin_function

from expression_lib._utils import LIB

if TYPE_CHECKING:
    from expression_lib._typing import IntoExprColumn


def pig_latinnify(expr: IntoExprColumn, capitalize: bool = False) -> pl.Expr:
    return register_plugin_function(
        plugin_path=LIB,
        args=[expr],
        function_name="pig_latinnify_column",
        is_elementwise=True,
        kwargs={"capitalize": capitalize},
    )


def materialize(expr: IntoExprColumn) -> pl.Expr:
    return register_plugin_function(
        plugin_path=LIB,
        args=[expr],
        function_name="materialize",
    )
