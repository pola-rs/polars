from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from polars.plugins import register_plugin_v2_function

from plugin_v2._utils import LIB

if TYPE_CHECKING:
    from expression_lib._typing import IntoExprColumn


def min_by(expr: IntoExprColumn, *, by: IntoExprColumn) -> pl.Expr:
    from plugin_v2 import plugin_v2

    return register_plugin_v2_function(
        plugin_path=LIB,
        args=[expr, by],
        info=plugin_v2.min_by(),
        function_name="min_by",
        returns_scalar=True,
        insert_has_output=False,
        states_combinable=True,
    )


def rolling_product(expr: IntoExprColumn, *, n: int) -> pl.Expr:
    from plugin_v2 import plugin_v2

    return register_plugin_v2_function(
        plugin_path=LIB,
        args=[expr],
        info=plugin_v2.rolling_product(n),
        function_name="rolling_product",
        length_preserving=True,
        needs_finalize=False,
        states_combinable=False,
    )


def byte_rev(expr: IntoExprColumn) -> pl.Expr:
    from plugin_v2 import plugin_v2

    return register_plugin_v2_function(
        plugin_path=LIB,
        args=[expr],
        info=plugin_v2.byte_rev(),
        function_name="byte_rev",
        length_preserving=True,
        row_separable=True,
        needs_finalize=False,
        states_combinable=False,
    )


def vertical_scan(expr: IntoExprColumn, *, init: int) -> pl.Expr:
    from plugin_v2 import plugin_v2

    return register_plugin_v2_function(
        plugin_path=LIB,
        args=[expr],
        info=plugin_v2.vertical_scan(init),
        function_name="vertical_scan",
        length_preserving=True,
        needs_finalize=False,
        states_combinable=False,
    )


def horizontal_count(*expr: pl.Expr) -> pl.Expr:
    from plugin_v2 import plugin_v2

    return register_plugin_v2_function(
        plugin_path=LIB,
        args=list(expr),
        info=plugin_v2.horizontal_count(),
        function_name="horizontal_count",
        length_preserving=True,
        row_separable=True,
        needs_finalize=False,
        states_combinable=False,
        selector_expansion=True,
    )
