from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from polars.plugins import register_plugin_function

from expression_lib._utils import LIB

if TYPE_CHECKING:
    from expression_lib._typing import IntoExprColumn


def hamming_distance(expr: IntoExprColumn, other: IntoExprColumn) -> pl.Expr:
    return register_plugin_function(
        plugin_path=LIB,
        args=[expr, other],
        function_name="hamming_distance",
        is_elementwise=True,
    )


def jaccard_similarity(expr: IntoExprColumn, other: IntoExprColumn) -> pl.Expr:
    return register_plugin_function(
        plugin_path=LIB,
        args=[expr, other],
        function_name="jaccard_similarity",
        is_elementwise=True,
    )


def haversine(
    start_lat: IntoExprColumn,
    start_long: IntoExprColumn,
    end_lat: IntoExprColumn,
    end_long: IntoExprColumn,
) -> pl.Expr:
    return register_plugin_function(
        plugin_path=LIB,
        args=[start_lat, start_long, end_lat, end_long],
        function_name="haversine",
        is_elementwise=True,
        cast_to_supertype=True,
    )
