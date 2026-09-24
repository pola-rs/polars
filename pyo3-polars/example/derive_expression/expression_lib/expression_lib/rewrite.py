from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from polars.plugins import register_plugin_rewrite

from expression_lib._utils import LIB

if TYPE_CHECKING:
    from expression_lib._typing import IntoExprColumn

# Lengths are stored as floats, the metadata holds the unit ("mm", "cm", "m" or "km").
pl.register_extension_type("expression_lib.length", pl.Extension)


def length(unit: str) -> pl.Extension:
    return pl.Extension(name="expression_lib.length", storage=pl.Float64, metadata=unit)


def to_unit(expr: IntoExprColumn, unit: str) -> pl.Expr:
    """Convert a `length` column to `unit`; the source unit is read from the dtype."""
    return register_plugin_rewrite(
        plugin_path=LIB,
        function_name="to_unit",
        args=[expr],
        kwargs={"unit": unit},
    )


def struct_median(expr: IntoExprColumn) -> pl.Expr:
    """Median per struct field, or a plain median for other dtypes."""
    return register_plugin_rewrite(
        plugin_path=LIB, function_name="struct_median", args=[expr]
    )


def is_leap_year_any(expr: IntoExprColumn) -> pl.Expr:
    """Call the `is_leap_year` kernel on dates or datetimes."""
    return register_plugin_rewrite(
        plugin_path=LIB, function_name="is_leap_year_any", args=[expr]
    )
