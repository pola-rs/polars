from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from polars.plugins import register_plugin
from polars.utils.udfs import _get_shared_lib_location

if TYPE_CHECKING:
    from polars.type_aliases import IntoExpr


def pig_latinnify(expr: IntoExpr) -> pl.Expr:
    """Pig-latinnify expression."""
    return register_plugin(
        plugin_location=Path(__file__).parent,
        function_name="pig_latinnify",
        inputs=expr,
        is_elementwise=True,
    )


def pig_latinnify_deprecated(expr: str) -> pl.Expr:
    """Pig-latinnify expression."""
    lib = _get_shared_lib_location(__file__)
    return pl.col(expr).register_plugin(
        lib=lib,
        symbol="pig_latinnify",
        is_elementwise=True,
    )
