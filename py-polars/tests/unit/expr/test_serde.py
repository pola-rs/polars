from __future__ import annotations

import io
from typing import TYPE_CHECKING

import pytest

import polars as pl
from polars.exceptions import ComputeError
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.mark.parametrize(
    "expr",
    [
        pl.col("foo").sum().over("bar"),
        pl.col("foo").rolling_quantile(0.25, window_size=5),
        pl.col("foo").rolling_var(window_size=4, ddof=2),
        pl.col("foo").rolling_min(window_size=2),
        pl.col("foo").rolling_quantile_by("bar", window_size="1mo", quantile=0.75),
    ],
)
def test_expr_serde_roundtrip_binary(expr: pl.Expr) -> None:
    json = expr.meta.serialize(format="binary")
    round_tripped = pl.Expr.deserialize(io.BytesIO(json), format="binary")
    assert round_tripped.meta == expr


@pytest.mark.parametrize(
    "expr",
    [
        pl.col("foo").sum().over("bar"),
        pl.col("foo").rolling_quantile(0.25, window_size=5),
        pl.col("foo").rolling_var(window_size=4, ddof=2),
        pl.col("foo").rolling_min(window_size=2),
        pl.col("foo").rolling_quantile_by("bar", window_size="1mo", quantile=0.75),
    ],
)
def test_expr_serde_roundtrip_json(expr: pl.Expr) -> None:
    expr = pl.col("foo").sum().over("bar")
    json = expr.meta.serialize(format="json")
    round_tripped = pl.Expr.deserialize(io.StringIO(json), format="json")
    assert round_tripped.meta == expr


def test_expr_deserialize_file_not_found() -> None:
    with pytest.raises(FileNotFoundError):
        pl.Expr.deserialize("abcdef")


def test_expr_deserialize_invalid_json() -> None:
    with pytest.raises(
        ComputeError, match="could not deserialize input into an expression"
    ):
        pl.Expr.deserialize(io.StringIO("abcdef"), format="json")


def test_expression_json_13991() -> None:
    expr = pl.col("foo").cast(pl.Decimal(38, 10))
    json = expr.meta.serialize(format="json")

    round_tripped = pl.Expr.deserialize(io.StringIO(json), format="json")
    assert round_tripped.meta == expr


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize(
    "make_expr",
    [
        lambda v: pl.lit(v),
        lambda v: pl.lit(v, dtype=pl.Float16),
        lambda v: pl.lit(v, dtype=pl.Float32),
        lambda v: pl.lit(v, dtype=pl.Float64),
        lambda v: pl.lit({"a": v}),
        lambda v: pl.col("x").fill_null(v),
    ],
)
def test_expr_serde_json_non_finite_float_29465(
    value: float, make_expr: Callable[[float], pl.Expr]
) -> None:
    expr = make_expr(value)
    json = expr.meta.serialize(format="json")
    round_tripped = pl.Expr.deserialize(io.StringIO(json), format="json")
    assert round_tripped.meta == expr

    df = pl.DataFrame({"x": [None]}, schema={"x": pl.Float64})
    assert_frame_equal(
        df.select(round_tripped.alias("out")), df.select(expr.alias("out"))
    )
