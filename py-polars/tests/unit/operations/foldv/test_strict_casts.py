from __future__ import annotations

import pytest

import polars as pl


def test_foldv_executes_strict_narrowing_casts_whose_values_fit() -> None:
    # A strict cast succeeds when every value fits, and nulls stay null.
    df = pl.DataFrame({"l": pl.Series([1, None, 3], dtype=pl.Int64)})
    expression = pl.foldv(
        lambda state: state + pl.col("l").cast(pl.Int32),
        initial_state=pl.lit(0, dtype=pl.Int32),
    )

    assert df.select(expression).to_series().to_list() == [1, None, None]


def test_foldv_raises_when_a_strict_cast_does_not_fit() -> None:
    # The error names the conversion, the first failing row, and its value.
    df = pl.DataFrame({"l": [1, 3_000_000_000]})
    expression = pl.foldv(
        lambda state: state + pl.col("l").cast(pl.Int32),
        initial_state=pl.lit(0, dtype=pl.Int32),
    )

    with pytest.raises(
        pl.exceptions.InvalidOperationError,
        match=r"conversion from `i64` to `i32` failed in foldv at row 1: \[3000000000\]",
    ):
        df.select(expression)
