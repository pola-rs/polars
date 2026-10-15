from __future__ import annotations

import math

import pytest

import polars as pl


def test_foldv_propagates_nan_input_values() -> None:
    # NaN is a value and remains in the carried state after arithmetic.
    df = pl.DataFrame({"l": [1.0, float("nan")]})
    expression = pl.foldv(lambda state: state + pl.col("l"), initial_state=0.0)

    result = df.select(expression).to_series().to_list()
    assert result[0] == 1.0
    assert math.isnan(result[1])


def test_foldv_propagates_null_state_through_arithmetic() -> None:
    # A null arithmetic result becomes the state supplied to later rows.
    result = pl.DataFrame({"l": [1, None, 3]}).select(
        pl.foldv(
            lambda state: state + pl.col("l"),
            initial_state=pl.lit(0, dtype=pl.Int64),
        )
    )

    assert result.to_series().to_list() == [1, None, None]


def test_foldv_recovers_from_a_null_initial_state() -> None:
    # A validity predicate can replace a null state before later arithmetic.
    result = pl.DataFrame({"l": [1, 2, 3]}).select(
        pl.foldv(
            lambda state: (
                pl.when(state.is_null())
                .then(pl.col("l"))
                .otherwise(state + pl.col("l"))
            ),
            initial_state=pl.lit(None, dtype=pl.Int64),
        )
    )

    assert result.to_series().to_list() == [1, 3, 6]


def test_foldv_supports_missing_equality_for_null_values() -> None:
    # Missing equality treats two nulls as equal and always returns a valid Boolean.
    result = pl.DataFrame({"l": [1, None, None]}).select(
        pl.foldv(
            lambda state: (
                pl.when(pl.col("l").eq_missing(pl.lit(None, dtype=pl.Int64)))
                .then(state + 1)
                .otherwise(state)
            ),
            initial_state=pl.lit(0, dtype=pl.Int64),
        )
    )

    assert result.to_series().to_list() == [0, 1, 2]


def test_foldv_asks_for_a_dtype_when_the_function_does_not_determine_the_state() -> (
    None
):
    # A body that never combines the state with a typed value leaves the state untyped.
    df = pl.DataFrame({"l": [1]})
    expression = pl.foldv(
        lambda state: pl.when(pl.col("l") > 0).then(state).otherwise(state),
        initial_state=None,
    )

    with pytest.raises(pl.exceptions.InvalidOperationError, match="untyped null"):
        df.select(expression)


def test_foldv_skips_nan_values_with_fill_nan() -> None:
    # fill_nan replaces NaN but keeps nulls, so a null still makes the state null.
    df = pl.DataFrame({"l": [1.0, float("nan"), 2.0]})
    result = df.select(
        pl.foldv(lambda s: s + pl.col("l").fill_nan(0.0), initial_state=0.0)
    )

    assert result.to_series().to_list() == [1.0, 1.0, 3.0]
