from __future__ import annotations

import polars as pl


def test_foldv_reads_multiple_columns_at_the_same_row() -> None:
    # Each iteration reads both column values at the current row index.
    result = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}).select(
        pl.foldv(lambda state: state + pl.col("a") * pl.col("b"), initial_state=0)
    )

    assert result.to_series().to_list() == [4, 14, 32]


def test_foldv_reads_multiple_columns_with_different_dtypes() -> None:
    # Each referenced column retains its own physical buffer type before coercion.
    result = pl.DataFrame(
        {
            "a": pl.Series([1, 2, 3], dtype=pl.Int32),
            "b": pl.Series([0.5, 1.5, 2.5], dtype=pl.Float64),
        }
    ).select(
        pl.foldv(lambda state: state + pl.col("a") + pl.col("b"), initial_state=0.0)
    )

    assert result.to_series().dtype == pl.Float64
    assert result.to_series().to_list() == [1.5, 5.0, 10.5]


def test_foldv_allows_a_body_that_does_not_read_the_previous_state() -> None:
    # The typed state binding remains available even when the body replaces it each row.
    result = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}).select(
        pl.foldv(lambda _state: pl.col("a") + pl.col("b"), initial_state=0)
    )

    assert result.to_series().to_list() == [5, 7, 9]
