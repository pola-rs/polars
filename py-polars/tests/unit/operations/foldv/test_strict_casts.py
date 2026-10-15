from __future__ import annotations

import pytest

import polars as pl


def test_foldv_rejects_narrowing_integer_casts() -> None:
    # Integer casts that may create nulls await validity tracking.
    df = pl.DataFrame({"l": pl.Series([1], dtype=pl.Int64)})
    expression = pl.foldv(
        lambda state: state + pl.col("l").cast(pl.Int32),
        s=pl.lit(0, dtype=pl.Int32),
    )

    with pytest.raises(pl.exceptions.InvalidOperationError, match="may produce null"):
        df.select(expression)
