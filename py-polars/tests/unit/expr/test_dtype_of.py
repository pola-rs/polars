from __future__ import annotations

import polars as pl


def test_dtype_of_scalar_literal_collect_dtype_26271() -> None:
    assert pl.dtype_of(pl.lit(1)).collect_dtype({}) == pl.Int32
    assert pl.dtype_of(pl.lit(1.0)).collect_dtype({}) == pl.Float64

    # These already worked but included for completeness
    assert pl.dtype_of(pl.lit("foo")).collect_dtype({}) == pl.String
    assert pl.dtype_of(pl.lit([1])).collect_dtype({}) == pl.List(pl.Int64)
