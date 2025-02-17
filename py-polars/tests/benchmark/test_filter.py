"""Benchmark tests for the filter operation."""

from __future__ import annotations

import pytest

import polars as pl

pytestmark = pytest.mark.benchmark()


def test_filter1(groupby_data: pl.DataFrame) -> None:
    (
        groupby_data.lazy()
        .filter(pl.col("id1").eq_missing(pl.lit("id046")))
        .select(
            pl.col("id6").cast(pl.Int64).sum(),
            pl.col("v3").sum(),
        )
        .collect()
    )


def test_filter2(groupby_data: pl.DataFrame) -> None:
    (
        groupby_data.lazy()
        .filter(~(pl.col("id1").eq_missing(pl.lit("id046"))))
        .select(
            pl.col("id6").cast(pl.Int64).sum(),
            pl.col("v3").sum(),
        )
        .collect()
    )
