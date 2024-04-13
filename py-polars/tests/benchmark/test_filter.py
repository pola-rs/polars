"""Benchmark tests for the filter operation."""

from __future__ import annotations

import pytest

import polars as pl

pytestmark = pytest.mark.benchmark()


def test_filter1(h2oai_groupby_data: pl.DataFrame) -> None:
    result = (
        h2oai_groupby_data.lazy()
        .filter(pl.col("id1").eq_missing(pl.lit("id046")))
        .select(
            pl.col("id6").cast(pl.Int64).sum(),
            pl.col("v3").sum(),
        )
        .collect()
    )
    assert result["id6"].item() == 4_762_459_723
    assert result["v3"].item() == pytest.approx(4766795.205196999)


def test_filter2(h2oai_groupby_data: pl.DataFrame) -> None:
    result = (
        h2oai_groupby_data.lazy()
        .filter(~(pl.col("id1").eq_missing(pl.lit("id046"))))
        .select(
            pl.col("id6").cast(pl.Int64).sum(),
            pl.col("v3").sum(),
        )
        .collect()
    )
    assert result["id6"].item() == 470_453_297_090
    assert result["v3"].item() == pytest.approx(470202778.84258103)
