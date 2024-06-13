"""Benchmark tests for the I/O operations."""

from pathlib import Path

import pytest

import polars as pl

pytestmark = pytest.mark.benchmark()


def test_write_read_scan_large_csv(groupby_data: pl.DataFrame, tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)

    data_path = tmp_path / "data.csv"
    groupby_data.write_csv(data_path)

    predicate = pl.col("v2") < 5

    shape_eager = pl.read_csv(data_path).filter(predicate).shape
    shape_lazy = pl.scan_csv(data_path).filter(predicate).collect().shape

    assert shape_lazy == shape_eager
