from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import polars as pl

if TYPE_CHECKING:
    from pathlib import Path


def _write_bloom_parquet(path: Path, row_groups: list[pa.Table], column: str) -> None:
    writer = pq.ParquetWriter(
        path,
        row_groups[0].schema,
        bloom_filter_options={column: {"ndv": 16, "fpp": 0.01}},
    )
    with writer:
        for rg in row_groups:
            writer.write_table(rg)


@pytest.mark.write_disk
def test_int8_bloom_prune_finds_present_value(tmp_path: Path) -> None:
    path = tmp_path / "int8_bloom.parquet"
    rg_with_needle = pa.table({"x": pa.array([42, 0, 1], type=pa.int8())})
    rg_without_needle = pa.table({"x": pa.array([2, 3, 4], type=pa.int8())})
    _write_bloom_parquet(path, [rg_with_needle, rg_without_needle], "x")

    got = (
        pl.scan_parquet(path)
        .filter(pl.col("x") == 42)
        .collect(engine="streaming")
    )
    assert got.to_dict(as_series=False) == {"x": [42]}


@pytest.mark.write_disk
def test_int16_bloom_prune_finds_present_value(tmp_path: Path) -> None:
    path = tmp_path / "int16_bloom.parquet"
    rg_with_needle = pa.table({"x": pa.array([-100, 0, 1], type=pa.int16())})
    rg_without_needle = pa.table({"x": pa.array([2, 3, 4], type=pa.int16())})
    _write_bloom_parquet(path, [rg_with_needle, rg_without_needle], "x")

    got = (
        pl.scan_parquet(path)
        .filter(pl.col("x") == -100)
        .collect(engine="streaming")
    )
    assert got.to_dict(as_series=False) == {"x": [-100]}
