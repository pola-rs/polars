from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import polars as pl

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.xdist_group("streaming")


@pytest.mark.write_disk()
def test_streaming_parquet_glob_5900(df: pl.DataFrame, tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)
    file_path = tmp_path / "small.parquet"
    df.write_parquet(file_path)

    path_glob = tmp_path / "small*.parquet"
    result = pl.scan_parquet(path_glob).select(pl.all().first()).collect(streaming=True)
    assert result.shape == (1, 16)


def test_scan_slice_streaming(io_files_path: Path) -> None:
    foods_file_path = io_files_path / "foods1.csv"
    df = pl.scan_csv(foods_file_path).head(5).collect(streaming=True)
    assert df.shape == (5, 4)


@pytest.mark.parametrize("dtype", [pl.Int8, pl.UInt8, pl.Int16, pl.UInt16])
def test_scan_csv_overwrite_small_dtypes(
    io_files_path: Path, dtype: pl.DataType
) -> None:
    file_path = io_files_path / "foods1.csv"
    df = pl.scan_csv(file_path, dtypes={"sugars_g": dtype}).collect(streaming=True)
    assert df.dtypes == [pl.Utf8, pl.Int64, pl.Float64, dtype]
