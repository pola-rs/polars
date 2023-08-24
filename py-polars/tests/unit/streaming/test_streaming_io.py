from __future__ import annotations

import unittest
from typing import TYPE_CHECKING

import pytest

import polars as pl
from polars.testing import assert_frame_equal

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


@pytest.mark.write_disk()
def test_sink_csv(io_files_path: Path, tmp_path: Path) -> None:
    source_file = io_files_path / "small.parquet"
    target_file = tmp_path / "sink.csv"

    pl.scan_parquet(source_file).sink_csv(target_file)

    with pl.StringCache():
        source_data = pl.read_parquet(source_file)
        target_data = pl.read_csv(target_file)
        assert_frame_equal(target_data, source_data)


def test_sink_csv_with_options() -> None:
    """
    Test with all possible options.

    As we already tested the main read/write functionality of the `sink_csv` method in
     the `test_sink_csv` method above, we only need to verify that all the options are
     passed into the rust-polars correctly.
    """
    df = pl.LazyFrame({"dummy": ["abc"]})
    with unittest.mock.patch.object(df, "_ldf") as ldf:
        df.sink_csv(
            "path",
            has_header=False,
            separator=";",
            line_terminator="|",
            quote="$",
            batch_size=42,
            datetime_format="%Y",
            date_format="%d",
            time_format="%H",
            float_precision=42,
            null_value="BOOM",
            quote_style="always",
            maintain_order=False,
        )

        ldf.optimization_toggle().sink_csv.assert_called_with(
            path="path",
            has_header=False,
            separator=ord(";"),
            line_terminator="|",
            quote=ord("$"),
            batch_size=42,
            datetime_format="%Y",
            date_format="%d",
            time_format="%H",
            float_precision=42,
            null_value="BOOM",
            quote_style="always",
            maintain_order=False,
        )


@pytest.mark.parametrize(("value"), ["abc", ""])
def test_sink_csv_exception_for_separator(value: str) -> None:
    df = pl.LazyFrame({"dummy": ["abc"]})
    with pytest.raises(ValueError, match="only single byte separator is allowed"):
        df.sink_csv("path", separator=value)


@pytest.mark.parametrize(("value"), ["abc", ""])
def test_sink_csv_exception_for_quote(value: str) -> None:
    df = pl.LazyFrame({"dummy": ["abc"]})
    with pytest.raises(ValueError, match="only single byte quote char is allowed"):
        df.sink_csv("path", quote=value)
