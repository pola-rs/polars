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
def test_sink_parquet(io_files_path: Path, tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)

    file = io_files_path / "small.parquet"

    file_path = tmp_path / "sink.parquet"

    df_scanned = pl.scan_parquet(file)
    df_scanned.sink_parquet(file_path)

    with pl.StringCache():
        result = pl.read_parquet(file_path)
        df_read = pl.read_parquet(file)
        assert_frame_equal(result, df_read)


@pytest.mark.write_disk()
def test_sink_parquet_10115(tmp_path: Path) -> None:
    in_path = tmp_path / "in.parquet"
    out_path = tmp_path / "out.parquet"

    # this fails if the schema will be incorrectly due to the projection
    # pushdown
    (pl.DataFrame([{"x": 1, "y": "foo"}]).write_parquet(in_path))

    joiner = pl.LazyFrame([{"y": "foo", "z": "_"}])

    (
        pl.scan_parquet(in_path)
        .join(joiner, how="left", on="y")
        .select("x", "y", "z")
        .sink_parquet(out_path)  #
    )

    assert pl.read_parquet(out_path).to_dict(False) == {
        "x": [1],
        "y": ["foo"],
        "z": ["_"],
    }


@pytest.mark.write_disk()
def test_sink_ipc(io_files_path: Path, tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)

    file = io_files_path / "small.parquet"

    file_path = tmp_path / "sink.ipc"

    df_scanned = pl.scan_parquet(file)
    df_scanned.sink_ipc(file_path)

    with pl.StringCache():
        result = pl.read_ipc(file_path)
        df_read = pl.read_parquet(file)
        assert_frame_equal(result, df_read)


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
            quote_char="$",
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
            quote_char=ord("$"),
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
    with pytest.raises(ValueError, match="should be a single byte character, but is"):
        df.sink_csv("path", separator=value)


@pytest.mark.parametrize(("value"), ["abc", ""])
def test_sink_csv_exception_for_quote(value: str) -> None:
    df = pl.LazyFrame({"dummy": ["abc"]})
    with pytest.raises(ValueError, match="should be a single byte character, but is"):
        df.sink_csv("path", quote_char=value)


def test_scan_csv_only_header_10792(io_files_path: Path) -> None:
    foods_file_path = io_files_path / "only_header.csv"
    df = pl.scan_csv(foods_file_path).collect(streaming=True)
    assert df.to_dict(False) == {"Name": [], "Address": []}


def test_scan_empty_csv_10818(io_files_path: Path) -> None:
    empty_file_path = io_files_path / "empty.csv"
    df = pl.scan_csv(empty_file_path, raise_if_empty=False).collect(streaming=True)
    assert df.is_empty()
