from pathlib import Path

import pyarrow.fs
import pytest

import polars as pl
from polars.testing import assert_frame_equal, assert_frame_not_equal


@pytest.fixture()
def delta_table_path(io_files_path: Path) -> Path:
    return io_files_path / "delta-table"


def test_scan_delta(delta_table_path: Path) -> None:
    ldf = pl.scan_delta(str(delta_table_path), version=0)

    expected = pl.DataFrame({"name": ["Joey", "Ivan"], "age": [14, 32]})
    assert_frame_equal(expected, ldf.collect(), check_dtype=False)


def test_scan_delta_version(delta_table_path: Path) -> None:
    df1 = pl.scan_delta(str(delta_table_path), version=0).collect()
    df2 = pl.scan_delta(str(delta_table_path), version=1).collect()

    assert_frame_not_equal(df1, df2)


def test_scan_delta_columns(delta_table_path: Path) -> None:
    ldf = pl.scan_delta(str(delta_table_path), version=0).select("name")

    expected = pl.DataFrame({"name": ["Joey", "Ivan"]})
    assert_frame_equal(expected, ldf.collect(), check_dtype=False)


def test_scan_delta_filesystem(delta_table_path: Path) -> None:
    fs = pyarrow.fs.LocalFileSystem()
    ldf = pl.scan_delta(str(delta_table_path), version=0, raw_filesystem=fs)

    expected = pl.DataFrame({"name": ["Joey", "Ivan"], "age": [14, 32]})
    assert_frame_equal(expected, ldf.collect(), check_dtype=False)


def test_scan_delta_relative(delta_table_path: Path) -> None:
    rel_delta_table_path = str(delta_table_path / ".." / "delta-table")

    ldf = pl.scan_delta(rel_delta_table_path, version=0)

    expected = pl.DataFrame({"name": ["Joey", "Ivan"], "age": [14, 32]})
    assert_frame_equal(expected, ldf.collect(), check_dtype=False)

    ldf = pl.scan_delta(rel_delta_table_path, version=1)
    assert_frame_not_equal(expected, ldf.collect())


def test_read_delta(delta_table_path: Path) -> None:
    df = pl.read_delta(str(delta_table_path), version=0)

    expected = pl.DataFrame({"name": ["Joey", "Ivan"], "age": [14, 32]})
    assert_frame_equal(expected, df, check_dtype=False)


def test_read_delta_version(delta_table_path: Path) -> None:
    df1 = pl.read_delta(str(delta_table_path), version=0)
    df2 = pl.read_delta(str(delta_table_path), version=1)

    assert_frame_not_equal(df1, df2)


def test_read_delta_columns(delta_table_path: Path) -> None:
    df = pl.read_delta(str(delta_table_path), version=0, columns=["name"])

    expected = pl.DataFrame({"name": ["Joey", "Ivan"]})
    assert_frame_equal(expected, df, check_dtype=False)


def test_read_delta_filesystem(delta_table_path: Path) -> None:
    raw_filesystem = pyarrow.fs.LocalFileSystem()
    fs = pyarrow.fs.SubTreeFileSystem(str(delta_table_path), raw_filesystem)

    df = pl.read_delta(
        str(delta_table_path), version=0, pyarrow_options={"filesystem": fs}
    )

    expected = pl.DataFrame({"name": ["Joey", "Ivan"], "age": [14, 32]})
    assert_frame_equal(expected, df, check_dtype=False)


def test_read_delta_relative(delta_table_path: Path) -> None:
    rel_delta_table_path = str(delta_table_path / ".." / "delta-table")

    df = pl.read_delta(rel_delta_table_path, version=0)

    expected = pl.DataFrame({"name": ["Joey", "Ivan"], "age": [14, 32]})
    assert_frame_equal(expected, df, check_dtype=False)
