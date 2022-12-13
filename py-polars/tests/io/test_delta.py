from pathlib import Path

import pyarrow.fs

import polars as pl
from polars.testing import assert_frame_equal


def test_scan_delta() -> None:
    table_path = str(Path(__file__).parent.parent / "files" / "delta-table")
    ldf = pl.scan_delta(table_path, version=0)

    expected = pl.DataFrame({"name": ["Joey", "Ivan"], "age": [14, 32]})
    assert_frame_equal(expected, ldf.collect(), check_dtype=False)


def test_scan_delta_version() -> None:
    table_path = Path(__file__).parent.parent / "files" / "delta-table"
    df1 = pl.scan_delta(str(table_path), version=0).collect()
    df2 = pl.scan_delta(str(table_path), version=1).collect()

    assert not df1.frame_equal(df2)


def test_scan_delta_columns() -> None:
    table_path = str(Path(__file__).parent.parent / "files" / "delta-table")
    ldf = pl.scan_delta(table_path, version=0).select("name")

    expected = pl.DataFrame({"name": ["Joey", "Ivan"]})
    assert_frame_equal(expected, ldf.collect(), check_dtype=False)


def test_scan_delta_filesystem() -> None:
    table_path = str(Path(__file__).parent.parent / "files" / "delta-table")
    fs = pyarrow.fs.LocalFileSystem()
    ldf = pl.scan_delta(table_path, version=0, raw_filesystem=fs)

    expected = pl.DataFrame({"name": ["Joey", "Ivan"], "age": [14, 32]})
    assert_frame_equal(expected, ldf.collect(), check_dtype=False)


def test_scan_delta_relative() -> None:
    table_path = Path(__file__).parent.parent / "files" / "delta-table"
    rel_table_path = str(table_path / ".." / "delta-table")

    ldf = pl.scan_delta(rel_table_path, version=0)

    expected = pl.DataFrame({"name": ["Joey", "Ivan"], "age": [14, 32]})
    assert_frame_equal(expected, ldf.collect(), check_dtype=False)

    ldf = pl.scan_delta(rel_table_path, version=1)
    assert not expected.frame_equal(ldf.collect())


def test_read_delta() -> None:
    table_path = str(Path(__file__).parent.parent / "files" / "delta-table")
    df = pl.read_delta(table_path, version=0)

    expected = pl.DataFrame({"name": ["Joey", "Ivan"], "age": [14, 32]})
    assert_frame_equal(expected, df, check_dtype=False)


def test_read_delta_version() -> None:
    table_path = Path(__file__).parent.parent / "files" / "delta-table"
    df1 = pl.read_delta(str(table_path), version=0)
    df2 = pl.read_delta(str(table_path), version=1)

    assert not df1.frame_equal(df2)


def test_read_delta_columns() -> None:
    table_path = str(Path(__file__).parent.parent / "files" / "delta-table")
    df = pl.read_delta(table_path, version=0, columns=["name"])

    expected = pl.DataFrame({"name": ["Joey", "Ivan"]})
    assert_frame_equal(expected, df, check_dtype=False)


def test_read_delta_filesystem() -> None:
    table_path = str(Path(__file__).parent.parent / "files" / "delta-table")

    raw_filesystem = pyarrow.fs.LocalFileSystem()
    fs = pyarrow.fs.SubTreeFileSystem(table_path, raw_filesystem)

    df = pl.read_delta(table_path, version=0, pyarrow_options={"filesystem": fs})

    expected = pl.DataFrame({"name": ["Joey", "Ivan"], "age": [14, 32]})
    assert_frame_equal(expected, df, check_dtype=False)


def test_read_delta_relative() -> None:
    table_path = Path(__file__).parent.parent / "files" / "delta-table"
    rel_table_path = str(table_path / ".." / "delta-table")

    df = pl.read_delta(rel_table_path, version=0)

    expected = pl.DataFrame({"name": ["Joey", "Ivan"], "age": [14, 32]})
    assert_frame_equal(expected, df, check_dtype=False)
