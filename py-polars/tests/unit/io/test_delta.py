from __future__ import annotations

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
    raw_filesystem = pyarrow.fs.LocalFileSystem()
    fs = pyarrow.fs.SubTreeFileSystem(str(delta_table_path), raw_filesystem)

    ldf = pl.scan_delta(
        str(delta_table_path), version=0, pyarrow_options={"filesystem": fs}
    )

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


@pytest.mark.write_disk()
def test_write_delta(df: pl.DataFrame, tmp_path: Path) -> None:
    from deltalake import DeltaTable

    v0 = df.select(pl.col(pl.Utf8))
    v1 = df.select(pl.col(pl.Int64))
    df_supported = df.drop(["cat", "time"])

    # Case: Success (version 0)
    v0.write_delta(tmp_path)

    # Case: Error if table exists
    with pytest.raises(ValueError):
        v1.write_delta(tmp_path)

    # Case: Overwrite with new version (version 1)
    v1.write_delta(tmp_path, mode="overwrite", overwrite_schema=True)

    # Case: Error if schema contains unsupported columns
    with pytest.raises(TypeError):
        df.write_delta(tmp_path, mode="overwrite", overwrite_schema=True)

    partitioned_tbl_uri = (tmp_path / ".." / "partitioned_table").resolve()

    # Case: Write new partitioned table (version 0)
    df_supported.write_delta(
        partitioned_tbl_uri, delta_write_options={"partition_by": "strings"}
    )

    # Case: Read back
    tbl = DeltaTable(tmp_path)
    partitioned_tbl = DeltaTable(partitioned_tbl_uri)

    pl_df_0 = pl.read_delta(tbl.table_uri, version=0)
    pl_df_1 = pl.read_delta(tbl.table_uri, version=1)
    pl_df_partitioned = pl.read_delta(str(partitioned_tbl_uri))

    assert v0.shape == pl_df_0.shape
    assert v0.columns == pl_df_0.columns
    assert v1.shape == pl_df_1.shape
    assert v1.columns == pl_df_1.columns

    assert df_supported.shape == pl_df_partitioned.shape
    assert df_supported.columns == pl_df_partitioned.columns

    assert tbl.version() == 1
    assert partitioned_tbl.version() == 0
    assert Path(partitioned_tbl.table_uri) == partitioned_tbl_uri
    assert partitioned_tbl.metadata().partition_columns == ["strings"]

    assert_frame_equal(v0, pl_df_0, check_row_order=False)
    assert_frame_equal(v1, pl_df_1, check_row_order=False)

    cols = [c for c in df_supported.columns if not c.startswith("list_")]
    assert_frame_equal(
        df_supported.select(cols),
        pl_df_partitioned.select(cols),
        check_row_order=False,
    )

    # Case: Append to existing tables
    v1.write_delta(tmp_path, mode="append")
    tbl = DeltaTable(tmp_path)
    pl_df_1 = pl.read_delta(tbl.table_uri, version=2)

    assert tbl.version() == 2
    assert pl_df_1.shape == (6, 2)  # Rows are doubled
    assert v1.columns == pl_df_1.columns

    df_supported.write_delta(partitioned_tbl_uri, mode="append")
    partitioned_tbl = DeltaTable(partitioned_tbl_uri)
    pl_df_partitioned = pl.read_delta(partitioned_tbl.table_uri, version=1)

    assert partitioned_tbl.version() == 1
    assert pl_df_partitioned.shape == (6, 14)  # Rows are doubled
    assert df_supported.columns == pl_df_partitioned.columns

    df_supported.write_delta(partitioned_tbl_uri, mode="overwrite")
