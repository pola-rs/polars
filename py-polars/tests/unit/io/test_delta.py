from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.fs
import pytest
from deltalake import DeltaTable
from deltalake.exceptions import DeltaError, TableNotFoundError
from deltalake.table import TableMerger

import polars as pl
from polars.testing import assert_frame_equal, assert_frame_not_equal


@pytest.fixture()
def delta_table_path(io_files_path: Path) -> Path:
    return io_files_path / "delta-table"


def test_scan_delta(delta_table_path: Path) -> None:
    ldf = pl.scan_delta(str(delta_table_path), version=0)

    expected = pl.DataFrame({"name": ["Joey", "Ivan"], "age": [14, 32]})
    assert_frame_equal(expected, ldf.collect(), check_dtypes=False)


def test_scan_delta_version(delta_table_path: Path) -> None:
    df1 = pl.scan_delta(str(delta_table_path), version=0).collect()
    df2 = pl.scan_delta(str(delta_table_path), version=1).collect()

    assert_frame_not_equal(df1, df2)


@pytest.mark.write_disk()
def test_scan_delta_timestamp_version(tmp_path: Path) -> None:
    df_sample = pl.DataFrame({"name": ["Joey"], "age": [14]})
    df_sample.write_delta(tmp_path, mode="append")

    df_sample2 = pl.DataFrame({"name": ["Ivan"], "age": [34]})
    df_sample2.write_delta(tmp_path, mode="append")

    log_dir = tmp_path / "_delta_log"
    log_mtime_pair = [
        ("00000000000000000000.json", datetime(2010, 1, 1).timestamp()),
        ("00000000000000000001.json", datetime(2024, 1, 1).timestamp()),
    ]
    for file_name, dt_epoch in log_mtime_pair:
        file_path = log_dir / file_name
        os.utime(str(file_path), (dt_epoch, dt_epoch))

    df1 = pl.scan_delta(
        str(tmp_path), version=datetime(2010, 1, 1, tzinfo=timezone.utc)
    ).collect()
    df2 = pl.scan_delta(
        str(tmp_path), version=datetime(2024, 1, 1, tzinfo=timezone.utc)
    ).collect()

    assert_frame_equal(df1, df_sample)
    assert_frame_equal(df2, pl.concat([df_sample, df_sample2]), check_row_order=False)


def test_scan_delta_columns(delta_table_path: Path) -> None:
    ldf = pl.scan_delta(str(delta_table_path), version=0).select("name")

    expected = pl.DataFrame({"name": ["Joey", "Ivan"]})
    assert_frame_equal(expected, ldf.collect(), check_dtypes=False)


def test_scan_delta_filesystem(delta_table_path: Path) -> None:
    raw_filesystem = pyarrow.fs.LocalFileSystem()
    fs = pyarrow.fs.SubTreeFileSystem(str(delta_table_path), raw_filesystem)

    ldf = pl.scan_delta(
        str(delta_table_path), version=0, pyarrow_options={"filesystem": fs}
    )

    expected = pl.DataFrame({"name": ["Joey", "Ivan"], "age": [14, 32]})
    assert_frame_equal(expected, ldf.collect(), check_dtypes=False)


def test_scan_delta_relative(delta_table_path: Path) -> None:
    rel_delta_table_path = str(delta_table_path / ".." / "delta-table")

    ldf = pl.scan_delta(rel_delta_table_path, version=0)

    expected = pl.DataFrame({"name": ["Joey", "Ivan"], "age": [14, 32]})
    assert_frame_equal(expected, ldf.collect(), check_dtypes=False)

    ldf = pl.scan_delta(rel_delta_table_path, version=1)
    assert_frame_not_equal(expected, ldf.collect())


def test_read_delta(delta_table_path: Path) -> None:
    df = pl.read_delta(str(delta_table_path), version=0)

    expected = pl.DataFrame({"name": ["Joey", "Ivan"], "age": [14, 32]})
    assert_frame_equal(expected, df, check_dtypes=False)


def test_read_delta_version(delta_table_path: Path) -> None:
    df1 = pl.read_delta(str(delta_table_path), version=0)
    df2 = pl.read_delta(str(delta_table_path), version=1)

    assert_frame_not_equal(df1, df2)


@pytest.mark.write_disk()
def test_read_delta_timestamp_version(tmp_path: Path) -> None:
    df_sample = pl.DataFrame({"name": ["Joey"], "age": [14]})
    df_sample.write_delta(tmp_path, mode="append")

    df_sample2 = pl.DataFrame({"name": ["Ivan"], "age": [34]})
    df_sample2.write_delta(tmp_path, mode="append")

    log_dir = tmp_path / "_delta_log"
    log_mtime_pair = [
        ("00000000000000000000.json", datetime(2010, 1, 1).timestamp()),
        ("00000000000000000001.json", datetime(2024, 1, 1).timestamp()),
    ]
    for file_name, dt_epoch in log_mtime_pair:
        file_path = log_dir / file_name
        os.utime(str(file_path), (dt_epoch, dt_epoch))

    df1 = pl.read_delta(
        str(tmp_path), version=datetime(2010, 1, 1, tzinfo=timezone.utc)
    )
    df2 = pl.read_delta(
        str(tmp_path), version=datetime(2024, 1, 1, tzinfo=timezone.utc)
    )

    assert_frame_equal(df1, df_sample)
    assert_frame_equal(df2, pl.concat([df_sample, df_sample2]), check_row_order=False)


def test_read_delta_columns(delta_table_path: Path) -> None:
    df = pl.read_delta(str(delta_table_path), version=0, columns=["name"])

    expected = pl.DataFrame({"name": ["Joey", "Ivan"]})
    assert_frame_equal(expected, df, check_dtypes=False)


def test_read_delta_filesystem(delta_table_path: Path) -> None:
    raw_filesystem = pyarrow.fs.LocalFileSystem()
    fs = pyarrow.fs.SubTreeFileSystem(str(delta_table_path), raw_filesystem)

    df = pl.read_delta(
        str(delta_table_path), version=0, pyarrow_options={"filesystem": fs}
    )

    expected = pl.DataFrame({"name": ["Joey", "Ivan"], "age": [14, 32]})
    assert_frame_equal(expected, df, check_dtypes=False)


def test_read_delta_relative(delta_table_path: Path) -> None:
    rel_delta_table_path = str(delta_table_path / ".." / "delta-table")

    df = pl.read_delta(rel_delta_table_path, version=0)

    expected = pl.DataFrame({"name": ["Joey", "Ivan"], "age": [14, 32]})
    assert_frame_equal(expected, df, check_dtypes=False)


@pytest.mark.write_disk()
def test_write_delta(df: pl.DataFrame, tmp_path: Path) -> None:
    v0 = df.select(pl.col(pl.String))
    v1 = df.select(pl.col(pl.Int64))
    df_supported = df.drop(["cat", "enum", "time"])

    # Case: Success (version 0)
    v0.write_delta(tmp_path)

    # Case: Error if table exists
    with pytest.raises(DeltaError, match="A table already exists"):
        v0.write_delta(tmp_path)

    # Case: Overwrite with new version (version 1)
    v1.write_delta(
        tmp_path, mode="overwrite", delta_write_options={"schema_mode": "overwrite"}
    )

    # Case: Error if schema contains unsupported columns
    with pytest.raises(TypeError):
        df.write_delta(
            tmp_path, mode="overwrite", delta_write_options={"schema_mode": "overwrite"}
        )

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


@pytest.mark.write_disk()
def test_write_delta_overwrite_schema_deprecated(
    df: pl.DataFrame, tmp_path: Path
) -> None:
    df = df.select(pl.col(pl.Int64))
    with pytest.deprecated_call():
        df.write_delta(tmp_path, mode="overwrite", overwrite_schema=True)
    result = pl.read_delta(str(tmp_path))
    assert_frame_equal(df, result)


@pytest.mark.write_disk()
@pytest.mark.parametrize(
    "series",
    [
        pl.Series("string", ["test"], dtype=pl.String),
        pl.Series("uint", [1], dtype=pl.UInt64),
        pl.Series("int", [1], dtype=pl.Int64),
        pl.Series(
            "uint_list",
            [[[[[1, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 3]]]]],
            dtype=pl.List(pl.List(pl.List(pl.List(pl.UInt16)))),
        ),
        pl.Series(
            "date_ns", [datetime(2010, 1, 1, 0, 0)], dtype=pl.Datetime(time_unit="ns")
        ).dt.replace_time_zone("Australia/Lord_Howe"),
        pl.Series(
            "date_us",
            [datetime(2010, 1, 1, 0, 0)],
            dtype=pl.Datetime(time_unit="us"),
        ),
        pl.Series(
            "list_date",
            [
                [
                    datetime(2010, 1, 1, 0, 0),
                    datetime(2010, 1, 2, 0, 0),
                ]
            ],
            dtype=pl.List(pl.Datetime(time_unit="ns")),
        ),
        pl.Series(
            "list_date_us",
            [
                [
                    datetime(2010, 1, 1, 0, 0),
                    datetime(2010, 1, 2, 0, 0),
                ]
            ],
            dtype=pl.List(pl.Datetime(time_unit="ms")),
        ),
        pl.Series(
            "nested_list_date",
            [
                [
                    [
                        datetime(2010, 1, 1, 0, 0),
                        datetime(2010, 1, 2, 0, 0),
                    ]
                ]
            ],
            dtype=pl.List(pl.List(pl.Datetime(time_unit="ns"))),
        ),
        pl.Series(
            "struct_with_list",
            [
                {
                    "date_range": [
                        datetime(2010, 1, 1, 0, 0),
                        datetime(2010, 1, 2, 0, 0),
                    ],
                    "date_us": [
                        datetime(2010, 1, 1, 0, 0),
                        datetime(2010, 1, 2, 0, 0),
                    ],
                    "date_range_nested": [
                        [
                            datetime(2010, 1, 1, 0, 0),
                            datetime(2010, 1, 2, 0, 0),
                        ]
                    ],
                    "string": "test",
                    "int": 1,
                }
            ],
            dtype=pl.Struct(
                [
                    pl.Field(
                        "date_range",
                        pl.List(pl.Datetime(time_unit="ms", time_zone="UTC")),
                    ),
                    pl.Field(
                        "date_us", pl.List(pl.Datetime(time_unit="ms", time_zone=None))
                    ),
                    pl.Field(
                        "date_range_nested",
                        pl.List(pl.List(pl.Datetime(time_unit="ms", time_zone=None))),
                    ),
                    pl.Field("string", pl.String),
                    pl.Field("int", pl.UInt32),
                ]
            ),
        ),
        pl.Series(
            "list_with_struct_with_list",
            [
                [
                    {
                        "date_range": [
                            datetime(2010, 1, 1, 0, 0),
                            datetime(2010, 1, 2, 0, 0),
                        ],
                        "date_ns": [
                            datetime(2010, 1, 1, 0, 0),
                            datetime(2010, 1, 2, 0, 0),
                        ],
                        "date_range_nested": [
                            [
                                datetime(2010, 1, 1, 0, 0),
                                datetime(2010, 1, 2, 0, 0),
                            ]
                        ],
                        "string": "test",
                        "int": 1,
                    }
                ]
            ],
            dtype=pl.List(
                pl.Struct(
                    [
                        pl.Field(
                            "date_range",
                            pl.List(pl.Datetime(time_unit="ns", time_zone=None)),
                        ),
                        pl.Field(
                            "date_ns",
                            pl.List(pl.Datetime(time_unit="ns", time_zone=None)),
                        ),
                        pl.Field(
                            "date_range_nested",
                            pl.List(
                                pl.List(pl.Datetime(time_unit="ns", time_zone=None))
                            ),
                        ),
                        pl.Field("string", pl.String),
                        pl.Field("int", pl.UInt32),
                    ]
                )
            ),
        ),
    ],
)
def test_write_delta_w_compatible_schema(series: pl.Series, tmp_path: Path) -> None:
    df = series.to_frame()

    # Create table
    df.write_delta(tmp_path, mode="append")

    # Write to table again, should pass with reconstructed schema
    df.write_delta(tmp_path, mode="append")

    tbl = DeltaTable(tmp_path)
    assert tbl.version() == 1


@pytest.mark.write_disk()
def test_write_delta_with_schema_10540(tmp_path: Path) -> None:
    df = pl.DataFrame({"a": [1, 2, 3]})

    pa_schema = pa.schema([("a", pa.int64())])
    df.write_delta(tmp_path, delta_write_options={"schema": pa_schema})


@pytest.mark.write_disk()
@pytest.mark.parametrize(
    "expr",
    [
        pl.datetime(2010, 1, 1, time_unit="us", time_zone="UTC"),
        pl.datetime(2010, 1, 1, time_unit="ns", time_zone="EST"),
        pl.datetime(2010, 1, 1, time_unit="ms", time_zone="Europe/Amsterdam"),
    ],
)
def test_write_delta_with_tz_in_df(expr: pl.Expr, tmp_path: Path) -> None:
    df = pl.select(expr)

    expected_dtype = pl.Datetime("us", "UTC")
    expected = pl.select(expr.cast(expected_dtype))

    df.write_delta(tmp_path, mode="append")
    # write second time because delta-rs also casts timestamp with tz to timestamp no tz
    df.write_delta(tmp_path, mode="append")

    # Check schema of DeltaTable object
    tbl = DeltaTable(tmp_path)
    assert tbl.schema().to_pyarrow() == expected.to_arrow().schema

    # Check result
    result = pl.read_delta(str(tmp_path), version=0)
    assert_frame_equal(result, expected)


def test_write_delta_with_merge_and_no_table(tmp_path: Path) -> None:
    df = pl.DataFrame({"a": [1, 2, 3]})

    with pytest.raises(TableNotFoundError):
        df.write_delta(
            tmp_path, mode="merge", delta_merge_options={"predicate": "a = a"}
        )


@pytest.mark.write_disk()
def test_write_delta_with_merge(tmp_path: Path) -> None:
    df = pl.DataFrame({"a": [1, 2, 3]})

    df.write_delta(tmp_path)

    merger = df.write_delta(
        tmp_path,
        mode="merge",
        delta_merge_options={
            "predicate": "s.a = t.a",
            "source_alias": "s",
            "target_alias": "t",
        },
    )

    assert isinstance(merger, TableMerger)
    assert merger.predicate == "s.a = t.a"
    assert merger.source_alias == "s"
    assert merger.target_alias == "t"

    merger.when_matched_delete(predicate="t.a > 2").execute()

    result = pl.read_delta(str(tmp_path))

    expected = df.filter(pl.col("a") <= 2)
    assert_frame_equal(result, expected, check_row_order=False)


@pytest.mark.write_disk()
def test_unsupported_dtypes(tmp_path: Path) -> None:
    df = pl.DataFrame({"a": [None]}, schema={"a": pl.Null})
    with pytest.raises(TypeError, match="unsupported data type"):
        df.write_delta(tmp_path / "null")

    df = pl.DataFrame({"a": [123]}, schema={"a": pl.Time})
    with pytest.raises(TypeError, match="unsupported data type"):
        df.write_delta(tmp_path / "time")


@pytest.mark.write_disk()
def test_categorical_becomes_string(tmp_path: Path) -> None:
    df = pl.DataFrame({"a": ["A", "B", "A"]}, schema={"a": pl.Categorical})
    df.write_delta(tmp_path)
    df2 = pl.read_delta(str(tmp_path))
    assert_frame_equal(df2, pl.DataFrame({"a": ["A", "B", "A"]}, schema={"a": pl.Utf8}))


@pytest.mark.write_disk()
@pytest.mark.parametrize("rechunk_and_expected_chunks", [(True, 1), (False, 3)])
def test_read_parquet_respects_rechunk_16982(
    rechunk_and_expected_chunks: tuple[bool, int], tmp_path: Path
) -> None:
    # Create a delta lake table with 3 chunks:
    df = pl.DataFrame({"a": [1]})
    df.write_delta(str(tmp_path))
    df.write_delta(str(tmp_path), mode="append")
    df.write_delta(str(tmp_path), mode="append")

    rechunk, expected_chunks = rechunk_and_expected_chunks
    result = pl.read_delta(str(tmp_path), rechunk=rechunk)
    assert result.n_chunks() == expected_chunks
