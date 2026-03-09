from __future__ import annotations

import os
import pickle
import warnings
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pytest
from deltalake import DeltaTable, write_deltalake
from deltalake.exceptions import DeltaError, TableNotFoundError
from deltalake.table import TableMerger

import polars as pl
from polars.io.cloud._utils import NoPickleOption
from polars.io.cloud.credential_provider._builder import (
    _init_credential_provider_builder,
)
from polars.io.delta._dataset import DeltaDataset
from polars.io.delta._utils import _extract_table_statistics_from_delta_add_actions
from polars.testing import assert_frame_equal, assert_frame_not_equal

if TYPE_CHECKING:
    from tests.conftest import PlMonkeyPatch


@pytest.fixture
def delta_table_path(io_files_path: Path) -> Path:
    return io_files_path / "delta-table"


def new_pl_delta_dataset(source: str | DeltaTable) -> DeltaDataset:
    return DeltaDataset(
        table_=NoPickleOption(source if isinstance(source, DeltaTable) else None),
        table_uri_=source if not isinstance(source, DeltaTable) else None,
        version=None,
        storage_options=None,
        credential_provider_builder=None,
        delta_table_options=None,
        use_pyarrow=False,
        pyarrow_options=None,
        rechunk=False,
    )


def test_scan_delta(delta_table_path: Path) -> None:
    ldf = pl.scan_delta(delta_table_path, version=0)

    expected = pl.DataFrame({"name": ["Joey", "Ivan"], "age": [14, 32]})
    assert_frame_equal(expected, ldf.collect(), check_dtypes=False)


def test_scan_delta_version(delta_table_path: Path) -> None:
    df1 = pl.scan_delta(delta_table_path, version=0).collect()
    df2 = pl.scan_delta(delta_table_path, version=1).collect()

    assert_frame_not_equal(df1, df2)


@pytest.mark.write_disk
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
    ldf = pl.scan_delta(delta_table_path, version=0).select("name")

    expected = pl.DataFrame({"name": ["Joey", "Ivan"]})
    assert_frame_equal(expected, ldf.collect(), check_dtypes=False)


def test_scan_delta_polars_storage_options_keys(
    delta_table_path: Path,
    plmonkeypatch: PlMonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    plmonkeypatch.setenv("POLARS_VERBOSE_SENSITIVE", "1")
    lf = pl.scan_delta(
        delta_table_path,
        version=0,
        storage_options={
            "file_cache_ttl": 7,
            "max_retries": 3,
            "retry_timeout_ms": 9873,
            "retry_init_backoff_ms": 9874,
            "retry_max_backoff_ms": 9875,
            "retry_base_multiplier": 3.14159,
        },
        credential_provider=None,
    ).select("name")

    lf.collect()

    capture = capfd.readouterr().err

    assert "file_cache_ttl: 7" in capture

    assert (
        """\
max_retries: Some(3), \
retry_timeout: Some(9.873s), \
retry_init_backoff: Some(9.874s), \
retry_max_backoff: Some(9.875s), \
retry_base_multiplier: Some(TotalOrdWrap(3.14159)) }"""
        in capture
    )


def test_scan_delta_relative(delta_table_path: Path) -> None:
    rel_delta_table_path = str(delta_table_path / ".." / "delta-table")

    ldf = pl.scan_delta(rel_delta_table_path, version=0)

    expected = pl.DataFrame({"name": ["Joey", "Ivan"], "age": [14, 32]})
    assert_frame_equal(expected, ldf.collect(), check_dtypes=False)

    ldf = pl.scan_delta(rel_delta_table_path, version=1)
    assert_frame_not_equal(expected, ldf.collect())


def test_read_delta(delta_table_path: Path) -> None:
    df = pl.read_delta(delta_table_path, version=0)

    expected = pl.DataFrame({"name": ["Joey", "Ivan"], "age": [14, 32]})
    assert_frame_equal(expected, df, check_dtypes=False)


def test_read_delta_version(delta_table_path: Path) -> None:
    df1 = pl.read_delta(delta_table_path, version=0)
    df2 = pl.read_delta(delta_table_path, version=1)

    assert_frame_not_equal(df1, df2)


@pytest.mark.write_disk
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
    df = pl.read_delta(delta_table_path, version=0, columns=["name"])

    expected = pl.DataFrame({"name": ["Joey", "Ivan"]})
    assert_frame_equal(expected, df, check_dtypes=False)


def test_read_delta_relative(delta_table_path: Path) -> None:
    rel_delta_table_path = str(delta_table_path / ".." / "delta-table")

    df = pl.read_delta(rel_delta_table_path, version=0)

    expected = pl.DataFrame({"name": ["Joey", "Ivan"], "age": [14, 32]})
    assert_frame_equal(expected, df, check_dtypes=False)


@pytest.mark.write_disk
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
    pl_df_partitioned = pl.read_delta(partitioned_tbl_uri)

    assert v0.shape == pl_df_0.shape
    assert v0.columns == pl_df_0.columns
    assert v1.shape == pl_df_1.shape
    assert v1.columns == pl_df_1.columns

    assert df_supported.shape == pl_df_partitioned.shape
    assert sorted(df_supported.columns) == sorted(pl_df_partitioned.columns)

    assert tbl.version() == 1
    assert partitioned_tbl.version() == 0

    uri = partitioned_tbl.table_uri.removeprefix("file://")
    if os.name == "nt" and uri.startswith("/"):
        uri = uri[1:]

    assert Path(uri) == partitioned_tbl_uri
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
    assert sorted(df_supported.columns) == sorted(pl_df_partitioned.columns)

    df_supported.write_delta(partitioned_tbl_uri, mode="overwrite")


@pytest.mark.write_disk
def test_sink_delta(df: pl.DataFrame, tmp_path: Path) -> None:
    v0 = df.lazy().select(pl.col(pl.String))
    v1 = df.lazy().select(pl.col(pl.Int64))
    df_supported = df.drop(["cat", "enum", "time"])

    # Case: Success (version 0)
    v0.sink_delta(tmp_path)

    # Case: Error if table exists
    with pytest.raises(DeltaError, match="A table already exists"):
        v0.sink_delta(tmp_path)

    # Case: Overwrite with new version (version 1)
    v1.sink_delta(
        tmp_path, mode="overwrite", delta_write_options={"schema_mode": "overwrite"}
    )

    # Case: Error if schema contains unsupported columns
    with pytest.raises(TypeError):
        df.lazy().sink_delta(
            tmp_path, mode="overwrite", delta_write_options={"schema_mode": "overwrite"}
        )

    partitioned_tbl_uri = (tmp_path / ".." / "partitioned_table_sink").resolve()

    # Case: Write new partitioned table (version 0)
    df_supported.lazy().sink_delta(
        partitioned_tbl_uri, delta_write_options={"partition_by": "strings"}
    )

    # Case: Read back
    tbl = DeltaTable(tmp_path)
    partitioned_tbl = DeltaTable(partitioned_tbl_uri)

    pl_df_0 = pl.read_delta(tbl.table_uri, version=0)
    pl_df_1 = pl.read_delta(tbl.table_uri, version=1)
    pl_df_partitioned = pl.read_delta(partitioned_tbl_uri)

    assert v0.collect().shape == pl_df_0.shape
    assert v0.collect_schema().names() == pl_df_0.columns
    assert v1.collect().shape == pl_df_1.shape
    assert v1.collect_schema().names() == pl_df_1.columns

    assert df_supported.shape == pl_df_partitioned.shape
    assert sorted(df_supported.columns) == sorted(pl_df_partitioned.columns)

    assert tbl.version() == 1
    assert partitioned_tbl.version() == 0

    uri = partitioned_tbl.table_uri.removeprefix("file://")
    if os.name == "nt" and uri.startswith("/"):
        uri = uri[1:]

    assert Path(uri) == partitioned_tbl_uri
    assert partitioned_tbl.metadata().partition_columns == ["strings"]

    assert_frame_equal(v0.collect(), pl_df_0, check_row_order=False)
    assert_frame_equal(v1.collect(), pl_df_1, check_row_order=False)

    cols = [c for c in df_supported.columns if not c.startswith("list_")]
    assert_frame_equal(
        df_supported.select(cols),
        pl_df_partitioned.select(cols),
        check_row_order=False,
    )

    # Case: Append to existing tables
    v1.sink_delta(tmp_path, mode="append")
    tbl = DeltaTable(tmp_path)
    pl_df_1 = pl.read_delta(tbl.table_uri, version=2)

    assert tbl.version() == 2
    assert pl_df_1.shape == (6, 2)  # Rows are doubled
    assert v1.collect_schema().names() == pl_df_1.columns

    df_supported.lazy().sink_delta(partitioned_tbl_uri, mode="append")
    partitioned_tbl = DeltaTable(partitioned_tbl_uri)
    pl_df_partitioned = pl.read_delta(partitioned_tbl.table_uri, version=1)

    assert partitioned_tbl.version() == 1
    assert pl_df_partitioned.shape == (6, 14)  # Rows are doubled
    assert sorted(df_supported.columns) == sorted(pl_df_partitioned.columns)

    df_supported.lazy().sink_delta(partitioned_tbl_uri, mode="overwrite")


@pytest.mark.write_disk
def test_write_delta_overwrite_schema_deprecated(
    df: pl.DataFrame, tmp_path: Path
) -> None:
    df = df.select(pl.col(pl.Int64))
    with pytest.deprecated_call():
        df.write_delta(tmp_path, mode="overwrite", overwrite_schema=True)
    result = pl.read_delta(tmp_path)
    assert_frame_equal(df, result)


@pytest.mark.write_disk
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


@pytest.mark.write_disk
@pytest.mark.parametrize(
    "expr",
    [
        pl.datetime(2010, 1, 1, time_unit="us", time_zone="UTC"),
        pl.datetime(2010, 1, 1, time_unit="ns", time_zone="America/New_York"),
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
    assert pa.schema(tbl.schema().to_arrow()) == expected.to_arrow().schema

    # Check result
    result = pl.read_delta(tmp_path, version=0)
    assert_frame_equal(result, expected)


def test_write_delta_with_merge_and_no_table(tmp_path: Path) -> None:
    df = pl.DataFrame({"a": [1, 2, 3]})

    with pytest.raises(TableNotFoundError):
        df.write_delta(
            tmp_path, mode="merge", delta_merge_options={"predicate": "a = a"}
        )


@pytest.mark.write_disk
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
    assert merger._builder.source_alias == "s"
    assert merger._builder.target_alias == "t"

    merger.when_matched_delete(predicate="t.a > 2").execute()

    result = pl.read_delta(tmp_path)

    expected = df.filter(pl.col("a") <= 2)
    assert_frame_equal(result, expected, check_row_order=False)


@pytest.mark.write_disk
def test_unsupported_dtypes(tmp_path: Path) -> None:
    df = pl.DataFrame({"a": [None]}, schema={"a": pl.Null})
    with pytest.raises(TypeError, match="unsupported data type"):
        df.write_delta(tmp_path / "null")

    df = pl.DataFrame({"a": [123]}, schema={"a": pl.Time})
    with pytest.raises(TypeError, match="unsupported data type"):
        df.write_delta(tmp_path / "time")


@pytest.mark.skip(
    reason="upstream bug in delta-rs causing categorical to be written as categorical in parquet"
)
@pytest.mark.write_disk
def test_categorical_becomes_string(tmp_path: Path) -> None:
    df = pl.DataFrame({"a": ["A", "B", "A"]}, schema={"a": pl.Categorical})
    df.write_delta(tmp_path)
    df2 = pl.read_delta(tmp_path)
    assert_frame_equal(df2, pl.DataFrame({"a": ["A", "B", "A"]}, schema={"a": pl.Utf8}))


def test_scan_delta_DT_input(delta_table_path: Path) -> None:
    DT = DeltaTable(delta_table_path, version=0)
    ldf = pl.scan_delta(DT)

    expected = pl.DataFrame({"name": ["Joey", "Ivan"], "age": [14, 32]})
    assert_frame_equal(expected, ldf.collect(), check_dtypes=False)


@pytest.mark.write_disk
def test_read_delta_empty(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)
    path = str(tmp_path)

    DeltaTable.create(path, pl.DataFrame(schema={"x": pl.Int64}).to_arrow().schema)
    assert_frame_equal(pl.read_delta(path), pl.DataFrame(schema={"x": pl.Int64}))


@pytest.mark.write_disk
def test_read_delta_arrow_map_type(tmp_path: Path) -> None:
    payload = [
        {"id": 1, "account_id": {17: "100.01.001 Cash"}},
        {"id": 2, "account_id": {18: "180.01.001 Cash", 19: "foo"}},
    ]

    schema = pa.schema(
        [
            pa.field("id", pa.int32()),
            pa.field("account_id", pa.map_(pa.int32(), pa.string())),
        ]
    )
    table = pa.Table.from_pylist(payload, schema)

    expect = pl.DataFrame(table)

    table_path = str(tmp_path)
    write_deltalake(
        table_path,
        table,
        mode="overwrite",
    )

    assert_frame_equal(pl.scan_delta(table_path).collect(), expect)
    assert_frame_equal(pl.read_delta(table_path), expect)


@pytest.mark.may_fail_cloud  # reason: inspects logs
@pytest.mark.write_disk
def test_scan_delta_nanosecond_timestamp(
    tmp_path: Path,
    plmonkeypatch: PlMonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    df = pl.DataFrame(
        {"timestamp": [datetime(2025, 1, 1), datetime(2025, 1, 2)]},
        schema={"timestamp": pl.Datetime("us", time_zone="UTC")},
    )

    df_nano_ts = pl.DataFrame(
        {"timestamp": [datetime(2025, 1, 1), datetime(2025, 1, 2)]},
        schema={"timestamp": pl.Datetime("ns", time_zone=None)},
    )

    root = tmp_path / "delta"

    import deltalake

    df.write_delta(
        root,
        delta_write_options={
            "writer_properties": deltalake.WriterProperties(
                default_column_properties=deltalake.ColumnProperties(
                    statistics_enabled="NONE"
                )
            )
        },
    )

    # Manually overwrite the file with one that has nanosecond timestamps.
    parquet_files = [x for x in root.iterdir() if x.suffix == ".parquet"]
    assert len(parquet_files) == 1
    parquet_file_path = parquet_files[0]

    df_nano_ts.write_parquet(parquet_file_path)

    # Baseline: The timestamp in the file is in nanoseconds.
    q = pl.scan_parquet(parquet_file_path)
    assert q.collect_schema() == {"timestamp": pl.Datetime("ns", time_zone=None)}
    assert_frame_equal(q.collect(), df_nano_ts)

    q = pl.scan_delta(root)

    assert q.collect_schema() == {"timestamp": pl.Datetime("us", time_zone="UTC")}
    assert_frame_equal(q.collect(), df)

    # Ensure row-group skipping is functioning.
    q = pl.scan_delta(root).filter(
        pl.col("timestamp")
        < pl.lit(datetime(2025, 1, 1), dtype=pl.Datetime("us", time_zone="UTC"))
    )
    plmonkeypatch.setenv("POLARS_VERBOSE", "1")
    capfd.readouterr()

    assert_frame_equal(q.collect(), df.clear())
    assert "reading 0 / 1 row groups" in capfd.readouterr().err


@pytest.mark.write_disk
def test_scan_delta_nanosecond_timestamp_nested(tmp_path: Path) -> None:
    df = pl.DataFrame(
        {
            "c1": [
                {"timestamp": datetime(2025, 1, 1)},
                {"timestamp": datetime(2025, 1, 2)},
            ]
        },
        schema={"c1": pl.Struct({"timestamp": pl.Datetime("us", time_zone="UTC")})},
    )

    df_nano_ts = pl.DataFrame(
        {
            "c1": [
                {"timestamp": datetime(2025, 1, 1)},
                {"timestamp": datetime(2025, 1, 2)},
            ]
        },
        schema={"c1": pl.Struct({"timestamp": pl.Datetime("ns", time_zone=None)})},
    )

    root = tmp_path / "delta"

    df.write_delta(root)

    # Manually overwrite the file with one that has nanosecond timestamps.
    parquet_files = [x for x in root.iterdir() if x.suffix == ".parquet"]
    assert len(parquet_files) == 1
    parquet_file_path = parquet_files[0]

    df_nano_ts.write_parquet(parquet_file_path)

    # Baseline: The timestamp in the file is in nanoseconds.
    q = pl.scan_parquet(parquet_file_path)
    assert q.collect_schema() == {
        "c1": pl.Struct({"timestamp": pl.Datetime("ns", time_zone=None)})
    }
    assert_frame_equal(q.collect(), df_nano_ts)

    q = pl.scan_delta(root)

    assert q.collect_schema() == {
        "c1": pl.Struct({"timestamp": pl.Datetime("us", time_zone="UTC")})
    }
    assert_frame_equal(q.collect(), df)


@pytest.mark.write_disk
def test_scan_delta_schema_evolution_nested_struct_field_19915(tmp_path: Path) -> None:
    (
        pl.DataFrame(
            {"a": ["test"], "properties": [{"property_key": {"item": 1}}]}
        ).write_delta(tmp_path)
    )

    (
        pl.DataFrame(
            {
                "a": ["test1"],
                "properties": [{"property_key": {"item": 50, "item2": 10}}],
            }
        ).write_delta(
            tmp_path,
            mode="append",
            delta_write_options={"schema_mode": "merge"},
        )
    )

    q = pl.scan_delta(tmp_path)

    expect = pl.DataFrame(
        {
            "a": ["test", "test1"],
            "properties": [
                {"property_key": {"item": 1, "item2": None}},
                {"property_key": {"item": 50, "item2": 10}},
            ],
        },
        schema={
            "a": pl.String,
            "properties": pl.Struct(
                {"property_key": pl.Struct({"item": pl.Int64, "item2": pl.Int64})}
            ),
        },
    )

    assert_frame_equal(q.sort("a").collect(), expect)


@pytest.mark.write_disk
def test_scan_delta_storage_options_from_delta_table(
    tmp_path: Path, plmonkeypatch: PlMonkeyPatch
) -> None:
    import polars.io.delta._dataset

    storage_options_checked = False

    def assert_scan_parquet_storage_options(*a: Any, **kw: Any) -> Any:
        nonlocal storage_options_checked

        assert kw["storage_options"] == {
            "aws_endpoint_url": "http://localhost:777",
            "aws_access_key_id": "...",
            "aws_secret_access_key": "...",
            "aws_session_token": "...",
            "endpoint_url": "...",
        }

        storage_options_checked = True

        return pl.scan_parquet(*a, **kw)

    plmonkeypatch.setattr(
        polars.io.delta._dataset, "scan_parquet", assert_scan_parquet_storage_options
    )

    df = pl.DataFrame({"a": ["test"], "properties": [{"property_key": {"item": 1}}]})

    df.write_delta(tmp_path)

    tbl = DeltaTable(
        tmp_path,
        storage_options={
            "aws_endpoint_url": "http://localhost:333",
            "aws_access_key_id": "...",
            "aws_secret_access_key": "...",
            "aws_session_token": "...",
        },
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        q = pl.scan_delta(
            tbl,
            storage_options={
                "aws_endpoint_url": "http://localhost:777",
                "endpoint_url": "...",
            },
        )

        assert_frame_equal(q.collect(), df)

    assert storage_options_checked


def test_scan_delta_loads_aws_profile_endpoint_url(
    tmp_path: Path,
    plmonkeypatch: PlMonkeyPatch,
) -> None:
    tmp_path.mkdir(exist_ok=True)

    cfg_file_path = tmp_path / "config"

    cfg_file_path.write_text("""\
[profile endpoint_333]
aws_access_key_id=A
aws_secret_access_key=A
endpoint_url = http://127.0.0.1:54321
""")

    plmonkeypatch.setenv("AWS_CONFIG_FILE", str(cfg_file_path))
    plmonkeypatch.setenv("AWS_PROFILE", "endpoint_333")

    assert (
        builder := _init_credential_provider_builder(
            "auto", "s3://.../...", storage_options=None, caller_name="test"
        )
    ) is not None

    assert isinstance(
        provider := builder.build_credential_provider(),
        pl.CredentialProviderAWS,
    )

    assert provider._can_use_as_provider()

    assert provider._storage_update_options() == {
        "endpoint_url": "http://127.0.0.1:54321"
    }

    with pytest.raises((DeltaError, OSError), match=r"http://127.0.0.1:54321"):
        pl.scan_delta("s3://.../...").collect()

    with pytest.raises((DeltaError, OSError), match=r"http://127.0.0.1:54321"):
        pl.DataFrame({"x": 1}).write_delta("s3://.../...", mode="append")


def _df_many_types() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "p": [10, 10, 20, 20, 30, 30],
            "a": [1, 2, 3, 4, 5, None],
            "bool": [False, False, True, True, True, None],
            "int": [1, 2, 3, 4, 5, None],
            "float": [1.0, 2.0, 3.0, 4.0, 5.0, None],
            "string": ["a", "b", "c", "cc", "ccc", None],
            "struct": [
                {"x": 1, "y": 10},
                {"x": 2, "y": 20},
                {"x": 3, "y": 30},
                {"x": 4, "y": 40},
                {"x": 5, "y": 50},
                None,
            ],
        }
    ).with_columns(
        decimal=pl.col.a.cast(pl.Decimal(10, 2)),
        date=pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 6), closed="both"),
        datetime=pl.datetime_range(
            pl.datetime(2020, 1, 1), pl.datetime(2020, 1, 6), closed="both"
        ),
    )


# TODO: uncomment dtype when fixed
@pytest.mark.parametrize(
    "expr",
    [
        # Bool
        # pl.col.bool == False,  ## see github issue #26290, to be confirmed
        # pl.col.bool <= False,
        # pl.col.bool < True,
        pl.col.bool.is_null(),
        # Integer
        pl.col.int == 2,
        pl.col.int <= 2,
        pl.col.int < 3,
        pl.col.int.is_null(),
        (pl.col.int < 2) & (pl.col.int.is_not_null()),
        # Float ## see github issue #26238
        # pl.col.float == 2.0,
        # pl.col.float <= 2.0,
        # pl.col.float < 3.0,
        pl.col.float.is_null(),
        # mixed
        (pl.col.int == 2) & (pl.col.float.is_not_null()),
        # String
        pl.col.string == "b",
        pl.col.string <= "b",
        pl.col.string.is_null(),
        # Decimal
        pl.col.decimal == pl.lit(2.0).cast(pl.Decimal(10, 2)),
        pl.col.decimal <= pl.lit(2.0).cast(pl.Decimal(10, 2)),
        pl.col.decimal < pl.lit(3.0).cast(pl.Decimal(10, 2)),
        pl.col.decimal.is_null(),
        # Struct # see github issue #26239
        # pl.col.struct == {"x": 2, "y": 20},
        # pl.col.struct.is_null(),
        # Date & datetime
        pl.col.date == pl.date(2020, 1, 1),
        pl.col.datetime == pl.datetime(2020, 1, 1),
        # on predicate
        pl.col.p == 10,
    ],
)
@pytest.mark.write_disk
def test_scan_delta_filter_delta_log_statistics_23780(
    tmp_path: Path,
    plmonkeypatch: PlMonkeyPatch,
    capfd: pytest.CaptureFixture[str],
    expr: pl.Expr,
) -> None:
    df = _df_many_types()
    root = tmp_path / "delta"
    df.write_delta(root, delta_write_options={"partition_by": "p"})

    plmonkeypatch.setenv("POLARS_VERBOSE", "1")
    capfd.readouterr()

    assert_frame_equal(
        pl.scan_delta(root).filter(expr).collect(),
        df.filter(expr),
        check_column_order=False,
        check_row_order=False,
    )
    assert "skipping 2 / 3 files" in capfd.readouterr().err


@pytest.mark.write_disk
def test_scan_delta_extract_table_statistics_df(tmp_path: Path) -> None:
    import datetime

    df = _df_many_types()
    root = tmp_path / "delta"
    df.write_delta(root, delta_write_options={"partition_by": "p"})

    statistics_df = _extract_table_statistics_from_delta_add_actions(
        pl.DataFrame(DeltaTable(tmp_path / "delta").get_add_actions()),
        filter_columns=df.columns,
        schema=df.schema,
        verbose=False,
    )

    assert statistics_df is not None

    assert_frame_equal(
        statistics_df,
        pl.DataFrame(
            [
                pl.Series('len', [2, 2, 2], dtype=pl.Int64),
                pl.Series('p_nc', [None, None, None], dtype=pl.UInt32),
                pl.Series('p_min', [None, None, None], dtype=pl.Int64),
                pl.Series('p_max', [None, None, None], dtype=pl.Int64),
                pl.Series('a_nc', [0, 1, 0], dtype=pl.Int64),
                pl.Series('a_min', [1, 5, 3], dtype=pl.Int64),
                pl.Series('a_max', [2, 5, 4], dtype=pl.Int64),
                pl.Series('bool_nc', [0, 1, 0], dtype=pl.Int64),
                pl.Series('bool_min', [None, None, None], dtype=pl.Boolean),
                pl.Series('bool_max', [None, None, None], dtype=pl.Boolean),
                pl.Series('int_nc', [0, 1, 0], dtype=pl.Int64),
                pl.Series('int_min', [1, 5, 3], dtype=pl.Int64),
                pl.Series('int_max', [2, 5, 4], dtype=pl.Int64),
                pl.Series('float_nc', [0, 1, 0], dtype=pl.Int64),
                pl.Series('float_min', [1.0, 5.0, 3.0], dtype=pl.Float64),
                pl.Series('float_max', [2.0, 5.0, 4.0], dtype=pl.Float64),
                pl.Series('string_nc', [0, 1, 0], dtype=pl.Int64),
                pl.Series('string_min', ['a', 'ccc', 'c'], dtype=pl.String),
                pl.Series('string_max', ['b', 'ccc', 'cc'], dtype=pl.String),
                pl.Series('struct_nc', [{'x': 0, 'y': 0}, {'x': 1, 'y': 1}, {'x': 0, 'y': 0}], dtype=pl.Struct({'x': pl.Int64, 'y': pl.Int64})),
                pl.Series('struct_min', [{'x': 1, 'y': 10}, {'x': 5, 'y': 50}, {'x': 3, 'y': 30}], dtype=pl.Struct({'x': pl.Int64, 'y': pl.Int64})),
                pl.Series('struct_max', [{'x': 2, 'y': 20}, {'x': 5, 'y': 50}, {'x': 4, 'y': 40}], dtype=pl.Struct({'x': pl.Int64, 'y': pl.Int64})),
                pl.Series('decimal_nc', [0, 1, 0], dtype=pl.Int64),
                pl.Series('decimal_min', [Decimal('1.00'), Decimal('5.00'), Decimal('3.00')], dtype=pl.Decimal(precision=10, scale=2)),
                pl.Series('decimal_max', [Decimal('2.00'), Decimal('5.00'), Decimal('4.00')], dtype=pl.Decimal(precision=10, scale=2)),
                pl.Series('date_nc', [0, 0, 0], dtype=pl.Int64),
                pl.Series('date_min', [datetime.date(2020, 1, 1), datetime.date(2020, 1, 5), datetime.date(2020, 1, 3)], dtype=pl.Date),
                pl.Series('date_max', [datetime.date(2020, 1, 2), datetime.date(2020, 1, 6), datetime.date(2020, 1, 4)], dtype=pl.Date),
                pl.Series('datetime_nc', [0, 0, 0], dtype=pl.Int64),
                pl.Series('datetime_min', [datetime.datetime(2020, 1, 1, 0, 0), datetime.datetime(2020, 1, 5, 0, 0), datetime.datetime(2020, 1, 3, 0, 0)], dtype=pl.Datetime(time_unit='us', time_zone=None)),
                pl.Series('datetime_max', [datetime.datetime(2020, 1, 2, 0, 0), datetime.datetime(2020, 1, 6, 0, 0), datetime.datetime(2020, 1, 4, 0, 0)], dtype=pl.Datetime(time_unit='us', time_zone=None)),
            ]
        ),
        check_row_order=False
    )  # fmt: skip


@pytest.mark.parametrize(
    ("expr", "n_cols", "expect_n_files_skipped"),
    [
        (pl.col.a == 2, "0", 0),
        (pl.col.a == 2, "1", 1),
        (pl.col.b == 2, "1", 0),
        (pl.col.a == 2, "2", 1),
    ],
)
@pytest.mark.write_disk
def test_scan_delta_filter_delta_log_statistics_partial_23780(
    tmp_path: Path,
    plmonkeypatch: PlMonkeyPatch,
    capfd: pytest.CaptureFixture[str],
    expr: pl.Expr,
    n_cols: str,
    expect_n_files_skipped: int,
) -> None:
    df = pl.DataFrame({"p": [10, 10, 20, 20], "a": [1, 2, 3, 4], "b": [1, 2, 3, 4]})

    root = tmp_path / "delta"
    df.write_delta(
        root,
        delta_write_options={
            "partition_by": "p",
            "configuration": {
                "delta.dataSkippingNumIndexedCols": n_cols  # Disable stats collection
            },
        },
    )

    plmonkeypatch.setenv("POLARS_VERBOSE", "1")
    capfd.readouterr()

    assert_frame_equal(
        pl.scan_delta(root).filter(expr).collect(),
        df.filter(expr),
        check_column_order=False,
        check_row_order=False,
    )
    assert f"skipping {expect_n_files_skipped} / 2 files" in capfd.readouterr().err


@pytest.mark.write_disk
def test_scan_delta_filter_delta_log_statistics_delete_partition_23780(
    tmp_path: Path,
    plmonkeypatch: PlMonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    df = pl.DataFrame(
        {
            "p": [10, 10, 20, 30],
            "a": [1, 2, 3, 4],
        }
    )
    root = tmp_path / "delta"

    df.write_delta(root, delta_write_options={"partition_by": "p"})

    plmonkeypatch.setenv("POLARS_VERBOSE", "1")
    capfd.readouterr()

    expr = pl.col.a == 2
    assert_frame_equal(
        pl.scan_delta(root).filter(expr).collect(),
        df.filter(expr),
        check_column_order=False,
        check_row_order=False,
    )
    assert "skipping 2 / 3 files" in capfd.readouterr().err

    from deltalake import DeltaTable

    dt = DeltaTable(root)
    dt.delete("p = 30")

    assert_frame_equal(
        pl.scan_delta(root).filter(expr).collect(),
        df.filter(expr),
        check_column_order=False,
        check_row_order=False,
    )
    assert "skipping 1 / 2 files" in capfd.readouterr().err


@pytest.mark.parametrize("use_pyarrow", [True, False])
@pytest.mark.write_disk
def test_scan_delta_use_pyarrow(tmp_path: Path, use_pyarrow: bool) -> None:
    df = pl.DataFrame({"year": [2025, 2026, 2026], "month": [0, 0, 0]})
    df.write_delta(tmp_path, delta_write_options={"partition_by": "year"})

    assert_frame_equal(
        pl.scan_delta(tmp_path, use_pyarrow=use_pyarrow)
        .filter(pl.col("year") == 2026)
        .collect(),
        pl.DataFrame({"year": [2026, 2026], "month": [0, 0]}),
    )

    assert_frame_equal(
        pl.scan_delta(tmp_path, use_pyarrow=use_pyarrow)
        .filter(pl.col("year") == 2026)
        .head(1)
        .collect(),
        pl.DataFrame({"year": [2026], "month": [0]}),
    )

    # Delta does not have stable file scan ordering.
    assert (
        pl.scan_delta(tmp_path, use_pyarrow=use_pyarrow).head(1).collect().height == 1
    )


@pytest.mark.parametrize("use_pyarrow", [True, False])
@pytest.mark.write_disk
def test_scan_delta_use_pyarrow_single_file(tmp_path: Path, use_pyarrow: bool) -> None:
    df = pl.DataFrame({"year": [2025, 2026, 2026], "month": [0, 0, 0]})
    df.write_delta(tmp_path)

    assert_frame_equal(
        pl.scan_delta(tmp_path, use_pyarrow=use_pyarrow)
        .filter(pl.col("year") == 2026)
        .collect(),
        pl.DataFrame({"year": [2026, 2026], "month": [0, 0]}),
    )

    assert_frame_equal(
        pl.scan_delta(tmp_path, use_pyarrow=use_pyarrow)
        .filter(pl.col("year") == 2026)
        .head(1)
        .collect(),
        pl.DataFrame({"year": [2026], "month": [0]}),
    )

    assert_frame_equal(
        pl.scan_delta(tmp_path, use_pyarrow=use_pyarrow).head(1).collect(),
        pl.DataFrame({"year": [2025], "month": [0]}),
    )

    assert_frame_equal(
        pl.scan_delta(tmp_path, use_pyarrow=use_pyarrow)
        .head(1)
        .filter(pl.col("year") == 2026)
        .collect(),
        pl.DataFrame(schema={"year": pl.Int64, "month": pl.Int64}),
    )


@pytest.mark.write_disk
def test_delta_dataset_does_not_pickle_table_object(tmp_path: Path) -> None:
    df = pl.DataFrame({"row_index": [0, 1, 2, 3, 4]})
    df.write_delta(tmp_path)

    dataset = new_pl_delta_dataset(DeltaTable(tmp_path))

    assert dataset.table_.get() is not None
    dataset = pickle.loads(pickle.dumps(dataset))
    assert dataset.table_.get() is None

    assert_frame_equal(dataset.to_dataset_scan()[0].collect(), df)  # type: ignore[index]


@pytest.mark.parametrize("use_pyarrow", [True, False])
@pytest.mark.write_disk
def test_delta_partition_filter(tmp_path: Path, use_pyarrow: bool) -> None:
    df = pl.DataFrame({"row_index": [0, 1, 2, 3, 4], "year": 2026})
    df.write_delta(tmp_path, delta_write_options={"partition_by": "year"})

    for path in DeltaTable(tmp_path).file_uris():
        Path(path).unlink()

    with pytest.raises((FileNotFoundError, OSError)):
        pl.scan_delta(tmp_path, use_pyarrow=use_pyarrow).collect()

    assert_frame_equal(
        pl.scan_delta(tmp_path, use_pyarrow=use_pyarrow)
        .filter(pl.col("year") < 0)
        .collect(),
        pl.DataFrame(schema=df.schema),
    )


@pytest.mark.write_disk
@pytest.mark.parametrize("use_pyarrow", [True, False])
def test_scan_delta_collect_without_version_scans_latest(
    tmp_path: Path,
    use_pyarrow: bool,
    plmonkeypatch: PlMonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    pl.DataFrame({"a": [0]}).write_delta(tmp_path)
    table = DeltaTable(tmp_path)

    q = pl.scan_delta(table, use_pyarrow=use_pyarrow)

    assert_frame_equal(q.collect(), pl.DataFrame({"a": [0]}))

    pl.DataFrame({"a": [1]}).write_delta(table, mode="append")

    assert_frame_equal(q.collect().sort("*"), pl.DataFrame({"a": [0, 1]}))

    version = table.version()

    q_with_id = pl.scan_delta(table, use_pyarrow=use_pyarrow, version=version)

    assert_frame_equal(q_with_id.collect().sort("*"), pl.DataFrame({"a": [0, 1]}))

    pl.DataFrame({"a": [2]}).write_delta(table, mode="append")

    assert_frame_equal(q.collect().sort("*"), pl.DataFrame({"a": [0, 1, 2]}))

    plmonkeypatch.setenv("POLARS_VERBOSE", "1")
    capfd.readouterr()

    assert_frame_equal(q_with_id.collect().sort("*"), pl.DataFrame({"a": [0, 1]}))

    capture = capfd.readouterr().err

    assert (
        "DeltaDataset: to_dataset_scan(): early return (version_key = '1')" in capture
    )


@pytest.mark.write_disk
def test_scan_delta_filter_delta_log_statistics_missing_26444(tmp_path: Path) -> None:
    pl.DataFrame({"x": [1, 2], "y": [True, False]}).write_delta(tmp_path)

    assert_frame_equal(
        pl.scan_delta(tmp_path).filter("y").collect(),
        pl.DataFrame({"x": 1, "y": True}),
    )

    schema = {
        "bool": pl.Boolean,
        "string": pl.String,
        "binary": pl.Binary,
        "int8": pl.Int8,
        "null": pl.Null,
    }

    for actions_df in [
        pl.DataFrame({"num_records": [1, 2, 3]}),
        pl.DataFrame({"num_records": [1, 2, 3], "min": [{}, {}, {}]}),
        pl.DataFrame({"num_records": [1, 2, 3], "max": [{}, {}, {}]}),
        pl.DataFrame({"num_records": [1, 2, 3], "null_count": [{}, {}, {}]}),
    ]:
        df = _extract_table_statistics_from_delta_add_actions(
            actions_df,
            filter_columns=[*schema],
            schema=schema,
            verbose=False,
        )

        assert df is not None

        assert_frame_equal(
            df,
            pl.DataFrame(
                [
                    pl.Series("len", [1, 2, 3], dtype=pl.Int64),
                    pl.Series("bool_nc", [None, None, None], dtype=pl.UInt32),
                    pl.Series("bool_min", [None, None, None], dtype=pl.Boolean),
                    pl.Series("bool_max", [None, None, None], dtype=pl.Boolean),
                    pl.Series("string_nc", [None, None, None], dtype=pl.UInt32),
                    pl.Series("string_min", [None, None, None], dtype=pl.String),
                    pl.Series("string_max", [None, None, None], dtype=pl.String),
                    pl.Series("binary_nc", [None, None, None], dtype=pl.UInt32),
                    pl.Series("binary_min", [None, None, None], dtype=pl.Binary),
                    pl.Series("binary_max", [None, None, None], dtype=pl.Binary),
                    pl.Series("int8_nc", [None, None, None], dtype=pl.UInt32),
                    pl.Series("int8_min", [None, None, None], dtype=pl.Int8),
                    pl.Series("int8_max", [None, None, None], dtype=pl.Int8),
                    pl.Series("null_nc", [None, None, None], dtype=pl.UInt32),
                    pl.Series("null_min", [None, None, None], dtype=pl.Null),
                    pl.Series("null_max", [None, None, None], dtype=pl.Null),
                ]
            ),
        )

        assert (
            _extract_table_statistics_from_delta_add_actions(
                pl.DataFrame(),
                filter_columns=[*schema],
                schema=schema,
                verbose=False,
            )
            is None
        )
