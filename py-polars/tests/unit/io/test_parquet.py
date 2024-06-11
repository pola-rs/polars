from __future__ import annotations

import io
import os
import sys
from datetime import datetime, time, timezone
from decimal import Decimal
from typing import TYPE_CHECKING, cast

import fsspec
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest

import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal

if TYPE_CHECKING:
    from pathlib import Path

    from polars.type_aliases import ParquetCompression
    from tests.unit.conftest import MemoryUsage


def test_round_trip(df: pl.DataFrame) -> None:
    f = io.BytesIO()
    df.write_parquet(f)
    f.seek(0)
    assert_frame_equal(pl.read_parquet(f), df)


COMPRESSIONS = [
    "lz4",
    "uncompressed",
    "snappy",
    "gzip",
    # "lzo",  # LZO compression currently not supported by Arrow backend
    "brotli",
    "zstd",
]


@pytest.mark.write_disk()
def test_write_parquet_using_pyarrow_9753(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)

    df = pl.DataFrame({"a": [1, 2, 3]})
    df.write_parquet(
        tmp_path / "test.parquet",
        compression="zstd",
        statistics=True,
        use_pyarrow=True,
        pyarrow_options={"coerce_timestamps": "us"},
    )


@pytest.mark.parametrize("compression", COMPRESSIONS)
def test_write_parquet_using_pyarrow_write_to_dataset_with_partitioning(
    tmp_path: Path,
    compression: ParquetCompression,
) -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "partition_col": ["one", "two", "two"]})
    path_to_write = tmp_path / "test_folder"
    path_to_write.mkdir(exist_ok=True)
    df.write_parquet(
        file=path_to_write,
        statistics=True,
        use_pyarrow=True,
        row_group_size=128,
        pyarrow_options={
            "partition_cols": ["partition_col"],
            "compression": compression,
        },
    )

    # cast is necessary as pyarrow writes partitions as categorical type
    read_df = pl.read_parquet(path_to_write, use_pyarrow=True).with_columns(
        pl.col("partition_col").cast(pl.String)
    )
    assert_frame_equal(df, read_df)


@pytest.fixture()
def small_parquet_path(io_files_path: Path) -> Path:
    return io_files_path / "small.parquet"


@pytest.mark.parametrize("compression", COMPRESSIONS)
@pytest.mark.parametrize("use_pyarrow", [True, False])
def test_to_from_buffer(
    df: pl.DataFrame, compression: ParquetCompression, use_pyarrow: bool
) -> None:
    df = df[["list_str"]]
    buf = io.BytesIO()
    df.write_parquet(buf, compression=compression, use_pyarrow=use_pyarrow)
    buf.seek(0)
    read_df = pl.read_parquet(buf, use_pyarrow=use_pyarrow)
    assert_frame_equal(df, read_df, categorical_as_str=True)


@pytest.mark.parametrize("use_pyarrow", [True, False])
@pytest.mark.parametrize("rechunk_and_expected_chunks", [(True, 1), (False, 3)])
def test_read_parquet_respects_rechunk_16416(
    use_pyarrow: bool, rechunk_and_expected_chunks: tuple[bool, int]
) -> None:
    # Create a dataframe with 3 chunks:
    df = pl.DataFrame({"a": [1]})
    df = pl.concat([df, df, df])
    buf = io.BytesIO()
    df.write_parquet(buf, row_group_size=1)
    buf.seek(0)

    rechunk, expected_chunks = rechunk_and_expected_chunks
    result = pl.read_parquet(buf, use_pyarrow=use_pyarrow, rechunk=rechunk)
    assert result.n_chunks() == expected_chunks


def test_to_from_buffer_lzo(df: pl.DataFrame) -> None:
    buf = io.BytesIO()
    # Writing lzo compressed parquet files is not supported for now.
    with pytest.raises(pl.ComputeError):
        df.write_parquet(buf, compression="lzo", use_pyarrow=False)
    buf.seek(0)
    # Invalid parquet file as writing failed.
    with pytest.raises(pl.ComputeError):
        _ = pl.read_parquet(buf)

    buf = io.BytesIO()
    with pytest.raises(OSError):
        # Writing lzo compressed parquet files is not supported for now.
        df.write_parquet(buf, compression="lzo", use_pyarrow=True)
    buf.seek(0)
    # Invalid parquet file as writing failed.
    with pytest.raises(pl.ComputeError):
        _ = pl.read_parquet(buf)


@pytest.mark.write_disk()
@pytest.mark.parametrize("compression", COMPRESSIONS)
def test_to_from_file(
    df: pl.DataFrame, compression: ParquetCompression, tmp_path: Path
) -> None:
    tmp_path.mkdir(exist_ok=True)

    file_path = tmp_path / "small.avro"
    df.write_parquet(file_path, compression=compression)
    read_df = pl.read_parquet(file_path)
    assert_frame_equal(df, read_df, categorical_as_str=True)


@pytest.mark.write_disk()
def test_to_from_file_lzo(df: pl.DataFrame, tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)

    file_path = tmp_path / "small.avro"

    # Writing lzo compressed parquet files is not supported for now.
    with pytest.raises(pl.ComputeError):
        df.write_parquet(file_path, compression="lzo", use_pyarrow=False)
    # Invalid parquet file as writing failed.
    with pytest.raises(pl.ComputeError):
        _ = pl.read_parquet(file_path)

    # Writing lzo compressed parquet files is not supported for now.
    with pytest.raises(OSError):
        df.write_parquet(file_path, compression="lzo", use_pyarrow=True)
    # Invalid parquet file as writing failed.
    with pytest.raises(FileNotFoundError):
        _ = pl.read_parquet(file_path)


def test_select_columns() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [True, False, True], "c": ["a", "b", "c"]})
    expected = pl.DataFrame({"b": [True, False, True], "c": ["a", "b", "c"]})

    f = io.BytesIO()
    df.write_parquet(f)
    f.seek(0)

    read_df = pl.read_parquet(f, columns=["b", "c"], use_pyarrow=False)
    assert_frame_equal(expected, read_df)


def test_select_projection() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [True, False, True], "c": ["a", "b", "c"]})
    expected = pl.DataFrame({"b": [True, False, True], "c": ["a", "b", "c"]})
    f = io.BytesIO()
    df.write_parquet(f)
    f.seek(0)

    read_df = pl.read_parquet(f, columns=[1, 2], use_pyarrow=False)
    assert_frame_equal(expected, read_df)


@pytest.mark.parametrize("compression", COMPRESSIONS)
@pytest.mark.parametrize("use_pyarrow", [True, False])
def test_parquet_datetime(compression: ParquetCompression, use_pyarrow: bool) -> None:
    # This failed because parquet writers cast datetime to Date
    f = io.BytesIO()
    data = {
        "datetime": [  # unix timestamp in ms
            1618354800000,
            1618354740000,
            1618354680000,
            1618354620000,
            1618354560000,
        ],
        "value1": [73.1999969482, 71.0999984741, 74.5, 69.5999984741, 69.6999969482],
        "value2": [59.5999984741, 61.0, 62.2999992371, 56.9000015259, 60.0],
    }
    df = pl.DataFrame(data)
    df = df.with_columns(df["datetime"].cast(pl.Datetime))

    df.write_parquet(f, use_pyarrow=use_pyarrow, compression=compression)
    f.seek(0)
    read = pl.read_parquet(f)
    assert_frame_equal(read, df)


def test_nested_parquet() -> None:
    f = io.BytesIO()
    data = [
        {"a": [{"b": 0}]},
        {"a": [{"b": 1}, {"b": 2}]},
    ]
    df = pd.DataFrame(data)
    df.to_parquet(f)

    read = pl.read_parquet(f, use_pyarrow=True)
    assert read.columns == ["a"]
    assert isinstance(read.dtypes[0], pl.datatypes.List)
    assert isinstance(read.dtypes[0].inner, pl.datatypes.Struct)


@pytest.mark.write_disk()
def test_glob_parquet(df: pl.DataFrame, tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)
    file_path = tmp_path / "small.parquet"
    df.write_parquet(file_path)

    path_glob = tmp_path / "small*.parquet"
    assert pl.read_parquet(path_glob).shape == (3, df.width)
    assert pl.scan_parquet(path_glob).collect().shape == (3, df.width)


def test_chunked_round_trip() -> None:
    df1 = pl.DataFrame(
        {
            "a": [1] * 2,
            "l": [[1] for j in range(2)],
        }
    )
    df2 = pl.DataFrame(
        {
            "a": [2] * 3,
            "l": [[2] for j in range(3)],
        }
    )

    df = df1.vstack(df2)

    f = io.BytesIO()
    df.write_parquet(f)
    f.seek(0)
    assert_frame_equal(pl.read_parquet(f), df)


@pytest.mark.write_disk()
def test_lazy_self_join_file_cache_prop_3979(df: pl.DataFrame, tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)

    file_path = tmp_path / "small.parquet"
    df.write_parquet(file_path)

    a = pl.scan_parquet(file_path)
    b = pl.DataFrame({"a": [1]}).lazy()
    assert a.join(b, how="cross").collect().shape == (3, df.width + b.width)
    assert b.join(a, how="cross").collect().shape == (3, df.width + b.width)


def test_recursive_logical_type() -> None:
    df = pl.DataFrame({"str": ["A", "B", "A", "B", "C"], "group": [1, 1, 2, 1, 2]})
    df = df.with_columns(pl.col("str").cast(pl.Categorical))

    df_groups = df.group_by("group").agg([pl.col("str").alias("cat_list")])
    f = io.BytesIO()
    df_groups.write_parquet(f, use_pyarrow=True)
    f.seek(0)
    read = pl.read_parquet(f, use_pyarrow=True)
    assert read.dtypes == [pl.Int64, pl.List(pl.Categorical)]
    assert read.shape == (2, 2)


def test_nested_dictionary() -> None:
    with pl.StringCache():
        df = (
            pl.DataFrame({"str": ["A", "B", "A", "B", "C"], "group": [1, 1, 2, 1, 2]})
            .with_columns(pl.col("str").cast(pl.Categorical))
            .group_by("group")
            .agg([pl.col("str").alias("cat_list")])
        )
        f = io.BytesIO()
        df.write_parquet(f)
        f.seek(0)

        read_df = pl.read_parquet(f)
        assert_frame_equal(df, read_df)


def test_row_group_size_saturation() -> None:
    df = pl.DataFrame({"a": [1, 2, 3]})
    f = io.BytesIO()

    # request larger chunk than rows in df
    df.write_parquet(f, row_group_size=1024)
    f.seek(0)
    assert_frame_equal(pl.read_parquet(f), df)


def test_nested_sliced() -> None:
    for df in [
        pl.Series([[1, 2], [3, 4], [5, 6]]).slice(2, 2).to_frame(),
        pl.Series([[None, 2], [3, 4], [5, 6]]).to_frame(),
        pl.Series([[None, 2], [3, 4], [5, 6]]).slice(2, 2).to_frame(),
        pl.Series([["a", "a"], ["", "a"], ["c", "de"]]).slice(3, 2).to_frame(),
        pl.Series([[None, True], [False, False], [True, True]]).slice(2, 2).to_frame(),
    ]:
        f = io.BytesIO()
        df.write_parquet(f)
        f.seek(0)
        assert_frame_equal(pl.read_parquet(f), df)


def test_parquet_5795() -> None:
    df_pd = pd.DataFrame(
        {
            "a": [
                "V",
                "V",
                "V",
                "V",
                "V",
                "V",
                "V",
                "V",
                "V",
                "V",
                "V",
                "V",
                "V",
                "V",
                None,
                None,
                None,
                None,
                None,
                None,
            ]
        }
    )
    f = io.BytesIO()
    df_pd.to_parquet(f)
    f.seek(0)
    assert_frame_equal(pl.read_parquet(f), pl.from_pandas(df_pd))


def test_parquet_nesting_structs_list() -> None:
    f = io.BytesIO()
    df = pl.from_records(
        [
            {
                "id": 1,
                "list_of_structs_col": [
                    {"a": 10, "b": [10, 11, 12]},
                    {"a": 11, "b": [13, 14, 15]},
                ],
            },
            {
                "id": 2,
                "list_of_structs_col": [
                    {"a": 44, "b": [12]},
                ],
            },
        ]
    )

    df.write_parquet(f)
    f.seek(0)

    assert_frame_equal(pl.read_parquet(f), df)


def test_parquet_nested_dictionaries_6217() -> None:
    _type = pa.dictionary(pa.int64(), pa.string())

    fields = [("a_type", _type)]
    struct_type = pa.struct(fields)

    col1 = pa.StructArray.from_arrays(
        [pa.DictionaryArray.from_arrays([0, 0, 1], ["A", "B"])],
        fields=struct_type,
    )

    table = pa.table({"Col1": col1})

    with pl.StringCache():
        df = pl.from_arrow(table)

        f = io.BytesIO()
        import pyarrow.parquet as pq

        pq.write_table(table, f, compression="snappy")
        f.seek(0)
        read = pl.read_parquet(f)
        assert_frame_equal(read, df)  # type: ignore[arg-type]


@pytest.mark.write_disk()
def test_fetch_union(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)

    df1 = pl.DataFrame({"a": [0, 1, 2], "b": [1, 2, 3]})
    df2 = pl.DataFrame({"a": [3, 4, 5], "b": [4, 5, 6]})

    file_path_1 = tmp_path / "df_fetch_1.parquet"
    file_path_2 = tmp_path / "df_fetch_2.parquet"
    file_path_glob = tmp_path / "df_fetch_*.parquet"

    df1.write_parquet(file_path_1)
    df2.write_parquet(file_path_2)

    result_one = pl.scan_parquet(file_path_1).fetch(1)
    result_glob = pl.scan_parquet(file_path_glob).fetch(1)

    expected = pl.DataFrame({"a": [0], "b": [1]})
    assert_frame_equal(result_one, expected)

    # Both fetch 1 per file or 1 per dataset would be ok, as we don't guarantee anything
    # currently we have one per dataset.
    expected = pl.DataFrame({"a": [0], "b": [1]})
    assert_frame_equal(result_glob, expected)


@pytest.mark.slow()
def test_struct_pyarrow_dataset_5796(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)

    num_rows = 2**17 + 1

    df = pl.from_records([{"id": i, "nested": {"a": i}} for i in range(num_rows)])
    file_path = tmp_path / "out.parquet"
    df.write_parquet(file_path, use_pyarrow=True)
    tbl = ds.dataset(file_path).to_table()
    result = pl.from_arrow(tbl)

    assert_frame_equal(result, df)  # type: ignore[arg-type]


@pytest.mark.slow()
@pytest.mark.parametrize("case", [1048576, 1048577])
def test_parquet_chunks_545(case: int) -> None:
    f = io.BytesIO()
    # repeat until it has case instances
    df = pd.DataFrame(
        np.tile([1.0, pd.to_datetime("2010-10-10")], [case, 1]),
        columns=["floats", "dates"],
    )

    # write as parquet
    df.to_parquet(f)
    f.seek(0)

    # read it with polars
    polars_df = pl.read_parquet(f)
    assert_frame_equal(pl.DataFrame(df), polars_df)


def test_nested_null_roundtrip() -> None:
    f = io.BytesIO()
    df = pl.DataFrame(
        {
            "experiences": [
                [
                    {"company": "Google", "years": None},
                    {"company": "Facebook", "years": None},
                ],
            ]
        }
    )

    df.write_parquet(f)
    f.seek(0)
    df_read = pl.read_parquet(f)
    assert_frame_equal(df_read, df)


def test_parquet_nested_list_pandas() -> None:
    # pandas/pyarrow writes as nested null dict
    df_pd = pd.DataFrame({"listcol": [[] * 10]})
    f = io.BytesIO()
    df_pd.to_parquet(f)
    f.seek(0)
    df = pl.read_parquet(f)
    assert df.dtypes == [pl.List(pl.Null)]
    assert df.to_dict(as_series=False) == {"listcol": [[]]}


def test_parquet_string_cache() -> None:
    f = io.BytesIO()

    df = pl.DataFrame({"a": ["a", "b", "c", "d"]}).with_columns(
        pl.col("a").cast(pl.Categorical)
    )

    df.write_parquet(f, row_group_size=2)

    # this file has 2 row groups and a categorical column
    # so polars should automatically set string cache
    f.seek(0)
    assert_series_equal(pl.read_parquet(f)["a"].cast(str), df["a"].cast(str))


def test_tz_aware_parquet_9586(io_files_path: Path) -> None:
    result = pl.read_parquet(io_files_path / "tz_aware.parquet")
    expected = pl.DataFrame(
        {"UTC_DATETIME_ID": [datetime(2023, 6, 26, 14, 15, 0, tzinfo=timezone.utc)]}
    ).select(pl.col("*").cast(pl.Datetime("ns", "UTC")))
    assert_frame_equal(result, expected)


def test_nested_list_page_reads_to_end_11548() -> None:
    df = pl.select(
        pl.repeat(pl.arange(0, 2048, dtype=pl.UInt64).implode(), 2).alias("x"),
    )

    f = io.BytesIO()

    pq.write_table(df.to_arrow(), f, data_page_size=1)

    f.seek(0)

    result = pl.read_parquet(f).select(pl.col("x").list.len())
    assert result.to_series().to_list() == [2048, 2048]


def test_parquet_nano_second_schema() -> None:
    value = time(9, 0, 0)
    f = io.BytesIO()
    df = pd.DataFrame({"Time": [value]})
    df.to_parquet(f)
    f.seek(0)
    assert pl.read_parquet(f).item() == value


def test_nested_struct_read_12610() -> None:
    n = 1_025
    expect = pl.select(a=pl.int_range(0, n), b=pl.repeat(1, n)).with_columns(
        struct=pl.struct(pl.all())
    )

    f = io.BytesIO()
    expect.write_parquet(
        f,
        use_pyarrow=True,
    )
    f.seek(0)

    actual = pl.read_parquet(f)
    assert_frame_equal(expect, actual)


@pytest.mark.write_disk()
def test_decimal_parquet(tmp_path: Path) -> None:
    path = tmp_path / "foo.parquet"
    df = pl.DataFrame(
        {
            "foo": [1, 2, 3],
            "bar": ["6", "7", "8"],
        }
    )

    df = df.with_columns(pl.col("bar").cast(pl.Decimal))

    df.write_parquet(path, statistics=True)
    out = pl.scan_parquet(path).filter(foo=2).collect().to_dict(as_series=False)
    assert out == {"foo": [2], "bar": [Decimal("7")]}


@pytest.mark.write_disk()
def test_enum_parquet(tmp_path: Path) -> None:
    path = tmp_path / "enum.parquet"
    df = pl.DataFrame(
        [pl.Series("e", ["foo", "bar", "ham"], dtype=pl.Enum(["foo", "bar", "ham"]))]
    )
    df.write_parquet(path)
    out = pl.read_parquet(path)
    assert_frame_equal(df, out)


def test_parquet_rle_non_nullable_12814() -> None:
    column = (
        pl.select(x=pl.arange(0, 1025, dtype=pl.Int64) // 10).to_series().to_arrow()
    )
    schema = pa.schema([pa.field("foo", pa.int64(), nullable=False)])
    table = pa.Table.from_arrays([column], schema=schema)

    f = io.BytesIO()
    pq.write_table(table, f, data_page_size=1)
    f.seek(0)

    expect = pl.DataFrame(table).tail(10)
    actual = pl.read_parquet(f).tail(10)

    assert_frame_equal(expect, actual)


@pytest.mark.slow()
def test_parquet_12831() -> None:
    n = 70_000
    df = pl.DataFrame({"x": ["aaaaaa"] * n})
    f = io.BytesIO()
    df.write_parquet(f, row_group_size=int(1e8), data_page_size=512)
    f.seek(0)
    assert_frame_equal(pl.from_arrow(pq.read_table(f)), df)  # type: ignore[arg-type]


@pytest.mark.write_disk()
def test_parquet_struct_categorical(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)

    df = pl.DataFrame(
        [
            pl.Series("a", ["bob"], pl.Categorical),
            pl.Series("b", ["foo"], pl.Categorical),
        ]
    )

    file_path = tmp_path / "categorical.parquet"
    df.write_parquet(file_path)

    with pl.StringCache():
        out = pl.read_parquet(file_path).select(pl.col("b").value_counts())
    assert out.to_dict(as_series=False) == {"b": [{"b": "foo", "count": 1}]}


@pytest.mark.write_disk()
def test_null_parquet(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)

    df = pl.DataFrame([pl.Series("foo", [], dtype=pl.Int8)])
    file_path = tmp_path / "null.parquet"
    df.write_parquet(file_path)
    out = pl.read_parquet(file_path)
    assert_frame_equal(out, df)


@pytest.mark.write_disk()
def test_write_parquet_with_null_col(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)

    df1 = pl.DataFrame({"nulls": [None] * 2, "ints": [1] * 2})
    df2 = pl.DataFrame({"nulls": [None] * 2, "ints": [1] * 2})
    df3 = pl.DataFrame({"nulls": [None] * 3, "ints": [1] * 3})
    df = df1.vstack(df2)
    df = df.vstack(df3)
    file_path = tmp_path / "with_null.parquet"
    df.write_parquet(file_path, row_group_size=3)
    out = pl.read_parquet(file_path)
    assert_frame_equal(out, df)


@pytest.mark.write_disk()
def test_read_parquet_binary_buffered_reader(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)

    df = pl.DataFrame({"a": [1, 2, 3]})
    file_path = tmp_path / "test.parquet"
    df.write_parquet(file_path)

    with file_path.open("rb") as f:
        out = pl.read_parquet(f)
    assert_frame_equal(out, df)


@pytest.mark.write_disk()
def test_read_parquet_binary_file_io(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)

    df = pl.DataFrame({"a": [1, 2, 3]})
    file_path = tmp_path / "test.parquet"
    df.write_parquet(file_path)

    with file_path.open("rb", buffering=0) as f:
        out = pl.read_parquet(f)
    assert_frame_equal(out, df)


# https://github.com/pola-rs/polars/issues/15760
@pytest.mark.write_disk()
def test_read_parquet_binary_fsspec(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)

    df = pl.DataFrame({"a": [1, 2, 3]})
    file_path = tmp_path / "test.parquet"
    df.write_parquet(file_path)

    with fsspec.open(file_path) as f:
        out = pl.read_parquet(f)
    assert_frame_equal(out, df)


def test_read_parquet_binary_bytes_io() -> None:
    df = pl.DataFrame({"a": [1, 2, 3]})
    f = io.BytesIO()
    df.write_parquet(f)
    f.seek(0)

    out = pl.read_parquet(f)
    assert_frame_equal(out, df)


def test_read_parquet_binary_bytes() -> None:
    df = pl.DataFrame({"a": [1, 2, 3]})
    f = io.BytesIO()
    df.write_parquet(f)
    bytes = f.getvalue()

    out = pl.read_parquet(bytes)
    assert_frame_equal(out, df)


def test_utc_timezone_normalization_13670(tmp_path: Path) -> None:
    """'+00:00' timezones becomes 'UTC' timezone."""
    utc_path = tmp_path / "utc.parquet"
    zero_path = tmp_path / "00_00.parquet"
    for tz, path in [("+00:00", zero_path), ("UTC", utc_path)]:
        pq.write_table(
            pa.table(
                {"c1": [1234567890123] * 10},
                schema=pa.schema([pa.field("c1", pa.timestamp("ms", tz=tz))]),
            ),
            path,
        )

    df = pl.scan_parquet([utc_path, zero_path]).head(5).collect()
    assert cast(pl.Datetime, df.schema["c1"]).time_zone == "UTC"
    df = pl.scan_parquet([zero_path, utc_path]).head(5).collect()
    assert cast(pl.Datetime, df.schema["c1"]).time_zone == "UTC"


def test_parquet_rle_14333() -> None:
    vals = [True, False, True, False, True, False, True, False, True, False]
    table = pa.table({"a": vals})

    f = io.BytesIO()
    pq.write_table(table, f, data_page_version="2.0")
    f.seek(0)
    assert pl.read_parquet(f)["a"].to_list() == vals


def test_parquet_rle_null_binary_read_14638() -> None:
    df = pl.DataFrame({"x": [None]}, schema={"x": pl.String})

    f = io.BytesIO()
    df.write_parquet(f, use_pyarrow=True)
    f.seek(0)
    assert "RLE_DICTIONARY" in pq.read_metadata(f).row_group(0).column(0).encodings
    f.seek(0)
    assert_frame_equal(df, pl.read_parquet(f))


def test_parquet_string_rle_encoding() -> None:
    n = 3
    data = {
        "id": ["abcdefgh"] * n,
    }

    df = pl.DataFrame(data)
    f = io.BytesIO()
    df.write_parquet(f, use_pyarrow=False)
    f.seek(0)

    assert (
        "RLE_DICTIONARY"
        in pq.ParquetFile(f).metadata.to_dict()["row_groups"][0]["columns"][0][
            "encodings"
        ]
    )


def test_sliced_dict_with_nulls_14904() -> None:
    df = (
        pl.DataFrame({"x": [None, None]})
        .cast(pl.Categorical)
        .with_columns(y=pl.concat_list("x"))
        .slice(0, 1)
    )
    test_round_trip(df)


def test_parquet_array_dtype() -> None:
    df = pl.DataFrame({"x": [[1, 2, 3]]})
    df = df.cast({"x": pl.Array(pl.Int64, shape=3)})
    test_round_trip(df)


@pytest.mark.write_disk()
def test_parquet_array_statistics(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)

    df = pl.DataFrame({"a": [[1, 2, 3], [4, 5, 6], [7, 8, 9]], "b": [1, 2, 3]})
    file_path = tmp_path / "test.parquet"

    df.with_columns(a=pl.col("a").list.to_array(3)).lazy().filter(
        pl.col("a") != [1, 2, 3]
    ).collect()
    df.with_columns(a=pl.col("a").list.to_array(3)).lazy().sink_parquet(file_path)

    result = pl.scan_parquet(file_path).filter(pl.col("a") != [1, 2, 3]).collect()
    assert result.to_dict(as_series=False) == {"a": [[4, 5, 6], [7, 8, 9]], "b": [2, 3]}


@pytest.mark.slow()
@pytest.mark.write_disk()
def test_read_parquet_only_loads_selected_columns_15098(
    memory_usage_without_pyarrow: MemoryUsage, tmp_path: Path
) -> None:
    """Only requested columns are loaded by ``read_parquet()``."""
    tmp_path.mkdir(exist_ok=True)

    # Each column will be about 8MB of RAM
    series = pl.arange(0, 1_000_000, dtype=pl.Int64, eager=True)

    file_path = tmp_path / "multicolumn.parquet"
    df = pl.DataFrame(
        {
            "a": series,
            "b": series,
        }
    )
    df.write_parquet(file_path)
    del df, series

    memory_usage_without_pyarrow.reset_tracking()

    # Only load one column:
    df = pl.read_parquet([file_path], columns=["b"], rechunk=False)
    del df
    # Only one column's worth of memory should be used; 2 columns would be
    # 16_000_000 at least, but there's some overhead.
    assert 8_000_000 < memory_usage_without_pyarrow.get_peak() < 13_000_000


@pytest.mark.release()
@pytest.mark.write_disk()
def test_max_statistic_parquet_writer(tmp_path: Path) -> None:
    # this hits the maximal page size
    # so the row group will be split into multiple pages
    # the page statistics need to be correctly reduced
    # for this query to make sense
    n = 150_000

    tmp_path.mkdir(exist_ok=True)

    # int64 is important to hit the page size
    df = pl.int_range(0, n, eager=True, dtype=pl.Int64).alias("int").to_frame()
    f = tmp_path / "tmp.parquet"
    df.write_parquet(f, statistics=True, use_pyarrow=False, row_group_size=n)
    result = pl.scan_parquet(f).filter(pl.col("int") > n - 3).collect()
    expected = pl.DataFrame({"int": [149998, 149999]})
    assert_frame_equal(result, expected)


@pytest.mark.write_disk()
@pytest.mark.skipif(os.environ.get("POLARS_FORCE_ASYNC") == "1", reason="only local")
@pytest.mark.skipif(
    sys.platform == "win32", reason="Windows filenames cannot contain an asterisk"
)
def test_no_glob(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)

    df = pl.DataFrame({"foo": 1})

    p1 = tmp_path / "*.parquet"
    df.write_parquet(str(p1))
    p2 = tmp_path / "*1.parquet"
    df.write_parquet(str(p2))

    assert_frame_equal(pl.scan_parquet(str(p1), glob=False).collect(), df)


@pytest.mark.write_disk()
@pytest.mark.skipif(os.environ.get("POLARS_FORCE_ASYNC") == "1", reason="only local")
def test_no_glob_windows(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)

    df = pl.DataFrame({"foo": 1})

    p1 = tmp_path / "hello[.parquet"
    df.write_parquet(str(p1))
    p2 = tmp_path / "hello[2.parquet"
    df.write_parquet(str(p2))

    assert_frame_equal(pl.scan_parquet(str(p1), glob=False).collect(), df)


@pytest.mark.slow()
def test_hybrid_rle() -> None:
    # 10_007 elements to test if not a nice multiple of 8
    n = 10_007
    literal_literal = []
    literal_rle = []
    for i in range(500):
        literal_literal.append(np.repeat(i, 5))
        literal_literal.append(np.repeat(i + 2, 11))
        literal_rle.append(np.repeat(i, 5))
        literal_rle.append(np.repeat(i + 2, 15))
    literal_literal.append(np.random.randint(0, 10, size=2007))
    literal_rle.append(np.random.randint(0, 10, size=7))
    literal_literal = np.concatenate(literal_literal)
    literal_rle = np.concatenate(literal_rle)
    df = pl.DataFrame(
        {
            # Primitive types
            "i64": pl.Series([1, 2], dtype=pl.Int64).sample(n, with_replacement=True),
            "u64": pl.Series([1, 2], dtype=pl.UInt64).sample(n, with_replacement=True),
            "i8": pl.Series([1, 2], dtype=pl.Int8).sample(n, with_replacement=True),
            "u8": pl.Series([1, 2], dtype=pl.UInt8).sample(n, with_replacement=True),
            "string": pl.Series(["abc", "def"], dtype=pl.String).sample(
                n, with_replacement=True
            ),
            "categorical": pl.Series(["aaa", "bbb"], dtype=pl.Categorical).sample(
                n, with_replacement=True
            ),
            # Fill up bit-packing buffer in middle of consecutive run
            "large_bit_pack": np.concatenate(
                [np.repeat(i, 5) for i in range(2000)]
                + [np.random.randint(0, 10, size=7)]
            ),
            # Literal run that is not a multiple of 8 followed by consecutive
            # run initially long enough to RLE but not after padding literal
            "literal_literal": literal_literal,
            # Literal run that is not a multiple of 8 followed by consecutive
            # run long enough to RLE even after padding literal
            "literal_rle": literal_rle,
            # Final run not long enough to RLE
            "final_literal": np.concatenate(
                [np.random.randint(0, 100, 10_000), np.repeat(-1, 7)]
            ),
            # Final run long enough to RLE
            "final_rle": np.concatenate(
                [np.random.randint(0, 100, 9_998), np.repeat(-1, 9)]
            ),
            # Test filling up bit-packing buffer for encode_bool,
            # which is only used to encode validities
            "large_bit_pack_validity": [0, None] * 4092
            + [0] * 9
            + [1] * 9
            + [2] * 10
            + [0] * 1795,
        }
    )
    f = io.BytesIO()
    df.write_parquet(f)
    f.seek(0)
    for column in pq.ParquetFile(f).metadata.to_dict()["row_groups"][0]["columns"]:
        assert "RLE_DICTIONARY" in column["encodings"]
    f.seek(0)
    assert_frame_equal(pl.read_parquet(f), df)


def test_parquet_statistics_uint64_16683() -> None:
    u64_max = (1 << 64) - 1
    df = pl.Series("a", [u64_max, 0], dtype=pl.UInt64).to_frame()
    file = io.BytesIO()
    df.write_parquet(file, statistics=True)
    file.seek(0)
    statistics = pq.read_metadata(file).row_group(0).column(0).statistics

    assert statistics.min == 0
    assert statistics.max == u64_max
