from __future__ import annotations

import io
import sys
import tempfile
import typing
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest

import polars as pl
from polars.testing import assert_frame_equal, assert_frame_equal_local_categoricals

if TYPE_CHECKING:
    from polars.internals.type_aliases import ParquetCompression

COMPRESSIONS = [
    "lz4",
    "uncompressed",
    "snappy",
    "gzip",
    # "lzo",  # LZO compression currently not supported by Arrow backend
    "brotli",
    "zstd",
]


@pytest.fixture()
def small_parquet_path(io_files_path: Path) -> Path:
    return io_files_path / "small.parquet"


@pytest.mark.parametrize("compression", COMPRESSIONS)
@pytest.mark.parametrize("use_pyarrow", [True, False])
def test_to_from_buffer(
    df: pl.DataFrame, compression: ParquetCompression, use_pyarrow: bool
) -> None:
    buf = io.BytesIO()
    df.write_parquet(buf, compression=compression, use_pyarrow=use_pyarrow)
    buf.seek(0)
    read_df = pl.read_parquet(buf, use_pyarrow=use_pyarrow)
    assert_frame_equal_local_categoricals(df, read_df)


def test_to_from_buffer_lzo(df: pl.DataFrame) -> None:
    buf = io.BytesIO()
    # Writing lzo compressed parquet files is not supported for now.
    with pytest.raises(pl.ArrowError):
        df.write_parquet(buf, compression="lzo", use_pyarrow=False)
    buf.seek(0)
    # Invalid parquet file as writing failed.
    with pytest.raises(pl.ArrowError):
        _ = pl.read_parquet(buf)

    buf = io.BytesIO()
    with pytest.raises(OSError):
        # Writing lzo compressed parquet files is not supported for now.
        df.write_parquet(buf, compression="lzo", use_pyarrow=True)
    buf.seek(0)
    # Invalid parquet file as writing failed.
    with pytest.raises(pl.ArrowError):
        _ = pl.read_parquet(buf)


@pytest.mark.parametrize("compression", COMPRESSIONS)
def test_to_from_file(df: pl.DataFrame, compression: ParquetCompression) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "small.avro"
        df.write_parquet(file_path, compression=compression)
        read_df = pl.read_parquet(file_path)
        assert_frame_equal_local_categoricals(df, read_df)


def test_to_from_file_lzo(df: pl.DataFrame) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "small.avro"

        # Writing lzo compressed parquet files is not supported for now.
        with pytest.raises(pl.ArrowError):
            df.write_parquet(file_path, compression="lzo", use_pyarrow=False)
        # Invalid parquet file as writing failed.
        with pytest.raises(pl.ArrowError):
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
        "laf_max": [73.1999969482, 71.0999984741, 74.5, 69.5999984741, 69.6999969482],
        "laf_eq": [59.5999984741, 61.0, 62.2999992371, 56.9000015259, 60.0],
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


def test_glob_parquet(df: pl.DataFrame) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "small.parquet"
        df.write_parquet(file_path)

        path_glob = Path(temp_dir) / "small*.parquet"
        assert pl.read_parquet(path_glob).shape == (3, 16)
        assert pl.scan_parquet(path_glob).collect().shape == (3, 16)


def test_streaming_parquet_glob_5900(df: pl.DataFrame) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "small.parquet"
        df.write_parquet(file_path)

        path_glob = Path(temp_dir) / "small*.parquet"
        result = (
            pl.scan_parquet(path_glob).select(pl.all().first()).collect(streaming=True)
        )
        assert result.shape == (1, 16)


def test_chunked_round_trip() -> None:
    df1 = pl.DataFrame(
        {
            "a": [1] * 2,
            "l": [[1] for j in range(0, 2)],
        }
    )
    df2 = pl.DataFrame(
        {
            "a": [2] * 3,
            "l": [[2] for j in range(0, 3)],
        }
    )

    df = df1.vstack(df2)

    f = io.BytesIO()
    df.write_parquet(f)
    f.seek(0)
    assert_frame_equal(pl.read_parquet(f), df)


def test_lazy_self_join_file_cache_prop_3979(df: pl.DataFrame) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "small.parquet"
        df.write_parquet(file_path)

        a = pl.scan_parquet(file_path)
        b = pl.DataFrame({"a": [1]}).lazy()
        assert a.join(b, how="cross").collect().shape == (3, 17)
        assert b.join(a, how="cross").collect().shape == (3, 17)


def test_recursive_logical_type() -> None:
    df = pl.DataFrame({"str": ["A", "B", "A", "B", "C"], "group": [1, 1, 2, 1, 2]})
    df = df.with_columns(pl.col("str").cast(pl.Categorical))

    df_groups = df.groupby("group").agg([pl.col("str").alias("cat_list")])
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
            .groupby("group")
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


@typing.no_type_check
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


@typing.no_type_check
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
        pq.write_table(table, f, compression="snappy")
        f.seek(0)
        read = pl.read_parquet(f)
        assert_frame_equal(read, df)


@pytest.mark.xfail(sys.platform == "win32", reason="Does not work on Windows")
def test_sink_parquet(io_files_path: Path) -> None:
    file = io_files_path / "small.parquet"

    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "sink.parquet"

        df_scanned = pl.scan_parquet(file)
        df_scanned.sink_parquet(file_path)

        with pl.StringCache():
            result = pl.read_parquet(file_path)
            df_read = pl.read_parquet(file)
            assert_frame_equal(result, df_read)


@pytest.mark.xfail(sys.platform == "win32", reason="Does not work on Windows")
def test_sink_ipc(io_files_path: Path) -> None:
    file = io_files_path / "small.parquet"

    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "sink.ipc"

        df_scanned = pl.scan_parquet(file)
        df_scanned.sink_ipc(file_path)

        with pl.StringCache():
            result = pl.read_ipc(file_path)
            df_read = pl.read_parquet(file)
            assert_frame_equal(result, df_read)


@pytest.mark.xfail(sys.platform == "win32", reason="Does not work on Windows")
def test_fetch_union() -> None:
    df1 = pl.DataFrame({"a": [0, 1, 2], "b": [1, 2, 3]})
    df2 = pl.DataFrame({"a": [3, 4, 5], "b": [4, 5, 6]})

    with tempfile.TemporaryDirectory() as temp_dir:
        file_path_1 = Path(temp_dir) / "df_fetch_1.parquet"
        file_path_2 = Path(temp_dir) / "df_fetch_2.parquet"
        file_path_glob = Path(temp_dir) / "df_fetch_*.parquet"

        df1.write_parquet(file_path_1)
        df2.write_parquet(file_path_2)

        result_one = pl.scan_parquet(file_path_1).fetch(1)
        result_glob = pl.scan_parquet(file_path_glob).fetch(1)

    expected = pl.DataFrame({"a": [0], "b": [1]})
    assert_frame_equal(result_one, expected)

    expected = pl.DataFrame({"a": [0, 3], "b": [1, 4]})
    assert_frame_equal(result_glob, expected)


@pytest.mark.slow()
@typing.no_type_check
@pytest.mark.xfail(sys.platform == "win32", reason="Does not work on Windows")
def test_struct_pyarrow_dataset_5796() -> None:
    num_rows = 2**17 + 1

    df = pl.from_records(
        [dict(id=i, nested=dict(a=i)) for i in range(num_rows)]  # noqa: C408
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "out.parquet"
        df.write_parquet(file_path, use_pyarrow=True)
        tbl = ds.dataset(file_path).to_table()
        result = pl.from_arrow(tbl)

    assert_frame_equal(result, df)


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
