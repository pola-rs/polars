from __future__ import annotations

import pickle
from datetime import datetime, timedelta

import pytest

import polars as pl
from polars import StringCache
from polars.exceptions import SchemaError
from polars.testing import assert_frame_equal, assert_series_equal


def test_pickling_simple_expression() -> None:
    e = pl.col("foo").sum()
    buf = pickle.dumps(e)
    assert str(pickle.loads(buf)) == str(e)


def test_pickling_as_struct_11100() -> None:
    e = pl.struct("a")
    buf = pickle.dumps(e)
    assert str(pickle.loads(buf)) == str(e)


def test_serde_time_unit() -> None:
    values = [datetime(2022, 1, 1) + timedelta(days=1) for _ in range(3)]
    s = pl.Series(values).cast(pl.Datetime("ns"))
    result = pickle.loads(pickle.dumps(s))
    assert result.dtype == pl.Datetime("ns")


def test_serde_duration() -> None:
    df = (
        pl.DataFrame(
            {
                "a": [
                    datetime(2021, 2, 1, 9, 20),
                    datetime(2021, 2, 2, 9, 20),
                ],
                "b": [4, 5],
            }
        )
        .with_columns([pl.col("a").cast(pl.Datetime("ns")).alias("a")])
        .select(pl.all())
    )
    df = df.with_columns([pl.col("a").diff(n=1).alias("a_td")])
    serde_df = pickle.loads(pickle.dumps(df))
    assert serde_df["a_td"].dtype == pl.Duration("ns")
    assert_series_equal(
        serde_df["a_td"],
        pl.Series("a_td", [None, timedelta(days=1)], dtype=pl.Duration("ns")),
    )


def test_serde_expression_5461() -> None:
    e = pl.col("a").sqrt() / pl.col("b").alias("c")
    assert pickle.loads(pickle.dumps(e)).meta == e.meta


def test_serde_binary() -> None:
    data = pl.Series(
        "binary_data",
        [
            b"\xba\x9b\xca\xd3y\xcb\xc9#",
            b"9\x04\xab\xe2\x11\xf3\x85",
            b"\xb8\xcb\xc9^\\\xa9-\x94\xe0H\x9d ",
            b"S\xbc:\xcb\xf0\xf5r\xfe\x18\xfeH",
            b",\xf5)y\x00\xe5\xf7",
            b"\xfd\xf6\xf1\xc2X\x0cn\xb9#",
            b"\x06\xef\xa6\xa2\xb7",
            b"@\xff\x95\xda\xff\xd2\x18",
        ],
    )
    assert_series_equal(
        data,
        pickle.loads(pickle.dumps(data)),
    )


def test_pickle_lazyframe() -> None:
    q = pl.LazyFrame({"a": [1, 4, 3]}).sort("a")

    s = pickle.dumps(q)
    assert_frame_equal(pickle.loads(s).collect(), pl.DataFrame({"a": [1, 3, 4]}))


def test_deser_empty_list() -> None:
    s = pickle.loads(pickle.dumps(pl.Series([[[42.0]], []])))
    assert s.dtype == pl.List(pl.List(pl.Float64))
    assert s.to_list() == [[[42.0]], []]


def times2(x: pl.Series) -> pl.Series:
    return x * 2


def test_pickle_udf_expression() -> None:
    df = pl.DataFrame({"a": [1, 2, 3]})

    e = pl.col("a").map_batches(times2)
    b = pickle.dumps(e)
    e = pickle.loads(b)

    result = df.select(e)
    expected = pl.DataFrame({"a": [2, 4, 6]})
    assert_frame_equal(result, expected)

    e = pl.col("a").map_batches(times2, return_dtype=pl.String)
    b = pickle.dumps(e)
    e = pickle.loads(b)

    # tests that 'GetOutput' is also deserialized
    with pytest.raises(
        SchemaError,
        match=r"expected output type 'String', got 'Int64'; set `return_dtype` to the proper datatype",
    ):
        df.select(e)


def test_pickle_small_integers() -> None:
    df = pl.DataFrame(
        [
            pl.Series([1, 2], dtype=pl.Int16),
            pl.Series([3, 2], dtype=pl.Int8),
            pl.Series([32, 2], dtype=pl.UInt8),
            pl.Series([3, 3], dtype=pl.UInt16),
        ]
    )
    b = pickle.dumps(df)
    assert_frame_equal(pickle.loads(b), df)


def df_times2(df: pl.DataFrame) -> pl.DataFrame:
    return df.select(pl.all() * 2)


def test_pickle_lazyframe_udf() -> None:
    df = pl.DataFrame({"a": [1, 2, 3]})

    q = df.lazy().map_batches(df_times2)
    b = pickle.dumps(q)

    q = pickle.loads(b)
    assert q.collect()["a"].to_list() == [2, 4, 6]


def test_pickle_lazyframe_nested_function_udf() -> None:
    df = pl.DataFrame({"a": [1, 2, 3]})

    # NOTE: This is only possible when we're using cloudpickle.
    def inner_df_times2(df: pl.DataFrame) -> pl.DataFrame:
        return df.select(pl.all() * 2)

    q = df.lazy().map_batches(inner_df_times2)
    b = pickle.dumps(q)

    q = pickle.loads(b)
    assert q.collect()["a"].to_list() == [2, 4, 6]


@StringCache()
def test_serde_categorical_series_10586() -> None:
    s = pl.Series(["a", "b", "b", "a", "c"], dtype=pl.Categorical)
    loaded_s = pickle.loads(pickle.dumps(s))
    assert_series_equal(loaded_s, s)


def test_serde_keep_dtype_empty_list() -> None:
    s = pl.Series([{"a": None}], dtype=pl.Struct([pl.Field("a", pl.List(pl.String))]))
    assert s.dtype == pickle.loads(pickle.dumps(s)).dtype


def test_serde_array_dtype() -> None:
    s = pl.Series(
        [[1, 2, 3], [None, None, None], [1, None, 3]],
        dtype=pl.Array(pl.Int32(), 3),
    )
    assert_series_equal(pickle.loads(pickle.dumps(s)), s)

    nested_s = pl.Series(
        [[[1, 2, 3], [4, None, 5]], None, [[None, None, 2]]],
        dtype=pl.List(pl.Array(pl.Int32(), 3)),
    )
    assert_series_equal(pickle.loads(pickle.dumps(nested_s)), nested_s)


def test_serde_data_type_class() -> None:
    dtype = pl.Datetime
    serialized = pickle.dumps(dtype)
    deserialized = pickle.loads(serialized)
    assert deserialized == dtype
    assert isinstance(deserialized, type)


def test_serde_data_type_instantiated() -> None:
    dtype = pl.Int8()
    serialized = pickle.dumps(dtype)
    deserialized = pickle.loads(serialized)
    assert deserialized == dtype
    assert isinstance(deserialized, pl.DataType)


def test_serde_data_type_instantiated_with_attributes() -> None:
    dtype = pl.Enum(["a", "b"])
    serialized = pickle.dumps(dtype)
    deserialized = pickle.loads(serialized)
    assert deserialized == dtype
    assert isinstance(deserialized, pl.DataType)
