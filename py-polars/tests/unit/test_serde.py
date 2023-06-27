from __future__ import annotations

import pickle
import typing
from datetime import datetime, timedelta

import pytest

import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal


def test_pickling_simple_expression() -> None:
    e = pl.col("foo").sum()
    buf = pickle.dumps(e)
    assert str(pickle.loads(buf)) == str(e)


def test_serde_lazy_frame_lp() -> None:
    lf = pl.DataFrame({"a": [1, 2, 3], "b": ["a", "b", "c"]}).lazy().select(pl.col("a"))
    json = lf.write_json()

    result = pl.LazyFrame.from_json(json).collect().to_series()
    assert_series_equal(result, pl.Series("a", [1, 2, 3]))


def test_serde_time_unit() -> None:
    assert pickle.loads(
        pickle.dumps(
            pl.Series(
                [datetime(2022, 1, 1) + timedelta(days=1) for _ in range(3)]
            ).cast(pl.Datetime("ns"))
        )
    ).dtype == pl.Datetime("ns")


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


def test_expression_json() -> None:
    e = pl.col("foo").sum().over("bar")
    json = e.meta.write_json()

    round_tripped = pl.Expr.from_json(json)
    assert round_tripped.meta == e


@typing.no_type_check
def times2(x):
    return x * 2


def test_pickle_udf_expression() -> None:
    df = pl.DataFrame({"a": [1, 2, 3]})

    e = pl.col("a").map(times2)
    b = pickle.dumps(e)
    e = pickle.loads(b)

    assert df.select(e).to_dict(False) == {"a": [2, 4, 6]}

    e = pl.col("a").map(times2, return_dtype=pl.Utf8)
    b = pickle.dumps(e)
    e = pickle.loads(b)

    # tests that 'GetOutput' is also deserialized
    with pytest.raises(
        pl.SchemaError,
        match=r"expected output type 'Utf8', got 'Int64'; set `return_dtype` to the proper datatype",
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
