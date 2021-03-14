from polars import DataFrame, Series
from polars.lazy import *
from polars.datatypes import *
import polars as pl
import pytest


def test_lazy():
    df = DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]})
    ldf = df.lazy().with_column(lit(1).alias("foo")).select([col("a"), col("foo")])

    print(ldf.collect())
    # test if it executes
    new = (
        df.lazy()
        .with_column(
            when(col("a").gt(lit(2))).then(lit(10)).otherwise(lit(1)).alias("new")
        )
        .collect()
    )


def test_apply():
    df = DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]})
    new = df.lazy().with_column(col("a").map(lambda s: s * 2).alias("foo")).collect()


def test_agg():
    df = DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]})
    ldf = df.lazy().min()
    assert ldf.collect().shape == (1, 2)


def test_fold():
    df = DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]})
    out = df.lazy().select(pl.sum(["a", "b"])).collect()
    assert out["sum"].series_equal(Series("sum", [2, 4, 6]))


def test_or():
    df = DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]})
    out = df.lazy().filter((pl.col("a") == 1) | (pl.col("b") > 2)).collect()
    assert out.shape[0] == 2


def test_groupby_apply():
    df = DataFrame({"a": [1, 1, 3], "b": [1.0, 2.0, 3.0]})
    ldf = df.lazy().groupby("a").apply(lambda df: df)
    assert ldf.collect().sort("b").frame_equal(df)


def test_binary_function():
    df = DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]})
    out = (
        df.lazy()
        .with_column(map_binary(col("a"), col("b"), lambda a, b: a + b))
        .collect()
    )
    assert out["binary_function"] == (out.a + out.b)


def test_filter_str():
    # use a str instead of a column expr
    df = pl.DataFrame(
        {
            "time": ["11:11:00", "11:12:00", "11:13:00", "11:14:00"],
            "bools": [True, False, True, False],
        }
    )
    q = df.lazy()
    # last row based on a filter
    q.filter(pl.col("bools")).select(pl.last("*"))
