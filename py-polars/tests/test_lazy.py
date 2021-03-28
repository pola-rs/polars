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
    out = df.lazy().select(pl.lazy.sum(["a", "b"])).collect()
    assert out["sum"].series_equal(Series("sum", [2, 4, 6]))


def test_or():
    df = DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]})
    out = df.lazy().filter((pl.lazy.col("a") == 1) | (pl.lazy.col("b") > 2)).collect()
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
    q.filter(pl.lazy.col("bools")).select(pl.lazy.last("*"))


def test_apply_custom_function():
    df = pl.DataFrame(
        {
            "A": [1, 2, 3, 4, 5],
            "fruits": ["banana", "banana", "apple", "apple", "banana"],
            "B": [5, 4, 3, 2, 1],
            "cars": ["beetle", "audi", "beetle", "beetle", "beetle"],
        }
    )

    # two ways to determine the length groups.
    a = (
        df.lazy()
        .groupby("fruits")
        .agg(
            [
                pl.lazy.col("cars").apply(lambda groups: groups.len()).alias("custom_1"),
                pl.lazy.col("cars").apply(lambda groups: groups.len()).alias("custom_2"),
                pl.lazy.count("cars"),
            ]
        )
        .sort("custom_1", reverse=True)
    ).collect()
    expected = pl.DataFrame(
        {
            "fruits": ["banana", "apple"],
            "custom_1": [3, 2],
            "custom_2": [3, 2],
            "cars_count": [3, 2],
        }
    )
    expected["cars_count"] = expected["cars_count"].cast(pl.UInt32)
    assert a.frame_equal(expected)


def test_groupby():
    df = pl.DataFrame({"a": [1.0, None, 3.0, 4.0], "groups": ["a", "a", "b", "b"]})
    out = df.lazy().groupby("groups").agg(pl.lazy.mean("a")).collect()


def test_shift_and_fill():
    df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": [1, 2, 3, 4, 5]})

    # use exprs
    out = df.lazy().with_column(col("a").shift_and_fill(-2, col("b").mean())).collect()
    assert out["a"].null_count() == 0

    # use df method
    out = df.lazy().shift_and_fill(2, col("b").std()).collect()
    assert out["a"].null_count() == 0
