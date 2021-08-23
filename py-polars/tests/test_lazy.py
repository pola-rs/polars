import numpy as np
import pytest

import polars as pl
from polars.datatypes import *
from polars.lazy import *


def test_lazy():
    df = pl.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]})
    ldf = df.lazy().with_column(lit(1).alias("foo")).select([col("a"), col("foo")])

    # test if it executes
    new = (
        df.lazy()
        .with_column(
            when(col("a").gt(lit(2))).then(lit(10)).otherwise(lit(1)).alias("new")
        )
        .collect()
    )

    # test if pl.list is available, this is `to_list` re-exported as list
    df.groupby("a").agg(pl.list("b"))


def test_apply():
    df = pl.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]})
    new = df.lazy().with_column(col("a").map(lambda s: s * 2).alias("foo")).collect()


def test_add_eager_column():
    df = pl.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]})
    out = df.lazy().with_column(pl.lit(pl.Series("c", [1, 2, 3]))).collect()
    assert out["c"].sum() == 6


def test_set_null():
    df = pl.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]})
    out = (
        df.lazy()
        .with_column(when(col("a") > 1).then(lit(None)).otherwise(100).alias("foo"))
        .collect()
    )
    s = out["foo"]
    assert s[0] == 100
    assert s[1] is None
    assert s[2] is None


def test_agg():
    df = pl.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]})
    ldf = df.lazy().min()
    assert ldf.collect().shape == (1, 2)


def test_fold():
    df = pl.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]})
    out = df.lazy().select(pl.sum(["a", "b"])).collect()
    assert out["sum"].series_equal(pl.Series("sum", [2, 4, 6]))

    out = df.select(
        pl.fold(acc=lit(0), f=lambda acc, x: acc + x, exprs=pl.col("*")).alias("foo")
    )
    assert out["foo"] == [2, 4, 6]


def test_or():
    df = pl.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]})
    out = df.lazy().filter((pl.col("a") == 1) | (pl.col("b") > 2)).collect()
    assert out.shape[0] == 2


def test_groupby_apply():
    df = pl.DataFrame({"a": [1, 1, 3], "b": [1.0, 2.0, 3.0]})
    ldf = df.lazy().groupby("a").apply(lambda df: df)
    assert ldf.collect().sort("b").frame_equal(df)


def test_binary_function():
    df = pl.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]})
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
                pl.col("cars").apply(lambda groups: groups.len()).alias("custom_1"),
                pl.col("cars").apply(lambda groups: groups.len()).alias("custom_2"),
                pl.count("cars"),
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
    out = df.lazy().groupby("groups").agg(pl.mean("a")).collect()


def test_shift_and_fill():
    df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": [1, 2, 3, 4, 5]})

    # use exprs
    out = df.lazy().with_column(col("a").shift_and_fill(-2, col("b").mean())).collect()
    assert out["a"].null_count() == 0

    # use df method
    out = df.lazy().shift_and_fill(2, col("b").std()).collect()
    assert out["a"].null_count() == 0


def test_arange():
    df = pl.DataFrame({"a": [1, 1, 1]}).lazy()
    result = df.filter(pl.lazy.col("a") >= pl.lazy.arange(0, 3)).collect()
    expected = pl.DataFrame({"a": [1, 1]})
    assert result.frame_equal(expected)


def test_arg_sort():
    df = pl.DataFrame({"a": [4, 1, 3]})
    assert df[col("a").arg_sort()]["a"] == [1, 2, 0]


def test_window_function():
    df = pl.DataFrame(
        {
            "A": [1, 2, 3, 4, 5],
            "fruits": ["banana", "banana", "apple", "apple", "banana"],
            "B": [5, 4, 3, 2, 1],
            "cars": ["beetle", "audi", "beetle", "beetle", "beetle"],
        }
    )

    q = df.lazy().with_columns(
        [
            pl.sum("A").over("fruits").alias("fruit_sum_A"),
            pl.first("B").over("fruits").alias("fruit_first_B"),
            pl.max("B").over("cars").alias("cars_max_B"),
        ]
    )
    out = q.collect()
    assert out["cars_max_B"] == [5, 4, 5, 5, 5]

    out = df[[pl.first("B").over(["fruits", "cars"]).alias("B_first")]]
    assert out["B_first"] == [5, 4, 3, 3, 5]


def test_when_then_flatten():
    df = pl.DataFrame({"foo": [1, 2, 3], "bar": [3, 4, 5]})

    assert df[
        when(col("foo") > 1)
        .then(col("bar"))
        .when(col("bar") < 3)
        .then(10)
        .otherwise(30)
    ]["bar"] == [30, 4, 5]


def test_describe_plan():
    pl.DataFrame({"a": [1]}).lazy().describe_optimized_plan()


def test_window_deadlock():
    np.random.seed(12)

    df = pl.DataFrame(
        {
            "nrs": [1, 2, 3, None, 5],
            "names": ["foo", "ham", "spam", "egg", None],
            "random": np.random.rand(5),
            "groups": ["A", "A", "B", "C", "B"],
        }
    )

    df = df[
        [
            col("*"),  # select all
            col("random").sum().over("groups").alias("sum[random]/groups"),
            col("random").list().over("names").alias("random/name"),
        ]
    ]


def test_concat_str():
    df = pl.DataFrame({"a": ["a", "b", "c"], "b": [1, 2, 3]})

    out = df[[pl.concat_str(["a", "b"], delimiter="-")]]
    assert out["a"] == ["a-1", "a-2", "a-3"]


def test_fold_filter():
    df = pl.DataFrame({"a": [1, 2, 3], "b": [0, 1, 2]})

    out = df.filter(
        pl.fold(
            acc=pl.lit(True),
            f=lambda a, b: a & b,
            exprs=[col(c) > 1 for c in df.columns],
        )
    )

    assert out.shape == (1, 2)

    out = df.filter(
        pl.fold(
            acc=pl.lit(True),
            f=lambda a, b: a | b,
            exprs=[col(c) > 1 for c in df.columns],
        )
    )

    assert out.shape == (3, 2)


def test_head_groupby():
    commodity_prices = {
        "commodity": [
            "Wheat",
            "Wheat",
            "Wheat",
            "Wheat",
            "Corn",
            "Corn",
            "Corn",
            "Corn",
            "Corn",
        ],
        "location": [
            "StPaul",
            "StPaul",
            "StPaul",
            "Chicago",
            "Chicago",
            "Chicago",
            "Chicago",
            "Chicago",
            "Chicago",
        ],
        "seller": [
            "Bob",
            "Charlie",
            "Susan",
            "Paul",
            "Ed",
            "Mary",
            "Paul",
            "Charlie",
            "Norman",
        ],
        "price": [1.0, 0.7, 0.8, 0.55, 2.0, 3.0, 2.4, 1.8, 2.1],
    }
    df = pl.DataFrame(commodity_prices)

    # this query flexes the wildcard exclusion quite a bit.
    keys = ["commodity", "location"]
    out = (
        df.sort(by="price")
        .groupby(keys)
        .agg([col("*").exclude(keys).head(2).list().keep_name()])
        .explode(col("*").exclude(keys))
    )

    assert out.shape == (5, 4)

    df = pl.DataFrame(
        {"letters": ["c", "c", "a", "c", "a", "b"], "nrs": [1, 2, 3, 4, 5, 6]}
    )

    out = df.groupby("letters").tail(2).sort("letters")
    assert out.frame_equal(
        pl.DataFrame({"str": ["a", "a", "b", "c", "c"], "nrs": [3, 5, 6, 2, 4]})
    )
    out = df.groupby("letters").head(2).sort("letters")
    assert out.frame_equal(
        pl.DataFrame({"str": ["a", "a", "b", "c", "c"], "nrs": [3, 5, 6, 1, 2]})
    )


def test_drop_nulls():
    df = pl.DataFrame({"nrs": [1, 2, 3, 4, 5, None]})
    assert df.select(col("nrs").drop_nulls()).shape == (5, 1)


def test_all_expr():
    df = pl.DataFrame({"nrs": [1, 2, 3, 4, 5, None]})
    assert df[[pl.all()]].frame_equal(df)


def test_lazy_columns():
    df = pl.DataFrame(
        {
            "a": [1],
            "b": [1],
            "c": [1],
        }
    ).lazy()

    assert df.select(["a", "c"]).columns == ["a", "c"]


def test_regex_selection():
    df = pl.DataFrame(
        {
            "foo": [1],
            "fooey": [1],
            "foobar": [1],
            "bar": [1],
        }
    ).lazy()

    assert df.select([col("^foo.*$")]).columns == ["foo", "fooey", "foobar"]


def test_literal_projection():
    df = pl.DataFrame({"a": [1, 2]})
    assert df.select([True, 1, 2.0]).dtypes == [pl.Boolean, pl.Int32, pl.Float64]


def test_to_python_datetime():
    df = pl.DataFrame({"a": [1, 2, 3]})
    assert (
        df.select(col("a").cast(pl.Date64).dt.to_python_datetime())["a"].dtype
        == pl.Object
    )
    assert df.select(col("a").cast(pl.Date64).dt.timestamp())["a"].dtype == pl.Int64


def test_interpolate():
    df = pl.DataFrame({"a": [1, None, 3]})
    assert df.select(col("a").interpolate())["a"] == [1, 2, 3]
    assert df["a"].interpolate() == [1, 2, 3]
    assert df.interpolate()["a"] == [1, 2, 3]
    assert df.lazy().interpolate().collect()["a"] == [1, 2, 3]


def test_fill_nan():
    df = pl.DataFrame({"a": [1.0, np.nan, 3.0]})
    assert df.fill_nan(2.0)["a"] == [1.0, 2.0, 3.0]
    assert df.lazy().fill_nan(2.0).collect()["a"] == [1.0, 2.0, 3.0]


def test_take(fruits_cars):
    df = fruits_cars

    # out of bounds error
    with pytest.raises(RuntimeError):
        (
            df.sort("fruits").select(
                [col("B").reverse().take([1, 2]).list().over("fruits"), "fruits"]
            )
        )

    out = df.sort("fruits").select(
        [col("B").reverse().take([0, 1]).list().over("fruits"), "fruits"]
    )

    out[0, "B"] == [2, 3]
    out[4, "B"] == [1, 4]


def test_select_by_col_list(fruits_cars):
    df = fruits_cars
    out = df.select(col(["A", "B"]).sum())
    out.columns == ["A", "B"]
    out.shape == (1, 2)


def test_rolling(fruits_cars):
    df = fruits_cars
    assert df.select(
        [
            col("A").rolling_min(3, min_periods=1).alias("1"),
            col("A").rolling_mean(3, min_periods=1).alias("2"),
            col("A").rolling_max(3, min_periods=1).alias("3"),
            col("A").rolling_sum(3, min_periods=1).alias("4"),
        ]
    ).frame_equal(
        pl.DataFrame(
            {
                "1": [1, 1, 1, 2, 3],
                "2": [1, 1, 2, 3, 4],
                "3": [1, 2, 3, 4, 5],
                "5": [1, 3, 6, 9, 12],
            }
        )
    )


def test_rolling_apply():
    s = pl.Series("A", [1.0, 2.0, 9.0, 2.0, 13.0])
    out = s.rolling_apply(window_size=3, function=lambda s: s.std())
    assert out[0] is None
    assert out[1] is None
    assert out[2] == 4.358898943540674
