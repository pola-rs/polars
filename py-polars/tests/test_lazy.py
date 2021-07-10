import polars as pl
from polars.datatypes import *
from polars.lazy import *


def test_lazy():
    df = pl.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]})
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

    out = df[[pl.first("B").over(["fruits", "cars"])]]
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
