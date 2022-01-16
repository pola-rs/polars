import numpy as np
import pytest
from _pytest.capture import CaptureFixture

import polars as pl
from polars import col, lit, map_binary, when


def test_lazy() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]})
    _ = df.lazy().with_column(lit(1).alias("foo")).select([col("a"), col("foo")])

    # test if it executes
    _ = (
        df.lazy()
        .with_column(
            when(col("a").gt(lit(2))).then(lit(10)).otherwise(lit(1)).alias("new")
        )
        .collect()
    )

    # test if pl.list is available, this is `to_list` re-exported as list
    df.groupby("a").agg(pl.list("b"))


def test_apply() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]})
    new = df.lazy().with_column(col("a").map(lambda s: s * 2).alias("foo")).collect()

    expected = df.clone()
    expected["foo"] = expected["a"] * 2

    assert new.frame_equal(expected)


def test_add_eager_column() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]})
    out = df.lazy().with_column(pl.lit(pl.Series("c", [1, 2, 3]))).collect()
    assert out["c"].sum() == 6


def test_set_null() -> None:
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


def test_agg() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]})
    ldf = df.lazy().min()
    assert ldf.collect().shape == (1, 2)


def test_fold() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]})
    out = df.select(
        [
            pl.sum(["a", "b"]),
            pl.max(["a", pl.col("b") ** 2]),
            pl.min(["a", pl.col("b") ** 2]),
        ]
    )
    assert out["sum"].series_equal(pl.Series("sum", [2.0, 4.0, 6.0]))
    assert out["max"].series_equal(pl.Series("max", [1.0, 4.0, 9.0]))
    assert out["min"].series_equal(pl.Series("min", [1.0, 2.0, 3.0]))

    out = df.select(
        pl.fold(acc=lit(0), f=lambda acc, x: acc + x, exprs=pl.col("*")).alias("foo")
    )
    assert out["foo"] == [2, 4, 6]


def test_or() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]})
    out = df.lazy().filter((pl.col("a") == 1) | (pl.col("b") > 2)).collect()
    assert out.shape[0] == 2


def test_groupby_apply() -> None:
    df = pl.DataFrame({"a": [1, 1, 3], "b": [1.0, 2.0, 3.0]})
    ldf = df.lazy().groupby("a").apply(lambda df: df)
    assert ldf.collect().sort("b").frame_equal(df)


def test_binary_function() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]})
    out = (
        df.lazy()
        .with_column(map_binary(col("a"), col("b"), lambda a, b: a + b))
        .collect()
    )
    assert out["binary_function"] == (out.a + out.b)

    # we can also avoid pl.col and insert column names directly
    out = df.lazy().with_column(map_binary("a", "b", lambda a, b: a + b)).collect()
    assert out["binary_function"] == (out.a + out.b)


def test_filter_str() -> None:
    # use a str instead of a column expr
    df = pl.DataFrame(
        {
            "time": ["11:11:00", "11:12:00", "11:13:00", "11:14:00"],
            "bools": [True, False, True, False],
        }
    )
    q = df.lazy()

    # last row based on a filter
    result = q.filter(pl.col("bools")).select(pl.last("*")).collect()
    expected = pl.DataFrame({"time": ["11:13:00"], "bools": [True]})
    assert result.frame_equal(expected)

    # last row based on a filter
    result = q.filter("bools").select(pl.last("*")).collect()
    assert result.frame_equal(expected)


def test_apply_custom_function() -> None:
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


def test_groupby() -> None:
    df = pl.DataFrame({"a": [1.0, None, 3.0, 4.0], "groups": ["a", "a", "b", "b"]})

    expected = pl.DataFrame({"groups": ["a", "b"], "a_mean": [1.0, 3.5]})

    out = df.lazy().groupby("groups").agg(pl.mean("a")).collect()
    assert out.sort(by="groups").frame_equal(expected)

    # refer to column via pl.Expr
    out = df.lazy().groupby(pl.col("groups")).agg(pl.mean("a")).collect()
    assert out.sort(by="groups").frame_equal(expected)


def test_shift(fruits_cars: pl.DataFrame) -> None:
    df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": [1, 2, 3, 4, 5]})
    out = df.select(col("a").shift(1))
    assert out["a"].series_equal(pl.Series("a", [None, 1, 2, 3, 4]), null_equal=True)

    res = fruits_cars.lazy().shift(2).collect()

    expected = pl.DataFrame(
        {
            "A": [None, None, 1, 2, 3],
            "fruits": [None, None, "banana", "banana", "apple"],
            "B": [None, None, 5, 4, 3],
            "cars": [None, None, "beetle", "audi", "beetle"],
        }
    )
    res.frame_equal(expected, null_equal=True)

    # negative value
    res = fruits_cars.lazy().shift(-2).collect()
    for rows in [3, 4]:
        for cols in range(4):
            assert res[rows, cols] is None


def test_shift_and_fill() -> None:
    df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": [1, 2, 3, 4, 5]})

    # use exprs
    out = df.lazy().with_column(col("a").shift_and_fill(-2, col("b").mean())).collect()
    assert out["a"].null_count() == 0

    # use df method
    out = df.lazy().shift_and_fill(2, col("b").std()).collect()
    assert out["a"].null_count() == 0


def test_arange() -> None:
    df = pl.DataFrame({"a": [1, 1, 1]}).lazy()
    result = df.filter(pl.col("a") >= pl.arange(0, 3)).collect()
    expected = pl.DataFrame({"a": [1, 1]})
    assert result.frame_equal(expected)


def test_arg_unique() -> None:
    df = pl.DataFrame({"a": [4, 1, 4]})
    assert df[col("a").arg_unique()]["a"].series_equal(pl.Series("a", [0, 1]))


def test_is_unique() -> None:
    df = pl.DataFrame({"a": [4, 1, 4]})
    assert df[col("a").is_unique()]["a"].series_equal(
        pl.Series("a", [False, True, False])
    )


def test_is_first() -> None:
    df = pl.DataFrame({"a": [4, 1, 4]})
    assert df[col("a").is_first()]["a"].series_equal(
        pl.Series("a", [True, True, False])
    )


def test_is_duplicated() -> None:
    df = pl.DataFrame({"a": [4, 1, 4]})
    assert df[col("a").is_duplicated()]["a"].series_equal(
        pl.Series("a", [True, False, True])
    )


def test_arg_sort() -> None:
    df = pl.DataFrame({"a": [4, 1, 3]})
    assert df[col("a").arg_sort()]["a"] == [1, 2, 0]


def test_window_function() -> None:
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


def test_when_then_flatten() -> None:
    df = pl.DataFrame({"foo": [1, 2, 3], "bar": [3, 4, 5]})

    assert df[
        when(col("foo") > 1)
        .then(col("bar"))
        .when(col("bar") < 3)
        .then(10)
        .otherwise(30)
    ]["bar"] == [30, 4, 5]


def test_describe_plan() -> None:
    assert isinstance(pl.DataFrame({"a": [1]}).lazy().describe_optimized_plan(), str)
    assert isinstance(pl.DataFrame({"a": [1]}).lazy().describe_plan(), str)


def test_inspect(capsys: CaptureFixture) -> None:
    df = pl.DataFrame({"a": [1]})
    df.lazy().inspect().collect()
    captured = capsys.readouterr()
    assert len(captured.out) > 0

    df.select(pl.col("a").cumsum().inspect().alias("bar"))
    res = capsys.readouterr()
    assert len(res.out) > 0


def test_fetch(fruits_cars: pl.DataFrame) -> None:
    res = fruits_cars.lazy().select("*").fetch(2)
    assert res.frame_equal(res[:2])


def test_window_deadlock() -> None:
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


def test_concat_str() -> None:
    df = pl.DataFrame({"a": ["a", "b", "c"], "b": [1, 2, 3]})

    out = df[[pl.concat_str(["a", "b"], sep="-")]]
    assert out["a"] == ["a-1", "a-2", "a-3"]

    out = df.select([pl.format("foo_{}_bar_{}", pl.col("a"), "b").alias("fmt")])

    assert out["fmt"].to_list() == ["foo_a_bar_1", "foo_b_bar_2", "foo_c_bar_3"]


def test_fold_filter() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [0, 1, 2]})

    out = df.filter(
        pl.fold(
            acc=pl.lit(True),
            f=lambda a, b: a & b,
            exprs=[pl.col(c) > 1 for c in df.columns],
        )
    )

    assert out.shape == (1, 2)

    out = df.filter(
        pl.fold(
            acc=pl.lit(True),
            f=lambda a, b: a | b,
            exprs=[pl.col(c) > 1 for c in df.columns],
        )
    )

    assert out.shape == (3, 2)


def test_head_groupby() -> None:
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
        pl.DataFrame({"letters": ["a", "a", "b", "c", "c"], "nrs": [3, 5, 6, 2, 4]})
    )
    out = df.groupby("letters").head(2).sort("letters")
    assert out.frame_equal(
        pl.DataFrame({"letters": ["a", "a", "b", "c", "c"], "nrs": [3, 5, 6, 1, 2]})
    )


def test_is_null_is_not_null() -> None:
    df = pl.DataFrame({"nrs": [1, 2, None]})
    assert df.select(col("nrs").is_null())["nrs"].to_list() == [False, False, True]
    assert df.select(col("nrs").is_not_null())["nrs"].to_list() == [True, True, False]


def test_is_nan_is_not_nan() -> None:
    df = pl.DataFrame({"nrs": np.array([1, 2, np.nan])})
    assert df.select(col("nrs").is_nan())["nrs"].to_list() == [False, False, True]
    assert df.select(col("nrs").is_not_nan())["nrs"].to_list() == [True, True, False]


def test_is_finite_is_infinite() -> None:
    df = pl.DataFrame({"nrs": np.array([1, 2, np.inf])})
    assert df.select(col("nrs").is_infinite())["nrs"].to_list() == [False, False, True]
    assert df.select(col("nrs").is_finite())["nrs"].to_list() == [True, True, False]


def test_len() -> None:
    df = pl.DataFrame({"nrs": [1, 2, 3]})
    assert df.select(col("nrs").len())[0, 0] == 3


def test_cum_agg() -> None:
    df = pl.DataFrame({"a": [1, 2, 3, 2]})
    assert df.select(pl.col("a").cumsum())["a"].series_equal(
        pl.Series("a", [1, 3, 6, 8])
    )
    assert df.select(pl.col("a").cummin())["a"].series_equal(
        pl.Series("a", [1, 1, 1, 1])
    )
    assert df.select(pl.col("a").cummax())["a"].series_equal(
        pl.Series("a", [1, 2, 3, 3])
    )
    assert df.select(pl.col("a").cumprod())["a"].series_equal(
        pl.Series("a", [1, 2, 6, 12])
    )


def test_floor() -> None:
    df = pl.DataFrame({"a": [1.8, 1.2, 3.0]})
    assert df.select(pl.col("a").floor())["a"].series_equal(pl.Series("a", [1, 1, 3]))


def test_round() -> None:
    df = pl.DataFrame({"a": [1.8, 1.2, 3.0]})
    assert df.select(pl.col("a").round(decimals=0))["a"].series_equal(
        pl.Series("a", [2, 1, 3])
    )


def test_dot() -> None:
    df = pl.DataFrame({"a": [1.8, 1.2, 3.0], "b": [3.2, 1, 2]})
    assert df.select(pl.col("a").dot(pl.col("b")))[0, 0] == 12.96


def test_sort() -> None:
    df = pl.DataFrame({"a": [1, 2, 3, 2]})
    assert df.select(pl.col("a").sort())["a"].series_equal(pl.Series("a", [1, 2, 2, 3]))


def test_drop_nulls() -> None:
    df = pl.DataFrame({"nrs": [1, 2, 3, 4, 5, None]})
    assert df.select(col("nrs").drop_nulls()).shape == (5, 1)

    df = pl.DataFrame({"foo": [1, 2, 3], "bar": [6, None, 8], "ham": ["a", "b", "c"]})
    expected = pl.DataFrame({"foo": [1, 3], "bar": [6, 8], "ham": ["a", "c"]})
    df.lazy().drop_nulls().collect().frame_equal(expected)


def test_all_expr() -> None:
    df = pl.DataFrame({"nrs": [1, 2, 3, 4, 5, None]})
    assert df[[pl.all()]].frame_equal(df)


def test_any_expr(fruits_cars: pl.DataFrame) -> None:
    assert fruits_cars.select(pl.any("A"))[0, 0]
    assert fruits_cars.select(pl.any([pl.col("A"), pl.col("B")]))[0, 0]


def test_lazy_columns() -> None:
    df = pl.DataFrame(
        {
            "a": [1],
            "b": [1],
            "c": [1],
        }
    ).lazy()

    assert df.select(["a", "c"]).columns == ["a", "c"]


def test_regex_selection() -> None:
    df = pl.DataFrame(
        {
            "foo": [1],
            "fooey": [1],
            "foobar": [1],
            "bar": [1],
        }
    ).lazy()

    assert df.select([col("^foo.*$")]).columns == ["foo", "fooey", "foobar"]


def test_exclude_selection() -> None:
    df = pl.DataFrame({"a": [1], "b": [1], "c": [True]}).lazy()

    assert df.select([pl.exclude("a")]).columns == ["b", "c"]
    assert df.select(pl.all().exclude(pl.Boolean)).columns == ["a", "b"]
    assert df.select(pl.all().exclude([pl.Boolean])).columns == ["a", "b"]


def test_col_series_selection() -> None:
    df = pl.DataFrame({"a": [1], "b": [1], "c": [1]}).lazy()
    srs = pl.Series(["b", "c"])

    assert df.select(pl.col(srs)).columns == ["b", "c"]


def test_literal_projection() -> None:
    df = pl.DataFrame({"a": [1, 2]})
    assert df.select([True]).dtypes == [pl.Boolean]
    assert df.select([1]).dtypes == [pl.Int32]
    assert df.select([2.0]).dtypes == [pl.Float64]


def test_interpolate() -> None:
    df = pl.DataFrame({"a": [1, None, 3]})
    assert df.select(col("a").interpolate())["a"] == [1, 2, 3]
    assert df["a"].interpolate() == [1, 2, 3]
    assert df.interpolate()["a"] == [1, 2, 3]
    assert df.lazy().interpolate().collect()["a"] == [1, 2, 3]


def test_fill_nan() -> None:
    df = pl.DataFrame({"a": [1.0, np.nan, 3.0]})
    assert df.fill_nan(2.0)["a"].series_equal(pl.Series("a", [1.0, 2.0, 3.0]))
    assert (
        df.lazy()
        .fill_nan(2.0)
        .collect()["a"]
        .series_equal(pl.Series("a", [1.0, 2.0, 3.0]))
    )
    assert df.select(pl.col("a").fill_nan(2))["literal"].series_equal(
        pl.Series("literal", [1.0, 2.0, 3.0])
    )


def test_fill_null() -> None:
    df = pl.DataFrame({"a": [1.0, None, 3.0]})
    assert df.select([pl.col("a").fill_null("min")])["a"][1] == 1.0
    assert df.lazy().fill_null(2).collect()["a"] == [1.0, 2.0, 3.0]


def test_backward_fill() -> None:
    df = pl.DataFrame({"a": [1.0, None, 3.0]})
    assert df.select([pl.col("a").backward_fill()])["a"].series_equal(
        pl.Series("a", [1, 3, 3])
    )


def test_take(fruits_cars: pl.DataFrame) -> None:
    df = fruits_cars

    # out of bounds error
    with pytest.raises(RuntimeError):
        (
            df.sort("fruits").select(
                [col("B").reverse().take([1, 2]).list().over("fruits"), "fruits"]  # type: ignore
            )
        )

    for index in [[0, 1], pl.Series([0, 1]), np.array([0, 1]), pl.lit(1)]:
        out = df.sort("fruits").select(
            [col("B").reverse().take(index).list().over("fruits"), "fruits"]  # type: ignore
        )

        assert out[0, "B"] == [2, 3]
        assert out[4, "B"] == [1, 4]


def test_select_by_col_list(fruits_cars: pl.DataFrame) -> None:
    df = fruits_cars
    out = df.select(col(["A", "B"]).sum())
    assert out.columns == ["A", "B"]
    assert out.shape == (1, 2)


def test_rolling(fruits_cars: pl.DataFrame) -> None:
    df = fruits_cars
    out = df.select(
        [
            pl.col("A").rolling_min(3, min_periods=1).alias("1"),
            pl.col("A").rolling_min(3).alias("1b"),
            pl.col("A").rolling_mean(3, min_periods=1).alias("2"),
            pl.col("A").rolling_mean(3).alias("2b"),
            pl.col("A").rolling_max(3, min_periods=1).alias("3"),
            pl.col("A").rolling_max(3).alias("3b"),
            pl.col("A").rolling_sum(3, min_periods=1).alias("4"),
            pl.col("A").rolling_sum(3).alias("4b"),
            # below we use .round purely for the ability to do .frame_equal()
            pl.col("A").rolling_std(3, min_periods=1).round(decimals=4).alias("std"),
            pl.col("A").rolling_std(3).alias("std2"),
            pl.col("A").rolling_var(3, min_periods=1).round(decimals=4).alias("var"),
            pl.col("A").rolling_var(3).alias("var2"),
        ]
    )

    # TODO: rolling_std & rolling_var return nan instead of null if it cant compute
    out[0, "std"] = None
    out[0, "var"] = None

    assert out.frame_equal(
        pl.DataFrame(
            {
                "1": [1, 1, 1, 2, 3],
                "1b": [None, None, 1, 2, 3],
                "2": [1.0, 1.5, 2.0, 3.0, 4.0],
                "2b": [None, None, 2.0, 3.0, 4.0],
                "3": [1, 2, 3, 4, 5],
                "3b": [None, None, 3, 4, 5],
                "4": [1, 3, 6, 9, 12],
                "4b": [None, None, 6, 9, 12],
                "std": [None, 0.7071, 1, 1, 1],
                "std2": [None, None, 1, 1, 1],
                "var": [None, 0.5, 1, 1, 1],
                "var2": [None, None, 1, 1, 1],
            }
        )
    )


def test_rolling_apply() -> None:
    s = pl.Series("A", [1.0, 2.0, 9.0, 2.0, 13.0], dtype=pl.Float64)
    out = s.rolling_apply(function=lambda s: s.std(), window_size=3)
    assert out[0] is None
    assert out[1] is None
    assert out[2] == 4.358898943540674
    assert out.dtype is pl.Float64

    s = pl.Series("A", [1.0, 2.0, 9.0, 2.0, 13.0], dtype=pl.Float32)
    out = s.rolling_apply(function=lambda s: s.std(), window_size=3)
    assert out[0] is None
    assert out[1] is None
    assert out[2] == 4.358899116516113
    assert out.dtype is pl.Float32

    s = pl.Series("A", [1, 2, 9, 2, 13], dtype=pl.Int32)
    out = s.rolling_apply(function=lambda s: s.sum(), window_size=3)
    assert out[0] is None
    assert out[1] is None
    assert out[2] == 12
    assert out.dtype is pl.Int32

    s = pl.Series("A", [1.0, 2.0, 9.0, 2.0, 13.0], dtype=pl.Float64)
    out = s.rolling_apply(
        function=lambda s: s.std(), window_size=3, weights=[1.0, 2.0, 3.0]
    )
    assert out[0] is None
    assert out[1] is None
    assert out[2] == 14.224392195567912
    assert out.dtype is pl.Float64

    s = pl.Series("A", [1.0, 2.0, 9.0, 2.0, 13.0], dtype=pl.Float32)
    out = s.rolling_apply(
        function=lambda s: s.std(), window_size=3, weights=[1.0, 2.0, 3.0]
    )
    assert out[0] is None
    assert out[1] is None
    assert out[2] == 14.22439193725586
    assert out.dtype is pl.Float32

    s = pl.Series("A", [1, 2, 9, None, 13], dtype=pl.Int32)
    out = s.rolling_apply(
        function=lambda s: s.sum(), window_size=3, weights=[1.0, 2.0, 3.0]
    )
    assert out[0] is None
    assert out[1] is None
    assert out[2] == 32.0
    assert out.dtype is pl.Float64
    s = pl.Series("A", [1, 2, 9, 2, 10])

    # compare rolling_apply to specific rolling functions
    s = pl.Series("A", list(range(5)), dtype=pl.Float64)
    roll_app_sum = s.rolling_apply(
        function=lambda s: s.sum(),
        window_size=3,
        weights=[1.0, 2.1, 3.2],
        min_periods=2,
        center=True,
    )

    roll_sum = s.rolling_sum(
        window_size=3, weights=[1.0, 2.1, 3.2], min_periods=2, center=True
    )

    assert (roll_app_sum - roll_sum).abs().sum() < 0.0001

    s = pl.Series("A", list(range(6)), dtype=pl.Float64)
    roll_app_sum = s.rolling_apply(
        function=lambda s: s.sum(),
        window_size=4,
        weights=[1.0, 2.0, 3.0, 0.1],
        min_periods=3,
        center=False,
    )

    roll_sum = s.rolling_sum(
        window_size=4, weights=[1.0, 2.0, 3.0, 0.1], min_periods=3, center=False
    )

    assert (roll_app_sum - roll_sum).abs().sum() < 0.0001


def test_arr_namespace(fruits_cars: pl.DataFrame) -> None:
    df = fruits_cars
    out = df.select(
        [  # type: ignore
            "fruits",
            col("B").over("fruits").arr.min().alias("B_by_fruits_min1"),
            col("B").min().over("fruits").alias("B_by_fruits_min2"),
            col("B").over("fruits").arr.max().alias("B_by_fruits_max1"),
            col("B").max().over("fruits").alias("B_by_fruits_max2"),
            col("B").over("fruits").arr.sum().alias("B_by_fruits_sum1"),
            col("B").sum().over("fruits").alias("B_by_fruits_sum2"),
            col("B").over("fruits").arr.mean().alias("B_by_fruits_mean1"),
            col("B").mean().over("fruits").alias("B_by_fruits_mean2"),
        ]
    )
    expected = pl.DataFrame(
        {
            "fruits": ["banana", "banana", "apple", "apple", "banana"],
            "B_by_fruits_min1": [1, 1, 2, 2, 1],
            "B_by_fruits_min2": [1, 1, 2, 2, 1],
            "B_by_fruits_max1": [5, 5, 3, 3, 5],
            "B_by_fruits_max2": [5, 5, 3, 3, 5],
            "B_by_fruits_sum1": [10, 10, 5, 5, 10],
            "B_by_fruits_sum2": [10, 10, 5, 5, 10],
            "B_by_fruits_mean1": [
                3.3333333333333335,
                3.3333333333333335,
                2.5,
                2.5,
                3.3333333333333335,
            ],
            "B_by_fruits_mean2": [
                3.3333333333333335,
                3.3333333333333335,
                2.5,
                2.5,
                3.3333333333333335,
            ],
        }
    )
    assert out.frame_equal(expected, null_equal=True)


def test_arithmetic() -> None:
    df = pl.DataFrame({"a": [1, 2, 3]})

    out = df.select(
        [
            (col("a") % 2).alias("1"),
            (2 % col("a")).alias("2"),
            (1 // col("a")).alias("3"),
            (1 * col("a")).alias("4"),
            (1 + col("a")).alias("5"),
            (1 - col("a")).alias("6"),
            (col("a") // 2).alias("7"),
            (col("a") * 2).alias("8"),
            (col("a") + 2).alias("9"),
            (col("a") - 2).alias("10"),
            (-col("a")).alias("11"),
        ]
    )
    expected = pl.DataFrame(
        {
            "1": [1, 0, 1],
            "2": [0, 0, 2],
            "3": [1, 0, 0],
            "4": [1, 2, 3],
            "5": [2, 3, 4],
            "6": [0, -1, -2],
            "7": [0, 1, 1],
            "8": [2, 4, 6],
            "9": [3, 4, 5],
            "10": [-1, 0, 1],
            "11": [-1, -2, -3],
        }
    )
    assert out.frame_equal(expected)


def test_ufunc() -> None:
    df = pl.DataFrame({"a": [1, 2]})
    out = df.select(np.log(col("a")))  # type: ignore
    assert out["a"][1] == 0.6931471805599453


def test_clip() -> None:
    df = pl.DataFrame({"a": [1, 2, 3, 4, 5]})
    assert df.select(pl.col("a").clip(2, 4))["a"].to_list() == [2, 2, 3, 4, 4]
    assert pl.Series([1, 2, 3, 4, 5]).clip(2, 4).to_list() == [2, 2, 3, 4, 4]


def test_argminmax() -> None:
    df = pl.DataFrame({"a": [1, 2, 3, 4, 5]})
    out = df.select(
        [
            pl.col("a").arg_min().alias("min"),
            pl.col("a").arg_max().alias("max"),
        ]
    )
    assert out["max"][0] == 4
    assert out["min"][0] == 0


def test_expr_bool_cmp() -> None:
    # Since expressions are lazy they should not be evaluated as
    # bool(x), this has the nice side effect of throwing an error
    # if someone tries to chain them via the and|or operators
    df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": [1, 2, 3, 4, 5]})

    with pytest.raises(ValueError):
        df[[pl.col("a").gt(pl.col("b")) and pl.col("b").gt(pl.col("b"))]]

    with pytest.raises(ValueError):
        df[[pl.col("a").gt(pl.col("b")) or pl.col("b").gt(pl.col("b"))]]


def test_is_in() -> None:
    df = pl.DataFrame({"a": [1, 2, 3]})
    assert df.select(pl.col("a").is_in([1, 2]))["a"].to_list() == [
        True,
        True,
        False,
    ]


def test_rename() -> None:
    lf = pl.DataFrame({"a": [1], "b": [2], "c": [3]}).lazy()
    out = lf.rename({"a": "foo", "b": "bar"}).collect()
    assert out.columns == ["foo", "bar", "c"]


def test_drop_columns() -> None:
    out = pl.DataFrame({"a": [1], "b": [2], "c": [3]}).lazy().drop(["a", "b"])
    assert out.columns == ["c"]

    out = pl.DataFrame({"a": [1], "b": [2], "c": [3]}).lazy().drop("a")
    assert out.columns == ["b", "c"]


def test_with_column_renamed(fruits_cars: pl.DataFrame) -> None:
    res = fruits_cars.lazy().with_column_renamed("A", "C").collect()
    assert res.columns[0] == "C"


def test_reverse() -> None:
    out = pl.DataFrame({"a": [1, 2], "b": [3, 4]}).lazy().reverse()
    expected = pl.DataFrame({"a": [2, 1], "b": [4, 3]})
    assert out.collect().frame_equal(expected)


def test_limit(fruits_cars: pl.DataFrame) -> None:
    assert fruits_cars.lazy().limit(1).collect().frame_equal(fruits_cars[0, :])


def test_head(fruits_cars: pl.DataFrame) -> None:
    assert fruits_cars.lazy().head(2).collect().frame_equal(fruits_cars[:2, :])


def test_tail(fruits_cars: pl.DataFrame) -> None:
    assert fruits_cars.lazy().tail(2).collect().frame_equal(fruits_cars[3:, :])


def test_last(fruits_cars: pl.DataFrame) -> None:
    assert (
        fruits_cars.lazy()
        .last()
        .collect()
        .frame_equal(fruits_cars[(len(fruits_cars) - 1) :, :])
    )


def test_first(fruits_cars: pl.DataFrame) -> None:
    assert fruits_cars.lazy().first().collect().frame_equal(fruits_cars[0, :])


def test_join_suffix() -> None:
    df_left = pl.DataFrame(
        {
            "a": ["a", "b", "a", "z"],
            "b": [1, 2, 3, 4],
            "c": [6, 5, 4, 3],
        }
    )
    df_right = pl.DataFrame(
        {
            "a": ["b", "c", "b", "a"],
            "b": [0, 3, 9, 6],
            "c": [1, 0, 2, 1],
        }
    )
    out = df_left.join(df_right, on="a", suffix="_bar")
    assert out.columns == ["a", "b", "c", "b_bar", "c_bar"]
    out = df_left.lazy().join(df_right.lazy(), on="a", suffix="_bar").collect()
    assert out.columns == ["a", "b", "c", "b_bar", "c_bar"]


def test_str_concat() -> None:
    df = pl.DataFrame({"foo": [1, None, 2]})
    df = df.select(pl.col("foo").str_concat("-"))
    assert df[0, 0] == "1-null-2"


@pytest.mark.parametrize("no_optimization", [False, True])
def test_collect_all(df: pl.DataFrame, no_optimization: bool) -> None:
    lf1 = df.lazy().select(pl.col("int").sum())
    lf2 = df.lazy().select((pl.col("floats") * 2).sum())
    out = pl.collect_all([lf1, lf2], no_optimization=no_optimization)
    assert out[0][0, 0] == 6
    assert out[1][0, 0] == 12.0


def test_spearman_corr() -> None:
    df = pl.DataFrame(
        {
            "era": [1, 1, 1, 2, 2, 2],
            "prediction": [2, 4, 5, 190, 1, 4],
            "target": [1, 3, 2, 1, 43, 3],
        }
    )

    out = (
        df.groupby("era", maintain_order=True).agg(
            pl.spearman_rank_corr(pl.col("prediction"), pl.col("target")).alias("c"),
        )
    )["c"]
    assert np.isclose(out[0], 0.5)
    assert np.isclose(out[1], -1.0)

    # we can also pass in column names directly
    out = (
        df.groupby("era", maintain_order=True).agg(
            pl.spearman_rank_corr("prediction", "target").alias("c"),
        )
    )["c"]
    assert np.isclose(out[0], 0.5)
    assert np.isclose(out[1], -1.0)


def test_pearson_corr() -> None:
    df = pl.DataFrame(
        {
            "era": [1, 1, 1, 2, 2, 2],
            "prediction": [2, 4, 5, 190, 1, 4],
            "target": [1, 3, 2, 1, 43, 3],
        }
    )

    out = (
        df.groupby("era", maintain_order=True).agg(
            pl.pearson_corr(pl.col("prediction"), pl.col("target")).alias("c"),
        )
    )["c"]
    assert out.to_list() == pytest.approx([0.6546536707079772, -5.477514993831792e-1])

    # we can also pass in column names directly
    out = (
        df.groupby("era", maintain_order=True).agg(
            pl.pearson_corr("prediction", "target").alias("c"),
        )
    )["c"]
    assert out.to_list() == pytest.approx([0.6546536707079772, -5.477514993831792e-1])


def test_cov(fruits_cars: pl.DataFrame) -> None:
    assert fruits_cars.select(pl.cov("A", "B"))[0, 0] == -2.5
    assert fruits_cars.select(pl.cov(pl.col("A"), pl.col("B")))[0, 0] == -2.5


def test_std(fruits_cars: pl.DataFrame) -> None:
    assert fruits_cars.lazy().std().collect()["A"][0] == pytest.approx(
        1.5811388300841898
    )


def test_var(fruits_cars: pl.DataFrame) -> None:
    assert fruits_cars.lazy().var().collect()["A"][0] == pytest.approx(2.5)


def test_max(fruits_cars: pl.DataFrame) -> None:
    assert fruits_cars.lazy().max().collect()["A"][0] == 5
    assert fruits_cars.select(pl.col("A").max())["A"][0] == 5


def test_min(fruits_cars: pl.DataFrame) -> None:
    assert fruits_cars.lazy().min().collect()["A"][0] == 1
    assert fruits_cars.select(pl.col("A").min())["A"][0] == 1


def test_median(fruits_cars: pl.DataFrame) -> None:
    assert fruits_cars.lazy().median().collect()["A"][0] == 3
    assert fruits_cars.select(pl.col("A").median())["A"][0] == 3


def test_quantile(fruits_cars: pl.DataFrame) -> None:
    assert fruits_cars.lazy().quantile(0.25, "nearest").collect()["A"][0] == 2
    assert fruits_cars.select(pl.col("A").quantile(0.25, "nearest"))["A"][0] == 2

    assert fruits_cars.lazy().quantile(0.24, "lower").collect()["A"][0] == 1
    assert fruits_cars.select(pl.col("A").quantile(0.24, "lower"))["A"][0] == 1

    assert fruits_cars.lazy().quantile(0.26, "higher").collect()["A"][0] == 3
    assert fruits_cars.select(pl.col("A").quantile(0.26, "higher"))["A"][0] == 3

    assert fruits_cars.lazy().quantile(0.24, "midpoint").collect()["A"][0] == 1
    assert fruits_cars.select(pl.col("A").quantile(0.24, "midpoint"))["A"][0] == 1

    assert fruits_cars.lazy().quantile(0.24, "linear").collect()["A"][0] == 1
    assert fruits_cars.select(pl.col("A").quantile(0.24, "linear"))["A"][0] == 1


def test_is_between(fruits_cars: pl.DataFrame) -> None:
    assert fruits_cars.select(pl.col("A").is_between(2, 4))["is_between"].series_equal(  # type: ignore
        pl.Series("is_between", [False, False, True, False, False])
    )
    assert fruits_cars.select(pl.col("A").is_between(2, 4, False))["is_between"].series_equal(  # type: ignore
        pl.Series("is_between", [False, False, True, False, False])
    )
    assert fruits_cars.select(pl.col("A").is_between(2, 4, [False, False]))["is_between"].series_equal(  # type: ignore
        pl.Series("is_between", [False, False, True, False, False])
    )
    assert fruits_cars.select(pl.col("A").is_between(2, 4, True))["is_between"].series_equal(  # type: ignore
        pl.Series("is_between", [False, True, True, True, False])
    )
    assert fruits_cars.select(pl.col("A").is_between(2, 4, [True, True]))["is_between"].series_equal(  # type: ignore
        pl.Series("is_between", [False, True, True, True, False])
    )
    assert fruits_cars.select(pl.col("A").is_between(2, 4, [False, True]))["is_between"].series_equal(  # type: ignore
        pl.Series("is_between", [False, False, True, True, False])
    )
    assert fruits_cars.select(pl.col("A").is_between(2, 4, [True, False]))["is_between"].series_equal(  # type: ignore
        pl.Series("is_between", [False, True, True, False, False])
    )


def test_drop_duplicates() -> None:
    df = pl.DataFrame({"a": [1, 2, 2], "b": [3, 3, 3]})

    expected = pl.DataFrame({"a": [1, 2], "b": [3, 3]})
    assert (
        df.lazy().drop_duplicates(maintain_order=True).collect().frame_equal(expected)
    )

    expected = pl.DataFrame({"a": [1], "b": [3]})
    assert (
        df.lazy()
        .drop_duplicates(subset="b", maintain_order=True)
        .collect()
        .frame_equal(expected)
    )


def test_lazy_concat(df: pl.DataFrame) -> None:
    shape = df.shape
    shape = (shape[0] * 2, shape[1])

    out = pl.concat([df.lazy(), df.lazy()]).collect()  # type: ignore
    assert out.shape == shape
    assert out.frame_equal(df.vstack(df.clone()), null_equal=True)


def test_max_min_multiple_columns(fruits_cars: pl.DataFrame) -> None:
    res = fruits_cars.select(pl.max(["A", "B"]).alias("max"))
    assert res.to_series(0).series_equal(pl.Series("max", [5, 4, 3, 4, 5]))

    res = fruits_cars.select(pl.min(["A", "B"]).alias("min"))
    assert res.to_series(0).series_equal(pl.Series("min", [1, 2, 3, 2, 1]))


def test_head_tail(fruits_cars: pl.DataFrame) -> None:
    res_expr = fruits_cars.select([pl.head("A", 2)])
    res_series = pl.head(fruits_cars["A"], 2)
    expected = pl.Series("A", [1, 2])
    assert res_expr.to_series(0).series_equal(expected)
    assert res_series.series_equal(expected)

    res_expr = fruits_cars.select([pl.tail("A", 2)])
    res_series = pl.tail(fruits_cars["A"], 2)
    expected = pl.Series("A", [4, 5])
    assert res_expr.to_series(0).series_equal(expected)
    assert res_series.series_equal(expected)


def test_lower_bound_upper_bound(fruits_cars: pl.DataFrame) -> None:
    res_expr = fruits_cars.select(pl.col("A").lower_bound())
    assert res_expr["A"][0] < -10_000_000
    res_expr = fruits_cars.select(pl.col("A").upper_bound())
    assert res_expr["A"][0] > 10_000_000
