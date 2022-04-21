import pandas as pd
from test_series import verify_series_and_expr_api

import polars as pl
from polars import testing


def test_list_arr_get() -> None:
    a = pl.Series("a", [[1, 2, 3], [4, 5], [6, 7, 8, 9]])
    out = a.arr.get(0)
    expected = pl.Series("a", [1, 4, 6])
    testing.assert_series_equal(out, expected)
    out = a.arr.first()
    testing.assert_series_equal(out, expected)
    out = pl.select(pl.lit(a).arr.first()).to_series()
    testing.assert_series_equal(out, expected)

    out = a.arr.get(-1)
    expected = pl.Series("a", [3, 5, 9])
    testing.assert_series_equal(out, expected)
    out = a.arr.last()
    testing.assert_series_equal(out, expected)
    out = pl.select(pl.lit(a).arr.last()).to_series()
    testing.assert_series_equal(out, expected)

    a = pl.Series("a", [[1, 2, 3], [4, 5], [6, 7, 8, 9]])
    out = a.arr.get(-3)
    expected = pl.Series("a", [1, None, 7])
    testing.assert_series_equal(out, expected)


def test_contains() -> None:
    a = pl.Series("a", [[1, 2, 3], [2, 5], [6, 7, 8, 9]])
    out = a.arr.contains(2)
    expected = pl.Series("a", [True, True, False])
    testing.assert_series_equal(out, expected)

    out = pl.select(pl.lit(a).arr.contains(2)).to_series()
    testing.assert_series_equal(out, expected)


def test_dtype() -> None:
    a = pl.Series("a", [[1, 2, 3], [2, 5], [6, 7, 8, 9]])
    assert a.dtype == pl.List
    assert a.inner_dtype == pl.Int64


def test_categorical() -> None:
    # https://github.com/pola-rs/polars/issues/2038
    df = pl.DataFrame(
        [
            pl.Series("a", [1, 1, 1, 1, 1, 1, 1, 1]),
            pl.Series("b", [8, 2, 3, 6, 3, 6, 2, 2]),
            pl.Series("c", ["a", "b", "c", "a", "b", "c", "a", "b"]).cast(
                pl.Categorical
            ),
        ]
    )
    out = (
        df.groupby(["a", "b"])
        .agg(
            [
                pl.col("c").count().alias("num_different_c"),
                pl.col("c").alias("c_values"),
            ]
        )
        .filter(pl.col("num_different_c") >= 2)
        .to_series(3)
    )

    assert out.inner_dtype == pl.Categorical


def test_list_concat_rolling_window() -> None:
    # inspired by: https://stackoverflow.com/questions/70377100/use-the-rolling-function-of-polars-to-get-a-list-of-all-values-in-the-rolling-wi
    # this tests if it works without specifically creating list dtype upfront.
    # note that the given answer is prefered over this snippet as that reuses the list array when shifting
    df = pl.DataFrame(
        {
            "A": [1.0, 2.0, 9.0, 2.0, 13.0],
        }
    )

    out = df.with_columns(
        [pl.col("A").shift(i).alias(f"A_lag_{i}") for i in range(3)]
    ).select(
        [pl.concat_list([f"A_lag_{i}" for i in range(3)][::-1]).alias("A_rolling")]
    )
    assert out.shape == (5, 1)
    assert out.to_series().dtype == pl.List

    # this test proper null behavior of concat list
    out = (
        df.with_column(pl.col("A").reshape((-1, 1)))  # first turn into a list
        .with_columns(
            [
                pl.col("A").shift(i).alias(f"A_lag_{i}")
                for i in range(3)  # slice the lists to a lag
            ]
        )
        .select(
            [
                pl.all(),
                pl.concat_list([f"A_lag_{i}" for i in range(3)][::-1]).alias(
                    "A_rolling"
                ),
            ]
        )
    )
    assert out.shape == (5, 5)
    assert out["A_rolling"].dtype == pl.List


def test_list_append() -> None:
    df = pl.DataFrame({"a": [[1, 2], [1], [1, 2, 3]]})

    out = df.select([pl.col("a").arr.concat(pl.Series([[1, 2]]))])
    assert out["a"][0].to_list() == [1, 2, 1, 2]

    out = df.select([pl.col("a").arr.concat([1, 4])])
    assert out["a"][0].to_list() == [1, 2, 1, 4]

    out_s = df["a"].arr.concat(([4, 1]))
    assert out_s[0].to_list() == [1, 2, 4, 1]


def test_list_arr_empty() -> None:
    df = pl.DataFrame({"cars": [[1, 2, 3], [2, 3], [4], []]})

    out = df.select(
        [
            pl.col("cars").arr.first().alias("cars_first"),
            pl.when(pl.col("cars").arr.first() == 2)
            .then(1)
            .when(pl.col("cars").arr.contains(2))
            .then(2)
            .otherwise(3)
            .alias("cars_literal"),
        ]
    )
    expected = pl.DataFrame(
        {"cars_first": [1, 2, 4, None], "cars_literal": [2, 1, 3, 3]}
    )
    assert out.frame_equal(expected)


def test_list_argminmax() -> None:
    s = pl.Series("a", [[1, 2], [3, 2, 1]])
    expected = pl.Series("a", [0, 2], dtype=pl.UInt32)
    verify_series_and_expr_api(s, expected, "arr.arg_min")
    expected = pl.Series("a", [1, 0], dtype=pl.UInt32)
    verify_series_and_expr_api(s, expected, "arr.arg_max")


def test_list_shift() -> None:
    s = pl.Series("a", [[1, 2], [3, 2, 1]])
    expected = pl.Series("a", [[None, 1], [None, 3, 2]])
    assert s.arr.shift().to_list() == expected.to_list()


def test_list_diff() -> None:
    s = pl.Series("a", [[1, 2], [10, 2, 1]])
    expected = pl.Series("a", [[None, 1], [None, -8, -1]])
    assert s.arr.diff().to_list() == expected.to_list()


def test_slice() -> None:
    vals = [[1, 2, 3, 4], [10, 2, 1]]
    s = pl.Series("a", vals)
    assert s.arr.head(2).to_list() == [[1, 2], [10, 2]]
    assert s.arr.tail(2).to_list() == [[3, 4], [2, 1]]
    assert s.arr.tail(200).to_list() == vals
    assert s.arr.head(200).to_list() == vals
    assert s.arr.slice(1, 2).to_list() == [[2, 3], [2, 1]]


def test_cast_inner() -> None:
    a = pl.Series([[1, 2]])
    for t in [bool, pl.Boolean]:
        b = a.cast(pl.List(t))
        assert b.dtype == pl.List(pl.Boolean)
        assert b.to_list() == [[True, True]]

    # this creates an inner null type
    df = pl.from_pandas(pd.DataFrame(data=[[[]], [[]]], columns=["A"]))
    assert df["A"].cast(pl.List(int)).dtype.inner == pl.Int64  # type: ignore


def test_list_eval_dtype_inference() -> None:
    grades = pl.DataFrame(
        {
            "student": ["bas", "laura", "tim", "jenny"],
            "arithmetic": [10, 5, 6, 8],
            "biology": [4, 6, 2, 7],
            "geography": [8, 4, 9, 7],
        }
    )

    rank_pct = pl.col("").rank(reverse=True) / pl.col("").count()

    # the .arr.first() would fail if .arr.eval did not correctly infer the output type
    assert grades.with_column(
        pl.concat_list(pl.all().exclude("student")).alias("all_grades")
    ).select(
        [
            pl.col("all_grades")
            .arr.eval(rank_pct, parallel=True)
            .alias("grades_rank")
            .arr.first()
        ]
    ).to_series().to_list() == [
        0.3333333432674408,
        0.6666666865348816,
        0.6666666865348816,
        0.3333333432674408,
    ]
