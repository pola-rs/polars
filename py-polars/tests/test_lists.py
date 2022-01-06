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
