import polars as pl
from polars.testing import assert_series_equal


def test_is_unique_series() -> None:
    s = pl.Series("a", [1, 2, 2, 3])
    assert_series_equal(s.is_unique(), pl.Series("a", [True, False, False, True]))

    # str
    assert pl.Series(["a", "b", "c", "a"]).is_duplicated().to_list() == [
        True,
        False,
        False,
        True,
    ]
    assert pl.Series(["a", "b", "c", "a"]).is_unique().to_list() == [
        False,
        True,
        True,
        False,
    ]


def test_is_unique() -> None:
    df = pl.DataFrame({"foo": [1, 2, 2], "bar": [6, 7, 7]})

    assert_series_equal(df.is_unique(), pl.Series("", [True, False, False]))
    assert df.unique(maintain_order=True).rows() == [(1, 6), (2, 7)]
    assert df.n_unique() == 2


def test_is_unique2() -> None:
    df = pl.DataFrame({"a": [4, 1, 4]})
    result = df.select(pl.col("a").is_unique())["a"]
    assert_series_equal(result, pl.Series("a", [False, True, False]))


def test_is_unique_null() -> None:
    s = pl.Series([])
    expected = pl.Series([], dtype=pl.Boolean)
    assert_series_equal(s.is_unique(), expected)

    s = pl.Series([None])
    expected = pl.Series([True], dtype=pl.Boolean)
    assert_series_equal(s.is_unique(), expected)

    s = pl.Series([None, None, None])
    expected = pl.Series([False, False, False], dtype=pl.Boolean)
    assert_series_equal(s.is_unique(), expected)


def test_is_unique_struct() -> None:
    assert pl.Series(
        [{"a": 1, "b": 1}, {"a": 2, "b": 1}, {"a": 1, "b": 1}]
    ).is_unique().to_list() == [False, True, False]
    assert pl.Series(
        [{"a": 1, "b": 1}, {"a": 2, "b": 1}, {"a": 1, "b": 1}]
    ).is_duplicated().to_list() == [True, False, True]


def test_is_duplicated_series() -> None:
    s = pl.Series("a", [1, 2, 2, 3])
    assert_series_equal(s.is_duplicated(), pl.Series("a", [False, True, True, False]))


def test_is_duplicated_df() -> None:
    df = pl.DataFrame({"foo": [1, 2, 2], "bar": [6, 7, 7]})
    assert_series_equal(df.is_duplicated(), pl.Series("", [False, True, True]))


def test_is_duplicated_lf() -> None:
    ldf = pl.LazyFrame({"a": [4, 1, 4]}).select(pl.col("a").is_duplicated())
    assert_series_equal(ldf.collect()["a"], pl.Series("a", [True, False, True]))


def test_is_duplicated_null() -> None:
    s = pl.Series([])
    expected = pl.Series([], dtype=pl.Boolean)
    assert_series_equal(s.is_duplicated(), expected)

    s = pl.Series([None])
    expected = pl.Series([False], dtype=pl.Boolean)
    assert_series_equal(s.is_duplicated(), expected)

    s = pl.Series([None, None, None])
    expected = pl.Series([True, True, True], dtype=pl.Boolean)
    assert_series_equal(s.is_duplicated(), expected)
