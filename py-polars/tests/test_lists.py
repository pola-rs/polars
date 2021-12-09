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
