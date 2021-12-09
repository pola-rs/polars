import polars as pl
from polars import testing


def test_list_arr_get() -> None:
    a = pl.Series("a", [[1, 2, 3], [4, 5], [6, 7, 8, 9]])
    out = a.arr.get(0)
    expected = pl.Series("a", [1, 4, 6])
    testing.assert_series_equal(out, expected)

    out = a.arr.get(-1)
    expected = pl.Series("a", [3, 5, 9])
    testing.assert_series_equal(out, expected)

    out = a.arr.get(-3)
    expected = pl.Series("a", [1, None, 7])
    testing.assert_series_equal(out, expected)
