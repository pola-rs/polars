import polars as pl
from polars.testing import assert_frame_equal


def test_fold_reduce() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]})

    out = df.select(
        pl.fold(acc=pl.lit(0), function=lambda acc, x: acc + x, exprs=pl.all()).alias(
            "foo"
        )
    )
    assert out["foo"].to_list() == [2, 4, 6]
    out = df.select(
        pl.reduce(function=lambda acc, x: acc + x, exprs=pl.all()).alias("foo")
    )
    assert out["foo"].to_list() == [2, 4, 6]


def test_cum_fold() -> None:
    df = pl.DataFrame(
        {
            "a": [1, 2, 3, 4],
            "b": [5, 6, 7, 8],
            "c": [10, 20, 30, 40],
        }
    )
    result = df.select(pl.cum_fold(pl.lit(0), lambda a, b: a + b, pl.all()))
    expected = pl.DataFrame(
        {
            "cum_fold": [
                {"a": 1, "b": 6, "c": 16},
                {"a": 2, "b": 8, "c": 28},
                {"a": 3, "b": 10, "c": 40},
                {"a": 4, "b": 12, "c": 52},
            ]
        }
    )
    assert_frame_equal(result, expected)


def test_cum_reduce() -> None:
    df = pl.DataFrame(
        {
            "a": [1, 2, 3, 4],
            "b": [5, 6, 7, 8],
            "c": [10, 20, 30, 40],
        }
    )
    result = df.select(pl.cum_reduce(lambda a, b: a + b, pl.all()))
    expected = pl.DataFrame(
        {
            "cum_reduce": [
                {"a": 1, "b": 6, "c": 16},
                {"a": 2, "b": 8, "c": 28},
                {"a": 3, "b": 10, "c": 40},
                {"a": 4, "b": 12, "c": 52},
            ]
        }
    )
    assert_frame_equal(result, expected)
