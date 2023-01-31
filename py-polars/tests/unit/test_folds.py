import polars as pl
from polars.testing import assert_series_equal


def test_fold() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]})
    out = df.select(
        [
            pl.sum(["a", "b"]),
            pl.max(["a", pl.col("b") ** 2]),
            pl.min(["a", pl.col("b") ** 2]),
        ]
    )
    assert_series_equal(out["sum"], pl.Series("sum", [2.0, 4.0, 6.0]))
    assert_series_equal(out["max"], pl.Series("max", [1.0, 4.0, 9.0]))
    assert_series_equal(out["min"], pl.Series("min", [1.0, 2.0, 3.0]))

    out = df.select(
        pl.fold(acc=pl.lit(0), f=lambda acc, x: acc + x, exprs=pl.all()).alias("foo")
    )
    assert out["foo"].to_list() == [2, 4, 6]
    out = df.select(pl.reduce(f=lambda acc, x: acc + x, exprs=pl.all()).alias("foo"))
    assert out["foo"].to_list() == [2, 4, 6]


def test_cumfold() -> None:
    df = pl.DataFrame(
        {
            "a": [1, 2, 3, 4],
            "b": [5, 6, 7, 8],
            "c": [10, 20, 30, 40],
        }
    )

    assert df.select(
        [pl.cumfold(pl.lit(0), lambda a, b: a + b, pl.all()).alias("folded")]
    ).unnest("folded").to_dict(False) == {
        "a": [1, 2, 3, 4],
        "b": [6, 8, 10, 12],
        "c": [16, 28, 40, 52],
    }
    assert df.select(
        [pl.cumreduce(lambda a, b: a + b, pl.all()).alias("folded")]
    ).unnest("folded").to_dict(False) == {
        "a": [1, 2, 3, 4],
        "b": [6, 8, 10, 12],
        "c": [16, 28, 40, 52],
    }


def test_cumsum_fold() -> None:
    df = pl.DataFrame(
        {
            "a": [1, 2],
            "b": [3, 4],
            "c": [5, 6],
        }
    )
    assert df.select(pl.cumsum(["a", "c"])).to_dict(False) == {
        "cumsum": [{"a": 1, "c": 6}, {"a": 2, "c": 8}]
    }
