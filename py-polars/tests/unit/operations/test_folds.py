import polars as pl


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
    ).unnest("folded").to_dict(as_series=False) == {
        "a": [1, 2, 3, 4],
        "b": [6, 8, 10, 12],
        "c": [16, 28, 40, 52],
    }
    assert df.select(
        [pl.cumreduce(lambda a, b: a + b, pl.all()).alias("folded")]
    ).unnest("folded").to_dict(as_series=False) == {
        "a": [1, 2, 3, 4],
        "b": [6, 8, 10, 12],
        "c": [16, 28, 40, 52],
    }
