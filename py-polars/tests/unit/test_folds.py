import polars as pl


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
