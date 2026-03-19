import pytest

import polars as pl


@pytest.mark.parametrize(
    "f",
    [
        "min",
        "max",
        "sum",
        "mean",
    ],
)
def test_shape_mismatch_19336(f: str) -> None:
    a = pl.Series([1, 2, 3])
    b = pl.Series([1, 2])
    fn = getattr(pl, f"{f}_horizontal")

    with pytest.raises(pl.exceptions.ShapeError):
        pl.select((fn)(a, b))


def test_fold_reduce_output_dtype_24011() -> None:
    df = pl.DataFrame(
        {
            "x": [0, 1, 2],
            "y": [1.1, 2.2, 3.3],
        }
    )

    def f(acc: pl.Series, x: pl.Series) -> pl.Series:
        return acc + x

    q = df.lazy().select(
        fold=pl.fold(acc=pl.lit(0), function=f, exprs=pl.col("*")),
        reduce=pl.reduce(function=f, exprs=pl.col("*")),
        cum_fold=pl.cum_fold(acc=pl.lit(0), function=f, exprs=pl.col("*")),
        cum_reduce=pl.cum_reduce(function=f, exprs=pl.col("*")),
    )

    assert q.collect_schema() == q.collect().schema
