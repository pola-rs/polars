from functools import reduce
from typing import List, Optional

import polars as pl


def test_apply_none() -> None:
    df = pl.DataFrame(
        {
            "g": [1, 1, 1, 2, 2, 2, 5],
            "a": [2, 4, 5, 190, 1, 4, 1],
            "b": [1, 3, 2, 1, 43, 3, 1],
        }
    )

    out = (
        df.groupby("g", maintain_order=True).agg(
            pl.apply(
                exprs=["a", pl.col("b") ** 4, pl.col("a") / 4],
                f=lambda x: x[0] * x[1] + x[2].sum(),
            ).alias("multiple")
        )
    )["multiple"]
    assert out[0].to_list() == [4.75, 326.75, 82.75]
    assert out[1].to_list() == [238.75, 3418849.75, 372.75]

    out = df.select(pl.map(exprs=["a", "b"], f=lambda s: s[0] * s[1]))
    assert out["a"].to_list() == (df["a"] * df["b"]).to_list()

    # check if we can return None
    def func(s: List) -> Optional[int]:
        if s[0][0] == 190:
            return None
        else:
            return s[0]

    out = (
        df.groupby("g", maintain_order=True).agg(
            pl.apply(exprs=["a", pl.col("b") ** 4, pl.col("a") / 4], f=func).alias(  # type: ignore
                "multiple"
            )
        )
    )["multiple"]
    assert out[1] is None


def test_apply_return_py_object() -> None:
    df = pl.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

    out = df.select([pl.all().map(lambda s: reduce(lambda a, b: a + b, s))])

    assert out.shape == (1, 2)
