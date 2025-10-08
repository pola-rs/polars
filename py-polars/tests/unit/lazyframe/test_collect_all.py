from typing import cast

import pytest

import polars as pl


def test_collect_all_type_coercion_21805() -> None:
    df = pl.LazyFrame({"A": [1.0, 2.0]})
    df = df.with_columns(pl.col("A").shift().fill_null(2))
    assert pl.collect_all([df])[0]["A"].to_list() == [2.0, 1.0]


@pytest.mark.parametrize("optimizations", [pl.QueryOptFlags(), pl.QueryOptFlags.none()])
def test_collect_all(df: pl.DataFrame, optimizations: pl.QueryOptFlags) -> None:
    lf1 = df.lazy().select(pl.col("int").sum())
    lf2 = df.lazy().select((pl.col("floats") * 2).sum())
    out = pl.collect_all([lf1, lf2], optimizations=optimizations)
    assert cast(int, out[0].item()) == 6
    assert cast(float, out[1].item()) == 12.0
