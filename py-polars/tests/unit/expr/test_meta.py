import pytest

import polars as pl


def test_expression_hash_set() -> None:
    a1 = pl.col("a")
    a2 = pl.col("a")
    b1 = pl.col("b") + 1
    b2 = pl.col("b") + 2
    b3 = pl.col("b") + 2

    s = {e.meta for e in [a1, a2, b1, b2, b3]}
    assert len(s) == 3


def test_hash_expr_hint() -> None:
    a = pl.col("a")

    with pytest.raises(
        TypeError, match=r"""unhashable type: 'Expr'\n\nConsider hashing \'col.*meta"""
    ):
        {a}
