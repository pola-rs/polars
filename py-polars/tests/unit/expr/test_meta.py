import pytest

import polars as pl
from polars.testing import assert_frame_equal


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


def test_meta_eq_tot_cmp_28469() -> None:
    a = pl.lit(float("nan"))
    b = pl.lit(float("nan"))
    assert a.meta.eq(b)


def test_meta_properties() -> None:
    e = pl.col("foo").sum()
    assert e.meta.is_scalar()
    assert e.meta.is_known_length()
    assert not e.meta.is_literal()
    assert not e.meta.is_row_separable()

    e = pl.col("foo") * pl.col("bar")
    assert e.meta.is_length_preserving()
    assert e.meta.is_row_separable()
    assert e.meta.is_known_length()
    e = (pl.col("foo") * pl.col("bar")).list.explode(empty_as_null=False)
    assert not e.meta.is_known_length()
    assert not e.meta.is_length_preserving()
    assert e.meta.is_row_separable()
    assert not e.meta.is_scalar()


a, b = pl.col("a"), pl.col("b")


@pytest.mark.parametrize(
    ("expr", "row_separable"),
    [
        (a.filter(a > 1) * 2, True),
        ((a + b).filter(b > 1), True),
        (a.filter(a > 1).is_in([1, 2]), True),
        (pl.when(a.filter(a > 1) > 2).then(1).otherwise(0), True),
        (a.sum() + 1, False),
        (a.filter(a > 1) + b.filter(b > 1), False),
        (a.drop_nulls() + b, False),
        (a.filter(b.filter(b > 1) > 1), False),
        (pl.struct(x=a.drop_nulls(), y=b.drop_nulls()), False),
    ],
)
def test_meta_is_row_separable(expr: pl.Expr, row_separable: bool) -> None:
    assert expr.meta.is_row_separable() == row_separable

    if row_separable:
        df = pl.DataFrame({"a": [0, 1, 2, 3, None, 5], "b": [3, 2, None, 0, 5, 4]})
        batches = pl.concat([df[:2].select(expr), df[2:].select(expr)])
        assert_frame_equal(batches, df.select(expr))
