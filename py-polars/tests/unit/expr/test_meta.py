import pytest

import polars as pl
import polars.selectors as cs
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


a, b, c = pl.col("a"), pl.col("b"), pl.col("c")


@pytest.mark.parametrize(
    ("expr", "row_separable"),
    [
        (a.filter(a > 1) * 2, True),
        ((a + b).filter(b > 1), True),
        (a.filter(a > 1).is_in([1, 2]), True),
        (pl.when(a.filter(a > 1) > 2).then(1).otherwise(0), True),
        (pl.lit(1).filter(a > 1), True),
        (a.sum() + 1, False),
        (a.filter(a > 1) + b.filter(b > 1), False),
        (a.drop_nulls() + b, False),
        (a.filter(b.filter(b > 1) > 1), False),
        (pl.struct(x=a.drop_nulls(), y=b.drop_nulls()), False),
        (pl.lit([1, 2]).explode(), False),
        (pl.lit(1).drop_nulls(), False),
        (pl.lit([1, 2]).list.eval(pl.element()).explode(), False),
        (pl.lit([1, 2]).list.eval(pl.len()).explode(), False),
        (
            pl.lit(1)
            .map_batches(lambda s: s, is_elementwise=True, return_dtype=pl.Int32)
            .drop_nulls(),
            False,
        ),
        (pl.lit(1).is_in(pl.Series([[1, 2]])).drop_nulls(), False),
        (a.is_in(pl.Series([[1], [2], [3], [4]])), False),
        (a.is_in(b), False),
        (a.replace(b, c), False),
    ],
)
def test_meta_is_row_separable(expr: pl.Expr, row_separable: bool) -> None:
    assert expr.meta.is_row_separable() == row_separable

    if row_separable:
        df = pl.DataFrame({"a": [0, 1, 2, 3, None, 5], "b": [3, 2, None, 0, 5, 4]})
        batches = pl.concat([df[:2].select(expr), df[2:].select(expr)])
        assert_frame_equal(batches, df.select(expr))


def test_meta_properties_schema() -> None:
    schema = {"a": pl.Int64, "s": pl.String}

    # A selector checks every column it expands to.
    assert cs.numeric().sum().meta.is_scalar(schema=schema)
    assert not cs.all().meta.is_scalar(schema=schema)
    assert pl.all().meta.is_length_preserving(schema=schema)
    assert pl.all().meta.is_known_length(schema=schema)
    assert (pl.col(pl.Int64) * 2).meta.is_row_separable(schema=schema)
    assert not cs.numeric().cum_sum().meta.is_row_separable(schema=schema)

    with pytest.raises(pl.exceptions.ColumnNotFoundError):
        pl.col("x").meta.is_scalar(schema=schema)


@pytest.mark.parametrize(
    "expr", [pl.col("s").struct.unnest(), pl.col("s").struct.field("x", "y")]
)
def test_meta_struct_fields_without_schema(expr: pl.Expr) -> None:
    with pytest.raises(pl.exceptions.InvalidOperationError):
        expr.meta.is_scalar()
    with pytest.raises(pl.exceptions.InvalidOperationError):
        expr.meta.tree_format(return_as_string=True)

    schema = {"s": pl.Struct({"x": pl.Int64, "y": pl.Int64})}
    assert expr.meta.is_length_preserving(schema=schema)


def test_meta_is_row_separable_is_in_list_column() -> None:
    expr = pl.col("a").is_in(pl.col("l"))
    # Only the schema tells a list column, zipped row by row, from a set.
    assert not expr.meta.is_row_separable()
    assert expr.meta.is_row_separable(schema={"a": pl.Int64, "l": pl.List(pl.Int64)})
    assert not expr.meta.is_row_separable(schema={"a": pl.Int64, "l": pl.Int64})
