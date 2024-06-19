from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import polars as pl
import polars.selectors as cs
from polars.exceptions import ComputeError
from tests.unit.conftest import NUMERIC_DTYPES

if TYPE_CHECKING:
    from pathlib import Path


def test_meta_pop_and_cmp() -> None:
    e = pl.col("foo").alias("bar")

    first = e.meta.pop()[0]
    assert first.meta == pl.col("foo")
    assert first.meta != pl.col("bar")

    assert first.meta.eq(pl.col("foo"))
    assert first.meta.ne(pl.col("bar"))


def test_root_and_output_names() -> None:
    e = pl.col("foo") * pl.col("bar")
    assert e.meta.output_name() == "foo"
    assert e.meta.root_names() == ["foo", "bar"]

    e = pl.col("foo").filter(bar=13)
    assert e.meta.output_name() == "foo"
    assert e.meta.root_names() == ["foo", "bar"]

    e = pl.sum("foo").over("groups")
    assert e.meta.output_name() == "foo"
    assert e.meta.root_names() == ["foo", "groups"]

    e = pl.sum("foo").slice(pl.len() - 10, pl.col("bar"))
    assert e.meta.output_name() == "foo"
    assert e.meta.root_names() == ["foo", "bar"]

    e = pl.len()
    assert e.meta.output_name() == "len"

    with pytest.raises(
        ComputeError,
        match="cannot determine output column without a context for this expression",
    ):
        pl.all().name.suffix("_").meta.output_name()

    assert (
        pl.all().name.suffix("_").meta.output_name(raise_if_undetermined=False) is None
    )


def test_undo_aliases() -> None:
    e = pl.col("foo").alias("bar")
    assert e.meta.undo_aliases().meta == pl.col("foo")

    e = pl.col("foo").sum().over("bar")
    assert e.name.keep().meta.undo_aliases().meta == e

    e.alias("bar").alias("foo")
    assert e.meta.undo_aliases().meta == e
    assert e.name.suffix("ham").meta.undo_aliases().meta == e


def test_meta_has_multiple_outputs() -> None:
    e = pl.col(["a", "b"]).name.suffix("_foo")
    assert e.meta.has_multiple_outputs()


def test_is_column() -> None:
    e = pl.col("foo")
    assert e.meta.is_column()

    e = pl.col("foo").alias("bar")
    assert not e.meta.is_column()

    e = pl.col("foo") * pl.col("bar")
    assert not e.meta.is_column()


@pytest.mark.parametrize(
    ("expr", "is_column_selection"),
    [
        # columns
        (pl.col("foo"), True),
        (pl.col("foo", "bar"), True),
        (pl.col(NUMERIC_DTYPES), True),
        # column expressions
        (pl.col("foo") + 100, False),
        (pl.col("foo").floordiv(10), False),
        (pl.col("foo") * pl.col("bar"), False),
        # selectors / expressions
        (cs.numeric() * 100, False),
        (cs.temporal() - cs.time(), True),
        (cs.numeric().exclude("value"), True),
        ((cs.temporal() - cs.time()).exclude("dt"), True),
        # top-level selection funcs
        (pl.nth(2), True),
        (pl.first(), True),
        (pl.last(), True),
    ],
)
def test_is_column_selection(
    expr: pl.Expr,
    is_column_selection: bool,
) -> None:
    if is_column_selection:
        assert expr.meta.is_column_selection()
        assert expr.meta.is_column_selection(allow_aliasing=True)
        expr = (
            expr.name.suffix("!")
            if expr.meta.has_multiple_outputs()
            else expr.alias("!")
        )
        assert not expr.meta.is_column_selection()
        assert expr.meta.is_column_selection(allow_aliasing=True)
    else:
        assert not expr.meta.is_column_selection()


def test_meta_is_regex_projection() -> None:
    e = pl.col("^.*$").name.suffix("_foo")
    assert e.meta.is_regex_projection()
    assert e.meta.has_multiple_outputs()

    e = pl.col("^.*")  # no trailing '$'
    assert not e.meta.is_regex_projection()
    assert not e.meta.has_multiple_outputs()
    assert e.meta.is_column()


def test_meta_tree_format(namespace_files_path: Path) -> None:
    with (namespace_files_path / "test_tree_fmt.txt").open("r", encoding="utf-8") as f:
        test_sets = f.read().split("---")
    for test_set in test_sets:
        expression = test_set.strip().split("\n")[0]
        tree_fmt = "\n".join(test_set.strip().split("\n")[1:])
        e = eval(expression)
        result = e.meta.tree_format(return_as_string=True)
        result = "\n".join(s.rstrip() for s in result.split("\n"))
        assert result.strip() == tree_fmt.strip()


def test_literal_output_name() -> None:
    e = pl.lit(1)
    assert e.meta.output_name() == "literal"

    e = pl.lit(pl.Series("abc", [1, 2, 3]))
    assert e.meta.output_name() == "abc"

    e = pl.lit(pl.Series([1, 2, 3]))
    assert e.meta.output_name() == ""
