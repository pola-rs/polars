from __future__ import annotations

import pytest

import polars as pl
from polars import col


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

    e = pl.col("foo").filter(pl.col("bar") == 13)
    assert e.meta.output_name() == "foo"
    assert e.meta.root_names() == ["foo", "bar"]

    e = pl.sum("foo").over("groups")
    assert e.meta.output_name() == "foo"
    assert e.meta.root_names() == ["foo", "groups"]

    e = pl.sum("foo").slice(pl.count() - 10, pl.col("bar"))
    assert e.meta.output_name() == "foo"
    assert e.meta.root_names() == ["foo", "bar"]

    e = pl.count()
    assert e.meta.output_name() == "count"

    with pytest.raises(
        pl.ComputeError,
        match="cannot determine output column without a context for this expression",
    ):
        pl.all().suffix("_").meta.output_name()


def test_undo_aliases() -> None:
    e = pl.col("foo").alias("bar")
    assert e.meta.undo_aliases().meta == pl.col("foo")

    e = pl.col("foo").sum().over("bar")
    assert e.keep_name().meta.undo_aliases().meta == e

    e.alias("bar").alias("foo")
    assert e.meta.undo_aliases().meta == e
    assert e.suffix("ham").meta.undo_aliases().meta == e


def test_meta_has_multiple_outputs() -> None:
    e = pl.col(["a", "b"]).alias("bar")
    assert e.meta.has_multiple_outputs()


def test_meta_is_regex_projection() -> None:
    e = pl.col("^.*$").alias("bar")
    assert e.meta.is_regex_projection()
    assert e.meta.has_multiple_outputs()


def test_meta_tree_format() -> None:
    e = (col("foo") * col("bar")).sum().over(col("ham")) / 2
    assert (
        e.meta.tree_format(return_as_string=True)
        == """    binary: /    \n\n      |              |       \n\n    lit(2)           window      \n\n                     |               |        \n\n                     col(ham)        sum          \n\n                                     |        \n\n                                     binary: *    \n\n                                     |                |       \n\n                                     col(bar)         col(foo)    \n"""
    )
