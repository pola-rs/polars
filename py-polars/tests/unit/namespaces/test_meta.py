from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import polars as pl

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
    assert e.suffix("ham").meta.undo_aliases().meta == e


def test_meta_has_multiple_outputs() -> None:
    e = pl.col(["a", "b"]).alias("bar")
    assert e.meta.has_multiple_outputs()


def test_meta_is_regex_projection() -> None:
    e = pl.col("^.*$").alias("bar")
    assert e.meta.is_regex_projection()
    assert e.meta.has_multiple_outputs()


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
