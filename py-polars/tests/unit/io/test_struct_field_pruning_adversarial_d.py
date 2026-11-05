"""Adversarial cases for struct-field pruning, from the review of the whole branch.

Both defects found showed only on the streaming engine: a struct read for validity
only is a `Struct({})`, whose length the streaming first/last reductions lost (their
`AnyValue` buffer gave a zero-field struct length 0); and a `with_fields` whose new
fields nobody reads became an empty `with_fields`, which the streaming engine rejects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import polars as pl
from polars.testing import assert_frame_equal
from tests.unit.io.test_struct_field_pruning import (
    NO_PUSHDOWN,
    assert_same_as_without_pushdown,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

l = pl.col("l")  # noqa: E741
s = pl.col("s")

DF = pl.DataFrame(
    {
        "id": [0, 1, 1, 2],
        "s": [{"a": 1, "b": 2}, None, {"a": None, "b": 3}, {"a": 4, "b": None}],
        "l": [[{"a": 1, "b": 2}], None, [], [None]],
    }
)

QUERIES: list[Callable[[pl.LazyFrame], pl.LazyFrame]] = [
    # Validity only, then first / last.
    lambda lf: lf.select(s.first().is_null()),
    lambda lf: lf.select(s.last().count()),
    lambda lf: lf.head(1).select(s.item().is_null()),
    lambda lf: lf.select(s.drop_nulls().first().is_null()),
    lambda lf: lf.sort("id", descending=True).select(s.first().is_null()),
    lambda lf: pl.concat([lf, lf]).select(s.last().is_null()),
    lambda lf: lf.join(lf.select("id", t=s), on="id", maintain_order="left").select(
        pl.col("t").first().is_null()
    ),
    lambda lf: lf.group_by("id", maintain_order=True).agg(s.first().is_null()),
    lambda lf: lf.unique(subset=["id"], maintain_order=True).select(s.count()),
    lambda lf: lf.unique(subset=["id"], keep="last", maintain_order=True).select(
        s.null_count()
    ),
    lambda lf: lf.explode("l").select(l.last().is_null()),
    lambda lf: lf.select(l.explode().last().is_null()),
    lambda lf: lf.select(l.list.first().first().is_null()),
    lambda lf: (
        lf.explode("l").unique(subset=["id"], maintain_order=True).select(l.count())
    ),
    # `with_fields` whose new fields nobody reads, on a non-elementwise input.
    lambda lf: lf.select(
        s.filter(pl.col("id") > 0)
        .struct.with_fields(pl.field("a").alias("zz"))
        .is_null()
    ),
    lambda lf: lf.select(
        s.filter(pl.col("id") > 0).struct.with_fields(pl.field("a").alias("zz")).count()
    ),
    lambda lf: lf.select(
        s.sort_by("id").struct.with_fields(pl.field("a").alias("zz")).is_null()
    ),
]


@pytest.mark.parametrize("fmt", ["parquet", "ipc", "ndjson", "memory"])
@pytest.mark.parametrize("query", QUERIES)
def test_struct_field_pruning_review_d(
    query: Callable[[pl.LazyFrame], pl.LazyFrame], fmt: str, tmp_path: Path
) -> None:
    path = tmp_path / f"data.{fmt}"
    if fmt == "parquet":
        DF.write_parquet(path)
        lf = pl.scan_parquet(path)
    elif fmt == "ipc":
        DF.write_ipc(path)
        lf = pl.scan_ipc(path)
    elif fmt == "ndjson":
        DF.write_ndjson(path)
        lf = pl.scan_ndjson(path, schema=DF.schema)
    else:
        lf = DF.lazy()
    assert_same_as_without_pushdown(query(lf))


def test_first_last_of_struct_without_fields_streaming() -> None:
    df = pl.DataFrame(
        {"id": [0, 1, 1, 2], "s": pl.Series([{}, None, {}, {}], dtype=pl.Struct({}))}
    )
    for q in [
        df.lazy().select(pl.col("s").first().is_null()),
        df.lazy().select(pl.col("s").last().is_null()),
        df.lazy().select(pl.col("s").drop_nulls().first().is_null()),
        df.lazy().select(pl.col("s").drop_nulls().last().is_null()),
        df.lazy()
        .group_by("id", maintain_order=True)
        .agg(pl.col("s").first().is_null(), pl.col("s").last().is_null().alias("t")),
        df.lazy()
        .unique(subset=["id"], maintain_order=True)
        .select(pl.col("s").count()),
    ]:
        assert_frame_equal(
            q.collect(engine="streaming", optimizations=NO_PUSHDOWN),
            q.collect(engine="in-memory", optimizations=NO_PUSHDOWN),
        )
