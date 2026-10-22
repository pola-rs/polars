from __future__ import annotations

import re

import pytest

import polars as pl
from polars.exceptions import ArgumentRemovedError


def test_lf_explain_format_tree() -> None:
    lf = pl.LazyFrame({"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]})
    plan = lf.select("a").select(pl.col("a").sum() + pl.len())

    result = plan.explain(format="tree")

    expected = """\
               0                           1
   ┌───────────────────────────────────────────────────────
   │
   │       ╭────────╮
 0 │       │ SELECT │
   │       ╰───┬┬───╯
   │           ││
   │           │╰──────────────────────────╮
   │           │                           │
   │  ╭────────┴─────────╮                 │
   │  │ expression:      │  ╭──────────────┴──────────────╮
   │  │ (col("a")        │  │ FROM:                       │
 1 │  │   .sum() + len() │  │ DF ["a", "b"]               │
   │  │   .cast(Int64))  │  │ PROJECT: ["a"]; 1/2 COLUMNS │
   │  ╰──────────────────╯  ╰─────────────────────────────╯
\
"""
    assert result == expected


def test_lf_explain_tree_format_removed() -> None:
    lf = pl.LazyFrame({"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]})

    with pytest.raises(ArgumentRemovedError, match=re.escape("'tree_format'")):
        lf.explain(tree_format=True)  # type: ignore[call-arg]


def test_tmp_column_names_are_tagged_by_crate() -> None:
    def tmp_tags(plan: str) -> set[str]:
        return set(re.findall(r"_POLARS_TMP_([A-Z]+)_\d+", plan))

    with pl.SQLContext(
        t=pl.LazyFrame({"a": [1, 1, 2], "b": [1, 2, 3]}), eager=False
    ) as ctx:
        lf = ctx.execute(
            "SELECT a FROM t WHERE b > (SELECT MIN(b) FROM t AS x WHERE x.a = t.a)"
        )
    assert tmp_tags(lf.explain(optimized=False)) == {"SQL"}

    lf = pl.LazyFrame({"g": [1, 1, 2], "x": [1, 2, 3]})
    windows = lf.select(
        pl.col("x").sum().over("g"), pl.col("x").min().over("g").alias("m")
    )
    assert tmp_tags(windows.explain(optimized=False)) == set()
    assert tmp_tags(windows.explain(engine="streaming")) == {"PLAN"}

    group_by = lf.group_by("g").agg(pl.col("x").mean())
    assert tmp_tags(group_by.explain(engine="streaming")) == set()
    physical = group_by.show_graph(
        engine="streaming", plan_stage="physical", raw_output=True
    )
    assert tmp_tags(physical) == {"PHYS"}
