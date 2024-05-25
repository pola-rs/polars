from __future__ import annotations

import pytest

import polars as pl


def test_lf_explain() -> None:
    lf = pl.LazyFrame({"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]})
    plan = lf.select("a").select(pl.col("a").sum() + pl.len())

    result = plan.explain()

    expected = """\
 SELECT [[(col("a").sum()) + (len().cast(Int64))]] FROM
  DF ["a", "b"]; PROJECT 1/2 COLUMNS; SELECTION: None\
"""
    assert result == expected


def test_lf_explain_format_tree() -> None:
    lf = pl.LazyFrame({"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]})
    plan = lf.select("a").select(pl.col("a").sum() + pl.len())

    result = plan.explain(format="tree")

    expected = """\
                0                        1
   ┌─────────────────────────────────────────────────
   │
   │        ╭────────╮
 0 │        │ SELECT │
   │        ╰───┬┬───╯
   │            ││
   │            │╰───────────────────────╮
   │            │                        │
   │  ╭─────────┴──────────╮             │
   │  │ expression:        │  ╭──────────┴──────────╮
   │  │ [(col("a")         │  │ FROM:               │
 1 │  │   .sum()) + (len() │  │ DF ["a", "b"]       │
   │  │   .cast(Int64))]   │  │ PROJECT 1/2 COLUMNS │
   │  ╰────────────────────╯  ╰─────────────────────╯
\
"""
    assert result == expected


def test_lf_explain_tree_format_deprecated() -> None:
    lf = pl.LazyFrame({"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]})

    with pytest.deprecated_call():
        lf.explain(tree_format=True)
