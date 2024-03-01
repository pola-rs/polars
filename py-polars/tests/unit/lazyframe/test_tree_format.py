from __future__ import annotations

import polars as pl


def test_logical_plan_tree_format() -> None:
    lf = (
        pl.LazyFrame(
            {
                "foo": [1, 2, 3],
                "bar": [6, 7, 8],
                "ham": ["a", "b", "c"],
            }
        )
        .select(foo=pl.col("foo") + 1, bar=pl.col("bar") + 2)
        .select(
            threshold=pl.when(pl.col("foo") + pl.col("bar") > 2).then(10).otherwise(0)
        )
    )

    expected = """
 SELECT [.when([([(col("foo")) + (col("bar"))]) > (2)]).then(10).otherwise(0).alias("threshold")] FROM
   SELECT [[(col("foo")) + (1)].alias("foo"), [(col("bar")) + (2)].alias("bar")] FROM
    DF ["foo", "bar", "ham"]; PROJECT 2/3 COLUMNS; SELECTION: "None"
"""
    assert lf.explain().strip() == expected.strip()

    expected = """
                              0                                      1                         2                           3
   ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   │
   │                      ╭────────╮
 0 │                      │ SELECT │
   │                      ╰───┬┬───╯
   │                          ││
   │                          │╰─────────────────────────────────────╮
   │                          │                                      │
   │  ╭───────────────────────┴────────────────────────╮             │
   │  │ expression:                                    │         ╭───┴────╮
   │  │ .when([([(col("foo")) + (col("bar"))]) > (2)]) │         │ FROM:  │
 1 │  │   .then(10)                                    │         │ SELECT │
   │  │   .otherwise(0)                                │         ╰───┬┬───╯
   │  │   .alias("threshold")                          │             ││
   │  ╰────────────────────────────────────────────────╯             ││
   │                                                                 ││
   │                                                                 │╰────────────────────────┬───────────────────────────╮
   │                                                                 │                         │                           │
   │                                                      ╭──────────┴───────────╮  ╭──────────┴───────────╮  ╭────────────┴─────────────╮
   │                                                      │ expression:          │  │ expression:          │  │ FROM:                    │
 2 │                                                      │ [(col("foo")) + (1)] │  │ [(col("bar")) + (2)] │  │ DF ["foo", "bar", "ham"] │
   │                                                      │   .alias("foo")      │  │   .alias("bar")      │  │ PROJECT 2/3 COLUMNS      │
   │                                                      ╰──────────────────────╯  ╰──────────────────────╯  ╰──────────────────────────╯
"""
    assert lf.explain(tree_format=True).strip() == expected.strip()
