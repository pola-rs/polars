from __future__ import annotations

import pytest

import polars as pl
from polars.testing import assert_frame_equal


def test_expression_literal_series_order() -> None:
    s = pl.Series([1, 2, 3])
    df = pl.DataFrame({"a": [1, 2, 3]})

    result = df.select(pl.col("a") + s)
    expected = pl.DataFrame({"a": [2, 4, 6]})
    assert_frame_equal(result, expected)

    result = df.select(pl.lit(s) + pl.col("a"))
    expected = pl.DataFrame({"": [2, 4, 6]})
    assert_frame_equal(result, expected)


def test_when_then_broadcast_nulls_12665() -> None:
    df = pl.DataFrame(
        {
            "val": [1, 2, 3, 4],
            "threshold": [4, None, None, 1],
        }
    )

    assert df.select(
        when=pl.when(pl.col("val") > pl.col("threshold")).then(1).otherwise(0),
    ).to_dict(as_series=False) == {"when": [0, 0, 0, 1]}


@pytest.mark.parametrize(
    ("needs_broadcast", "expect_contains"),
    [
        (pl.lit("a"), [True, False, False]),
        (pl.col("name").head(1), [True, False, False]),
        (pl.lit(None, dtype=pl.String), [None, None, None]),
        (pl.col("null_utf8").head(1), [None, None, None]),
    ],
)
@pytest.mark.parametrize("literal", [True, False])
@pytest.mark.parametrize(
    "df",
    [
        pl.DataFrame(
            {
                "name": ["a", "b", "c"],
                "null_utf8": pl.Series([None, None, None], dtype=pl.String),
            }
        )
    ],
)
def test_broadcast_string_ops_12632(
    df: pl.DataFrame,
    needs_broadcast: pl.Expr,
    expect_contains: list[bool],
    literal: bool,
) -> None:
    assert (
        df.select(needs_broadcast.str.contains(pl.col("name"), literal=literal))
        .to_series()
        .to_list()
        == expect_contains
    )

    assert (
        df.select(needs_broadcast.str.starts_with(pl.col("name"))).to_series().to_list()
        == expect_contains
    )

    assert (
        df.select(needs_broadcast.str.ends_with(pl.col("name"))).to_series().to_list()
        == expect_contains
    )

    assert df.select(needs_broadcast.str.strip_chars(pl.col("name"))).height == 3
    assert df.select(needs_broadcast.str.strip_chars_start(pl.col("name"))).height == 3
    assert df.select(needs_broadcast.str.strip_chars_end(pl.col("name"))).height == 3


def test_negate_inlined_14278() -> None:
    df = pl.DataFrame(
        {"group": ["A", "A", "B", "B", "B", "C", "C"], "value": [1, 2, 3, 4, 5, 6, 7]}
    )

    agg_expr = [
        pl.struct("group", "value").tail(2).alias("list"),
        pl.col("value").sort().tail(2).count().alias("count"),
    ]

    q = df.lazy().group_by("group").agg(agg_expr)
    assert q.collect().sort("group").to_dict(as_series=False) == {
        "group": ["A", "B", "C"],
        "list": [
            [{"group": "A", "value": 1}, {"group": "A", "value": 2}],
            [{"group": "B", "value": 4}, {"group": "B", "value": 5}],
            [{"group": "C", "value": 6}, {"group": "C", "value": 7}],
        ],
        "count": [2, 2, 2],
    }


def test_nested_level_literals_17377() -> None:
    df = pl.LazyFrame({"group": [1, 2], "value": [1, 2]})

    df2 = df.group_by("group").agg(
        [
            pl.when((pl.col("value") < 0).all())
            .then(None)
            .otherwise(pl.col("value").mean())
            .alias("res")
        ]
    )

    assert df2.collect_schema() == pl.Schema({"group": pl.Int64(), "res": pl.Float64()})
