from __future__ import annotations

from typing import Any

import pytest

import polars as pl
from polars.testing import assert_frame_equal
from tests.unit.conftest import NUMERIC_DTYPES


def test_regex_exclude() -> None:
    df = pl.DataFrame({f"col_{i}": [i] for i in range(5)})

    assert df.select(pl.col("^col_.*$").exclude("col_0")).columns == [
        "col_1",
        "col_2",
        "col_3",
        "col_4",
    ]


def test_regex_in_filter() -> None:
    df = pl.DataFrame(
        {
            "nrs": [1, 2, 3, None, 5],
            "names": ["foo", "ham", "spam", "egg", None],
            "flt": [1.0, None, 3.0, 1.0, None],
        }
    )

    res = df.filter(
        pl.fold(
            acc=False, function=lambda acc, s: acc | s, exprs=(pl.col("^nrs|flt*$") < 3)
        )
    ).row(0)
    expected = (1, "foo", 1.0)
    assert res == expected


def test_regex_selection() -> None:
    lf = pl.LazyFrame(
        {
            "foo": [1],
            "fooey": [1],
            "foobar": [1],
            "bar": [1],
        }
    )
    result = lf.select([pl.col("^foo.*$")])
    assert result.collect_schema().names() == ["foo", "fooey", "foobar"]


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        (pl.exclude("a"), ["b", "c"]),
        (pl.all().exclude(pl.Boolean), ["a", "b"]),
        (pl.all().exclude([pl.Boolean]), ["a", "b"]),
        (pl.all().exclude(NUMERIC_DTYPES), ["c"]),
    ],
)
def test_exclude_selection(expr: pl.Expr, expected: list[str]) -> None:
    lf = pl.LazyFrame({"a": [1], "b": [1], "c": [True]})

    assert lf.select(expr).collect_schema().names() == expected


def test_struct_name_resolving_15430() -> None:
    q = pl.LazyFrame([{"a": {"b": "c"}}])
    a = (
        q.with_columns(pl.col("a").struct.field("b"))
        .drop("a")
        .collect(projection_pushdown=True)
    )

    b = (
        q.with_columns(pl.col("a").struct[0])
        .drop("a")
        .collect(projection_pushdown=True)
    )

    assert a["b"].item() == "c"
    assert b["b"].item() == "c"
    assert a.columns == ["b"]
    assert b.columns == ["b"]


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        (pl.all().name.prefix("agg_"), ["A", "agg_B", "agg_C"]),
        (pl.col("B", "C").name.prefix("agg_"), ["A", "agg_B", "agg_C"]),
        (pl.col("A", "C").name.prefix("agg_"), ["A", "agg_A", "agg_C"]),
    ],
)
def test_exclude_keys_in_aggregation_16170(expr: pl.Expr, expected: list[str]) -> None:
    df = pl.DataFrame({"A": [4, 4, 3], "B": [1, 2, 3], "C": [5, 6, 7]})

    # wildcard excludes aggregation column
    result = df.lazy().group_by("A").agg(expr)
    assert result.collect_schema().names() == expected


@pytest.mark.parametrize(
    "field",
    [
        ["aaa", "ccc"],
        [["aaa", "ccc"]],
        [["aaa"], "ccc"],
        [["^aa.+|cc.+$"]],
    ],
)
def test_struct_field_expand(field: Any) -> None:
    df = pl.DataFrame(
        {
            "aaa": [1, 2],
            "bbb": ["ab", "cd"],
            "ccc": [True, None],
            "ddd": [[1, 2], [3]],
        }
    )
    struct_df = df.select(pl.struct(["aaa", "bbb", "ccc", "ddd"]).alias("struct_col"))
    res_df = struct_df.select(pl.col("struct_col").struct.field(*field))
    assert_frame_equal(res_df, df.select("aaa", "ccc"))


def test_struct_field_expand_star() -> None:
    df = pl.DataFrame(
        {
            "aaa": [1, 2],
            "bbb": ["ab", "cd"],
            "ccc": [True, None],
            "ddd": [[1, 2], [3]],
        }
    )
    struct_df = df.select(pl.struct(["aaa", "bbb", "ccc", "ddd"]).alias("struct_col"))
    assert_frame_equal(struct_df.select(pl.col("struct_col").struct.field("*")), df)


def test_struct_field_expand_rewrite() -> None:
    df = pl.DataFrame({"A": [1], "B": [2]})
    assert df.select(
        pl.struct(["A", "B"]).struct.field("*").name.prefix("foo_")
    ).to_dict(as_series=False) == {"foo_A": [1], "foo_B": [2]}


def test_struct_field_expansion_16410() -> None:
    q = pl.LazyFrame({"coords": [{"x": 4, "y": 4}]})

    assert q.with_columns(
        pl.col("coords").struct.with_fields(pl.field("x").sqrt()).struct.field("*")
    ).collect().to_dict(as_series=False) == {
        "coords": [{"x": 4, "y": 4}],
        "x": [2.0],
        "y": [4],
    }


def test_field_and_column_expansion() -> None:
    df = pl.DataFrame({"a": [{"x": 1, "y": 2}], "b": [{"i": 3, "j": 4}]})

    assert df.select(pl.col("a", "b").struct.field("*")).to_dict(as_series=False) == {
        "x": [1],
        "y": [2],
        "i": [3],
        "j": [4],
    }


def test_struct_field_exclude_and_wildcard_expansion() -> None:
    df = pl.DataFrame({"a": [{"x": 1, "y": 2}], "b": [{"i": 3, "j": 4}]})

    assert df.select(pl.exclude("foo").struct.field("*")).to_dict(as_series=False) == {
        "x": [1],
        "y": [2],
        "i": [3],
        "j": [4],
    }
    assert df.select(pl.all().struct.field("*")).to_dict(as_series=False) == {
        "x": [1],
        "y": [2],
        "i": [3],
        "j": [4],
    }
