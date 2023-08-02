import re
from datetime import date
from tempfile import NamedTemporaryFile
from typing import Any

import pytest

import polars as pl
from polars.testing import assert_frame_equal


def test_cse_rename_cross_join_5405() -> None:
    right = pl.DataFrame({"A": [1, 2], "B": [3, 4], "D": [5, 6]}).lazy()

    left = pl.DataFrame({"C": [3, 4]}).lazy().join(right.select("A"), how="cross")

    out = left.join(right.rename({"B": "C"}), on=["A", "C"], how="left")

    assert out.collect(comm_subplan_elim=True).to_dict(False) == {
        "C": [3, 3, 4, 4],
        "A": [1, 2, 1, 2],
        "D": [5, None, None, 6],
    }


def test_union_duplicates() -> None:
    n_dfs = 10
    df_lazy = pl.DataFrame({}).lazy()
    lazy_dfs = [df_lazy for _ in range(n_dfs)]
    assert (
        len(
            re.findall(
                r".*CACHE\[id: .*, count: 9].*",
                pl.concat(lazy_dfs).explain(),
                flags=re.MULTILINE,
            )
        )
        == 10
    )


def test_cse_schema_6081() -> None:
    df = pl.DataFrame(
        data=[
            [date(2022, 12, 12), 1, 1],
            [date(2022, 12, 12), 1, 2],
            [date(2022, 12, 13), 5, 2],
        ],
        schema=["date", "id", "value"],
        orient="row",
    ).lazy()

    min_value_by_group = df.groupby(["date", "id"]).agg(
        pl.col("value").min().alias("min_value")
    )

    result = df.join(min_value_by_group, on=["date", "id"], how="left")
    assert result.collect(comm_subplan_elim=True, projection_pushdown=True).to_dict(
        False
    ) == {
        "date": [date(2022, 12, 12), date(2022, 12, 12), date(2022, 12, 13)],
        "id": [1, 1, 5],
        "value": [1, 2, 2],
        "min_value": [1, 1, 2],
    }


def test_cse_9630() -> None:
    df1 = pl.DataFrame(
        {
            "key": [1],
            "x": [1],
        }
    ).lazy()

    df2 = pl.DataFrame(
        {
            "key": [1],
            "y": [2],
        }
    ).lazy()

    joined_df2 = df1.join(df2, on="key")

    all_subsections = (
        pl.concat(
            [
                df1.select("key", pl.col("x").alias("value")),
                joined_df2.select("key", pl.col("y").alias("value")),
            ]
        )
        .groupby("key")
        .agg(
            [
                pl.col("value"),
            ]
        )
    )

    intersected_df1 = all_subsections.join(df1, on="key")
    intersected_df2 = all_subsections.join(df2, on="key")

    assert intersected_df1.join(intersected_df2, on=["key"], how="left").collect(
        comm_subplan_elim=True
    ).to_dict(False) == {
        "key": [1],
        "value": [[1, 2]],
        "x": [1],
        "value_right": [[1, 2]],
        "y": [2],
    }


@pytest.mark.write_disk()
def test_schema_row_count_cse() -> None:
    csv_a = NamedTemporaryFile()
    csv_a.write(
        b"""
    A,B
    Gr1,A
    Gr1,B
    """.strip()
    )
    csv_a.seek(0)

    df_a = pl.scan_csv(csv_a.name).with_row_count("Idx")
    assert df_a.join(df_a, on="B").groupby(
        "A", maintain_order=True
    ).all().collect().to_dict(False) == {
        "A": ["Gr1"],
        "Idx": [[0, 1]],
        "B": [["A", "B"]],
        "Idx_right": [[0, 1]],
        "A_right": [["Gr1", "Gr1"]],
    }
    csv_a.close()


def test_cse_expr_selection_context(monkeypatch: Any, capfd: Any) -> None:
    monkeypatch.setenv("POLARS_VERBOSE", "1")
    q = pl.LazyFrame(
        {
            "a": [1, 2, 3, 4],
            "b": [1, 2, 3, 4],
            "c": [1, 2, 3, 4],
        }
    )

    derived = (pl.col("a") * pl.col("b")).sum()
    derived2 = derived * derived

    exprs = [
        derived.alias("d1"),
        (derived * pl.col("c").sum() - 1).alias("foo"),
        derived2.alias("d2"),
        (derived2 * 10).alias("d3"),
    ]

    assert q.select(exprs).collect(comm_subexpr_elim=True).to_dict(False) == {
        "d1": [30],
        "foo": [299],
        "d2": [900],
        "d3": [9000],
    }
    assert q.with_columns(exprs).collect(comm_subexpr_elim=True).to_dict(False) == {
        "a": [1, 2, 3, 4],
        "b": [1, 2, 3, 4],
        "c": [1, 2, 3, 4],
        "d1": [30, 30, 30, 30],
        "foo": [299, 299, 299, 299],
        "d2": [900, 900, 900, 900],
        "d3": [9000, 9000, 9000, 9000],
    }

    out = capfd.readouterr().out
    assert "run ProjectionExec with 2 CSE" in out
    assert "run StackExec with 2 CSE" in out


def test_cse_expr_selection_streaming(monkeypatch: Any, capfd: Any) -> None:
    monkeypatch.setenv("POLARS_VERBOSE", "1")
    q = pl.LazyFrame(
        {
            "a": [1, 2, 3, 4],
            "b": [1, 2, 3, 4],
            "c": [1, 2, 3, 4],
        }
    )

    derived = pl.col("a") * pl.col("b")
    derived2 = derived * derived

    exprs = [
        derived.alias("d1"),
        derived2.alias("d2"),
        (derived2 * 10).alias("d3"),
    ]

    assert q.select(exprs).collect(comm_subexpr_elim=True, streaming=True).to_dict(
        False
    ) == {"d1": [1, 4, 9, 16], "d2": [1, 16, 81, 256], "d3": [10, 160, 810, 2560]}
    assert q.with_columns(exprs).collect(
        comm_subexpr_elim=True, streaming=True
    ).to_dict(False) == {
        "a": [1, 2, 3, 4],
        "b": [1, 2, 3, 4],
        "c": [1, 2, 3, 4],
        "d1": [1, 4, 9, 16],
        "d2": [1, 16, 81, 256],
        "d3": [10, 160, 810, 2560],
    }
    err = capfd.readouterr().err
    assert "df -> projection[cse] -> ordered_sink" in err
    assert "df -> hstack[cse] -> ordered_sink" in err


@pytest.mark.skip(reason="activate once fixed")
def test_cse_expr_groupby() -> None:
    q = pl.LazyFrame(
        {
            "a": [1, 2, 3, 4],
            "b": [1, 2, 3, 4],
            "c": [1, 2, 3, 4],
        }
    )

    derived = pl.col("a") * pl.col("b")

    q = (
        q.groupby("a")
        .agg(derived.sum().alias("sum"), derived.min().alias("min"))
        .sort("min")
    )

    assert "__POLARS_CSER" in q.explain(comm_subexpr_elim=True, optimized=True)

    s = q.explain(
        comm_subexpr_elim=True, optimized=True, streaming=True, comm_subplan_elim=False
    )
    # check if it uses CSE_expr
    # and is a complete pipeline
    assert "__POLARS_CSER" in s
    assert s.startswith("--- PIPELINE")

    expected = pl.DataFrame(
        {"a": [1, 2, 3, 4], "sum": [1, 4, 9, 16], "min": [1, 4, 9, 16]}
    )
    for streaming in [True, False]:
        out = q.collect(comm_subexpr_elim=True, streaming=streaming)
        assert_frame_equal(out, expected)


def test_windows_cse_excluded() -> None:
    lf = pl.LazyFrame(
        data=[
            ("a", "aaa", 1),
            ("a", "bbb", 3),
            ("a", "ccc", 1),
            ("c", "xxx", 2),
            ("c", "yyy", 3),
            ("c", "zzz", 4),
            ("b", "qqq", 0),
        ],
        schema=["a", "b", "c"],
    )
    assert lf.select(
        c_diff=pl.col("c").diff(1),
        c_diff_by_a=pl.col("c").diff(1).over("a"),
    ).collect(comm_subexpr_elim=True).to_dict(False) == {
        "c_diff": [None, 2, -2, 1, 1, 1, -4],
        "c_diff_by_a": [None, 2, -2, None, 1, 1, None],
    }


def test_cse_groupby_10215() -> None:
    assert (
        pl.DataFrame(
            {
                "a": [1, 2, 3],
                "b": [1, 1, 1],
            }
        )
        .lazy()
        .groupby(
            "a",
        )
        .agg(
            (pl.col("a").sum() * pl.col("a").sum()).alias("x"),
            (pl.col("b").sum() * pl.col("b").sum()).alias("y"),
        )
        .collect()
        .sort("a")
    ).to_dict(False) == {"a": [1, 2, 3], "x": [1, 4, 9], "y": [1, 1, 1]}
