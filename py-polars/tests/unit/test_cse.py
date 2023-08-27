import re
from datetime import date
from tempfile import NamedTemporaryFile
from typing import Any

import pytest

import polars as pl


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

    min_value_by_group = df.group_by(["date", "id"]).agg(
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
        .group_by("key")
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
    assert df_a.join(df_a, on="B").group_by(
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


def test_cse_group_by_10215() -> None:
    q = (
        pl.DataFrame(
            {
                "a": [1],
                "b": [1],
            }
        )
        .lazy()
        .group_by(
            "b",
        )
        .agg(
            (pl.col("a").sum() * pl.col("a").sum()).alias("x"),
            (pl.col("b").sum() * pl.col("b").sum()).alias("y"),
            (pl.col("a").sum() * pl.col("a").sum()).alias("x2"),
            ((pl.col("a") + 2).sum() * pl.col("a").sum()).alias("x3"),
            ((pl.col("a") + 2).sum() * pl.col("b").sum()).alias("x4"),
            ((pl.col("a") + 2).sum() * pl.col("b").sum()),
        )
    )
    out = q.collect(comm_subexpr_elim=True).to_dict(False)
    assert "__POLARS_CSER" in q.explain(comm_subexpr_elim=True)
    assert out == {
        "b": [1],
        "x": [1],
        "y": [1],
        "x2": [1],
        "x3": [3],
        "x4": [3],
        "a": [3],
    }


def test_cse_mixed_window_functions() -> None:
    # checks if the window caches are cleared
    # there are windows in the cse's and the default expressions
    assert pl.DataFrame(
        {
            "a": [1],
            "b": [1],
            "c": [1],
        }
    ).lazy().select(
        pl.col("a"),
        pl.col("b"),
        pl.col("c"),
        pl.col("b").rank().alias("rank"),
        pl.col("b").rank().alias("d_rank"),
        pl.col("b").first().over([pl.col("a")]).alias("b_first"),
        pl.col("b").last().over([pl.col("a")]).alias("b_last"),
        pl.col("b").shift().alias("b_lag_1"),
        pl.col("b").shift().alias("b_lead_1"),
        pl.col("c").cumsum().alias("c_cumsum"),
        pl.col("c").cumsum().over([pl.col("a")]).alias("c_cumsum_by_a"),
        pl.col("c").diff().alias("c_diff"),
        pl.col("c").diff().over([pl.col("a")]).alias("c_diff_by_a"),
    ).collect().to_dict(False) == {
        "a": [1],
        "b": [1],
        "c": [1],
        "rank": [1.0],
        "d_rank": [1.0],
        "b_first": [1],
        "b_last": [1],
        "b_lag_1": [None],
        "b_lead_1": [None],
        "c_cumsum": [1],
        "c_cumsum_by_a": [1],
        "c_diff": [None],
        "c_diff_by_a": [None],
    }


def test_cse_10401() -> None:
    df = pl.DataFrame({"clicks": [1.0, float("nan"), None]})

    q = df.lazy().with_columns(pl.all().fill_null(0).fill_nan(0))
    assert r"""col("clicks").fill_null([0]).alias("__POLARS_CSER""" in q.explain()
    assert q.collect().to_dict(False) == {"clicks": [1.0, 0.0, 0.0]}


def test_cse_10441() -> None:
    assert pl.LazyFrame({"a": [1, 2, 3], "b": [3, 2, 1]}).select(
        pl.col("a").sum() + pl.col("a").sum() + pl.col("b").sum()
    ).collect(comm_subexpr_elim=True).to_dict(False) == {"a": [18]}


def test_cse_10452() -> None:
    q = pl.LazyFrame({"a": [1, 2, 3], "b": [3, 2, 1]}).select(
        pl.col("b").sum() + pl.col("a").sum().over([pl.col("b")]) + pl.col("b").sum()
    )
    assert "__POLARS_CSE" in q.explain(comm_subexpr_elim=True)
    assert q.collect(comm_subexpr_elim=True).to_dict(False) == {"b": [13, 14, 15]}


def test_cse_group_by_ternary_10490() -> None:
    df = pl.DataFrame(
        {
            "a": [1, 1, 2, 2],
            "b": [1, 2, 3, 4],
            "c": [2, 3, 4, 5],
        }
    )

    assert (
        df.lazy()
        .group_by("a")
        .agg(
            [
                pl.when(pl.col(col).is_null().all()).then(None).otherwise(1).alias(col)
                for col in ["b", "c"]
            ]
            + [
                (pl.col("a").sum() * pl.col("a").sum()).alias("x"),
                (pl.col("b").sum() * pl.col("b").sum()).alias("y"),
                (pl.col("a").sum() * pl.col("a").sum()).alias("x2"),
                ((pl.col("a") + 2).sum() * pl.col("a").sum()).alias("x3"),
                ((pl.col("a") + 2).sum() * pl.col("b").sum()).alias("x4"),
            ]
        )
        .collect(comm_subexpr_elim=True)
        .sort("a")
        .to_dict(False)
    ) == {
        "a": [1, 2],
        "b": [1, 1],
        "c": [1, 1],
        "x": [4, 16],
        "y": [9, 49],
        "x2": [4, 16],
        "x3": [12, 32],
        "x4": [18, 56],
    }
