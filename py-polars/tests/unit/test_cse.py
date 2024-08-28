from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np
import pytest

import polars as pl
from polars.testing import assert_frame_equal


def num_cse_occurrences(explanation: str) -> int:
    """The number of unique CSE columns in an explain string."""
    return len(set(re.findall('__POLARS_CSER_0x[^"]+"', explanation)))


def test_cse_rename_cross_join_5405() -> None:
    # https://github.com/pola-rs/polars/issues/5405

    right = pl.DataFrame({"A": [1, 2], "B": [3, 4], "D": [5, 6]}).lazy()
    left = pl.DataFrame({"C": [3, 4]}).lazy().join(right.select("A"), how="cross")

    result = left.join(right.rename({"B": "C"}), on=["A", "C"], how="left").collect(
        comm_subplan_elim=True
    )

    expected = pl.DataFrame(
        {
            "C": [3, 3, 4, 4],
            "A": [1, 2, 1, 2],
            "D": [5, None, None, 6],
        }
    )
    assert_frame_equal(result, expected)


def test_union_duplicates() -> None:
    n_dfs = 10
    df_lazy = pl.DataFrame({}).lazy()
    lazy_dfs = [df_lazy for _ in range(n_dfs)]

    result = len(
        re.findall(
            r".*CACHE\[id: .*, cache_hits: 9].*",
            pl.concat(lazy_dfs).explain(),
            flags=re.MULTILINE,
        )
    )
    assert result


def test_cse_with_struct_expr_11116() -> None:
    # https://github.com/pola-rs/polars/issues/11116

    df = pl.DataFrame([{"s": {"a": 1, "b": 4}, "c": 3}]).lazy()

    result = df.with_columns(
        pl.col("s").struct.field("a").alias("s_a"),
        pl.col("s").struct.field("b").alias("s_b"),
        (
            (pl.col("s").struct.field("a") <= pl.col("c"))
            & (pl.col("s").struct.field("b") > pl.col("c"))
        ).alias("c_between_a_and_b"),
    ).collect(comm_subexpr_elim=True)

    expected = pl.DataFrame(
        {
            "s": [{"a": 1, "b": 4}],
            "c": [3],
            "s_a": [1],
            "s_b": [4],
            "c_between_a_and_b": [True],
        }
    )
    assert_frame_equal(result, expected)


def test_cse_schema_6081() -> None:
    # https://github.com/pola-rs/polars/issues/6081

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

    result = df.join(min_value_by_group, on=["date", "id"], how="left").collect(
        comm_subplan_elim=True, projection_pushdown=True
    )
    expected = pl.DataFrame(
        {
            "date": [date(2022, 12, 12), date(2022, 12, 12), date(2022, 12, 13)],
            "id": [1, 1, 5],
            "value": [1, 2, 2],
            "min_value": [1, 1, 2],
        }
    )
    assert_frame_equal(result, expected)


def test_cse_9630() -> None:
    lf1 = pl.LazyFrame({"key": [1], "x": [1]})
    lf2 = pl.LazyFrame({"key": [1], "y": [2]})

    joined_lf2 = lf1.join(lf2, on="key")

    all_subsections = (
        pl.concat(
            [
                lf1.select("key", pl.col("x").alias("value")),
                joined_lf2.select("key", pl.col("y").alias("value")),
            ]
        )
        .group_by("key")
        .agg(pl.col("value"))
    )

    intersected_df1 = all_subsections.join(lf1, on="key")
    intersected_df2 = all_subsections.join(lf2, on="key")

    result = intersected_df1.join(intersected_df2, on=["key"], how="left").collect(
        comm_subplan_elim=True
    )

    expected = pl.DataFrame(
        {
            "key": [1],
            "value": [[1, 2]],
            "x": [1],
            "value_right": [[1, 2]],
            "y": [2],
        }
    )
    assert_frame_equal(result, expected)


@pytest.mark.write_disk()
def test_schema_row_index_cse() -> None:
    csv_a = NamedTemporaryFile()
    csv_a.write(
        b"""
A,B
Gr1,A
Gr1,B
    """.strip()
    )
    csv_a.seek(0)

    df_a = pl.scan_csv(csv_a.name).with_row_index("Idx")

    result = (
        df_a.join(df_a, on="B")
        .group_by("A", maintain_order=True)
        .all()
        .collect(comm_subexpr_elim=True)
    )

    csv_a.close()

    expected = pl.DataFrame(
        {
            "A": ["Gr1"],
            "Idx": [[0, 1]],
            "B": [["A", "B"]],
            "Idx_right": [[0, 1]],
            "A_right": [["Gr1", "Gr1"]],
        },
        schema_overrides={"Idx": pl.List(pl.UInt32), "Idx_right": pl.List(pl.UInt32)},
    )
    assert_frame_equal(result, expected)


@pytest.mark.debug()
def test_cse_expr_selection_context() -> None:
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

    result = q.select(exprs).collect(comm_subexpr_elim=True)
    assert num_cse_occurrences(q.select(exprs).explain(comm_subexpr_elim=True)) == 2
    expected = pl.DataFrame(
        {
            "d1": [30],
            "foo": [299],
            "d2": [900],
            "d3": [9000],
        }
    )
    assert_frame_equal(result, expected)

    result = q.with_columns(exprs).collect(comm_subexpr_elim=True)
    assert (
        num_cse_occurrences(q.with_columns(exprs).explain(comm_subexpr_elim=True)) == 2
    )
    expected = pl.DataFrame(
        {
            "a": [1, 2, 3, 4],
            "b": [1, 2, 3, 4],
            "c": [1, 2, 3, 4],
            "d1": [30, 30, 30, 30],
            "foo": [299, 299, 299, 299],
            "d2": [900, 900, 900, 900],
            "d3": [9000, 9000, 9000, 9000],
        }
    )
    assert_frame_equal(result, expected)


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
        orient="row",
    )

    result = lf.select(
        c_diff=pl.col("c").diff(1),
        c_diff_by_a=pl.col("c").diff(1).over("a"),
    ).collect(comm_subexpr_elim=True)

    expected = pl.DataFrame(
        {
            "c_diff": [None, 2, -2, 1, 1, 1, -4],
            "c_diff_by_a": [None, 2, -2, None, 1, 1, None],
        }
    )
    assert_frame_equal(result, expected)


def test_cse_group_by_10215() -> None:
    lf = pl.LazyFrame({"a": [1], "b": [1]})

    result = lf.group_by("b").agg(
        (pl.col("a").sum() * pl.col("a").sum()).alias("x"),
        (pl.col("b").sum() * pl.col("b").sum()).alias("y"),
        (pl.col("a").sum() * pl.col("a").sum()).alias("x2"),
        ((pl.col("a") + 2).sum() * pl.col("a").sum()).alias("x3"),
        ((pl.col("a") + 2).sum() * pl.col("b").sum()).alias("x4"),
        ((pl.col("a") + 2).sum() * pl.col("b").sum()),
    )

    assert "__POLARS_CSER" in result.explain(comm_subexpr_elim=True)
    expected = pl.DataFrame(
        {
            "b": [1],
            "x": [1],
            "y": [1],
            "x2": [1],
            "x3": [3],
            "x4": [3],
            "a": [3],
        }
    )
    assert_frame_equal(result.collect(comm_subexpr_elim=True), expected)


def test_cse_mixed_window_functions() -> None:
    # checks if the window caches are cleared
    # there are windows in the cse's and the default expressions
    lf = pl.LazyFrame({"a": [1], "b": [1], "c": [1]})

    result = lf.select(
        pl.col("a"),
        pl.col("b"),
        pl.col("c"),
        pl.col("b").rank().alias("rank"),
        pl.col("b").rank().alias("d_rank"),
        pl.col("b").first().over([pl.col("a")]).alias("b_first"),
        pl.col("b").last().over([pl.col("a")]).alias("b_last"),
        pl.col("b").shift().alias("b_lag_1"),
        pl.col("b").shift().alias("b_lead_1"),
        pl.col("c").cum_sum().alias("c_cumsum"),
        pl.col("c").cum_sum().over([pl.col("a")]).alias("c_cumsum_by_a"),
        pl.col("c").diff().alias("c_diff"),
        pl.col("c").diff().over([pl.col("a")]).alias("c_diff_by_a"),
    ).collect(comm_subexpr_elim=True)

    expected = pl.DataFrame(
        {
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
        },
    ).with_columns(pl.col(pl.Null).cast(pl.Int64))
    assert_frame_equal(result, expected)


def test_cse_10401() -> None:
    df = pl.LazyFrame({"clicks": [1.0, float("nan"), None]})

    q = df.with_columns(pl.all().fill_null(0).fill_nan(0))

    assert r"""col("clicks").fill_null([0.0]).alias("__POLARS_CSER""" in q.explain()

    expected = pl.DataFrame({"clicks": [1.0, 0.0, 0.0]})
    assert_frame_equal(q.collect(comm_subexpr_elim=True), expected)


def test_cse_10441() -> None:
    lf = pl.LazyFrame({"a": [1, 2, 3], "b": [3, 2, 1]})

    result = lf.select(
        pl.col("a").sum() + pl.col("a").sum() + pl.col("b").sum()
    ).collect(comm_subexpr_elim=True)

    expected = pl.DataFrame({"a": [18]})
    assert_frame_equal(result, expected)


def test_cse_10452() -> None:
    lf = pl.LazyFrame({"a": [1, 2, 3], "b": [3, 2, 1]})
    q = lf.select(
        pl.col("b").sum() + pl.col("a").sum().over(pl.col("b")) + pl.col("b").sum()
    )

    assert "__POLARS_CSE" in q.explain(comm_subexpr_elim=True)

    expected = pl.DataFrame({"b": [13, 14, 15]})
    assert_frame_equal(q.collect(comm_subexpr_elim=True), expected)


def test_cse_group_by_ternary_10490() -> None:
    lf = pl.LazyFrame(
        {
            "a": [1, 1, 2, 2],
            "b": [1, 2, 3, 4],
            "c": [2, 3, 4, 5],
        }
    )

    result = (
        lf.group_by("a")
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
    )

    expected = pl.DataFrame(
        {
            "a": [1, 2],
            "b": [1, 1],
            "c": [1, 1],
            "x": [4, 16],
            "y": [9, 49],
            "x2": [4, 16],
            "x3": [12, 32],
            "x4": [18, 56],
        },
        schema_overrides={"b": pl.Int32, "c": pl.Int32},
    )
    assert_frame_equal(result, expected)


def test_cse_quantile_10815() -> None:
    np.random.seed(1)
    a = np.random.random(10)
    b = np.random.random(10)
    df = pl.DataFrame({"a": a, "b": b})
    cols = ["a", "b"]
    q = df.lazy().select(
        *(
            pl.col(c).quantile(0.75, interpolation="midpoint").name.suffix("_3")
            for c in cols
        ),
        *(
            pl.col(c).quantile(0.25, interpolation="midpoint").name.suffix("_1")
            for c in cols
        ),
    )
    assert "__POLARS_CSE" not in q.explain()
    assert q.collect().to_dict(as_series=False) == {
        "a_3": [0.40689473946662197],
        "b_3": [0.6145786693120769],
        "a_1": [0.16650805109739197],
        "b_1": [0.2012768694081981],
    }


def test_cse_nan_10824() -> None:
    v = pl.col("a") / pl.col("b")
    magic = pl.when(v > 0).then(pl.lit(float("nan"))).otherwise(v)
    assert (
        str(
            (
                pl.DataFrame(
                    {
                        "a": [1.0],
                        "b": [1.0],
                    }
                )
                .lazy()
                .select(magic)
                .collect(comm_subexpr_elim=True)
            ).to_dict(as_series=False)
        )
        == "{'literal': [nan]}"
    )


def test_cse_10901() -> None:
    df = pl.DataFrame(data=range(6), schema={"a": pl.Int64})
    a = pl.col("a").rolling_sum(window_size=2)
    b = pl.col("a").rolling_sum(window_size=3)
    exprs = {
        "ax1": a,
        "ax2": a * 2,
        "bx1": b,
        "bx2": b * 2,
    }

    expected = pl.DataFrame(
        {
            "a": [0, 1, 2, 3, 4, 5],
            "ax1": [None, 1, 3, 5, 7, 9],
            "ax2": [None, 2, 6, 10, 14, 18],
            "bx1": [None, None, 3, 6, 9, 12],
            "bx2": [None, None, 6, 12, 18, 24],
        }
    )

    assert_frame_equal(df.lazy().with_columns(**exprs).collect(), expected)


def test_cse_count_in_group_by() -> None:
    q = (
        pl.LazyFrame({"a": [1, 1, 2], "b": [1, 2, 3], "c": [40, 51, 12]})
        .group_by("a")
        .agg(pl.all().slice(0, pl.len() - 1))
    )

    assert "POLARS_CSER" not in q.explain()
    assert q.collect().sort("a").to_dict(as_series=False) == {
        "a": [1, 2],
        "b": [[1], []],
        "c": [[40], []],
    }


def test_cse_slice_11594() -> None:
    df = pl.LazyFrame({"a": [1, 2, 1, 2, 1, 2]})

    q = df.select(
        pl.col("a").slice(offset=1, length=pl.len() - 1).alias("1"),
        pl.col("a").slice(offset=1, length=pl.len() - 1).alias("2"),
    )

    assert "__POLARS_CSE" in q.explain(comm_subexpr_elim=True)

    assert q.collect(comm_subexpr_elim=True).to_dict(as_series=False) == {
        "1": [2, 1, 2, 1, 2],
        "2": [2, 1, 2, 1, 2],
    }

    q = df.select(
        pl.col("a").slice(offset=1, length=pl.len() - 1).alias("1"),
        pl.col("a").slice(offset=0, length=pl.len() - 1).alias("2"),
    )

    assert "__POLARS_CSE" in q.explain(comm_subexpr_elim=True)

    assert q.collect(comm_subexpr_elim=True).to_dict(as_series=False) == {
        "1": [2, 1, 2, 1, 2],
        "2": [1, 2, 1, 2, 1],
    }


def test_cse_is_in_11489() -> None:
    df = pl.DataFrame(
        {"cond": [1, 2, 3, 2, 1], "x": [1.0, 0.20, 3.0, 4.0, 0.50]}
    ).lazy()
    any_cond = (
        pl.when(pl.col("cond").is_in([2, 3]))
        .then(True)
        .when(pl.col("cond").is_in([1]))
        .then(False)
        .otherwise(None)
        .alias("any_cond")
    )
    val = (
        pl.when(any_cond)
        .then(1.0)
        .when(~any_cond)
        .then(0.0)
        .otherwise(None)
        .alias("val")
    )
    assert df.select("cond", any_cond, val).collect().to_dict(as_series=False) == {
        "cond": [1, 2, 3, 2, 1],
        "any_cond": [False, True, True, True, False],
        "val": [0.0, 1.0, 1.0, 1.0, 0.0],
    }


def test_cse_11958() -> None:
    df = pl.LazyFrame({"a": [1, 2, 3, 4, 5]})
    vector_losses = []
    for lag in range(1, 5):
        difference = pl.col("a") - pl.col("a").shift(lag)
        component_loss = pl.when(difference >= 0).then(difference * 10)
        vector_losses.append(component_loss.alias(f"diff{lag}"))

    q = df.select(vector_losses)
    assert "__POLARS_CSE" in q.explain(comm_subexpr_elim=True)
    assert q.collect(comm_subexpr_elim=True).to_dict(as_series=False) == {
        "diff1": [None, 10, 10, 10, 10],
        "diff2": [None, None, 20, 20, 20],
        "diff3": [None, None, None, 30, 30],
        "diff4": [None, None, None, None, 40],
    }


def test_cse_14047() -> None:
    ldf = pl.LazyFrame(
        {
            "timestamp": pl.datetime_range(
                datetime(2024, 1, 12),
                datetime(2024, 1, 12, 0, 0, 0, 150_000),
                "10ms",
                eager=True,
                closed="left",
            ),
            "price": list(range(15)),
        }
    )

    def count_diff(
        price: pl.Expr, upper_bound: float = 0.1, lower_bound: float = 0.001
    ) -> pl.Expr:
        span_end_to_curr = (
            price.count()
            .cast(int)
            .rolling("timestamp", period=timedelta(seconds=lower_bound))
        )
        span_start_to_curr = (
            price.count()
            .cast(int)
            .rolling("timestamp", period=timedelta(seconds=upper_bound))
        )
        return (span_start_to_curr - span_end_to_curr).alias(
            f"count_diff_{upper_bound}_{lower_bound}"
        )

    def s_per_count(count_diff: pl.Expr, span: tuple[float, float]) -> pl.Expr:
        return (span[1] * 1000 - span[0] * 1000) / count_diff

    spans = [(0.001, 0.1), (1, 10)]
    count_diff_exprs = [count_diff(pl.col("price"), span[0], span[1]) for span in spans]
    s_per_count_exprs = [
        s_per_count(count_diff, span).alias(f"zz_{span}")
        for count_diff, span in zip(count_diff_exprs, spans)
    ]

    exprs = count_diff_exprs + s_per_count_exprs
    ldf = ldf.with_columns(*exprs)
    assert_frame_equal(
        ldf.collect(comm_subexpr_elim=True), ldf.collect(comm_subexpr_elim=False)
    )


def test_cse_15536() -> None:
    source = pl.DataFrame({"a": range(10)})

    data = source.lazy().filter(pl.col("a") >= 5)

    assert pl.concat(
        [
            data.filter(pl.lit(True) & (pl.col("a") == 6) | (pl.col("a") == 9)),
            data.filter(pl.lit(True) & (pl.col("a") == 7) | (pl.col("a") == 8)),
        ]
    ).collect()["a"].to_list() == [6, 9, 7, 8]


def test_cse_15548() -> None:
    ldf = pl.LazyFrame({"a": [1, 2, 3]})
    ldf2 = ldf.filter(pl.col("a") == 1).cache()
    ldf3 = pl.concat([ldf, ldf2])

    assert len(ldf3.collect(comm_subplan_elim=False)) == 4
    assert len(ldf3.collect(comm_subplan_elim=True)) == 4


@pytest.mark.debug()
def test_cse_and_schema_update_projection_pd() -> None:
    df = pl.LazyFrame({"a": [1, 2], "b": [99, 99]})

    q = (
        df.lazy()
        .with_row_index()
        .select(
            pl.when(pl.col("b") < 10)
            .then(0.1 * pl.col("b"))
            .when(pl.col("b") < 100)
            .then(0.2 * pl.col("b"))
        )
    )
    assert q.collect(comm_subplan_elim=False).to_dict(as_series=False) == {
        "literal": [19.8, 19.8]
    }
    assert num_cse_occurrences(q.explain(comm_subexpr_elim=True)) == 1


@pytest.mark.debug()
def test_cse_predicate_self_join(capfd: Any, monkeypatch: Any) -> None:
    monkeypatch.setenv("POLARS_VERBOSE", "1")
    y = pl.LazyFrame({"a": [1], "b": [2], "y": [3]})

    xf = y.filter(pl.col("y") == 2).select(["a", "b"])
    y_xf = y.join(xf, on=["a", "b"], how="left")

    y_xf_c = y_xf.select("a", "b")
    assert y_xf_c.collect().to_dict(as_series=False) == {"a": [1], "b": [2]}
    captured = capfd.readouterr().err
    assert "CACHE HIT" in captured


def test_cse_manual_cache_15688() -> None:
    df = pl.LazyFrame(
        {"a": [1, 2, 3, 1, 2, 3], "b": [1, 1, 1, 1, 1, 1], "id": [1, 1, 1, 2, 2, 2]}
    )

    df1 = df.filter(id=1).join(df.filter(id=2), on=["a", "b"], how="semi")
    df2 = df.filter(id=1).join(df1, on=["a", "b"], how="semi")
    df2 = df2.cache()
    res = df2.group_by("b").agg(pl.all().sum())
    assert res.cache().with_columns(foo=1).collect().to_dict(as_series=False) == {
        "b": [1],
        "a": [6],
        "id": [3],
        "foo": [1],
    }


def test_cse_drop_nulls_15795() -> None:
    A = pl.LazyFrame({"X": 1})
    B = pl.LazyFrame({"X": 1, "Y": 0}).filter(pl.col("Y").is_not_null())
    C = A.join(B, on="X").select("X")
    D = B.select("X")
    assert C.join(D, on="X").collect().shape == (1, 1)


def test_cse_no_projection_15980() -> None:
    df = pl.LazyFrame({"x": "a", "y": 1})
    df = pl.concat(df.with_columns(pl.col("y").add(n)) for n in range(2))

    assert df.filter(pl.col("x").eq("a")).select("x").collect().to_dict(
        as_series=False
    ) == {"x": ["a", "a"]}


@pytest.mark.debug()
def test_cse_series_collision_16138() -> None:
    holdings = pl.DataFrame(
        {
            "fund_currency": ["CLP", "CLP"],
            "asset_currency": ["EUR", "USA"],
        }
    )

    usd = ["USD"]
    eur = ["EUR"]
    clp = ["CLP"]

    currency_factor_query_dict = [
        pl.col("asset_currency").is_in(eur) & pl.col("fund_currency").is_in(clp),
        pl.col("asset_currency").is_in(eur) & pl.col("fund_currency").is_in(usd),
        pl.col("asset_currency").is_in(clp) & pl.col("fund_currency").is_in(clp),
        pl.col("asset_currency").is_in(usd) & pl.col("fund_currency").is_in(usd),
    ]

    factor_holdings = holdings.lazy().with_columns(
        pl.coalesce(currency_factor_query_dict).alias("currency_factor"),
    )

    assert factor_holdings.collect(comm_subexpr_elim=True).to_dict(as_series=False) == {
        "fund_currency": ["CLP", "CLP"],
        "asset_currency": ["EUR", "USA"],
        "currency_factor": [True, False],
    }
    assert num_cse_occurrences(factor_holdings.explain(comm_subexpr_elim=True)) == 3


def test_nested_cache_no_panic_16553() -> None:
    assert pl.LazyFrame().select(a=[[[1]]]).collect(comm_subexpr_elim=True).to_dict(
        as_series=False
    ) == {"a": [[[[1]]]]}


def test_hash_empty_series_16577() -> None:
    s = pl.Series(values=None)
    out = pl.LazyFrame().select(s).collect()
    assert out.equals(s.to_frame())


def test_cse_non_scalar_length_mismatch_17732() -> None:
    df = pl.LazyFrame({"a": pl.Series(range(30), dtype=pl.Int32)})
    got = (
        df.lazy()
        .with_columns(
            pl.col("a").head(5).min().alias("b"),
            pl.col("a").head(5).max().alias("c"),
        )
        .collect(comm_subexpr_elim=True)
    )
    expect = pl.DataFrame(
        {
            "a": pl.Series(range(30), dtype=pl.Int32),
            "b": pl.Series([0] * 30, dtype=pl.Int32),
            "c": pl.Series([4] * 30, dtype=pl.Int32),
        }
    )

    assert_frame_equal(expect, got)


def test_cse_chunks_18124() -> None:
    df = pl.DataFrame(
        {
            "ts_diff": [timedelta(seconds=60)] * 2,
            "ts_diff_after": [timedelta(seconds=120)] * 2,
        }
    )
    df = pl.concat([df, df], rechunk=False)
    assert (
        df.lazy()
        .with_columns(
            ts_diff_sign=pl.col("ts_diff") > pl.duration(seconds=0),
            ts_diff_after_sign=pl.col("ts_diff_after") > pl.duration(seconds=0),
        )
        .filter(pl.col("ts_diff") > 1)
    ).collect().shape == (4, 4)


def test_eager_cse_during_struct_expansion_18411() -> None:
    df = pl.DataFrame({"foo": [0, 0, 0, 1, 1]})
    vc = pl.col("foo").value_counts()
    classes = vc.struct[0]
    counts = vc.struct[1]
    # Check if output is stable
    assert (
        df.select(pl.col("foo").replace(classes, counts))
        == df.select(pl.col("foo").replace(classes, counts))
    )["foo"].all()
