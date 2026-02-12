from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import TYPE_CHECKING, Any, TypeVar
from unittest.mock import Mock

import numpy as np
import pytest

import polars as pl
from polars.io.plugins import register_io_source
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from collections.abc import Iterator

    from tests.conftest import PlMonkeyPatch


def num_cse_occurrences(explanation: str) -> int:
    """The number of unique CSE columns in an explain string."""
    return len(set(re.findall(r'__POLARS_CSER_0x[^"]+"', explanation)))


def create_dataframe_source(
    source_df: pl.DataFrame,
    is_pure: bool,
    validate_schame: bool = False,
) -> pl.LazyFrame:
    """Generates a custom io source based on the provided pl.DataFrame."""

    def dataframe_source(
        with_columns: list[str] | None,
        predicate: pl.Expr | None,
        _n_rows: int | None,
        _batch_size: int | None,
    ) -> Iterator[pl.DataFrame]:
        df = source_df.clone()
        if predicate is not None:
            df = df.filter(predicate)
        if with_columns is not None:
            df = df.select(with_columns)
        yield df

    return register_io_source(
        dataframe_source,
        schema=source_df.schema,
        validate_schema=validate_schame,
        is_pure=is_pure,
    )


@pytest.mark.parametrize("use_custom_io_source", [True, False])
def test_cse_rename_cross_join_5405(use_custom_io_source: bool) -> None:
    # https://github.com/pola-rs/polars/issues/5405

    right = pl.DataFrame({"A": [1, 2], "B": [3, 4], "D": [5, 6]}).lazy()
    if use_custom_io_source:
        right = create_dataframe_source(right.collect(), is_pure=True)
    left = pl.DataFrame({"C": [3, 4]}).lazy().join(right.select("A"), how="cross")

    result = left.join(right.rename({"B": "C"}), on=["A", "C"], how="left").collect(
        optimizations=pl.QueryOptFlags(comm_subplan_elim=True)
    )

    expected = pl.DataFrame(
        {
            "C": [3, 3, 4, 4],
            "A": [1, 2, 1, 2],
            "D": [5, None, None, 6],
        }
    )
    assert_frame_equal(result, expected, check_row_order=False)


def test_union_duplicates() -> None:
    n_dfs = 10
    df_lazy = pl.DataFrame({}).lazy()
    lazy_dfs = [df_lazy for _ in range(n_dfs)]

    matches = re.findall(r"CACHE\[id: (.*)]", pl.concat(lazy_dfs).explain())

    assert len(matches) == 10
    assert len(set(matches)) == 1


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
    ).collect(optimizations=pl.QueryOptFlags(comm_subexpr_elim=True))

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
        optimizations=pl.QueryOptFlags(comm_subplan_elim=True, projection_pushdown=True)
    )
    expected = pl.DataFrame(
        {
            "date": [date(2022, 12, 12), date(2022, 12, 12), date(2022, 12, 13)],
            "id": [1, 1, 5],
            "value": [1, 2, 2],
            "min_value": [1, 1, 2],
        }
    )
    assert_frame_equal(result, expected, check_row_order=False)


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
        optimizations=pl.QueryOptFlags(comm_subplan_elim=True)
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


@pytest.mark.write_disk
@pytest.mark.parametrize("maintain_order", [False, True])
def test_schema_row_index_cse(maintain_order: bool) -> None:
    with NamedTemporaryFile() as csv_a:
        csv_a.write(b"A,B\nGr1,A\nGr1,B")
        csv_a.seek(0)

        df_a = pl.scan_csv(csv_a.name).with_row_index("Idx")

        result = (
            df_a.join(df_a, on="B", maintain_order="left" if maintain_order else "none")
            .group_by("A", maintain_order=maintain_order)
            .all()
            .collect(optimizations=pl.QueryOptFlags(comm_subexpr_elim=True))
        )

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
    assert_frame_equal(result, expected, check_row_order=maintain_order)


@pytest.mark.debug
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

    result = q.select(exprs).collect(
        optimizations=pl.QueryOptFlags(comm_subexpr_elim=True)
    )
    assert (
        num_cse_occurrences(
            q.select(exprs).explain(
                optimizations=pl.QueryOptFlags(comm_subexpr_elim=True)
            )
        )
        == 2
    )
    expected = pl.DataFrame(
        {
            "d1": [30],
            "foo": [299],
            "d2": [900],
            "d3": [9000],
        }
    )
    assert_frame_equal(result, expected)

    result = q.with_columns(exprs).collect(
        optimizations=pl.QueryOptFlags(comm_subexpr_elim=True)
    )
    assert (
        num_cse_occurrences(
            q.with_columns(exprs).explain(
                optimizations=pl.QueryOptFlags(comm_subexpr_elim=True)
            )
        )
        == 2
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
    ).collect(optimizations=pl.QueryOptFlags(comm_subexpr_elim=True))

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

    assert "__POLARS_CSER" in result.explain(
        optimizations=pl.QueryOptFlags(comm_subexpr_elim=True)
    )
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
    assert_frame_equal(
        result.collect(optimizations=pl.QueryOptFlags(comm_subexpr_elim=True)), expected
    )


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
        pl.col("b").item().over([pl.col("a")]).alias("b_item"),
        pl.col("b").shift().alias("b_lag_1"),
        pl.col("b").shift().alias("b_lead_1"),
        pl.col("c").cum_sum().alias("c_cumsum"),
        pl.col("c").cum_sum().over([pl.col("a")]).alias("c_cumsum_by_a"),
        pl.col("c").diff().alias("c_diff"),
        pl.col("c").diff().over([pl.col("a")]).alias("c_diff_by_a"),
    ).collect(optimizations=pl.QueryOptFlags(comm_subexpr_elim=True))

    expected = pl.DataFrame(
        {
            "a": [1],
            "b": [1],
            "c": [1],
            "rank": [1.0],
            "d_rank": [1.0],
            "b_first": [1],
            "b_last": [1],
            "b_item": [1],
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
    assert_frame_equal(
        q.collect(optimizations=pl.QueryOptFlags(comm_subexpr_elim=True)), expected
    )


def test_cse_10441() -> None:
    lf = pl.LazyFrame({"a": [1, 2, 3], "b": [3, 2, 1]})

    result = lf.select(
        pl.col("a").sum() + pl.col("a").sum() + pl.col("b").sum()
    ).collect(optimizations=pl.QueryOptFlags(comm_subexpr_elim=True))

    expected = pl.DataFrame({"a": [18]})
    assert_frame_equal(result, expected)


def test_cse_10452() -> None:
    lf = pl.LazyFrame({"a": [1, 2, 3], "b": [3, 2, 1]})
    q = lf.select(
        pl.col("b").sum() + pl.col("a").sum().over(pl.col("b")) + pl.col("b").sum()
    )

    assert "__POLARS_CSE" in q.explain(
        optimizations=pl.QueryOptFlags(comm_subexpr_elim=True)
    )

    expected = pl.DataFrame({"b": [13, 14, 15]})
    assert_frame_equal(
        q.collect(optimizations=pl.QueryOptFlags(comm_subexpr_elim=True)), expected
    )


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
        .collect(optimizations=pl.QueryOptFlags(comm_subexpr_elim=True))
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
                .collect(optimizations=pl.QueryOptFlags(comm_subexpr_elim=True))
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

    assert "__POLARS_CSE" in q.explain(
        optimizations=pl.QueryOptFlags(comm_subexpr_elim=True)
    )

    assert q.collect(optimizations=pl.QueryOptFlags(comm_subexpr_elim=True)).to_dict(
        as_series=False
    ) == {
        "1": [2, 1, 2, 1, 2],
        "2": [2, 1, 2, 1, 2],
    }

    q = df.select(
        pl.col("a").slice(offset=1, length=pl.len() - 1).alias("1"),
        pl.col("a").slice(offset=0, length=pl.len() - 1).alias("2"),
    )

    assert "__POLARS_CSE" in q.explain(
        optimizations=pl.QueryOptFlags(comm_subexpr_elim=True)
    )

    assert q.collect(optimizations=pl.QueryOptFlags(comm_subexpr_elim=True)).to_dict(
        as_series=False
    ) == {
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
    assert "__POLARS_CSE" in q.explain(
        optimizations=pl.QueryOptFlags(comm_subexpr_elim=True)
    )
    assert q.collect(optimizations=pl.QueryOptFlags(comm_subexpr_elim=True)).to_dict(
        as_series=False
    ) == {
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
        for count_diff, span in zip(count_diff_exprs, spans, strict=True)
    ]

    exprs = count_diff_exprs + s_per_count_exprs
    ldf = ldf.with_columns(*exprs)
    assert_frame_equal(
        ldf.collect(optimizations=pl.QueryOptFlags(comm_subexpr_elim=True)),
        ldf.collect(optimizations=pl.QueryOptFlags(comm_subexpr_elim=False)),
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

    assert (
        len(ldf3.collect(optimizations=pl.QueryOptFlags(comm_subplan_elim=False))) == 4
    )
    assert (
        len(ldf3.collect(optimizations=pl.QueryOptFlags(comm_subplan_elim=True))) == 4
    )


@pytest.mark.debug
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
    assert q.collect(optimizations=pl.QueryOptFlags(comm_subplan_elim=False)).to_dict(
        as_series=False
    ) == {"literal": [19.8, 19.8]}
    assert (
        num_cse_occurrences(
            q.explain(optimizations=pl.QueryOptFlags(comm_subexpr_elim=True))
        )
        == 1
    )


@pytest.mark.debug
@pytest.mark.may_fail_auto_streaming
@pytest.mark.parametrize("use_custom_io_source", [True, False])
def test_cse_predicate_self_join(
    capfd: Any, plmonkeypatch: PlMonkeyPatch, use_custom_io_source: bool
) -> None:
    plmonkeypatch.setenv("POLARS_VERBOSE", "1")
    y = pl.LazyFrame({"a": [1], "b": [2], "y": [3]})
    if use_custom_io_source:
        y = create_dataframe_source(y.collect(), is_pure=True)

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


@pytest.mark.debug
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

    assert factor_holdings.collect(
        optimizations=pl.QueryOptFlags(comm_subexpr_elim=True)
    ).to_dict(as_series=False) == {
        "fund_currency": ["CLP", "CLP"],
        "asset_currency": ["EUR", "USA"],
        "currency_factor": [True, False],
    }
    assert (
        num_cse_occurrences(
            factor_holdings.explain(
                optimizations=pl.QueryOptFlags(comm_subexpr_elim=True)
            )
        )
        == 3
    )


def test_nested_cache_no_panic_16553() -> None:
    assert pl.LazyFrame().select(a=[[[1]]]).collect(
        optimizations=pl.QueryOptFlags(comm_subexpr_elim=True)
    ).to_dict(as_series=False) == {"a": [[[[1]]]]}


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
        .collect(optimizations=pl.QueryOptFlags(comm_subexpr_elim=True))
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


@pytest.mark.may_fail_auto_streaming
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


def test_cse_as_struct_19253() -> None:
    df = pl.LazyFrame({"x": [1, 2], "y": [4, 5]})

    assert (
        df.with_columns(
            q1=pl.struct(pl.col.x - pl.col.y.mean()),
            q2=pl.struct(pl.col.x - pl.col.y.mean().over("y")),
        ).collect()
    ).to_dict(as_series=False) == {
        "x": [1, 2],
        "y": [4, 5],
        "q1": [{"x": -3.5}, {"x": -2.5}],
        "q2": [{"x": -3.0}, {"x": -3.0}],
    }


@pytest.mark.may_fail_auto_streaming
def test_cse_as_struct_value_counts_20927() -> None:
    assert pl.DataFrame({"x": [i for i in range(1, 6) for _ in range(i)]}).select(
        pl.struct("x").value_counts().struct.unnest()
    ).sort("count").to_dict(as_series=False) == {
        "x": [{"x": 1}, {"x": 2}, {"x": 3}, {"x": 4}, {"x": 5}],
        "count": [1, 2, 3, 4, 5],
    }


def test_cse_union_19227() -> None:
    lf = pl.LazyFrame({"A": [1], "B": [2]})
    lf_1 = lf.select(C="A", B="B")
    lf_2 = lf.select(C="A", A="B")

    direct = lf_2.join(lf, on=["A"]).select("C", "A", "B")

    indirect = lf_1.join(direct, on=["C", "B"]).select("C", "A", "B")

    out = pl.concat([direct, indirect])
    assert out.collect().schema == pl.Schema(
        [("C", pl.Int64), ("A", pl.Int64), ("B", pl.Int64)]
    )


def test_cse_21115() -> None:
    lf = pl.LazyFrame({"x": 1, "y": 5})

    assert lf.with_columns(
        pl.all().exp() + pl.min_horizontal(pl.all().exp())
    ).collect().to_dict(as_series=False) == {
        "x": [5.43656365691809],
        "y": [151.13144093103566],
    }


@pytest.mark.parametrize("use_custom_io_source", [True, False])
def test_cse_cache_leakage_22339(use_custom_io_source: bool) -> None:
    lf1 = pl.LazyFrame({"x": [True] * 2})
    lf2 = pl.LazyFrame({"x": [True] * 3})
    if use_custom_io_source:
        lf1 = create_dataframe_source(lf1.collect(), is_pure=True)
        lf2 = create_dataframe_source(lf2.collect(), is_pure=True)

    a = lf1
    b = lf1.filter(pl.col("x").not_().over(1))
    c = lf2.filter(pl.col("x").not_().over(1))

    ab = a.join(b, on="x")
    bc = b.join(c, on="x")
    ac = a.join(c, on="x")

    assert pl.concat([ab, bc, ac]).collect().to_dict(as_series=False) == {"x": []}


@pytest.mark.write_disk
def test_multiplex_predicate_pushdown() -> None:
    ldf = pl.LazyFrame({"a": [1, 1, 2, 2], "b": [1, 2, 3, 4]})
    with TemporaryDirectory() as f:
        tmppath = Path(f)
        ldf.sink_parquet(
            pl.PartitionBy(tmppath, key="a", include_key=True),
            sync_on_close="all",
            mkdir=True,
        )
        ldf = pl.scan_parquet(tmppath, hive_partitioning=True)
        ldf = ldf.filter(pl.col("a").eq(1)).select("b")
        assert 'SELECTION: [(col("a")) == (1)]' in pl.explain_all([ldf, ldf])


def test_cse_custom_io_source_same_object() -> None:
    df = pl.DataFrame({"a": [1, 2, 3, 4, 5]})

    io_source = Mock(wraps=lambda *_: iter([df]))

    lf = register_io_source(
        io_source,
        schema=df.schema,
        validate_schema=True,
        is_pure=True,
    )

    lfs = [lf, lf]

    plan = pl.explain_all(lfs)
    caches: list[str] = [
        x for x in map(str.strip, plan.splitlines()) if x.startswith("CACHE[")
    ]
    assert len(caches) == 2
    assert len(set(caches)) == 1

    assert io_source.call_count == 0

    assert_frame_equal(
        pl.concat(pl.collect_all(lfs)),
        pl.DataFrame({"a": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]}),
    )

    assert io_source.call_count == 1

    io_source = Mock(wraps=lambda *_: iter([df]))

    # Without explicit is_pure parameter should default to False
    lf = register_io_source(
        io_source,
        schema=df.schema,
        validate_schema=True,
    )

    lfs = [lf, lf]

    plan = pl.explain_all(lfs)

    caches = [x for x in map(str.strip, plan.splitlines()) if x.startswith("CACHE[")]
    assert len(caches) == 0

    assert io_source.call_count == 0

    assert_frame_equal(
        pl.concat(pl.collect_all(lfs)),
        pl.DataFrame({"a": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]}),
    )

    assert io_source.call_count == 2

    io_source = Mock(wraps=lambda *_: iter([df]))

    # LazyFrames constructed from separate calls do not CSE even if the
    # io_source function is the same.
    #
    # Note: This behavior is achieved by having `register_io_source` wrap
    # the user-provided io plugin with a locally constructed wrapper before
    # passing to the Rust-side.
    lfs = [
        register_io_source(
            io_source,
            schema=df.schema,
            validate_schema=True,
            is_pure=True,
        ),
        register_io_source(
            io_source,
            schema=df.schema,
            validate_schema=True,
            is_pure=True,
        ),
    ]

    caches = [x for x in map(str.strip, plan.splitlines()) if x.startswith("CACHE[")]
    assert len(caches) == 0

    assert io_source.call_count == 0

    assert_frame_equal(
        pl.concat(pl.collect_all(lfs)),
        pl.DataFrame({"a": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]}),
    )

    assert io_source.call_count == 2


@pytest.mark.write_disk
def test_cse_preferred_over_slice() -> None:
    # This test asserts that even if we slice disjoint sections of a lazyframe, caching
    # is preferred, and slicing is not pushed down
    df = pl.DataFrame({"a": list(range(1, 21))})
    with NamedTemporaryFile() as f:
        val = df.write_csv()
        f.write(val.encode())
        f.seek(0)
        ldf = pl.scan_csv(f.name)
        left = ldf.slice(0, 5)
        right = ldf.slice(6, 5)
        q = left.join(right, on="a", how="inner")
        assert "CACHE[id:" in q.explain(
            optimizations=pl.QueryOptFlags(comm_subplan_elim=True)
        )


def test_cse_preferred_over_slice_custom_io_source() -> None:
    # This test asserts that even if we slice disjoint sections of a custom io source,
    # caching is preferred, and slicing is not pushed down
    df = pl.DataFrame({"a": list(range(1, 21))})
    lf = create_dataframe_source(df, is_pure=True)
    left = lf.slice(0, 5)
    right = lf.slice(6, 5)
    q = left.join(right, on="a", how="inner")
    assert "CACHE[id:" in q.explain(
        optimizations=pl.QueryOptFlags(comm_subplan_elim=True)
    )

    lf = create_dataframe_source(df, is_pure=False)
    left = lf.slice(0, 5)
    right = lf.slice(6, 5)
    q = left.join(right, on="a", how="inner")
    assert "CACHE[id:" not in q.explain(
        optimizations=pl.QueryOptFlags(comm_subplan_elim=True)
    )


def test_cse_custom_io_source_diff_columns() -> None:
    df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": [10, 11, 12, 13, 14]})
    lf = create_dataframe_source(df, is_pure=True)
    collection = [lf.select("a"), lf.select("b")]
    assert "CACHE[id:" in pl.explain_all(collection)
    collected = pl.collect_all(
        collection, optimizations=pl.QueryOptFlags(comm_subplan_elim=True)
    )
    assert_frame_equal(df.select("a"), collected[0])
    assert_frame_equal(df.select("b"), collected[1])


def test_cse_custom_io_source_diff_filters() -> None:
    df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": [10, 11, 12, 13, 14]})
    lf = create_dataframe_source(df, is_pure=True)

    # We use this so that the true type of the input is passed through
    # to the output
    PolarsFrame = TypeVar("PolarsFrame", pl.DataFrame, pl.LazyFrame)

    def left_pipe(df_or_lf: PolarsFrame) -> PolarsFrame:
        return df_or_lf.select("a").filter(pl.col("a").is_between(2, 6))

    def right_pipe(df_or_lf: PolarsFrame) -> PolarsFrame:
        return df_or_lf.select("b").filter(pl.col("b").is_between(10, 13))

    collection = [lf.pipe(left_pipe), lf.pipe(right_pipe)]
    explanation = pl.explain_all(collection)
    # we prefer predicate pushdown over CSE
    assert "CACHE[id:" not in explanation
    assert 'SELECTION: col("a").is_between([2, 6])' in explanation
    assert 'SELECTION: col("b").is_between([10, 13])' in explanation

    res = pl.collect_all(collection)
    expected = [df.pipe(left_pipe), df.pipe(right_pipe)]
    assert_frame_equal(expected[0], res[0])
    assert_frame_equal(expected[1], res[1])


@pytest.mark.skip
def test_cspe_recursive_24744() -> None:
    df_a = pl.DataFrame([pl.Series("x", [0, 1, 2, 3], dtype=pl.UInt32)])

    def convoluted_inner_join(
        lf_left: pl.LazyFrame,
        lf_right: pl.LazyFrame,
    ) -> pl.LazyFrame:
        lf_left = lf_left.with_columns(pl.col("x").alias("index"))

        lf_joined = lf_left.join(
            lf_right,
            how="inner",
            on=["x"],
        )

        lf_joined_final = lf_left.join(
            lf_joined,
            how="inner",
            on=["index", "x"],
        ).drop("index")
        return lf_joined_final

    lf_a = df_a.lazy()
    lf_j1 = convoluted_inner_join(lf_left=lf_a, lf_right=lf_a)
    lf_j2 = convoluted_inner_join(lf_left=lf_j1, lf_right=lf_a)
    lf_j3 = convoluted_inner_join(lf_left=lf_j2, lf_right=lf_a).sort("x")

    assert lf_j3.explain().count("CACHE") == 14
    assert_frame_equal(
        lf_j3.collect(),
        lf_j3.collect(optimizations=pl.QueryOptFlags(comm_subplan_elim=False)),
    )
    assert (
        lf_j3.show_graph(  # type: ignore[union-attr]
            engine="streaming", plan_stage="physical", raw_output=True
        ).count("multiplexer")
        == 3
    )
    assert (
        lf_j3.show_graph(  # type: ignore[union-attr]
            engine="in-memory", plan_stage="physical", raw_output=True
        ).count("CACHE")
        == 3
    )


def test_cpse_predicates_25030() -> None:
    df = pl.LazyFrame({"key": [1, 2, 2], "x": [6, 2, 3], "y": [0, 1, 4]})

    q1 = df.group_by("key").len().filter(pl.col("len") > 1)
    q2 = df.filter(pl.col.x > pl.col.y)

    q3 = q1.join(q2, on="key")

    q4 = q3.group_by("key").len().join(q3, on="key")

    got = q4.collect()
    expected = q4.collect(optimizations=pl.QueryOptFlags(comm_subplan_elim=False))

    assert_frame_equal(got, expected)
    assert q4.explain().count("CACHE") == 2


def test_asof_join_25699() -> None:
    df = pl.LazyFrame({"a": [10], "b": [10]})

    df = df.with_columns(pl.col("a"))
    df = df.with_columns(pl.col("b"))

    assert_frame_equal(
        df.join_asof(df, on="b").collect(),
        pl.DataFrame({"a": [10], "b": [10], "a_right": [10]}),
    )


def test_csee_python_function() -> None:
    # Make sure to use the same expression
    # This only works for functions on the same address
    expr = pl.col("a").map_elements(lambda x: hash(x))
    q = pl.LazyFrame({"a": [10], "b": [10]}).with_columns(
        a=expr * 10,
        b=expr * 100,
    )

    assert "__POLARS_CSER" in q.explain()
    assert_frame_equal(
        q.collect(), q.collect(optimizations=pl.QueryOptFlags(comm_subexpr_elim=False))
    )


def test_csee_streaming() -> None:
    lf = pl.LazyFrame({"a": [10], "b": [10]})

    # elementwise is allowed
    expr = pl.col("a") * pl.col("b")
    q = lf.with_columns(
        a=expr * 10,
        b=expr * 100,
    )
    assert "__POLARS_CSER" in q.explain(engine="streaming")

    # non-elementwise not
    expr = pl.col("a").sum()
    q = lf.with_columns(
        a=expr * 10,
        b=expr * 100,
    )
    assert "__POLARS_CSER" not in q.explain(engine="streaming")
