from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

import polars as pl
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from pathlib import Path

    from tests.conftest import PlMonkeyPatch


def _left() -> pl.LazyFrame:
    return pl.LazyFrame(
        {
            # Key 1 repeats, key 9 has no match, one key is null.
            "k": [1, 1, 2, 3, 9, None],
            "g": ["a", "a", "b", "b", "c", "c"],
            "v": [10, 20, 30, 40, 50, 60],
        }
    )


def _right() -> pl.LazyFrame:
    return pl.LazyFrame(
        {
            # Key 4 has no match in the left side, and one key is null.
            "k": [1, 1, 1, 2, 2, 3, 4, None],
            "x": [1, None, 3, 4, 5, None, 7, 8],
            "f": [0.5, 1.5, None, 2.5, 3.5, 4.5, 5.5, 6.5],
            "d": [date(2020, 1, i) for i in range(1, 9)],
            "v": [100, 200, 300, 400, 500, 600, 700, 800],
        }
    )


def _plans(
    lf: pl.LazyFrame, plmonkeypatch: PlMonkeyPatch, *, skip_gate: bool = True
) -> tuple[str, str]:
    plmonkeypatch.setenv(
        "POLARS_EAGER_AGGREGATION_SKIP_GATE", "1" if skip_gate else "0"
    )
    plmonkeypatch.setenv("POLARS_EAGER_AGGREGATION", "1")
    on = lf.explain(engine="streaming")
    plmonkeypatch.setenv("POLARS_EAGER_AGGREGATION", "0")
    off = lf.explain(engine="streaming")
    plmonkeypatch.setenv("POLARS_EAGER_AGGREGATION", "1")
    return on, off


def _fired(on: str, off: str) -> bool:
    return on.count("AGGREGATE") == off.count("AGGREGATE") + 1


def _collect_both(
    lf: pl.LazyFrame, plmonkeypatch: PlMonkeyPatch
) -> tuple[pl.DataFrame, pl.DataFrame]:
    plmonkeypatch.setenv("POLARS_EAGER_AGGREGATION_SKIP_GATE", "1")
    plmonkeypatch.setenv("POLARS_EAGER_AGGREGATION", "1")
    on = lf.collect(engine="streaming")
    plmonkeypatch.setenv("POLARS_EAGER_AGGREGATION", "0")
    off = lf.collect(engine="streaming")
    plmonkeypatch.setenv("POLARS_EAGER_AGGREGATION", "1")
    return on, off


def _assert_rewrite(
    lf: pl.LazyFrame, plmonkeypatch: PlMonkeyPatch, *, fires: bool, sort_by: Any
) -> None:
    on, off = _plans(lf, plmonkeypatch)
    assert _fired(on, off) == fires, on
    result, expected = _collect_both(lf, plmonkeypatch)
    assert_frame_equal(result.sort(sort_by), expected.sort(sort_by), rel_tol=1e-9)


AGGS = {
    "count": pl.col("x").count(),
    "len": pl.len(),
    "sum_int": pl.col("x").sum(),
    "sum_float": pl.col("f").sum(),
    "min_int": pl.col("x").min(),
    "max_int": pl.col("x").max(),
    "min_float": pl.col("f").min(),
    "max_float": pl.col("f").nan_max(),
    "min_date": pl.col("d").min(),
    "max_date": pl.col("d").max(),
    "suffixed": pl.col("v_right").sum(),
    "top_part": (pl.col("x").sum() * 2 + pl.col("x").count()).alias("t"),
    "cast": pl.col("x").count().cast(pl.Int64),
    "elementwise_on_top": pl.col("x").max().fill_null(pl.col("x").min()),
    "sql_sum_guard": pl.when(pl.col("x").count() > 0)
    .then(pl.col("x").sum())
    .otherwise(None)
    .alias("s"),
}


@pytest.mark.parametrize("how", ["inner", "left"])
@pytest.mark.parametrize("agg", AGGS.values(), ids=AGGS.keys())
def test_eager_aggregation_splits(
    how: Any, agg: pl.Expr, plmonkeypatch: PlMonkeyPatch
) -> None:
    lf = _left().join(_right(), on="k", how=how).group_by("g").agg(agg)
    _assert_rewrite(lf, plmonkeypatch, fires=True, sort_by="g")


@pytest.mark.parametrize("how", ["inner", "left"])
def test_eager_aggregation_group_keys(how: Any, plmonkeypatch: PlMonkeyPatch) -> None:
    # Keys that are the join key, several keys, and a group whose left rows all lack a
    # match.
    for keys in (["k"], ["g", "k"], ["g", "v"]):
        lf = (
            _left()
            .join(_right(), on="k", how=how)
            .group_by(keys)
            .agg(
                pl.col("x").count().alias("c"),
                pl.len(),
                pl.col("x").sum().alias("s"),
            )
        )
        _assert_rewrite(lf, plmonkeypatch, fires=True, sort_by=keys)


@pytest.mark.parametrize("how", ["inner", "left"])
def test_eager_aggregation_multi_column_keys(
    how: Any, plmonkeypatch: PlMonkeyPatch
) -> None:
    left = pl.LazyFrame(
        {"a": [1, 1, 2, None], "b": [1, 2, 1, 1], "g": ["p", "q", "p", "q"]}
    )
    right = pl.LazyFrame(
        {"a": [1, 1, 1, 2, None, 2], "b": [1, 1, 2, 2, 1, 1], "x": [1, 2, 3, 4, 5, 6]}
    )
    lf = (
        left.join(right, on=["a", "b"], how=how)
        .group_by("g")
        .agg(pl.col("x").sum().alias("s"), pl.len(), pl.col("x").min().alias("m"))
    )
    _assert_rewrite(lf, plmonkeypatch, fires=True, sort_by="g")


def test_eager_aggregation_unmatched_len_counts_once(
    plmonkeypatch: PlMonkeyPatch,
) -> None:
    left = pl.LazyFrame({"k": [1, 2, 3], "g": ["a", "a", "b"]})
    right = pl.LazyFrame({"k": [1, 1, 1], "x": [None, None, 5]})
    lf = (
        left.join(right, on="k", how="left")
        .group_by("g")
        .agg(
            pl.len().alias("n"),
            pl.col("x").count().alias("c"),
            pl.col("x").sum().alias("s"),
            pl.col("x").max().alias("m"),
        )
    )
    _assert_rewrite(lf, plmonkeypatch, fires=True, sort_by="g")
    result, _ = _collect_both(lf, plmonkeypatch)
    assert result.sort("g").to_dict(as_series=False) == {
        "g": ["a", "b"],
        "n": [4, 1],
        "c": [1, 0],
        "s": [5, 0],
        "m": [5, None],
    }


def test_eager_aggregation_sql(plmonkeypatch: PlMonkeyPatch) -> None:
    ctx = pl.SQLContext(customer=_left(), orders=_right())
    lf = ctx.execute(
        """
        SELECT g, COUNT(x) AS c, SUM(x) AS s, SUM(f) AS sf
        FROM customer LEFT JOIN orders ON customer.k = orders.k
        GROUP BY g
        """
    )
    _assert_rewrite(lf, plmonkeypatch, fires=True, sort_by="g")

    # Q13 shape.
    lf = ctx.execute(
        """
        SELECT c_count, COUNT(*) AS custdist FROM (
            SELECT customer.k, COUNT(x) AS c_count
            FROM customer LEFT OUTER JOIN orders ON customer.k = orders.k
            GROUP BY customer.k
        ) AS c_orders
        GROUP BY c_count
        """
    )
    _assert_rewrite(lf, plmonkeypatch, fires=True, sort_by="c_count")


NOT_SPLIT = {
    "fill_null": pl.col("x").fill_null(1).sum(),
    "is_null": pl.col("x").is_null().sum(),
    "literal": pl.lit(1).sum(),
    "head": pl.col("x").head(1).sum(),
    "computed_input": (pl.col("x") + 1).sum(),
    "strict_cast": pl.col("x").cast(pl.Int8, strict=True).sum(),
    "first": pl.col("x").first(),
    "count_with_nulls": pl.col("x").count() + pl.col("x").null_count(),
    "count_include_nulls": pl.col("x").len(),
    "mean": pl.col("x").mean(),
    "bare_column_on_top": pl.col("x").sum() + pl.col("x"),
}


@pytest.mark.parametrize("agg", NOT_SPLIT.values(), ids=NOT_SPLIT.keys())
def test_eager_aggregation_does_not_split(
    agg: pl.Expr, plmonkeypatch: PlMonkeyPatch
) -> None:
    lf = _left().join(_right(), on="k", how="left").group_by("g").agg(agg)
    on, off = _plans(lf, plmonkeypatch)
    assert not _fired(on, off), on


def test_eager_aggregation_decimal_sum_does_not_split(
    plmonkeypatch: PlMonkeyPatch,
) -> None:
    left = pl.LazyFrame({"k": [1], "g": ["a"]})
    right = pl.LazyFrame(
        {
            "k": [1, 2, 2],
            "x": pl.Series(
                [Decimal(1), Decimal("9e37"), Decimal("9e37")],
                dtype=pl.Decimal(38, 0),
            ),
        }
    )
    for how in ("inner", "left"):
        lf = left.join(right, on="k", how=how).group_by("g").agg(pl.col("x").sum())
        on, off = _plans(lf, plmonkeypatch)
        assert not _fired(on, off), on
        result, _ = _collect_both(lf, plmonkeypatch)
        assert result["x"].to_list() == [Decimal(1)]


def test_eager_aggregation_rejected_shapes(plmonkeypatch: PlMonkeyPatch) -> None:
    left, right = _left(), _right()
    agg = pl.col("x").sum()
    shapes = {
        "computed_join_key": left.join(
            right, left_on="k", right_on=pl.col("k") + 1, how="left"
        )
        .group_by("g")
        .agg(agg),
        "maintain_order": left.join(right, on="k", how="left")
        .group_by("g", maintain_order=True)
        .agg(agg),
        "full_join": left.join(right, on="k", how="full").group_by("g").agg(agg),
        "key_from_right": left.join(right, on="k", how="left")
        .group_by("d")
        .agg(pl.col("v").sum()),
        "nulls_equal": left.join(right, on="k", how="left", nulls_equal=True)
        .group_by("g")
        .agg(agg),
    }
    for name, lf in shapes.items():
        on, off = _plans(lf, plmonkeypatch)
        assert not _fired(on, off), (name, on)


def _physical(lf: pl.LazyFrame) -> str:
    out = lf.show_graph(engine="streaming", plan_stage="physical", raw_output=True)
    assert out is not None
    return out


# A right side that is a join has no bound on its rows.
@pytest.mark.parametrize("right_is_join", [False, True])
def test_eager_aggregation_execution_paths(
    right_is_join: bool,
    plmonkeypatch: PlMonkeyPatch,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aggs = [pl.col("x").count().alias("c"), pl.len().alias("n")]
    plmonkeypatch.setenv("POLARS_EAGER_AGGREGATION_SKIP_GATE", "1")
    if right_is_join:
        right = _right().join(pl.LazyFrame({"k": [1, 1, 2, 4]}), on="k")
    else:
        right = _right()

    # Streaming hash group by.
    lf = _left().join(right, on="k", how="left").group_by("g").agg(aggs)
    _assert_rewrite(lf, plmonkeypatch, fires=True, sort_by="g")
    assert "sum_counts" in _physical(lf)

    # An object column sends the group by to the in-memory engine.
    left = _left().with_columns(o=pl.Series([object()] * 6, dtype=pl.Object))
    lf = left.join(right, on="k", how="left").group_by("g", "o").agg(aggs)
    on, off = _plans(lf, plmonkeypatch)
    assert _fired(on, off), on
    physical = _physical(lf)
    assert "in-memory-map" in physical
    assert "sum_counts" in physical
    plmonkeypatch.setenv("POLARS_EAGER_AGGREGATION", "1")
    result = lf.collect(engine="streaming").drop("o").sort("g")
    plmonkeypatch.setenv("POLARS_EAGER_AGGREGATION", "0")
    expected = lf.collect(engine="streaming").drop("o").sort("g")
    plmonkeypatch.setenv("POLARS_EAGER_AGGREGATION", "1")
    assert_frame_equal(result, expected)

    # Sorted group by.
    monkeypatch.setenv("POLARS_FORCE_SORTED_GROUP_BY", "1")
    lf = _left().join(right, on="k", how="left").group_by("g").agg(aggs)
    physical = _physical(lf)
    assert "sorted-group-by" in physical
    assert "sum_counts" in physical
    _assert_rewrite(lf, plmonkeypatch, fires=True, sort_by="g")


def test_eager_aggregation_right_null_keys_filtered(
    plmonkeypatch: PlMonkeyPatch,
) -> None:
    lf = _left().join(_right(), on="k", how="left").group_by("g").agg(pl.len())
    on, _ = _plans(lf, plmonkeypatch)
    assert "is_not_null" in on


def _scan(tmp_path: Path, name: str, df: pl.DataFrame) -> pl.LazyFrame:
    path = tmp_path / f"{name}.parquet"
    df.write_parquet(path)
    return pl.scan_parquet(path)


def _gate_fires(lf: pl.LazyFrame, plmonkeypatch: PlMonkeyPatch) -> bool:
    on, off = _plans(lf, plmonkeypatch, skip_gate=False)
    return _fired(on, off)


def _count_by_group(left: pl.LazyFrame, right: pl.LazyFrame, how: Any) -> pl.LazyFrame:
    return left.join(right, on="k", how=how).group_by("g").agg(pl.col("x").count())


def test_eager_aggregation_gate(plmonkeypatch: PlMonkeyPatch, tmp_path: Path) -> None:
    rng = np.random.default_rng(0)

    def left_frame(keys: np.ndarray) -> pl.DataFrame:
        return pl.DataFrame({"k": keys, "g": keys % 7, "y": keys % 5})

    def right_frame(keys: np.ndarray) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "k": keys,
                "x": rng.integers(0, 9, len(keys)),
                "c": rng.choice(["plain", "special requests"], len(keys)),
            }
        )

    # TPC-H Q13 shape: every order has a customer, many orders per customer, and a
    # `NOT LIKE` filter the estimator only knows as a flat 20 %.
    left = _scan(tmp_path, "q13_l", left_frame(np.arange(30_000)))
    right = _scan(tmp_path, "q13_r", right_frame(rng.integers(0, 30_000, 600_000)))
    right = right.filter(~pl.col("c").str.contains("%special%requests%"))
    assert _gate_fires(_count_by_group(left, right, "left"), plmonkeypatch)

    # L covers 80 % of R's dense key range.
    left = _scan(tmp_path, "part_l", left_frame(np.arange(80_000)))
    right = _scan(tmp_path, "part_r", right_frame(np.repeat(np.arange(100_000), 2)))
    assert _gate_fires(_count_by_group(left, right, "inner"), plmonkeypatch)

    # 90 % of R's keys are null; L covers all of the others.
    left = _scan(tmp_path, "null_l", left_frame(np.arange(10_000)))
    keys = pl.Series("k", np.repeat(np.arange(10_000), 12)).extend(
        pl.Series("k", [None] * 1_080_000, dtype=pl.Int64)
    )
    right = _scan(
        tmp_path,
        "null_r",
        right_frame(np.zeros(1_200_000, dtype=np.int64)).with_columns(keys),
    )
    lf = _count_by_group(left, right, "left")
    assert _gate_fires(lf, plmonkeypatch)
    assert "is_not_null" in lf.explain(engine="streaming")

    # Only 1 % of R's keys are in L, although duplicate L keys make the join output as
    # large as R.
    left = _scan(tmp_path, "dup_l", left_frame(np.repeat(np.arange(1_000), 100)))
    right = _scan(tmp_path, "dup_r", right_frame(rng.integers(0, 100_000, 1_000_000)))
    assert not _gate_fires(_count_by_group(left, right, "left"), plmonkeypatch)

    # Both sides filtered to 20 % over the same keys: only 59 % of R's rows still match.
    left = _scan(tmp_path, "both_l", left_frame(np.repeat(np.arange(100_000), 4)))
    right = _scan(tmp_path, "both_r", right_frame(np.repeat(np.arange(100_000), 6)))
    lf = _count_by_group(
        left.filter(pl.col("y") * 3 != 7), right.filter(pl.col("x") * 3 != 7), "inner"
    )
    assert not _gate_fires(lf, plmonkeypatch)

    # Disjoint key ranges.
    left = _scan(tmp_path, "disj_l", left_frame(np.arange(10_000)))
    right = _scan(
        tmp_path, "disj_r", right_frame(np.repeat(np.arange(100_000, 200_000), 2))
    )
    assert not _gate_fires(_count_by_group(left, right, "inner"), plmonkeypatch)

    # R's key is unique, so nothing folds.
    left = _scan(tmp_path, "uniq_l", left_frame(np.arange(200_000)))
    right = _scan(tmp_path, "uniq_r", right_frame(np.arange(200_000)))
    assert not _gate_fires(_count_by_group(left, right, "inner"), plmonkeypatch)

    # A selective inner join: L keeps a small part of the key range.
    left = _scan(tmp_path, "sel_l", left_frame(np.arange(100_000)))
    right = _scan(tmp_path, "sel_r", right_frame(rng.integers(0, 100_000, 1_000_000)))
    lf = _count_by_group(left.filter(pl.col("y") * 3 != 7), right, "inner")
    assert not _gate_fires(lf, plmonkeypatch)
