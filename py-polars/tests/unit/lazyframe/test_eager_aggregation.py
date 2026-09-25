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
    lf: pl.LazyFrame,
    plmonkeypatch: PlMonkeyPatch,
    *,
    skip_gate: bool = True,
    optimizations: pl.QueryOptFlags | None = None,
) -> tuple[str, str]:
    optimizations = optimizations or pl.QueryOptFlags()
    plmonkeypatch.setenv(
        "POLARS_EAGER_AGGREGATION_SKIP_GATE", "1" if skip_gate else "0"
    )
    plmonkeypatch.setenv("POLARS_EAGER_AGGREGATION", "1")
    on = lf.explain(engine="streaming", optimizations=optimizations)
    plmonkeypatch.setenv("POLARS_EAGER_AGGREGATION", "0")
    off = lf.explain(engine="streaming", optimizations=optimizations)
    plmonkeypatch.setenv("POLARS_EAGER_AGGREGATION", "1")
    return on, off


def _fired(on: str, off: str) -> bool:
    return on.count("AGGREGATE") == off.count("AGGREGATE") + 1


def _collect_both(
    lf: pl.LazyFrame,
    plmonkeypatch: PlMonkeyPatch,
    optimizations: pl.QueryOptFlags | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    optimizations = optimizations or pl.QueryOptFlags()
    plmonkeypatch.setenv("POLARS_EAGER_AGGREGATION_SKIP_GATE", "1")
    plmonkeypatch.setenv("POLARS_EAGER_AGGREGATION", "1")
    on = lf.collect(engine="streaming", optimizations=optimizations)
    plmonkeypatch.setenv("POLARS_EAGER_AGGREGATION", "0")
    off = lf.collect(engine="streaming", optimizations=optimizations)
    plmonkeypatch.setenv("POLARS_EAGER_AGGREGATION", "1")
    return on, off


def _assert_rewrite(
    lf: pl.LazyFrame,
    plmonkeypatch: PlMonkeyPatch,
    *,
    fires: bool,
    sort_by: Any,
    optimizations: pl.QueryOptFlags | None = None,
) -> None:
    on, off = _plans(lf, plmonkeypatch, optimizations=optimizations)
    assert _fired(on, off) == fires, on
    result, expected = _collect_both(lf, plmonkeypatch, optimizations)
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
    "sum_arithmetic": (pl.col("f") * (1 - pl.col("x"))).sum(),
    "sum_divide": (pl.col("x") / pl.col("f")).sum(),
    "sum_plus_literal": (pl.col("x") + 1).sum(),
    "count_arithmetic": (pl.col("x") + pl.col("f")).count(),
    "min_arithmetic": (pl.col("x") - 1).min(),
    "max_arithmetic": (pl.col("f") * 2).max(),
    "non_strict_cast": (pl.col("x").cast(pl.Float32, strict=False) * 2).sum(),
    "sql_sum_guard_arithmetic": pl.when((pl.col("f") * (1 - pl.col("x"))).count() > 0)
    .then((pl.col("f") * (1 - pl.col("x"))).sum())
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
    "fill_null_in_arithmetic": (pl.col("x").fill_null(0) * pl.col("f")).sum(),
    "boolean": ((pl.col("x") > 1) | (pl.col("f") > 1)).sum(),
    "boolean_cast": (pl.col("x") > 1).cast(pl.Int64).sum(),
    "floor_divide": (pl.col("x") // 2).sum(),
    "modulo": (pl.col("x") % 2).sum(),
    "strict_cast_in_arithmetic": (pl.col("x").cast(pl.Float64) * 2).sum(),
    "literals_only": (pl.lit(2) * pl.lit(3)).sum(),
    "function_in_arithmetic": (pl.col("x").abs() * 2).sum(),
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


def test_eager_aggregation_equal_leaves_share_a_partial(
    plmonkeypatch: PlMonkeyPatch,
) -> None:
    e = pl.col("f") * (1 - pl.col("x"))
    lf = (
        _left()
        .join(_right(), on="k")
        .group_by("g")
        .agg(e.sum().alias("a"), (e.sum() * 2).alias("b"), e.count().alias("c"))
    )
    _assert_rewrite(lf, plmonkeypatch, fires=True, sort_by="g")
    on, _ = _plans(lf, plmonkeypatch)
    partial = next(line for line in on.splitlines() if 'BY [col("k")]' in line)
    assert partial.count(".sum()") == 1, partial
    assert partial.count(".count()") == 1, partial


def test_eager_aggregation_decimal_arithmetic_does_not_split(
    plmonkeypatch: PlMonkeyPatch,
) -> None:
    # Key 2 has no match, and dividing its Decimal values raises. The original never
    # divides that row, so it must still return a result.
    left = pl.LazyFrame({"k": [1], "g": ["a"]})
    dec = pl.Decimal(10, 2)
    right = pl.LazyFrame(
        {
            "k": [1, 2],
            "a": pl.Series([Decimal("2"), Decimal("1")], dtype=dec),
            "b": pl.Series([Decimal("1"), Decimal("0")], dtype=dec),
        }
    )
    agg = (pl.col("a") / pl.col("b")).cast(pl.Float64, strict=False).sum()
    lf = left.join(right, on="k").group_by("g").agg(agg)
    _assert_rewrite(lf, plmonkeypatch, fires=False, sort_by="g")
    assert lf.collect(engine="streaming")["a"].to_list() == [2.0]


def _decimal_right() -> pl.LazyFrame:
    dec = pl.Decimal(7, 2)
    values = [Decimal("1.25"), None, Decimal("-3.10"), Decimal("4.00")]
    return pl.LazyFrame(
        {
            "k": [1, 1, 1, 2, 2, 3, 4, None],
            "a": pl.Series(values * 2, dtype=dec),
            "b": pl.Series(values[::-1] * 2, dtype=dec),
            "c": pl.Series(
                [Decimal(v) for v in ("0.50", "7.75", "-2.00", "9.99")] * 2, dtype=dec
            ),
            "d": pl.Series(
                [Decimal(v) for v in ("10.01", "-0.25", "3.33", "6.40")] * 2, dtype=dec
            ),
            "w": pl.Series(values * 2, dtype=pl.Decimal(38, 2)),
            "x": [1, None, 3, 4, 5, None, 7, 8],
        }
    )


_DECIMAL_Q04 = ((pl.col("a") - pl.col("b") - pl.col("c")) + pl.col("d")) / pl.lit(
    Decimal("2.00")
)
DECIMAL_AGGS = {
    "sum": pl.col("a").sum(),
    "q04_arithmetic": _DECIMAL_Q04.sum(),
    "multiply": (pl.col("a") * pl.col("b")).sum(),
    "int_mixed": (pl.col("a") * pl.col("x") + 1).sum(),
    "int_times_literal": (pl.col("x") * pl.lit(Decimal("1.5"))).sum(),
    "cast": pl.col("a").cast(pl.Decimal(38, 1), strict=False).sum(),
    "min": pl.col("a").min(),
    "max": (pl.col("a") - pl.col("b")).max(),
    "count": (pl.col("a") * 2).count(),
    "sql_sum_guard": pl.when(_DECIMAL_Q04.count() > 0)
    .then(_DECIMAL_Q04.sum())
    .otherwise(None)
    .alias("s"),
}


@pytest.mark.parametrize("how", ["inner", "left"])
@pytest.mark.parametrize("agg", DECIMAL_AGGS.values(), ids=DECIMAL_AGGS.keys())
def test_eager_aggregation_decimal_splits(
    how: Any, agg: pl.Expr, plmonkeypatch: PlMonkeyPatch
) -> None:
    # In a left join, group "c" only has an unmatched row: a bare sum gives 0 there and
    # the guarded sum null, with the pass on as with it off.
    lf = _left().join(_decimal_right(), on="k", how=how).group_by("g").agg(agg)
    _assert_rewrite(lf, plmonkeypatch, fires=True, sort_by="g")


DECIMAL_REJECTED = {
    # The values are small, but a Decimal(38, 2) column can hold ones that overflow.
    "wide_multiply": (pl.col("w") * pl.col("w")).sum(),
    "divide_by_column": (pl.col("a") / pl.col("b")).sum(),
    "divide_by_zero": (pl.col("a") / pl.lit(Decimal("0.00"))).sum(),
    "float_mixed": (pl.col("a") * pl.lit(0.5, dtype=pl.Float64))
    .cast(pl.Decimal(38, 2), strict=False)
    .sum(),
}


@pytest.mark.parametrize("agg", DECIMAL_REJECTED.values(), ids=DECIMAL_REJECTED.keys())
def test_eager_aggregation_decimal_does_not_split(
    agg: pl.Expr, plmonkeypatch: PlMonkeyPatch
) -> None:
    lf = _left().join(_decimal_right(), on="k").group_by("g").agg(agg)
    on, off = _plans(lf, plmonkeypatch)
    assert not _fired(on, off), on


def _only_key_1_matches(right: pl.LazyFrame, agg: pl.Expr) -> pl.LazyFrame:
    left = pl.LazyFrame({"k": [1], "g": ["a"]})
    return left.join(right, on="k").group_by("g").agg(agg.alias("s"))


def test_eager_aggregation_decimal_sum_join_multiplicity(
    plmonkeypatch: PlMonkeyPatch,
) -> None:
    # R's 98 rows alone keep the sum small enough, but each R row meets 48 L rows.
    # Summed per key first, the final sum would overflow; the original sums to 0.
    dec = pl.Decimal(35, 0)
    big = Decimal("9e34")
    right = pl.LazyFrame(
        {
            "k": [1, 2] * 49,
            "x": pl.Series([big, -big] * 49, dtype=dec),
        }
    )
    left = pl.LazyFrame({"k": [1, 2] * 48, "g": ["a"] * 96})
    lf = left.join(right, on="k").group_by("g").agg(pl.col("x").sum())
    on, off = _plans(lf, plmonkeypatch)
    assert not _fired(on, off), on
    result, _ = _collect_both(lf, plmonkeypatch)
    assert result["x"].to_list() == [Decimal(0)]


def test_eager_aggregation_decimal_chained_rounding(
    plmonkeypatch: PlMonkeyPatch,
) -> None:
    # At scale 1, 0.1 * 0.6 rounds back to 0.1, so key 2's value does not shrink and
    # overflows at the end. The original never evaluates key 2.
    right = pl.LazyFrame(
        {
            "k": [1, 2],
            "x": pl.Series([Decimal("0.0"), Decimal("0.1")], dtype=pl.Decimal(2, 1)),
        }
    )
    e = pl.col("x")
    for _ in range(20):
        e = e * pl.lit(Decimal("0.6"))
    e = e * pl.lit(Decimal("1e35"), dtype=pl.Decimal(38, 0)) * 20_000
    lf = _only_key_1_matches(right, e.sum())
    on, off = _plans(lf, plmonkeypatch)
    assert not _fired(on, off), on
    result, _ = _collect_both(lf, plmonkeypatch)
    assert result["s"].to_list() == [Decimal("0.0")]


def test_eager_aggregation_decimal_large_intermediate(
    plmonkeypatch: PlMonkeyPatch,
) -> None:
    # The literals add up to 10**38 - 5, which fits, but key 2's 9 on top does not. In
    # f64 the sum rounds to just below 10**38.
    right = pl.LazyFrame(
        {
            "k": [1, 2],
            "x": pl.Series([Decimal(0), Decimal(9)], dtype=pl.Decimal(1, 0)),
        }
    )
    dec = pl.Decimal(38, 0)
    quarter = 25 * 10**36
    e = pl.col("x")
    for value in (quarter, quarter, quarter, quarter - 5):
        e = e + pl.lit(Decimal(value), dtype=dec)
    e = e / pl.lit(Decimal(10**37), dtype=dec)
    lf = _only_key_1_matches(right, e.sum())
    on, off = _plans(lf, plmonkeypatch)
    assert not _fired(on, off), on
    result, _ = _collect_both(lf, plmonkeypatch)
    assert result["s"].to_list() == [Decimal(10)]


@pytest.mark.parametrize(
    ("values", "agg"),
    [
        # A non-strict float to Decimal cast still raises when the value has too many
        # digits.
        ([1.0, 1e6], pl.col("x").cast(pl.Decimal(7, 2), strict=False).sum()),
        # These casts are not supported and raise.
        ([[1], [2]], pl.col("x").cast(pl.Int64, strict=False).sum()),
        ([b"1", b"2"], pl.col("x").cast(pl.Int64, strict=False).sum()),
    ],
    ids=["float_to_decimal", "list_to_int", "binary_to_int"],
)
def test_eager_aggregation_raising_cast_does_not_split(
    values: list[Any], agg: pl.Expr, plmonkeypatch: PlMonkeyPatch
) -> None:
    right = pl.LazyFrame({"k": [1, 2], "x": values})
    lf = _only_key_1_matches(right, agg)
    on, off = _plans(lf, plmonkeypatch)
    assert not _fired(on, off), on


def _fires_for(right: pl.LazyFrame, agg: pl.Expr, plmonkeypatch: PlMonkeyPatch) -> bool:
    lf = _left().join(right, on="k").group_by("g").agg(agg)
    return _fired(*_plans(lf, plmonkeypatch))


def test_eager_aggregation_decimal_sum_needs_a_row_bound(
    plmonkeypatch: PlMonkeyPatch,
) -> None:
    # Without a bound on R's rows, a Decimal sum does not split; an integer sum does, so
    # the missing bound is what blocks the rewrite.
    dec_sum, int_sum = pl.col("a").sum(), pl.col("x").sum()
    right = _decimal_right()
    assert _fires_for(right, dec_sum, plmonkeypatch)

    unknown = right.map_batches(lambda df: df)
    # One row expanded to 100 by the select.
    expanded = (
        right.head(1)
        .select(pl.int_range(0, 100).alias("k"), pl.col("a"), pl.col("x"))
        .with_columns(pl.col("k") % 5)
    )
    applied = right.group_by("k").map_groups(
        lambda df: df, schema=right.collect_schema()
    )
    dynamic = (
        right.drop_nulls("k")
        .sort("k")
        .with_columns(t=pl.int_range(pl.len()))
        .group_by_dynamic("t", every="2i")
        .agg(pl.col("k").first(), pl.col("a").sum(), pl.col("x").sum())
    )
    for shape in (unknown, expanded, applied, dynamic):
        assert not _fires_for(shape, dec_sum, plmonkeypatch)
        assert _fires_for(shape, int_sum, plmonkeypatch)


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
    on, _ = _plans(lf, plmonkeypatch)
    # Without a row bound, partial counts are summed as u64 so they cannot overflow.
    assert ("strict_cast(UInt64).sum()" in on) == right_is_join, on

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


def _gate_fires(
    lf: pl.LazyFrame,
    plmonkeypatch: PlMonkeyPatch,
    optimizations: pl.QueryOptFlags | None = None,
) -> bool:
    on, off = _plans(lf, plmonkeypatch, skip_gate=False, optimizations=optimizations)
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


# Frames without shared column names, like SQL tables: `fact` is aggregated, `cust`
# holds the group keys, and `nation` sits above them.
def _fact() -> pl.LazyFrame:
    return pl.LazyFrame(
        {
            # Customer 4 does not exist, and one key is null.
            "f_ck": [1, 1, 1, 2, 2, 3, 4, None],
            "f_price": [1.0, 2.0, None, 4.0, 5.0, 6.0, 7.0, 8.0],
            "f_disc": [0.1, 0.2, 0.3, None, 0.5, 0.6, 0.7, 0.8],
            "f_qty": [1, 2, 3, 4, None, 6, 7, 8],
            "f_dk": [10, 10, 20, 20, 30, 30, 10, 20],
        }
    )


def _cust() -> pl.LazyFrame:
    return pl.LazyFrame(
        {
            # Customer 5 has no rows in `fact`.
            "c_ck": [1, 2, 3, 5],
            "c_name": ["a", "b", "c", "e"],
            "c_nk": [0, 0, 1, 1],
        }
    )


def _nation() -> pl.LazyFrame:
    return pl.LazyFrame({"n_nk": [0, 1], "n_name": ["x", "y"]})


def _revenue() -> list[pl.Expr]:
    e = pl.col("f_price") * (1 - pl.col("f_disc"))
    return [
        pl.when(e.count() > 0).then(e.sum()).otherwise(None).alias("revenue"),
        pl.col("f_qty").max().alias("q"),
        pl.len().alias("n"),
    ]


def _join(
    left: pl.LazyFrame, right: pl.LazyFrame, left_on: str, right_on: str, **kwargs: Any
) -> pl.LazyFrame:
    return left.join(
        right, left_on=left_on, right_on=right_on, coalesce=False, **kwargs
    )


@pytest.mark.parametrize("fact_is_left", [True, False])
def test_eager_aggregation_right_side_either_input(
    fact_is_left: bool, plmonkeypatch: PlMonkeyPatch
) -> None:
    if fact_is_left:
        joined = _join(_fact(), _cust(), "f_ck", "c_ck")
    else:
        joined = _join(_cust(), _fact(), "c_ck", "f_ck")
    lf = joined.group_by("c_ck", "c_name").agg(_revenue())
    _assert_rewrite(lf, plmonkeypatch, fires=True, sort_by="c_ck")


@pytest.mark.parametrize("how", ["inner", "left"])
def test_eager_aggregation_group_by_right_join_key(
    how: Any, plmonkeypatch: PlMonkeyPatch
) -> None:
    # Default coalescing keeps only the left key, here R's.
    lf = (
        _fact()
        .join(_cust(), left_on="f_ck", right_on="c_ck")
        .group_by("f_ck", "c_name")
        .agg(_revenue())
    )
    _assert_rewrite(lf, plmonkeypatch, fires=True, sort_by="f_ck")

    # R on the right, grouped by its retained key.
    lf = (
        _join(_cust(), _fact(), "c_ck", "f_ck", how=how)
        .group_by("f_ck", "c_name")
        .agg(_revenue())
    )
    _assert_rewrite(lf, plmonkeypatch, fires=True, sort_by=["f_ck", "c_name"])


def test_eager_aggregation_right_side_left_input_rejected(
    plmonkeypatch: PlMonkeyPatch,
) -> None:
    # A left join keeps the aggregated side's unmatched rows.
    lf = (
        _join(_fact(), _cust(), "f_ck", "c_ck", how="left")
        .group_by("c_ck", "c_name")
        .agg(pl.col("f_qty").sum())
    )
    _assert_rewrite(lf, plmonkeypatch, fires=False, sort_by="c_ck")

    # A name both sides use, and read from both, is suffixed on the other side.
    fact = _fact().with_columns(c_nk=pl.col("f_qty"))
    lf = (
        _join(fact, _cust(), "f_ck", "c_ck")
        .group_by("c_ck", "c_nk_right")
        .agg(pl.col("c_nk").sum())
    )
    _assert_rewrite(lf, plmonkeypatch, fires=False, sort_by="c_ck")


@pytest.mark.parametrize("how", ["inner", "left"])
def test_eager_aggregation_group_key_from_aggregated_side(
    how: Any, plmonkeypatch: PlMonkeyPatch
) -> None:
    # R on the right: `x` has nulls, `v_right` is suffixed, and in a left join the
    # unmatched left rows give null for both.
    for keys in (["g", "x"], ["g", "v_right"], ["x"]):
        lf = (
            _left()
            .join(_right(), on="k", how=how)
            .group_by(keys)
            .agg(
                pl.col("f").sum().alias("s"),
                pl.col("f").count().alias("c"),
                pl.len(),
                pl.col("d").min().alias("m"),
            )
        )
        _assert_rewrite(lf, plmonkeypatch, fires=True, sort_by=keys)


@pytest.mark.parametrize("fact_is_left", [True, False])
def test_eager_aggregation_group_key_from_aggregated_side_either_input(
    fact_is_left: bool, plmonkeypatch: PlMonkeyPatch
) -> None:
    if fact_is_left:
        joined = _join(_fact(), _cust(), "f_ck", "c_ck")
    else:
        joined = _join(_cust(), _fact(), "c_ck", "f_ck")
    lf = joined.group_by("c_name", "f_dk").agg(_revenue())
    _assert_rewrite(lf, plmonkeypatch, fires=True, sort_by=["c_name", "f_dk"])


def test_eager_aggregation_group_key_through_a_join_below(
    plmonkeypatch: PlMonkeyPatch,
) -> None:
    # TPC-DS q04: R is the fact table joined to a dimension that gives a group key.
    dim = pl.LazyFrame({"d_dk": [10, 20, 30], "d_year": [2001, 2002, None]})
    lf = (
        _join(_join(_fact(), dim, "f_dk", "d_dk"), _cust(), "f_ck", "c_ck")
        .group_by("c_name", "d_year")
        .agg(_revenue())
    )
    _assert_rewrite(
        lf,
        plmonkeypatch,
        fires=True,
        sort_by=["c_name", "d_year"],
        optimizations=_KEEP_JOIN_ORDER,
    )

    # A join above keyed on a column of R that is not a group key: the partial
    # aggregation goes under that join instead, grouped by both keys.
    lf = (
        _join(_join(_fact(), _cust(), "f_ck", "c_ck"), dim, "f_dk", "d_dk")
        .group_by("c_ck", "d_year")
        .agg(pl.col("f_qty").sum())
    )
    _assert_rewrite(lf, plmonkeypatch, fires=True, sort_by=["c_ck", "d_year"])


def test_eager_aggregation_gate_counts_partial_groups(
    plmonkeypatch: PlMonkeyPatch, tmp_path: Path
) -> None:
    rng = np.random.default_rng(0)
    n = 200_000
    left = _scan(
        tmp_path,
        "left",
        pl.DataFrame({"k": np.arange(1_000), "g": np.arange(1_000) % 7}),
    )
    right = _scan(
        tmp_path,
        "right",
        pl.DataFrame(
            {
                "k": rng.integers(0, 1_000, n),
                "few": rng.integers(0, 3, n),
                "row": np.arange(n),
                "x": rng.integers(0, 100, n),
            }
        ),
    )
    joined = left.join(right, on="k")
    agg = pl.col("x").sum()
    assert _gate_fires(joined.group_by("g").agg(agg), plmonkeypatch)
    assert _gate_fires(joined.group_by("g", "few").agg(agg), plmonkeypatch)
    # Grouping R by its row number leaves nothing to fold.
    assert not _gate_fires(joined.group_by("g", "row").agg(agg), plmonkeypatch)


@pytest.mark.parametrize(
    ("restriction", "groups"),
    [
        ((pl.col("year") >= 2001) & (pl.col("year") <= 2002), 2),
        (pl.col("year").is_between(2001, 2003, closed="left"), 2),
        ((pl.col("year") == 2001) | (pl.col("year") == 2005), 2),
        (pl.col("year").is_in([2001, 2002, 2003]), 3),
    ],
)
def test_eager_aggregation_gate_caps_filtered_group_keys(
    restriction: pl.Expr,
    groups: int,
    plmonkeypatch: PlMonkeyPatch,
    capfd: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    # A filtered group key from a dimension joined into R, as the year in TPC-DS q04:
    # its column statistics still cover every year.
    rng = np.random.default_rng(0)
    n = 200_000
    fact = _scan(
        tmp_path,
        "fact",
        pl.DataFrame(
            {
                "k": rng.integers(0, 1_000, n),
                "dk": rng.integers(0, 3_000, n),
                "x": rng.integers(0, 100, n),
            }
        ),
    )
    dim = _scan(
        tmp_path,
        "dim",
        pl.DataFrame({"d_dk": np.arange(3_000), "year": 1900 + np.arange(3_000) // 15}),
    )
    left = _scan(tmp_path, "left", pl.DataFrame({"k": np.arange(1_000), "g": "a"}))
    r = fact.join(dim.filter(restriction), left_on="dk", right_on="d_dk")
    lf = left.join(r, on="k").group_by("g", "year").agg(pl.col("x").sum())
    plmonkeypatch.setenv("POLARS_VERBOSE", "1")
    capfd.readouterr()
    _plans(lf, plmonkeypatch, skip_gate=False, optimizations=_KEEP_JOIN_ORDER)
    err = capfd.readouterr().err
    assert f"groups per join key {groups}.0" in err, err


def test_eager_aggregation_gate_needs_the_join_to_do_more_than_filter(
    plmonkeypatch: PlMonkeyPatch, tmp_path: Path
) -> None:
    # TPC-DS q53: every group key comes from R and L has one row per key, so the join
    # only filters R and there is nothing for a partial aggregation to save.
    n = 200_000
    rng = np.random.default_rng(0)
    right = _scan(
        tmp_path,
        "right",
        pl.DataFrame(
            {
                "k": rng.integers(0, 100, n),
                "g": rng.integers(0, 50, n),
                "x": rng.integers(0, 100, n),
            }
        ),
    )
    keys = np.arange(100)
    unique = _scan(tmp_path, "unique", pl.DataFrame({"k": keys, "name": keys % 7}))
    repeated = _scan(
        tmp_path, "repeated", pl.DataFrame({"k": np.repeat(keys, 3), "name": 0})
    )
    agg = pl.col("x").sum()
    assert not _gate_fires(
        unique.join(right, on="k").group_by("g").agg(agg), plmonkeypatch
    )
    # The join repeats R's rows.
    assert _gate_fires(
        repeated.join(right, on="k").group_by("g").agg(agg), plmonkeypatch
    )
    # A group key from L.
    assert _gate_fires(
        unique.join(right, on="k").group_by("g", "name").agg(agg), plmonkeypatch
    )


def test_eager_aggregation_gate_ignores_restrictions_an_outer_join_undoes(
    plmonkeypatch: PlMonkeyPatch, tmp_path: Path
) -> None:
    # Only the left input of the full join is restricted to year 0; the right one fills
    # in a year per row, so grouping by year leaves nothing to fold.
    n = 200_000
    rng = np.random.default_rng(0)
    left = _scan(tmp_path, "left", pl.DataFrame({"k": np.arange(1_000), "g": "a"}))
    fact = _scan(
        tmp_path,
        "fact",
        pl.DataFrame(
            {"k": rng.integers(0, 1_000, n), "year": np.arange(n), "x": np.ones(n)}
        ),
    )
    zero = _scan(tmp_path, "zero", pl.DataFrame({"year": np.arange(n)}))
    r = zero.filter(pl.col("year") == 0).join(
        fact, on="year", how="full", coalesce=True
    )
    lf = left.join(r, on="k").group_by("g", "year").agg(pl.col("x").sum())
    assert not _gate_fires(lf, plmonkeypatch, _KEEP_JOIN_ORDER)


I128_MIN, I128_MAX = -(2**127), 2**127 - 1


def test_eager_aggregation_gate_int128_range(
    plmonkeypatch: PlMonkeyPatch, tmp_path: Path
) -> None:
    # The full Int128 range is too wide to count, so it does not cap the unique keys.
    n = 500_000
    rng = np.random.default_rng(0)
    left = _scan(tmp_path, "left", pl.DataFrame({"k": np.arange(1_000), "g": "a"}))
    right = _scan(
        tmp_path,
        "right",
        pl.DataFrame(
            {
                "k": rng.integers(0, 1_000, n),
                "u": pl.Series(np.arange(n), dtype=pl.Int128),
                "x": np.ones(n),
            }
        ),
    )
    lo, hi = pl.lit(I128_MIN, dtype=pl.Int128), pl.lit(I128_MAX, dtype=pl.Int128)
    r = right.filter((pl.col("u") >= lo) & (pl.col("u") <= hi))
    lf = left.join(r, on="k").group_by("g", "u").agg(pl.col("x").sum())
    assert not _gate_fires(lf, plmonkeypatch)


@pytest.mark.parametrize(
    "restriction",
    [
        pl.col("u") > pl.lit(I128_MAX, dtype=pl.Int128),
        pl.col("u") < pl.lit(I128_MIN, dtype=pl.Int128),
        pl.col("u").is_between(
            pl.lit(I128_MIN, dtype=pl.Int128),
            pl.lit(I128_MAX, dtype=pl.Int128),
            closed="none",
        ),
    ],
)
def test_eager_aggregation_int128_boundary_restrictions(
    restriction: pl.Expr, plmonkeypatch: PlMonkeyPatch
) -> None:
    right = pl.LazyFrame(
        {"k": [1, 2], "u": pl.Series([0, 1], dtype=pl.Int128), "x": [1, 2]}
    ).filter(restriction)
    lf = _left().join(right, on="k").group_by("g", "u").agg(pl.col("x").sum())
    # Planning with the gate on reads the restriction.
    _plans(lf, plmonkeypatch, skip_gate=False)
    result, expected = _collect_both(lf, plmonkeypatch)
    assert_frame_equal(result.sort("g", "u"), expected.sort("g", "u"))


# Join ordering may move a join from above the target into its other input.
_KEEP_JOIN_ORDER = pl.QueryOptFlags(join_order=False)


def _stacked(**kwargs: Any) -> pl.LazyFrame:
    return _join(
        _join(_fact(), _cust(), "f_ck", "c_ck"), _nation(), "c_nk", "n_nk", **kwargs
    )


def test_eager_aggregation_stacked_joins(plmonkeypatch: PlMonkeyPatch) -> None:
    # A group key from the join above, as in TPC-H Q10.
    lf = _stacked().group_by("c_ck", "c_name", "n_name").agg(_revenue())
    _assert_rewrite(lf, plmonkeypatch, fires=True, sort_by="c_ck")
    on, _ = _plans(lf, plmonkeypatch)
    # The partial aggregation goes under the customer join, not the nation join.
    partial = next(line for line in on.splitlines() if 'BY [col("f_ck")]' in line)
    assert "sum()" in partial

    # Three levels: a region above the nation.
    region = pl.LazyFrame({"r_nk": [0, 1], "r_name": ["p", "q"]})
    lf = (
        _join(_stacked(), region, "n_nk", "r_nk")
        .group_by("r_name", "c_name")
        .agg(_revenue())
    )
    _assert_rewrite(lf, plmonkeypatch, fires=True, sort_by="c_name")

    # The same in SQL.
    ctx = pl.SQLContext(fact=_fact(), cust=_cust(), nation=_nation())
    lf = ctx.execute(
        """
        SELECT c_ck, c_name, n_name,
               SUM(f_price * (1 - f_disc)) AS revenue, COUNT(*) AS n
        FROM fact, cust, nation
        WHERE f_ck = c_ck AND c_nk = n_nk
        GROUP BY c_ck, c_name, n_name
        """
    )
    _assert_rewrite(lf, plmonkeypatch, fires=True, sort_by="c_ck")


def test_eager_aggregation_stacked_joins_rejected(
    plmonkeypatch: PlMonkeyPatch,
) -> None:
    group = ["c_ck", "c_name", "n_name"]
    agg = pl.col("f_qty").sum()

    # An upper left join.
    lf = _stacked(how="left").group_by(group).agg(agg)
    _assert_rewrite(lf, plmonkeypatch, fires=False, sort_by="c_ck")

    # An upper join with an extra condition.
    lf = (
        _join(_fact(), _cust(), "f_ck", "c_ck")
        .join_where(
            _nation(),
            pl.col("c_nk") == pl.col("n_nk"),
            pl.col("c_name") != pl.col("n_name"),
        )
        .group_by(group)
        .agg(agg)
    )
    _assert_rewrite(
        lf, plmonkeypatch, fires=False, sort_by="c_ck", optimizations=_KEEP_JOIN_ORDER
    )

    # Upper joins keyed on the lower join's key, through either of its names: the lower
    # join cannot take the partial aggregation, so it goes under the upper join, grouped
    # by that key and `c_name`.
    for key in ["c_ck", "f_ck"]:
        above = pl.LazyFrame({"a_ck": [1, 2, 3], "a_name": ["i", "j", "k"]})
        lf = (
            _join(_join(_fact(), _cust(), "f_ck", "c_ck"), above, key, "a_ck")
            .group_by("c_name", "a_name")
            .agg(agg)
        )
        _assert_rewrite(
            lf,
            plmonkeypatch,
            fires=True,
            sort_by=["c_name", "a_name"],
            optimizations=_KEEP_JOIN_ORDER,
        )
        on, _ = _plans(lf, plmonkeypatch, optimizations=_KEEP_JOIN_ORDER)
        assert f'BY [col("{key}"), col("c_name")]' in on, on


@pytest.mark.parametrize(
    "kwargs",
    [
        {"validate": "1:1"},
        {"validate": "1:m"},
        {"validate": "m:1"},
        {"maintain_order": "left"},
    ],
)
def test_eager_aggregation_constrained_upper_join(
    kwargs: dict[str, Any], plmonkeypatch: PlMonkeyPatch
) -> None:
    # Duplicate nation keys make `validate` fail, in the original and after the rewrite.
    nation = pl.LazyFrame({"n_nk": [0, 0, 1], "n_name": ["x", "x2", "y"]})
    lf = (
        _join(_join(_fact(), _cust(), "f_ck", "c_ck"), nation, "c_nk", "n_nk", **kwargs)
        .group_by("c_ck", "n_name")
        .agg(pl.col("f_qty").sum())
    )
    on, off = _plans(lf, plmonkeypatch)
    assert not _fired(on, off), on
    if "validate" in kwargs and kwargs["validate"] != "1:m":
        plmonkeypatch.setenv("POLARS_EAGER_AGGREGATION", "1")
        with pytest.raises(pl.exceptions.ComputeError):
            lf.collect(engine="streaming")


def test_eager_aggregation_sliced_upper_join(plmonkeypatch: PlMonkeyPatch) -> None:
    # Which rows the slice keeps is not fixed, so only the plan is compared.
    lf = _stacked().head(3).group_by("c_ck", "n_name").agg(pl.col("f_qty").sum())
    on, off = _plans(lf, plmonkeypatch)
    assert not _fired(on, off), on


def test_eager_aggregation_gate_above_the_target(
    plmonkeypatch: PlMonkeyPatch, tmp_path: Path
) -> None:
    rng = np.random.default_rng(1)
    n_cust = 100_000
    fact = _scan(
        tmp_path,
        "fact",
        pl.DataFrame(
            {
                "f_ck": rng.integers(0, n_cust, 1_000_000),
                "f_qty": rng.integers(0, 100, 1_000_000),
            }
        ),
    )

    def cust(nation_keys: Any) -> pl.LazyFrame:
        df = pl.DataFrame({"c_ck": np.arange(n_cust), "c_nk": nation_keys})
        return _scan(tmp_path, f"cust_{rng.integers(1 << 30)}", df)

    def query(cust: pl.LazyFrame, nation: pl.LazyFrame) -> pl.LazyFrame:
        return (
            _join(_join(fact, cust, "f_ck", "c_ck"), nation, "c_nk", "n_nk")
            .group_by("c_ck", "n_name")
            .agg(pl.col("f_qty").sum())
        )

    def nation(keys: Any) -> pl.LazyFrame:
        df = pl.DataFrame({"n_nk": keys, "n_name": [str(k) for k in keys]})
        return _scan(tmp_path, f"nation_{rng.integers(1 << 30)}", df)

    all_nations = cust(np.arange(n_cust) % 25)
    # The nation join keeps every customer.
    assert _gate_fires(
        query(all_nations, nation(np.arange(25))), plmonkeypatch, _KEEP_JOIN_ORDER
    )
    # It keeps one nation of 25.
    one_nation = nation(np.arange(25)).filter(pl.col("n_name") == "3")
    assert not _gate_fires(
        query(all_nations, one_nation), plmonkeypatch, _KEEP_JOIN_ORDER
    )
    # One matching customer, repeated many times above: the join above is as large as
    # its input but keeps 1 % of the customers.
    dup_nation = nation(np.zeros(100, dtype=np.int64))
    one_in_100 = cust(np.where(np.arange(n_cust) % 100 == 0, 0, 1_000))
    assert not _gate_fires(
        query(one_in_100, dup_nation), plmonkeypatch, _KEEP_JOIN_ORDER
    )
    # Most customers have no nation key.
    mostly_null = cust(
        pl.Series(np.arange(n_cust) % 25).scatter(range(5_000, n_cust), None)
    )
    assert not _gate_fires(
        query(mostly_null, nation(np.arange(25))), plmonkeypatch, _KEEP_JOIN_ORDER
    )
