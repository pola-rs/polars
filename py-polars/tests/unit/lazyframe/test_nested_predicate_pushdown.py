from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import polars as pl
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from pathlib import Path

    from polars._typing import EngineType


@pytest.fixture
def nested_scan(tmp_path: Path) -> pl.LazyFrame:
    path = tmp_path / "nested.parquet"
    pl.DataFrame(
        {
            "s": [
                {"a": {"b": 1}, "h": 0},
                None,
                {"a": None, "h": 4},
                {"a": {"b": 5}, "h": 6},
            ],
            "x": [9, 8, 7, 6],
            "g": [0, 0, 0, 0],
        },
        schema={
            "s": pl.Struct({"a": pl.Struct({"b": pl.Int64}), "h": pl.Int64}),
            "x": pl.Int64,
            "g": pl.Int64,
        },
    ).write_parquet(path)
    return pl.scan_parquet(path)


def assert_equivalent(q: pl.LazyFrame, engine: EngineType) -> None:
    assert_frame_equal(
        q.collect(engine=engine),
        q.collect(
            engine=engine,
            optimizations=pl.QueryOptFlags(
                predicate_pushdown=False, projection_pushdown=False
            ),
        ),
    )


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("shape", range(10))
def test_nested_output_predicate_reaches_scan(
    nested_scan: pl.LazyFrame, engine: EngineType, empty: bool, shape: int
) -> None:
    lf = nested_scan.filter(pl.col("x") < 0) if empty else nested_scan
    s = pl.col("s")
    leaf = s.struct.field("a").struct.field("b")
    queries = [
        lf.unnest("s").filter(pl.col("h") > 3),
        lf.unnest("s").unnest("a").filter(pl.col("b") > 3),
        lf.select(b=leaf, y="x").filter(pl.col("b") > 3),
        lf.with_columns(b=leaf).filter(pl.col("b") > 3),
        lf.select(s.struct.unnest()).filter(pl.col("h") > 3),
        lf.with_columns(t=pl.struct(k="x")).filter(
            pl.col("t").struct.field("k") > 6
        ),
        lf.select(s.struct.field("a", "h")).filter(pl.col("h") > 3),
        lf.unnest("s", separator="_").unnest("s_a", separator="_").filter(
            pl.col("s_a_b") > 3
        ),
        lf.with_columns(s=s.struct.field("a")).filter(
            pl.col("s").struct.field("b") > 3
        ),
        # Both replacements must use the original input names.
        lf.select(x=leaf, s="x").filter((pl.col("x") > 3) & (pl.col("s") > 5)),
    ]
    q = queries[shape]
    plan = q.explain()
    assert "FILTER" not in plan, plan
    assert "SELECTION" in plan, plan
    assert_equivalent(q, engine)


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
def test_nested_substitutions_preserve_window_partition(
    nested_scan: pl.LazyFrame, engine: EngineType
) -> None:
    q = nested_scan.with_columns(
        total=pl.col("x").sum().over("g"),
        t=pl.struct(k="x"),
        b=pl.col("s").struct.field("a").struct.field("b"),
    ).filter((pl.col("t").struct.field("k") > 6) & (pl.col("b") > 0))
    assert "FILTER" in q.explain()
    assert_equivalent(q, engine)


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
@pytest.mark.parametrize("empty", [False, True])
def test_struct_scalar_selected_child_preserves_height(
    nested_scan: pl.LazyFrame, engine: EngineType, empty: bool
) -> None:
    lf = nested_scan.filter(pl.col("x") < 0) if empty else nested_scan
    q = lf.select(t=pl.struct(k=pl.lit(1), v="x")).filter(
        pl.col("t").struct.field("k") == 1
    )
    assert_equivalent(q, engine)
    assert q.collect(engine=engine).height == (0 if empty else 4)


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
def test_nested_substitution_preserves_non_elementwise_barrier(
    nested_scan: pl.LazyFrame, engine: EngineType
) -> None:
    q = nested_scan.select(
        b=pl.col("s").struct.field("h"), reordered=pl.col("x").sort()
    ).filter(pl.col("b") > 3)
    assert "FILTER" in q.explain()
    assert_equivalent(q, engine)


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
def test_field_derived_join_key_keeps_runtime_filter_operand_column(
    tmp_path: Path, engine: EngineType
) -> None:
    path = tmp_path / "join.parquet"
    pl.DataFrame({"a": range(1000)}).select(s=pl.struct("a")).write_parquet(
        path, row_group_size=100
    )
    probe = pl.scan_parquet(path).select(k=pl.col("s").struct.field("a"))
    build = pl.LazyFrame({"k": [220, 240], "v": [1, 2]}).filter(pl.col("v") > 0)
    q = probe.join(build, on="k").sort("k")
    # Optimizing runtime filters must not reinterpret a field expression as Column.
    q.explain()
    assert_equivalent(q, engine)
    assert q.collect(engine=engine)["k"].to_list() == [220, 240]
