from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import polars as pl
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from polars._typing import EngineType


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
def test_unique_id(engine: EngineType) -> None:
    lf = pl.LazyFrame({"x": ["A", "B", "A", "C", "C", "B"]})
    ids = lf.select(pl.col("x").unique_id()).collect(engine=engine)["x"]

    assert ids.dtype == pl.UInt32
    # Dense 0..n_unique-1, equal values share an ID, distinct values differ.
    assert set(ids.to_list()) == {0, 1, 2}
    assert ids[0] == ids[2]  # A
    assert ids[1] == ids[5]  # B
    assert ids[3] == ids[4]  # C


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
def test_unique_id_maintain_order(engine: EngineType) -> None:
    lf = pl.LazyFrame({"x": ["A", "B", "A", "C", "C", "B"]})
    ids = lf.select(pl.col("x").unique_id(maintain_order=True)).collect(engine=engine)
    assert ids["x"].to_list() == [0, 1, 0, 2, 2, 1]


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
def test_unique_id_nulls(engine: EngineType) -> None:
    lf = pl.LazyFrame({"x": ["A", None, "A", None, "B"]})
    ids = lf.select(pl.col("x").unique_id()).collect(engine=engine)["x"]

    assert ids.n_unique() == 3  # "A", null, "B"
    assert ids[1] == ids[3]  # nulls share an ID


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
def test_unique_id_empty(engine: EngineType) -> None:
    lf = pl.LazyFrame({"x": pl.Series([], dtype=pl.Int64)})
    result = lf.select(pl.col("x").unique_id()).collect(engine=engine)
    assert_frame_equal(result, pl.DataFrame({"x": pl.Series([], dtype=pl.UInt32)}))


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
@pytest.mark.parametrize("maintain_order", [False, True])
def test_unique_id_many_distinct_are_unique(
    engine: EngineType, maintain_order: bool
) -> None:
    n = 100_000
    lf = pl.LazyFrame({"x": range(n)})
    ids = lf.select(pl.col("x").unique_id(maintain_order=maintain_order)).collect(
        engine=engine
    )["x"]
    assert ids.n_unique() == n


def test_unique_id_over() -> None:
    df = pl.DataFrame({"group": ["a", "a", "b", "b"], "x": [1, 2, 1, 2]})
    result = df.group_by("group", maintain_order=True).agg(pl.col("x").unique_id())
    for ids in result["x"].to_list():
        assert len(set(ids)) == 2
