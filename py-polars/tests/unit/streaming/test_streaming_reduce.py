"""Short-circuiting reductions in the streaming engine (#27586)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import polars as pl
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from tests.conftest import PlMonkeyPatch

pytestmark = pytest.mark.xdist_group("streaming")


@pytest.mark.parametrize(
    "expr",
    [
        # `any` locks to True on the first True; the all-False case never locks.
        pl.col("a").any(),
        # `all` locks to False on the first False; the all-True case never locks.
        pl.col("a").all(),
        # Kleene variants (ignore_nulls=False) with nulls present.
        pl.col("a").any(ignore_nulls=False),
        pl.col("a").all(ignore_nulls=False),
    ],
)
@pytest.mark.parametrize(
    "values",
    [
        [True] + [False] * 999,  # locks early
        [False] + [True] * 999,  # locks early
        [True] * 1000,  # never locks (all-True)
        [False] * 1000,  # never locks (all-False)
        [True, None] + [False] * 998,  # nulls present
        [None] * 1000,  # all-null
    ],
)
def test_streaming_bool_reduce_short_circuit_correctness(
    plmonkeypatch: PlMonkeyPatch,
    expr: pl.Expr,
    values: list[bool | None],
) -> None:
    # Morsel size 1 forces many morsels, exercising the short-circuit path.
    plmonkeypatch.setenv("POLARS_IDEAL_MORSEL_SIZE", "1")
    lf = pl.LazyFrame({"a": values}, schema={"a": pl.Boolean}).select(expr)
    assert_frame_equal(
        lf.collect(engine="streaming"),
        lf.collect(engine="in-memory"),
    )


@pytest.mark.slow
def test_streaming_reduce_any_short_circuits() -> None:
    # `any` over all-False never locks, so every row flows through the upstream
    # map_batches; over all-True it locks and the reduce stops the source early,
    # reading strictly fewer rows. Size n above the in-flight overshoot (roughly
    # one morsel per pipeline) so the margin holds regardless of CI core count.
    n = min(max(20_000_000, pl.thread_pool_size() * 2_000_000), 50_000_000)

    def rows_read(*, all_true: bool) -> tuple[bool, int]:
        seen = {"rows": 0}

        def count(df: pl.DataFrame) -> pl.DataFrame:
            seen["rows"] += df.height
            return df

        col = pl.repeat(all_true, n, dtype=pl.Boolean, eager=True)
        result = (
            pl.LazyFrame({"a": col})
            .map_batches(count, schema={"a": pl.Boolean}, streamable=True)
            .select(pl.col("a").any())
            .collect(engine="streaming")
        )
        return result.item(), seen["rows"]

    res_full, rows_full = rows_read(all_true=False)
    assert res_full is False
    assert rows_full == n  # never locks -> reads every row

    res_short, rows_short = rows_read(all_true=True)
    assert res_short is True
    assert rows_short < rows_full  # locks early -> reads strictly fewer
