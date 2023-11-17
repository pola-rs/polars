from __future__ import annotations

from typing import Any

import pytest

import polars as pl
from polars.testing import assert_frame_equal

pytestmark = pytest.mark.xdist_group("streaming")


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

    result = q.select(exprs).collect(comm_subexpr_elim=True, streaming=True)
    expected = pl.DataFrame(
        {"d1": [1, 4, 9, 16], "d2": [1, 16, 81, 256], "d3": [10, 160, 810, 2560]}
    )
    assert_frame_equal(result, expected)

    result = q.with_columns(exprs).collect(comm_subexpr_elim=True, streaming=True)
    expected = pl.DataFrame(
        {
            "a": [1, 2, 3, 4],
            "b": [1, 2, 3, 4],
            "c": [1, 2, 3, 4],
            "d1": [1, 4, 9, 16],
            "d2": [1, 16, 81, 256],
            "d3": [10, 160, 810, 2560],
        }
    )
    assert_frame_equal(result, expected)

    err = capfd.readouterr().err
    assert "df -> projection[cse] -> ordered_sink" in err
    assert "df -> hstack[cse] -> ordered_sink" in err


@pytest.mark.skip(reason="activate once fixed")
def test_cse_expr_group_by() -> None:
    q = pl.LazyFrame(
        {
            "a": [1, 2, 3, 4],
            "b": [1, 2, 3, 4],
            "c": [1, 2, 3, 4],
        }
    )

    derived = pl.col("a") * pl.col("b")

    q = (
        q.group_by("a")
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
    assert s.startswith("--- STREAMING")

    expected = pl.DataFrame(
        {"a": [1, 2, 3, 4], "sum": [1, 4, 9, 16], "min": [1, 4, 9, 16]}
    )
    for streaming in [True, False]:
        out = q.collect(comm_subexpr_elim=True, streaming=streaming)
        assert_frame_equal(out, expected)
