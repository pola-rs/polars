from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from pathlib import Path

    from tests.conftest import PlMonkeyPatch


def test_ooc_spill(tmp_path: Path, plmonkeypatch: PlMonkeyPatch, capfd: Any) -> None:
    plmonkeypatch.setenv("POLARS_OOC_SPILL_POLICY", "spill")
    # Tiny budget so every store() exceeds it and triggers spilling.
    plmonkeypatch.setenv("POLARS_OOC_MEMORY_BUDGET_FRACTION", "0.000000001")
    plmonkeypatch.setenv("POLARS_OOC_SPILL_DIR", str(tmp_path))
    plmonkeypatch.setenv("POLARS_OOC_DRIFT_THRESHOLD", "1")
    plmonkeypatch.setenv("POLARS_OOC_SPILL_MIN_BYTES", "1")
    plmonkeypatch.setenv("POLARS_VERBOSE", "1")

    capfd.readouterr()

    lf = pl.LazyFrame({"a": [1, 2, 1, 2], "b": [10, 20, 30, 40]})

    q = (
        lf.group_by("a")
        .agg(pl.col("b").sum().alias("b_sum"))
        .sort("a")
        .join(lf.group_by("a").agg(pl.col("b").mean().alias("b_mean")), on="a")
        .sort("a")
    )

    result = q.collect(engine="streaming")
    expected = q.collect(engine="in-memory")
    assert_frame_equal(result, expected)

    captured = capfd.readouterr().err
    assert "[ooc] spill_trigger" in captured
    assert "[ooc] spill" in captured
