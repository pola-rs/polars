from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import polars as pl
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from pathlib import Path

    from tests.conftest import PlMonkeyPatch


def _force_spill(tmp_path: Path, plmonkeypatch: PlMonkeyPatch) -> None:
    plmonkeypatch.setenv("POLARS_OOC_SPILL_POLICY", "spill")
    # Tiny budget so every store() exceeds it and triggers spilling.
    plmonkeypatch.setenv("POLARS_OOC_MEMORY_BUDGET_FRACTION", "0.000000001")
    plmonkeypatch.setenv("POLARS_OOC_SPILL_DIR", str(tmp_path))
    plmonkeypatch.setenv("POLARS_OOC_DRIFT_THRESHOLD", "1")
    plmonkeypatch.setenv("POLARS_OOC_SPILL_MIN_BYTES", "1")
    plmonkeypatch.setenv("POLARS_VERBOSE", "1")


def _assert_spill_reload_clean(captured: str, label: str = "") -> None:
    suffix = f" ({label})" if label else ""
    assert "[ooc] spill_trigger" in captured, f"no spill_trigger{suffix}"
    assert "[ooc] spill" in captured, f"no spill{suffix}"
    assert "[ooc] reload" in captured, f"no reload{suffix}"
    assert "[ooc] clean" in captured, f"no clean{suffix}"


@pytest.mark.skip
def test_ooc_spill(tmp_path: Path, plmonkeypatch: PlMonkeyPatch, capfd: Any) -> None:
    _force_spill(tmp_path, plmonkeypatch)

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

    _assert_spill_reload_clean(capfd.readouterr().err)


@pytest.mark.skip
def test_ooc_spill_multiple_queries(
    tmp_path: Path, plmonkeypatch: PlMonkeyPatch, capfd: Any
) -> None:
    """Verify spill lifecycle across three streaming queries."""
    _force_spill(tmp_path, plmonkeypatch)

    lf = pl.LazyFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
    q = lf.join(lf, on="a").sort("a")
    expected = q.collect(engine="in-memory")

    for i in range(3):
        capfd.readouterr()
        assert_frame_equal(q.collect(engine="streaming"), expected)
        _assert_spill_reload_clean(capfd.readouterr().err, f"query {i}")
