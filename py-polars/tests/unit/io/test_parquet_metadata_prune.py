from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

import polars as pl
from polars._plr import _bench_parquet_metadata_pruned_json

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.write_disk
def test_pruned_metadata(tmp_path: Path) -> None:
    path = tmp_path / "t.parquet"
    pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}).write_parquet(path)

    # Project [a, b], predicate [a]: only a, b survive; stats only on a.
    meta = json.loads(_bench_parquet_metadata_pruned_json(str(path), ["a", "b"], ["a"]))
    assert [f["name"] for f in meta["schema_descr"]["fields"]] == ["a", "b"]
    cols = meta["row_groups"][0]["columns"]
    assert cols[0]["statistics"] is not None
    assert cols[1]["statistics"] is None

    # Empty predicate drops all stats.
    meta = json.loads(
        _bench_parquet_metadata_pruned_json(str(path), ["a", "b", "c"], [])
    )
    assert all(c["statistics"] is None for c in meta["row_groups"][0]["columns"])
