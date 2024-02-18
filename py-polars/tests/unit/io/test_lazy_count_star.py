from __future__ import annotations

from pathlib import Path

import pytest

import polars as pl
from polars.testing import assert_frame_equal

def test_count_csv(io_files_path: Path) -> None:
    lf = pl.scan_csv(io_files_path / "foods1.csv").select(pl.len())

    expected = pl.DataFrame(pl.Series("len", [27], dtype=pl.UInt32))

    # Check if we are using our fast count star
    assert "FAST COUNT(*)" in lf.explain()
    assert_frame_equal(lf.collect(), expected)


@pytest.mark.parametrize(("pattern","n_rows"),[("small.parquet",4),("foods*.parquet",54)])
def test_count_parquet(io_files_path: Path,pattern: str, n_rows: int) -> None:
    lf = pl.scan_parquet(io_files_path / pattern).select(pl.len())

    expected = pl.DataFrame(pl.Series("len", [n_rows], dtype=pl.UInt32))

    # Check if we are using our fast count star
    assert "FAST COUNT(*)" in lf.explain()
    assert_frame_equal(lf.collect(), expected)


# IPC globbing returns a UNION of plans which is not yet supported
def test_count_ipc(io_files_path: Path) -> None:
    lf = pl.scan_ipc(io_files_path / "foods2.ipc").select(pl.len())

    expected = pl.DataFrame(pl.Series("len", [27], dtype=pl.UInt32))

    # Check if we are using our fast count star
    assert "FAST COUNT(*)" in lf.explain()
    assert_frame_equal(lf.collect(), expected)
