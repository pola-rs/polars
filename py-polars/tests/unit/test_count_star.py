from __future__ import annotations

from pathlib import Path

import pytest

import polars as pl
from polars.testing import assert_frame_equal


@pytest.fixture()
def io_files_path() -> Path:
    return Path(__file__).parent / "io/files"


def test_count_csv(io_files_path: Path) -> None:
    lf = pl.scan_csv(io_files_path / "foods1.csv").select(pl.len())

    expected = pl.DataFrame(pl.Series("len", [27], dtype=pl.UInt32))

    # Check if we are using our fast count star
    assert "FAST COUNT(*)" in lf.explain()
    assert_frame_equal(lf.collect(), expected)


def test_count_parquet(io_files_path: Path) -> None:
    lf = pl.scan_parquet(io_files_path / "small.parquet").select(pl.len())

    expected = pl.DataFrame(pl.Series("len", [4], dtype=pl.UInt32))

    # Check if we are using our fast count star
    assert "FAST COUNT(*)" in lf.explain()
    assert_frame_equal(lf.collect(), expected)
