from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from tempfile import NamedTemporaryFile

import pytest

import polars as pl
from polars.testing import assert_frame_equal


@pytest.mark.parametrize(
    ("path", "n_rows"), [("foods1.csv", 27), ("foods*.csv", 27 * 5)]
)
def test_count_csv(io_files_path: Path, path: str, n_rows: int) -> None:
    lf = pl.scan_csv(io_files_path / path).select(pl.len())

    expected = pl.DataFrame(pl.Series("len", [n_rows], dtype=pl.UInt32))

    # Check if we are using our fast count star
    assert "FAST COUNT(*)" in lf.explain()
    assert_frame_equal(lf.collect(), expected)


@pytest.mark.write_disk()
def test_commented_csv() -> None:
    csv_a = NamedTemporaryFile()
    csv_a.write(
        b"""
A,B
Gr1,A
Gr1,B
# comment line
    """.strip()
    )
    csv_a.seek(0)

    expected = pl.DataFrame(pl.Series("len", [2], dtype=pl.UInt32))
    lf = pl.scan_csv(csv_a.name, comment_prefix="#").select(pl.len())
    assert "FAST COUNT(*)" in lf.explain()
    assert_frame_equal(lf.collect(), expected)


@pytest.mark.parametrize(
    ("pattern", "n_rows"), [("small.parquet", 4), ("foods*.parquet", 54)]
)
def test_count_parquet(io_files_path: Path, pattern: str, n_rows: int) -> None:
    lf = pl.scan_parquet(io_files_path / pattern).select(pl.len())

    expected = pl.DataFrame(pl.Series("len", [n_rows], dtype=pl.UInt32))

    # Check if we are using our fast count star
    assert "FAST COUNT(*)" in lf.explain()
    assert_frame_equal(lf.collect(), expected)


@pytest.mark.parametrize(
    ("path", "n_rows"), [("foods1.ipc", 27), ("foods*.ipc", 27 * 2)]
)
def test_count_ipc(io_files_path: Path, path: str, n_rows: int) -> None:
    lf = pl.scan_ipc(io_files_path / path).select(pl.len())

    expected = pl.DataFrame(pl.Series("len", [n_rows], dtype=pl.UInt32))

    # Check if we are using our fast count star
    assert "FAST COUNT(*)" in lf.explain()
    assert_frame_equal(lf.collect(), expected)
