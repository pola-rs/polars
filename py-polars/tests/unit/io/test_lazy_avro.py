from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import polars as pl
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def foods_avro_path(io_files_path: Path) -> Path:
    return io_files_path / "foods1.avro"


def test_scan_avro(foods_avro_path: Path) -> None:
    df = pl.scan_avro(foods_avro_path, row_index_name="row_index").collect()
    assert df["row_index"].to_list() == list(range(27))

    df = (
        pl.scan_avro(foods_avro_path, row_index_name="row_index")
        .filter(pl.col("category") == pl.lit("vegetables"))
        .collect()
    )

    assert df["row_index"].to_list() == [0, 6, 11, 13, 14, 20, 25]

    df = (
        pl.scan_avro(foods_avro_path, row_index_name="row_index")
        .with_row_index("foo", 10)
        .filter(pl.col("category") == pl.lit("vegetables"))
        .collect()
    )

    assert df["foo"].to_list() == [10, 16, 21, 23, 24, 30, 35]


def test_projection_pushdown_avro(io_files_path: Path) -> None:
    file_path = io_files_path / "foods1.avro"
    df = pl.scan_avro(file_path).select(pl.col.calories)

    explain = df.explain()

    assert "simple π" not in explain
    assert "PROJECT 1/4 COLUMNS" in explain

    assert_frame_equal(df.collect(no_optimization=True), df.collect())


def test_predicate_pushdown_avro(io_files_path: Path) -> None:
    file_path = io_files_path / "foods1.avro"
    df = pl.scan_avro(file_path).filter(pl.col.calories > 80)

    explain = df.explain()

    assert "FILTER" not in explain
    assert """SELECTION: [(col("calories")) > (80)]""" in explain

    assert_frame_equal(df.collect(no_optimization=True), df.collect())


def test_glob_n_rows(io_files_path: Path) -> None:
    file_path = io_files_path / "foods*.avro"
    df = pl.scan_avro(file_path, n_rows=40).collect()

    # 27 rows from foods1.avro and 13 from foods2.avro
    assert df.shape == (40, 4)

    # take first and last rows
    assert df[[0, 39]].to_dict(as_series=False) == {
        "category": ["vegetables", "seafood"],
        "calories": [45, 146],
        "fats_g": [0.5, 6.0],
        "sugars_g": [2, 2],
    }


def test_avro_list_arg(io_files_path: Path) -> None:
    first = io_files_path / "foods1.avro"
    second = io_files_path / "foods2.avro"

    df = pl.scan_avro(source=[first, second]).collect()
    assert df.shape == (54, 4)
    assert df.row(-1) == ("seafood", 194, 12.0, 1)
    assert df.row(0) == ("vegetables", 45, 0.5, 2)


def test_glob_single_scan(io_files_path: Path) -> None:
    file_path = io_files_path / "foods*.avro"
    df = pl.scan_avro(file_path, n_rows=40)

    explain = df.explain()

    assert explain.count("SCAN") == 1
    assert "UNION" not in explain
