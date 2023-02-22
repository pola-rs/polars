from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

import polars as pl
from polars.testing import assert_frame_equal


@pytest.fixture()
def foods_file_path(io_files_path: Path) -> Path:
    return io_files_path / "foods1.csv"


def test_scan_csv(io_files_path: Path) -> None:
    df = pl.scan_csv(io_files_path / "small.csv")
    assert df.collect().shape == (4, 3)


def test_scan_empty_csv(io_files_path: Path) -> None:
    with pytest.raises(Exception) as excinfo:
        pl.scan_csv(io_files_path / "empty.csv").collect()
    assert "empty csv" in str(excinfo.value)


def test_invalid_utf8() -> None:
    np.random.seed(1)
    bts = bytes(np.random.randint(0, 255, 200))

    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "nonutf8.csv"
        with open(file_path, "wb") as f:
            f.write(bts)

        a = pl.read_csv(file_path, has_header=False, encoding="utf8-lossy")
        b = pl.scan_csv(file_path, has_header=False, encoding="utf8-lossy").collect()

    assert_frame_equal(a, b)


def test_row_count(foods_file_path: Path) -> None:
    df = pl.read_csv(foods_file_path, row_count_name="row_count")
    assert df["row_count"].to_list() == list(range(27))

    df = (
        pl.scan_csv(foods_file_path, row_count_name="row_count")
        .filter(pl.col("category") == pl.lit("vegetables"))
        .collect()
    )

    assert df["row_count"].to_list() == [0, 6, 11, 13, 14, 20, 25]

    df = (
        pl.scan_csv(foods_file_path, row_count_name="row_count")
        .with_row_count("foo", 10)
        .filter(pl.col("category") == pl.lit("vegetables"))
        .collect()
    )

    assert df["foo"].to_list() == [10, 16, 21, 23, 24, 30, 35]


@pytest.mark.parametrize("file_name", ["foods1.csv", "foods*.csv"])
def test_scan_csv_schema_overwrite_and_dtypes_overwrite(
    io_files_path: Path, file_name: str
) -> None:
    file_path = io_files_path / file_name
    df = pl.scan_csv(
        file_path,
        dtypes={"calories_foo": pl.Utf8, "fats_g_foo": pl.Float32},
        with_column_names=lambda names: [f"{a}_foo" for a in names],
    ).collect()
    assert df.dtypes == [pl.Utf8, pl.Utf8, pl.Float32, pl.Int64]
    assert df.columns == [
        "category_foo",
        "calories_foo",
        "fats_g_foo",
        "sugars_g_foo",
    ]


def test_lazy_n_rows(foods_file_path: Path) -> None:
    df = (
        pl.scan_csv(foods_file_path, n_rows=4, row_count_name="idx")
        .filter(pl.col("idx") > 2)
        .collect()
    )
    assert df.to_dict(False) == {
        "idx": [3],
        "category": ["fruit"],
        "calories": [60],
        "fats_g": [0.0],
        "sugars_g": [11],
    }


def test_scan_slice_streaming(foods_file_path: Path) -> None:
    df = pl.scan_csv(foods_file_path).head(5).collect(streaming=True)
    assert df.shape == (5, 4)


def test_glob_skip_rows() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        for i in range(2):
            file_path = Path(temp_dir) / f"test_{i}.csv"
            with open(file_path, "w") as f:
                f.write(
                    f"""
metadata goes here
file number {i}
foo,bar,baz
1,2,3
4,5,6
7,8,9
        """
                )
        file_path = Path(temp_dir) / "*.csv"
        assert pl.read_csv(file_path, skip_rows=2).to_dict(False) == {
            "foo": [1, 4, 7, 1, 4, 7],
            "bar": [2, 5, 8, 2, 5, 8],
            "baz": [3, 6, 9, 3, 6, 9],
        }
