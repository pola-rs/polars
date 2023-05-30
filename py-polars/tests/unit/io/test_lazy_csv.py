from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import polars as pl
from polars.exceptions import PolarsPanicError
from polars.testing import assert_frame_equal
from polars.testing._tempdir import TemporaryDirectory


@pytest.fixture()
def foods_file_path(io_files_path: Path) -> Path:
    return io_files_path / "foods1.csv"


def test_scan_csv(io_files_path: Path) -> None:
    df = pl.scan_csv(io_files_path / "small.csv")
    assert df.collect().shape == (4, 3)


def test_scan_csv_no_cse_deadlock(io_files_path: Path) -> None:
    dfs = [pl.scan_csv(io_files_path / "small.csv")] * (pl.threadpool_size() + 1)
    pl.concat(dfs, parallel=True).collect(common_subplan_elimination=False)


def test_scan_empty_csv(io_files_path: Path) -> None:
    with pytest.raises(Exception) as excinfo:
        pl.scan_csv(io_files_path / "empty.csv").collect()
    assert "empty csv" in str(excinfo.value)


@pytest.mark.write_disk()
def test_invalid_utf8() -> None:
    np.random.seed(1)
    bts = bytes(np.random.randint(0, 255, 200))

    with TemporaryDirectory() as temp_dir:
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


@pytest.mark.parametrize("file_name", ["foods1.csv", "foods*.csv"])
@pytest.mark.parametrize("dtype", [pl.Int8, pl.UInt8, pl.Int16, pl.UInt16])
def test_scan_csv_schema_overwrite_and_small_dtypes_overwrite(
    io_files_path: Path, file_name: str, dtype: pl.DataType
) -> None:
    file_path = io_files_path / file_name
    df = pl.scan_csv(
        file_path,
        dtypes={"calories_foo": pl.Utf8, "sugars_g_foo": dtype},
        with_column_names=lambda names: [f"{a}_foo" for a in names],
    ).collect()
    assert df.dtypes == [pl.Utf8, pl.Utf8, pl.Float64, dtype]
    assert df.columns == [
        "category_foo",
        "calories_foo",
        "fats_g_foo",
        "sugars_g_foo",
    ]


@pytest.mark.parametrize("file_name", ["foods1.csv", "foods*.csv"])
def test_scan_csv_schema_new_columns_dtypes(
    io_files_path: Path, file_name: str
) -> None:
    file_path = io_files_path / file_name

    for dtype in [pl.Int8, pl.UInt8, pl.Int16, pl.UInt16]:
        # assign 'new_columns', providing partial dtype overrides
        df1 = pl.scan_csv(
            file_path,
            dtypes={"calories": pl.Utf8, "sugars": dtype},
            new_columns=["category", "calories", "fats", "sugars"],
        ).collect()
        assert df1.dtypes == [pl.Utf8, pl.Utf8, pl.Float64, dtype]
        assert df1.columns == ["category", "calories", "fats", "sugars"]

        # assign 'new_columns' with 'dtypes' list
        df2 = pl.scan_csv(
            file_path,
            dtypes=[pl.Utf8, pl.Utf8, pl.Float64, dtype],
            new_columns=["category", "calories", "fats", "sugars"],
        ).collect()
        assert df1.rows() == df2.rows()

    # rename existing columns, then lazy-select disjoint cols
    df3 = pl.scan_csv(
        file_path,
        new_columns=["colw", "colx", "coly", "colz"],
    )
    assert df3.dtypes == [pl.Utf8, pl.Int64, pl.Float64, pl.Int64]
    assert df3.columns == ["colw", "colx", "coly", "colz"]
    assert (
        df3.select(["colz", "colx"]).collect().rows()
        == df1.select(["sugars", pl.col("calories").cast(pl.Int64)]).rows()
    )

    # expect same number of column names as there are columns in the file
    with pytest.raises(PolarsPanicError, match="should be equal"):
        pl.scan_csv(
            file_path,
            dtypes=[pl.Utf8, pl.Utf8],
            new_columns=["category", "calories"],
        ).collect()

    # cannot set both 'new_columns' and 'with_column_names'
    with pytest.raises(ValueError, match="mutually.exclusive"):
        pl.scan_csv(
            file_path,
            dtypes=[pl.Utf8, pl.Utf8],
            new_columns=["category", "calories", "fats", "sugars"],
            with_column_names=lambda cols: [col.capitalize() for col in cols],
        ).collect()


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


@pytest.mark.write_disk()
def test_glob_skip_rows() -> None:
    with TemporaryDirectory() as temp_dir:
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


def test_glob_n_rows(io_files_path: Path) -> None:
    file_path = io_files_path / "foods*.csv"
    df = pl.scan_csv(file_path, n_rows=40).collect()

    # 27 rows from foods1.csv and 13 from foods2.csv
    assert df.shape == (40, 4)

    # take first and last rows
    assert df[[0, 39]].to_dict(False) == {
        "category": ["vegetables", "seafood"],
        "calories": [45, 146],
        "fats_g": [0.5, 6.0],
        "sugars_g": [2, 2],
    }


def test_scan_csv_schema_overwrite_not_projected_8483(foods_file_path: str) -> None:
    df = (
        pl.scan_csv(
            foods_file_path,
            dtypes={"calories": pl.Utf8, "sugars_g": pl.Int8},
        )
        .select(pl.count())
        .collect()
    )
    expected = pl.DataFrame({"count": 27}, schema={"count": pl.UInt32})
    assert_frame_equal(df, expected)
