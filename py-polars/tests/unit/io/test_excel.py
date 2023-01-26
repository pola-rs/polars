from pathlib import Path

import pytest

import polars as pl
from polars.testing import assert_frame_equal


@pytest.fixture()
def excel_file_path(io_files_path: Path) -> Path:
    return io_files_path / "example.xlsx"


def test_read_excel(excel_file_path: Path) -> None:
    df = pl.read_excel(excel_file_path, sheet_name="Sheet1", sheet_id=None)

    expected = pl.DataFrame({"hello": ["Row 1", "Row 2"]})

    assert_frame_equal(df, expected)


def test_read_excel_all_sheets(excel_file_path: Path) -> None:
    df = pl.read_excel(excel_file_path, sheet_id=None)  # type: ignore[call-overload]

    expected1 = pl.DataFrame({"hello": ["Row 1", "Row 2"]})
    expected2 = pl.DataFrame({"world": ["Row 3", "Row 4"]})

    assert_frame_equal(df["Sheet1"], expected1)
    assert_frame_equal(df["Sheet2"], expected2)
