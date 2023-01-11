import os
import datetime
from pathlib import Path

import polars as pl
from polars.testing import assert_frame_equal


def test_read_excel() -> None:
    example_file = Path(__file__).parent.parent / "files" / "example.xlsx"
    df = pl.read_excel(example_file, sheet_name="Sheet1", sheet_id=None)

    expected = pl.DataFrame({"hello": ["Row 1", "Row 2"]})

    assert_frame_equal(df, expected)


def test_read_excel_all_sheets() -> None:
    example_file = Path(__file__).parent.parent / "files" / "example.xlsx"
    df = pl.read_excel(example_file, sheet_id=None)  # type: ignore[call-overload]

    expected1 = pl.DataFrame({"hello": ["Row 1", "Row 2"]})
    expected2 = pl.DataFrame({"world": ["Row 3", "Row 4"]})

    assert_frame_equal(df["Sheet1"], expected1)
    assert_frame_equal(df["Sheet2"], expected2)


def test_basic_datatypes_write_excel() -> None:
    df = pl.DataFrame(
        {
            "A": [1, 2, 3, 4, 5],
            "fruits": ["banana", "banana", "apple", "apple", "banana"],
            "floats": [1.1, 1.2, 1.3, 1.4, 1.5],
            "datetime": [datetime.datetime(2023, 1, x) for x in range(1, 6)],
            "nulls": [1, None, None, None, 1]
        })
    filename = "test.xlsx"
    df.write_excel(filename)
    # check if can be read as it was written
    # we use openpyxl because type inference is better
    df_reread = pl.read_excel(filename, driver="openpyxl")
    assert_frame_equal(df, df_reread)
    os.remove(filename)
