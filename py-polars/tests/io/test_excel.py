from pathlib import Path

import polars as pl


def test_read_excel() -> None:
    example_file = Path(__file__).parent.parent / "files" / "example.xlsx"
    df = pl.read_excel(example_file, sheet_name="Sheet1", sheet_id=None)

    expected = pl.DataFrame({"hello": ["Row 1", "Row 2"]})

    pl.testing.assert_frame_equal(df, expected)


def test_read_excel_all_sheets() -> None:
    example_file = Path(__file__).parent.parent / "files" / "example.xlsx"
    df = pl.read_excel(example_file, sheet_id=None)  # type: ignore[call-overload]

    expected1 = pl.DataFrame({"hello": ["Row 1", "Row 2"]})
    expected2 = pl.DataFrame({"world": ["Row 3", "Row 4"]})

    pl.testing.assert_frame_equal(df["Sheet1"], expected1)
    pl.testing.assert_frame_equal(df["Sheet2"], expected2)
