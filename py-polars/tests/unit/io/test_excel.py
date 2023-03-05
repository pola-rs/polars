from __future__ import annotations

from datetime import date
from io import BytesIO
from typing import TYPE_CHECKING, Any

import pytest

import polars as pl
from polars.datatypes import FLOAT_DTYPES, INTEGER_DTYPES
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from pathlib import Path


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


# the parameters don't change the data, only the formatting, so we expect
# the same result each time. however, it's important to validate that the
# parameter permutations don't raise exceptions, or interfere wth the
# values written to the worksheet, so test multiple variations.
@pytest.mark.parametrize(
    "write_params",
    [
        # default parameters
        {},
        # basic formatting
        {
            "autofit": True,
            "table_style": "Table Style Light 16",
            "column_totals": True,
            "float_precision": 0,
        },
        # advanced formatting #1
        {
            "position": (0, 0),
            "table_style": {
                "style": "Table Style Medium 25",
                "first_column": True,
            },
            "conditional_formats": {
                "val": {
                    "type": "data_bar",
                    "data_bar_2010": True,
                    "bar_color": "#9bbb59",
                },
            },
            "column_formats": {"val": "#,##0.000;[White]-#,##0.000"},
            "column_widths": {"val": 100},
        },
        # advanced formatting #2
        {
            "conditional_formats": {
                "val": {
                    "type": "3_color_scale",
                    "min_color": "#76933c",
                    "mid_color": "#c4d79b",
                    "max_color": "#ebf1de",
                },
            },
            "dtype_formats": {
                FLOAT_DTYPES: '_(£* #,##0.00_);_(£* (#,##0.00);_(£* "-"??_);_(@_)'
            },
            "column_formats": {
                "dtm": {"font_color": "#31869c", "bg_color": "#b7dee8"},
            },
            "column_totals": {"val": "average"},
            "hidden_columns": ["str"],
            "hide_gridlines": True,
            "has_header": False,
        },
    ],
)
def test_excel_round_trip(write_params: dict[str, Any]) -> None:
    df = pl.DataFrame(
        {
            "dtm": [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3)],
            "str": ["aaa", "bbb", "ccc"],
            "val": [100.5, 55.0, -99.5],
        }
    )
    header_opts = (
        {}
        if write_params.get("has_header", True)
        else {"has_header": False, "new_columns": ["dtm", "str", "val"]}
    )

    # write to an xlsx with polars, using various parameters...
    xls = BytesIO()
    _wb = df.write_excel(workbook=xls, worksheet="data", **write_params)

    # ...and read it back again:
    xldf = pl.read_excel(  # type: ignore[call-overload]
        file=xls,
        sheet_name="data",
        read_csv_options=header_opts,
    ).with_columns(pl.col("dtm").str.strptime(pl.Date, "%Y-%m-%d"))

    if "column_totals" in write_params:
        xldf = xldf[:3]

    assert_frame_equal(df, xldf)


def test_excel_sparklines() -> None:
    from xlsxwriter import Workbook

    # note that we don't (quite) expect sparkline export to round-trip
    # as we have to inject additional empty columns to hold them...
    df = pl.DataFrame(
        {
            "id": ["aaa", "bbb", "ccc", "ddd", "eee"],
            "q1": [100, 55, -20, 0, 35],
            "q2": [30, -10, 15, 60, 20],
            "q3": [-50, 0, 40, 80, 80],
            "q4": [75, 55, 25, -10, -55],
        }
    )

    # also: confirm that we can use a Workbook directly with "write_excel"
    xls = BytesIO()
    with Workbook(xls) as wb:
        df.write_excel(
            workbook=wb,
            worksheet="frame_data",
            table_style="Table Style Light 2",
            dtype_formats={INTEGER_DTYPES: "#,##0_);(#,##0)"},
            sparklines={
                "trend": ["q1", "q2", "q3", "q4"],
                "+/-": {
                    "columns": ["q1", "q2", "q3", "q4"],
                    "insert_after": "id",
                    "type": "win_loss",
                },
            },
            hide_gridlines=True,
        )

    xldf = pl.read_excel(file=xls, sheet_name="frame_data")  # type: ignore[call-overload]
    # ┌──────┬──────┬─────┬─────┬─────┬─────┬───────┐
    # │ id   ┆ +/-  ┆ q1  ┆ q2  ┆ q3  ┆ q4  ┆ trend │
    # │ ---  ┆ ---  ┆ --- ┆ --- ┆ --- ┆ --- ┆ ---   │
    # │ str  ┆ str  ┆ i64 ┆ i64 ┆ i64 ┆ i64 ┆ str   │
    # ╞══════╪══════╪═════╪═════╪═════╪═════╪═══════╡
    # │ aaa  ┆ null ┆ 100 ┆ 30  ┆ -50 ┆ 75  ┆ null  │
    # │ bbb  ┆ null ┆ 55  ┆ -10 ┆ 0   ┆ 55  ┆ null  │
    # │ ccc  ┆ null ┆ -20 ┆ 15  ┆ 40  ┆ 25  ┆ null  │
    # │ ddd  ┆ null ┆ 0   ┆ 60  ┆ 80  ┆ -10 ┆ null  │
    # │ eee  ┆ null ┆ 35  ┆ 20  ┆ 80  ┆ -55 ┆ null  │
    # └──────┴──────┴─────┴─────┴─────┴─────┴───────┘

    for sparkline_col in ("+/-", "trend"):
        assert set(xldf[sparkline_col]) == {None}

    assert xldf.columns == ["id", "+/-", "q1", "q2", "q3", "q4", "trend"]
    assert_frame_equal(df, xldf.drop("+/-", "trend"))
