from __future__ import annotations

from datetime import date
from io import BytesIO
from typing import TYPE_CHECKING, Any

import pytest

import polars as pl
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
    df = pl.read_excel(excel_file_path, sheet_id=0)

    expected1 = pl.DataFrame({"hello": ["Row 1", "Row 2"]})
    expected2 = pl.DataFrame({"world": ["Row 3", "Row 4"]})

    assert_frame_equal(df["Sheet1"], expected1)
    assert_frame_equal(df["Sheet2"], expected2)


def test_read_excel_all_sheets_with_sheet_name(excel_file_path: Path) -> None:
    with pytest.raises(
        ValueError, match="Cannot specify both `sheet_name` and `sheet_id`"
    ):
        pl.read_excel(excel_file_path, sheet_id=1, sheet_name="Sheet1")


# the parameters don't change the data, only the formatting, so we expect
# the same result each time. however, it's important to validate that the
# parameter permutations don't raise exceptions, or interfere with the
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
        # slightly customised formatting, with some formulas
        {
            "position": (0, 0),
            "table_style": {
                "style": "Table Style Medium 23",
                "first_column": True,
            },
            "conditional_formats": {"val": "data_bar"},
            "column_formats": {
                "val": "#,##0.000;[White]-#,##0.000",
                ("day", "month", "year"): {"align": "left", "num_format": "0"},
            },
            "column_widths": {"val": 100},
            "row_heights": {0: 35},
            "formulas": {
                # string: formula added to the end of the table (but before row_totals)
                "day": "=DAY([@dtm])",
                "month": "=MONTH([@dtm])",
                "year": {
                    # dict: full control over formula positioning/dtype
                    "formula": "=YEAR([@dtm])",
                    "insert_after": "month",
                    "return_type": pl.Int16,
                },
            },
            "column_totals": True,
            "row_totals": True,
        },
        # heavily customised formatting/definition
        {
            "position": "A1",
            "table_name": "PolarsFrameData",
            "table_style": "Table Style Light 9",
            "conditional_formats": {
                # single dict format
                "str": {
                    "type": "duplicate",
                    "format": {"bg_color": "#ff0000", "font_color": "#ffffff"},
                },
                # multiple dict formats
                "val": [
                    {
                        "type": "3_color_scale",
                        "min_color": "#4bacc6",
                        "mid_color": "#ffffff",
                        "max_color": "#daeef3",
                    },
                    {
                        "type": "cell",
                        "criteria": "<",
                        "value": -90,
                        "format": {"font_color": "white"},
                    },
                ],
                "dtm": [
                    {
                        "type": "top",
                        "value": 1,
                        "format": {"bold": True, "font_color": "green"},
                    },
                    {
                        "type": "bottom",
                        "value": 1,
                        "format": {"bold": True, "font_color": "red"},
                    },
                ],
            },
            "dtype_formats": {
                pl.FLOAT_DTYPES: '_(£* #,##0.00_);_(£* (#,##0.00);_(£* "-"??_);_(@_)',
                pl.Date: "dd-mm-yyyy",
            },
            "column_formats": {"dtm": {"font_color": "#31869c", "bg_color": "#b7dee8"}},
            "column_totals": {"val": "average", "dtm": "min"},
            "column_widths": {("str", "val"): 60, "dtm": 80},
            "row_totals": {"tot": True},
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
            "str": ["xxx", "yyy", "xxx"],
            "val": [100.5, 55.0, -99.5],
        }
    )
    header_opts = (
        {}
        if write_params.get("has_header", True)
        else {"has_header": False, "new_columns": ["dtm", "str", "val"]}
    )
    fmt_strptime = "%Y-%m-%d"
    if write_params.get("dtype_formats", {}).get(pl.Date) == "dd-mm-yyyy":
        fmt_strptime = "%d-%m-%Y"

    # write to an xlsx with polars, using various parameters...
    xls = BytesIO()
    _wb = df.write_excel(workbook=xls, worksheet="data", **write_params)

    # ...and read it back again:
    xldf = pl.read_excel(
        xls,
        sheet_name="data",
        read_csv_options=header_opts,
    )[:3]
    xldf = xldf.select(xldf.columns[:3]).with_columns(
        pl.col("dtm").str.strptime(pl.Date, fmt_strptime)
    )
    assert_frame_equal(df, xldf)


def test_excel_compound_types() -> None:
    df = pl.DataFrame(
        {"x": [[1, 2], [3, 4], [5, 6]], "y": ["a", "b", "c"], "z": [9, 8, 7]}
    ).select("x", pl.struct(["y", "z"]))

    xls = BytesIO()
    df.write_excel(xls, worksheet="data")

    xldf = pl.read_excel(xls, sheet_name="data")
    assert xldf.rows() == [
        ("[1, 2]", "{'y': 'a', 'z': 9}"),
        ("[3, 4]", "{'y': 'b', 'z': 8}"),
        ("[5, 6]", "{'y': 'c', 'z': 7}"),
    ]


def test_excel_sparklines() -> None:
    from xlsxwriter import Workbook

    # note that we don't (quite) expect sparkline export to round-trip as we
    # inject additional empty columns to hold them (which will read as nulls).
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
            dtype_formats={pl.INTEGER_DTYPES: "#,##0_);(#,##0)"},
            column_formats={("h1", "h2"): "#,##0_);(#,##0)"},
            sparklines={
                "trend": ["q1", "q2", "q3", "q4"],
                "+/-": {
                    "columns": ["q1", "q2", "q3", "q4"],
                    "insert_after": "id",
                    "type": "win_loss",
                },
            },
            conditional_formats={
                ("q1", "q2", "q3", "q4", "h1", "h2"): {
                    "type": "2_color_scale",
                    "min_color": "#95b3d7",
                    "max_color": "#ffffff",
                }
            },
            column_widths={("q1", "q2", "q3", "q4", "h1", "h2"): 40},
            row_totals={
                "h1": ("q1", "q2"),
                "h2": ("q3", "q4"),
            },
            hide_gridlines=True,
            row_heights=35,
            sheet_zoom=125,
        )

    tables = {tbl["name"] for tbl in wb.get_worksheet_by_name("frame_data").tables}
    assert "Frame0" in tables

    xldf = pl.read_excel(xls, sheet_name="frame_data")
    # ┌─────┬──────┬─────┬─────┬─────┬─────┬───────┬─────┬─────┐
    # │ id  ┆ +/-  ┆ q1  ┆ q2  ┆ q3  ┆ q4  ┆ trend ┆ h1  ┆ h2  │
    # │ --- ┆ ---  ┆ --- ┆ --- ┆ --- ┆ --- ┆ ---   ┆ --- ┆ --- │
    # │ str ┆ str  ┆ i64 ┆ i64 ┆ i64 ┆ i64 ┆ str   ┆ i64 ┆ i64 │
    # ╞═════╪══════╪═════╪═════╪═════╪═════╪═══════╪═════╪═════╡
    # │ aaa ┆ null ┆ 100 ┆ 30  ┆ -50 ┆ 75  ┆ null  ┆ 0   ┆ 0   │
    # │ bbb ┆ null ┆ 55  ┆ -10 ┆ 0   ┆ 55  ┆ null  ┆ 0   ┆ 0   │
    # │ ccc ┆ null ┆ -20 ┆ 15  ┆ 40  ┆ 25  ┆ null  ┆ 0   ┆ 0   │
    # │ ddd ┆ null ┆ 0   ┆ 60  ┆ 80  ┆ -10 ┆ null  ┆ 0   ┆ 0   │
    # │ eee ┆ null ┆ 35  ┆ 20  ┆ 80  ┆ -55 ┆ null  ┆ 0   ┆ 0   │
    # └─────┴──────┴─────┴─────┴─────┴─────┴───────┴─────┴─────┘

    for sparkline_col in ("+/-", "trend"):
        assert set(xldf[sparkline_col]) == {None}

    assert xldf.columns == ["id", "+/-", "q1", "q2", "q3", "q4", "trend", "h1", "h2"]
    assert_frame_equal(df, xldf.drop("+/-", "trend", "h1", "h2"))


def test_excel_write_multiple_tables() -> None:
    from xlsxwriter import Workbook

    # note: checks that empty tables don't error on write
    df1 = pl.DataFrame(schema={"colx": pl.Date, "coly": pl.Utf8, "colz": pl.Float64})
    df2 = pl.DataFrame(schema={"colx": pl.Date, "coly": pl.Utf8, "colz": pl.Float64})
    df3 = pl.DataFrame(schema={"colx": pl.Date, "coly": pl.Utf8, "colz": pl.Float64})
    df4 = pl.DataFrame(schema={"colx": pl.Date, "coly": pl.Utf8, "colz": pl.Float64})

    xls = BytesIO()
    with Workbook(xls) as wb:
        df1.write_excel(workbook=wb, worksheet="sheet1", position="A1")
        df2.write_excel(workbook=wb, worksheet="sheet1", position="A6")
        df3.write_excel(workbook=wb, worksheet="sheet2", position="A1")

        # validate integration of externally-added formats
        fmt = wb.add_format({"bg_color": "#ffff00"})
        df4.write_excel(
            workbook=wb,
            worksheet="sheet3",
            position="A1",
            conditional_formats={
                "colz": {
                    "type": "formula",
                    "criteria": "=C2=B2",
                    "format": fmt,
                }
            },
        )

    table_names: set[str] = set()
    for sheet in ("sheet1", "sheet2", "sheet3"):
        table_names.update(
            tbl["name"] for tbl in wb.get_worksheet_by_name(sheet).tables
        )
    assert table_names == {f"Frame{n}" for n in range(4)}
    assert pl.read_excel(xls, sheet_name="sheet3").rows() == []
