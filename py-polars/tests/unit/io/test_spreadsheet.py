from __future__ import annotations

import warnings
from collections import OrderedDict
from datetime import date, datetime
from io import BytesIO
from typing import TYPE_CHECKING, Any, Callable, Literal

import pytest

import polars as pl
import polars.selectors as cs
from polars.exceptions import NoDataError, ParameterCollisionError
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from pathlib import Path

    from polars.type_aliases import SchemaDict, SelectorType

pytestmark = pytest.mark.slow()


@pytest.fixture()
def path_xlsx(io_files_path: Path) -> Path:
    return io_files_path / "example.xlsx"


@pytest.fixture()
def path_xlsx_empty(io_files_path: Path) -> Path:
    return io_files_path / "empty.xlsx"


@pytest.fixture()
def path_xlsb(io_files_path: Path) -> Path:
    return io_files_path / "example.xlsb"


@pytest.fixture()
def path_xlsb_empty(io_files_path: Path) -> Path:
    return io_files_path / "empty.xlsb"


@pytest.fixture()
def path_ods(io_files_path: Path) -> Path:
    return io_files_path / "example.ods"


@pytest.fixture()
def path_ods_empty(io_files_path: Path) -> Path:
    return io_files_path / "empty.ods"


@pytest.mark.parametrize(
    ("read_spreadsheet", "source", "engine_params"),
    [
        (pl.read_excel, "path_xlsx", {"engine": "xlsx2csv"}),
        (pl.read_excel, "path_xlsx", {"engine": "openpyxl"}),
        (pl.read_excel, "path_xlsb", {"engine": "pyxlsb"}),
        (pl.read_ods, "path_ods", {}),
    ],
)
def test_read_spreadsheet(
    read_spreadsheet: Callable[..., pl.DataFrame],
    source: str,
    engine_params: dict[str, str],
    request: pytest.FixtureRequest,
) -> None:
    sheet_params: dict[str, Any]

    for sheet_params in (  # type: ignore[assignment]
        {"sheet_name": None, "sheet_id": None},
        {"sheet_name": "test1"},
        {"sheet_id": 1},
    ):
        df = read_spreadsheet(
            source=request.getfixturevalue(source),
            **engine_params,
            **sheet_params,
        )
        expected = pl.DataFrame({"hello": ["Row 1", "Row 2"]})
        assert_frame_equal(df, expected)


@pytest.mark.parametrize(
    ("read_spreadsheet", "source", "params"),
    [
        (pl.read_excel, "path_xlsx", {"engine": "xlsx2csv"}),
        (pl.read_excel, "path_xlsx", {"engine": "openpyxl"}),
        (pl.read_excel, "path_xlsb", {"engine": "pyxlsb"}),
        (pl.read_ods, "path_ods", {}),
    ],
)
def test_read_excel_multi_sheets(
    read_spreadsheet: Callable[..., dict[str, pl.DataFrame]],
    source: str,
    params: dict[str, str],
    request: pytest.FixtureRequest,
) -> None:
    spreadsheet_path = request.getfixturevalue(source)
    frames_by_id = read_spreadsheet(
        spreadsheet_path,
        sheet_id=[2, 1],
        sheet_name=None,
        **params,
    )
    frames_by_name = read_spreadsheet(
        spreadsheet_path,
        sheet_id=None,
        sheet_name=["test2", "test1"],
        **params,
    )
    for frames in (frames_by_id, frames_by_name):
        assert list(frames_by_name) == ["test2", "test1"]

        expected1 = pl.DataFrame({"hello": ["Row 1", "Row 2"]})
        expected2 = pl.DataFrame({"world": ["Row 3", "Row 4"]})

        assert_frame_equal(frames["test1"], expected1)
        assert_frame_equal(frames["test2"], expected2)


@pytest.mark.parametrize(
    ("read_spreadsheet", "source", "params"),
    [
        (pl.read_excel, "path_xlsx", {"engine": "xlsx2csv"}),
        (pl.read_excel, "path_xlsx", {"engine": "openpyxl"}),
        (pl.read_excel, "path_xlsb", {"engine": "pyxlsb"}),
        (pl.read_ods, "path_ods", {}),
    ],
)
def test_read_excel_all_sheets(
    read_spreadsheet: Callable[..., dict[str, pl.DataFrame]],
    source: str,
    params: dict[str, str],
    request: pytest.FixtureRequest,
) -> None:
    spreadsheet_path = request.getfixturevalue(source)
    frames = read_spreadsheet(
        spreadsheet_path,
        sheet_id=0,
        **params,
    )
    assert len(frames) == (4 if str(spreadsheet_path).endswith("ods") else 5)

    expected1 = pl.DataFrame({"hello": ["Row 1", "Row 2"]})
    expected2 = pl.DataFrame({"world": ["Row 3", "Row 4"]})
    expected3 = pl.DataFrame(
        {
            "cardinality": [1, 3, 15, 30, 150, 300],
            "rows_by_key": [0.05059, 0.04478, 0.04414, 0.05245, 0.05395, 0.05677],
            "iter_groups": [0.04806, 0.04223, 0.04774, 0.04864, 0.0572, 0.06945],
        }
    )
    assert_frame_equal(frames["test1"], expected1)
    assert_frame_equal(frames["test2"], expected2)
    if params.get("engine") == "openpyxl":
        # TODO: flag that trims trailing all-null rows?
        assert_frame_equal(frames["test3"], expected3)
        assert_frame_equal(frames["test4"].drop_nulls(), expected3)


@pytest.mark.parametrize(
    ("engine", "schema_overrides"),
    [
        ("xlsx2csv", {"datetime": pl.Datetime}),
        ("openpyxl", None),
    ],
)
def test_read_excel_basic_datatypes(
    engine: Literal["xlsx2csv", "openpyxl", "pyxlsb"],
    schema_overrides: SchemaDict | None,
) -> None:
    df = pl.DataFrame(
        {
            "A": [1, 2, 3, 4, 5],
            "fruits": ["banana", "banana", "apple", "apple", "banana"],
            "floats": [1.1, 1.2, 1.3, 1.4, 1.5],
            "datetime": [datetime(2023, 1, x) for x in range(1, 6)],
            "nulls": [1, None, None, None, 1],
        }
    )
    xls = BytesIO()
    df.write_excel(xls, position="C5")

    # check if can be read as it was written
    for sheet_id, sheet_name in ((None, None), (1, None), (None, "Sheet1")):
        df = pl.read_excel(
            xls,
            sheet_id=sheet_id,
            sheet_name=sheet_name,
            engine=engine,
            schema_overrides=schema_overrides,
        )
        assert_frame_equal(df, df)


@pytest.mark.parametrize(
    ("read_spreadsheet", "source", "params"),
    [
        (pl.read_excel, "path_xlsx", {"engine": "xlsx2csv"}),
        (pl.read_excel, "path_xlsx", {"engine": "openpyxl"}),
        (pl.read_excel, "path_xlsb", {"engine": "pyxlsb"}),
        (pl.read_ods, "path_ods", {}),
    ],
)
def test_read_invalid_worksheet(
    read_spreadsheet: Callable[..., dict[str, pl.DataFrame]],
    source: str,
    params: dict[str, str],
    request: pytest.FixtureRequest,
) -> None:
    spreadsheet_path = request.getfixturevalue(source)
    for param, sheet_id, sheet_name in (
        ("id", 999, None),
        ("name", None, "not_a_sheet_name"),
    ):
        value = sheet_id if param == "id" else sheet_name
        with pytest.raises(
            ValueError,
            match=f"no matching sheet found when `sheet_{param}` is {value!r}",
        ):
            read_spreadsheet(
                spreadsheet_path, sheet_id=sheet_id, sheet_name=sheet_name, **params
            )


@pytest.mark.parametrize("engine", ["xlsx2csv", "openpyxl"])
def test_write_excel_bytes(engine: Literal["xlsx2csv", "openpyxl", "pyxlsb"]) -> None:
    df = pl.DataFrame({"A": [1, 2, 3, 4, 5]})

    excel_bytes = BytesIO()
    df.write_excel(excel_bytes)

    df_read = pl.read_excel(excel_bytes, engine=engine)
    assert_frame_equal(df, df_read)


def test_schema_overrides(path_xlsx: Path, path_xlsb: Path, path_ods: Path) -> None:
    df1 = pl.read_excel(
        path_xlsx,
        sheet_name="test4",
        schema_overrides={"cardinality": pl.UInt16},
    ).drop_nulls()
    assert df1.schema == {
        "cardinality": pl.UInt16,
        "rows_by_key": pl.Float64,
        "iter_groups": pl.Float64,
    }

    df2 = pl.read_excel(
        path_xlsx,
        sheet_name="test4",
        read_csv_options={"dtypes": {"cardinality": pl.UInt16}},
    ).drop_nulls()
    assert df2.schema == {
        "cardinality": pl.UInt16,
        "rows_by_key": pl.Float64,
        "iter_groups": pl.Float64,
    }

    df3 = pl.read_excel(
        path_xlsx,
        sheet_name="test4",
        schema_overrides={"cardinality": pl.UInt16},
        read_csv_options={
            "dtypes": {
                "rows_by_key": pl.Float32,
                "iter_groups": pl.Float32,
            },
        },
    ).drop_nulls()
    assert df3.schema == {
        "cardinality": pl.UInt16,
        "rows_by_key": pl.Float32,
        "iter_groups": pl.Float32,
    }

    for workbook_path in (path_xlsx, path_xlsb, path_ods):
        df4 = pl.read_excel(
            workbook_path,
            sheet_name="test5",
            schema_overrides={"dtm": pl.Datetime("ns"), "dt": pl.Date},
        )
        assert_frame_equal(
            df4,
            pl.DataFrame(
                {
                    "dtm": [
                        datetime(1999, 12, 31, 10, 30, 45),
                        datetime(2010, 10, 11, 12, 13, 14),
                    ],
                    "dt": [date(2024, 1, 1), date(2018, 8, 7)],
                    "val": [1.5, -0.5],
                },
                schema={"dtm": pl.Datetime("ns"), "dt": pl.Date, "val": pl.Float64},
            ),
        )

    with pytest.raises(ParameterCollisionError):
        # cannot specify 'cardinality' in both schema_overrides and read_csv_options
        pl.read_excel(
            path_xlsx,
            sheet_name="test4",
            schema_overrides={"cardinality": pl.UInt16},
            read_csv_options={"dtypes": {"cardinality": pl.Int32}},
        )

    # read multiple sheets in conjunction with 'schema_overrides'
    # (note: reading the same sheet twice simulates the issue in #11850)
    overrides = OrderedDict(
        [
            ("cardinality", pl.UInt32),
            ("rows_by_key", pl.Float32),
            ("iter_groups", pl.Float64),
        ]
    )
    df = pl.read_excel(  # type: ignore[call-overload]
        path_xlsx,
        sheet_name=["test4", "test4"],
        schema_overrides=overrides,
    )
    assert df["test4"].schema == overrides


def test_unsupported_engine() -> None:
    with pytest.raises(NotImplementedError):
        pl.read_excel(None, engine="foo")  # type: ignore[call-overload]


def test_unsupported_binary_workbook(path_xlsx: Path, path_xlsb: Path) -> None:
    with pytest.raises(Exception, match="invalid Excel Binary Workbook"):
        pl.read_excel(path_xlsx, engine="pyxlsb")

    with pytest.raises(Exception, match="does not support binary format"):
        pl.read_excel(path_xlsb, engine="openpyxl")


@pytest.mark.parametrize("engine", ["xlsx2csv", "openpyxl"])
def test_read_excel_all_sheets_with_sheet_name(path_xlsx: Path, engine: str) -> None:
    with pytest.raises(
        ValueError,
        match=r"cannot specify both `sheet_name` \('Sheet1'\) and `sheet_id` \(1\)",
    ):
        pl.read_excel(  # type: ignore[call-overload]
            path_xlsx,
            sheet_id=1,
            sheet_name="Sheet1",
            engine=engine,
        )


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
            "header_format": {"italic": True, "bg_color": "#d9d9d9"},
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
            "include_header": False,
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
        if write_params.get("include_header", True)
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


@pytest.mark.parametrize("engine", ["xlsx2csv", "openpyxl"])
def test_excel_compound_types(
    engine: Literal["xlsx2csv", "openpyxl", "pyxlsb"],
) -> None:
    df = pl.DataFrame(
        {"x": [[1, 2], [3, 4], [5, 6]], "y": ["a", "b", "c"], "z": [9, 8, 7]}
    ).select("x", pl.struct(["y", "z"]))

    xls = BytesIO()
    df.write_excel(xls, worksheet="data")

    xldf = pl.read_excel(xls, sheet_name="data", engine=engine)
    assert xldf.rows() == [
        ("[1, 2]", "{'y': 'a', 'z': 9}"),
        ("[3, 4]", "{'y': 'b', 'z': 8}"),
        ("[5, 6]", "{'y': 'c', 'z': 7}"),
    ]


@pytest.mark.parametrize("engine", ["xlsx2csv", "openpyxl"])
def test_excel_sparklines(engine: Literal["xlsx2csv", "openpyxl", "pyxlsb"]) -> None:
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
            column_formats={cs.starts_with("h"): "#,##0_);(#,##0)"},
            sparklines={
                "trend": ["q1", "q2", "q3", "q4"],
                "+/-": {
                    "columns": ["q1", "q2", "q3", "q4"],
                    "insert_after": "id",
                    "type": "win_loss",
                },
            },
            conditional_formats={
                cs.starts_with("q", "h"): {
                    "type": "2_color_scale",
                    "min_color": "#95b3d7",
                    "max_color": "#ffffff",
                }
            },
            column_widths={cs.starts_with("q", "h"): 40},
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

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        xldf = pl.read_excel(xls, sheet_name="frame_data", engine=engine)

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
    df1 = pl.DataFrame(schema={"colx": pl.Date, "coly": pl.String, "colz": pl.Float64})
    df2 = pl.DataFrame(schema={"colx": pl.Date, "coly": pl.String, "colz": pl.Float64})
    df3 = pl.DataFrame(schema={"colx": pl.Date, "coly": pl.String, "colz": pl.Float64})
    df4 = pl.DataFrame(schema={"colx": pl.Date, "coly": pl.String, "colz": pl.Float64})

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


def test_excel_freeze_panes() -> None:
    from xlsxwriter import Workbook

    # note: checks that empty tables don't error on write
    df1 = pl.DataFrame(schema={"colx": pl.Date, "coly": pl.String, "colz": pl.Float64})
    df2 = pl.DataFrame(schema={"colx": pl.Date, "coly": pl.String, "colz": pl.Float64})
    df3 = pl.DataFrame(schema={"colx": pl.Date, "coly": pl.String, "colz": pl.Float64})

    xls = BytesIO()

    # use all three freeze_pane notations
    with Workbook(xls) as wb:
        df1.write_excel(workbook=wb, worksheet="sheet1", freeze_panes=(1, 0))
        df2.write_excel(workbook=wb, worksheet="sheet2", freeze_panes=(1, 0, 3, 4))
        df3.write_excel(workbook=wb, worksheet="sheet3", freeze_panes=("B2"))

    table_names: set[str] = set()
    for sheet in ("sheet1", "sheet2", "sheet3"):
        table_names.update(
            tbl["name"] for tbl in wb.get_worksheet_by_name(sheet).tables
        )
    assert table_names == {f"Frame{n}" for n in range(3)}
    assert pl.read_excel(xls, sheet_name="sheet3").rows() == []


@pytest.mark.parametrize(
    ("read_spreadsheet", "source"),
    [
        (pl.read_excel, "path_xlsx_empty"),
        (pl.read_excel, "path_xlsb_empty"),
        (pl.read_ods, "path_ods_empty"),
    ],
)
def test_excel_empty_sheet(
    read_spreadsheet: Callable[..., pl.DataFrame],
    source: str,
    request: pytest.FixtureRequest,
) -> None:
    empty_spreadsheet_path = request.getfixturevalue(source)
    with pytest.raises(NoDataError, match="empty Excel sheet"):
        pl.read_excel(empty_spreadsheet_path)

    df = pl.read_excel(empty_spreadsheet_path, raise_if_empty=False)
    assert_frame_equal(df, pl.DataFrame())


@pytest.mark.parametrize(
    ("engine", "hidden_columns"),
    [
        ("xlsx2csv", ["a"]),
        ("openpyxl", ["a", "b"]),
        ("xlsx2csv", cs.numeric()),
        ("openpyxl", cs.last()),
    ],
)
def test_excel_hidden_columns(
    hidden_columns: list[str] | SelectorType,
    engine: Literal["xlsx2csv", "openpyxl", "pyxlsb"],
) -> None:
    df = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})

    xls = BytesIO()
    df.write_excel(xls, hidden_columns=hidden_columns)

    read_df = pl.read_excel(xls)
    assert_frame_equal(df, read_df)


def test_invalid_engine_options() -> None:
    with pytest.raises(ValueError, match="cannot specify `read_csv_options`"):
        pl.read_excel(
            "",
            engine="openpyxl",
            read_csv_options={"sep": "\t"},
        )

    with pytest.raises(ValueError, match="cannot specify `xlsx2csv_options`"):
        pl.read_excel(
            "",
            engine="openpyxl",
            xlsx2csv_options={"skip_empty_lines": True},
        )
