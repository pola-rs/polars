from __future__ import annotations

import warnings
from collections import OrderedDict
from datetime import date, datetime
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence

import pytest

import polars as pl
import polars.selectors as cs
from polars.exceptions import NoDataError, ParameterCollisionError
from polars.io.spreadsheet.functions import _identify_workbook
from polars.testing import assert_frame_equal, assert_series_equal
from tests.unit.conftest import FLOAT_DTYPES, NUMERIC_DTYPES

if TYPE_CHECKING:
    from polars._typing import ExcelSpreadsheetEngine, SelectorType

pytestmark = pytest.mark.slow()


@pytest.fixture()
def path_xls(io_files_path: Path) -> Path:
    # old excel 97-2004 format
    return io_files_path / "example.xls"


@pytest.fixture()
def path_xlsx(io_files_path: Path) -> Path:
    # modern excel format
    return io_files_path / "example.xlsx"


@pytest.fixture()
def path_xlsb(io_files_path: Path) -> Path:
    # excel binary format
    return io_files_path / "example.xlsb"


@pytest.fixture()
def path_ods(io_files_path: Path) -> Path:
    # open document spreadsheet
    return io_files_path / "example.ods"


@pytest.fixture()
def path_xls_empty(io_files_path: Path) -> Path:
    return io_files_path / "empty.xls"


@pytest.fixture()
def path_xlsx_empty(io_files_path: Path) -> Path:
    return io_files_path / "empty.xlsx"


@pytest.fixture()
def path_xlsx_mixed(io_files_path: Path) -> Path:
    return io_files_path / "mixed.xlsx"


@pytest.fixture()
def path_xlsb_empty(io_files_path: Path) -> Path:
    return io_files_path / "empty.xlsb"


@pytest.fixture()
def path_xlsb_mixed(io_files_path: Path) -> Path:
    return io_files_path / "mixed.xlsb"


@pytest.fixture()
def path_ods_empty(io_files_path: Path) -> Path:
    return io_files_path / "empty.ods"


@pytest.fixture()
def path_ods_mixed(io_files_path: Path) -> Path:
    return io_files_path / "mixed.ods"


@pytest.mark.parametrize(
    ("read_spreadsheet", "source", "engine_params"),
    [
        # xls file
        (pl.read_excel, "path_xls", {"engine": "calamine"}),
        # xlsx file
        (pl.read_excel, "path_xlsx", {"engine": "xlsx2csv"}),
        (pl.read_excel, "path_xlsx", {"engine": "openpyxl"}),
        (pl.read_excel, "path_xlsx", {"engine": "calamine"}),
        # xlsb file (binary)
        (pl.read_excel, "path_xlsb", {"engine": "calamine"}),
        # open document
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

    for sheet_params in (
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
        # xls file
        (pl.read_excel, "path_xls", {"engine": "calamine"}),
        # xlsx file
        (pl.read_excel, "path_xlsx", {"engine": "xlsx2csv"}),
        (pl.read_excel, "path_xlsx", {"engine": "openpyxl"}),
        (pl.read_excel, "path_xlsx", {"engine": "calamine"}),
        # xlsb file (binary)
        (pl.read_excel, "path_xlsb", {"engine": "calamine"}),
        # open document
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
        # xls file
        (pl.read_excel, "path_xls", {"engine": "calamine"}),
        # xlsx file
        (pl.read_excel, "path_xlsx", {"engine": "xlsx2csv"}),
        (pl.read_excel, "path_xlsx", {"engine": "openpyxl"}),
        (pl.read_excel, "path_xlsx", {"engine": "calamine"}),
        # xlsb file (binary)
        (pl.read_excel, "path_xlsb", {"engine": "calamine"}),
        # open document
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
    "engine",
    ["xlsx2csv", "calamine", "openpyxl"],
)
def test_read_excel_basic_datatypes(engine: ExcelSpreadsheetEngine) -> None:
    df = pl.DataFrame(
        {
            "A": [1, 2, 3, 4, 5],
            "fruits": ["banana", "banana", "apple", "apple", "banana"],
            "floats": [1.1, 1.2, 1.3, 1.4, 1.5],
            "datetime": [datetime(2023, 1, x) for x in range(1, 6)],
            "nulls": [1, None, None, None, 0],
        },
    )
    xls = BytesIO()
    df.write_excel(xls, position="C5")

    schema_overrides = {"datetime": pl.Datetime, "nulls": pl.Boolean}
    df_compare = df.with_columns(
        pl.col(nm).cast(tp) for nm, tp in schema_overrides.items()
    )
    for sheet_id, sheet_name in ((None, None), (1, None), (None, "Sheet1")):
        df_from_excel = pl.read_excel(
            xls,
            sheet_id=sheet_id,
            sheet_name=sheet_name,
            engine=engine,
            schema_overrides=schema_overrides,
        )
        assert_frame_equal(df_compare, df_from_excel)

    # check some additional overrides
    # (note: xlsx2csv can't currently convert datetime with trailing '00:00:00' to date)
    dt_override = {"datetime": pl.Date} if engine != "xlsx2csv" else {}
    df = pl.read_excel(
        xls,
        sheet_id=sheet_id,
        sheet_name=sheet_name,
        engine=engine,
        schema_overrides={"A": pl.Float32, **dt_override},
    )
    assert_series_equal(
        df["A"],
        pl.Series(name="A", values=[1.0, 2.0, 3.0, 4.0, 5.0], dtype=pl.Float32),
    )
    if dt_override:
        assert_series_equal(
            df["datetime"],
            pl.Series(
                name="datetime",
                values=[date(2023, 1, x) for x in range(1, 6)],
                dtype=pl.Date,
            ),
        )


@pytest.mark.parametrize(
    ("read_spreadsheet", "source", "params"),
    [
        # xls file
        (pl.read_excel, "path_xls", {"engine": "calamine"}),
        # xlsx file
        (pl.read_excel, "path_xlsx", {"engine": "xlsx2csv"}),
        (pl.read_excel, "path_xlsx", {"engine": "openpyxl"}),
        (pl.read_excel, "path_xlsx", {"engine": "calamine"}),
        # xlsb file (binary)
        (pl.read_excel, "path_xlsb", {"engine": "calamine"}),
        # open document
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


@pytest.mark.parametrize(
    ("read_spreadsheet", "source", "additional_params"),
    [
        (pl.read_excel, "path_xlsx_mixed", {"engine": "openpyxl"}),
        (pl.read_ods, "path_ods_mixed", {}),
    ],
)
def test_read_mixed_dtype_columns(
    read_spreadsheet: Callable[..., dict[str, pl.DataFrame]],
    source: str,
    additional_params: dict[str, str],
    request: pytest.FixtureRequest,
) -> None:
    spreadsheet_path = request.getfixturevalue(source)
    schema_overrides = {
        "Employee ID": pl.Utf8,
        "Employee Name": pl.Utf8,
        "Date": pl.Date,
        "Details": pl.Categorical,
        "Asset ID": pl.Utf8,
    }

    df = read_spreadsheet(
        spreadsheet_path,
        sheet_id=0,
        schema_overrides=schema_overrides,
        **additional_params,
    )["Sheet1"]

    assert_frame_equal(
        df,
        pl.DataFrame(
            {
                "Employee ID": ["123456", "44333", "US00011", "135967", "IN86868"],
                "Employee Name": ["Test1", "Test2", "Test4", "Test5", "Test6"],
                "Date": [
                    date(2023, 7, 21),
                    date(2023, 7, 21),
                    date(2023, 7, 21),
                    date(2023, 7, 21),
                    date(2023, 7, 21),
                ],
                "Details": [
                    "Healthcare",
                    "Healthcare",
                    "Healthcare",
                    "Healthcare",
                    "Something",
                ],
                "Asset ID": ["84444", "84444", "84444", "84444", "ABC123"],
            },
            schema_overrides=schema_overrides,
        ),
    )


@pytest.mark.parametrize("engine", ["xlsx2csv", "openpyxl", "calamine"])
def test_write_excel_bytes(engine: ExcelSpreadsheetEngine) -> None:
    df = pl.DataFrame({"colx": [1.5, -2, 0], "coly": ["a", None, "c"]})

    excel_bytes = BytesIO()
    df.write_excel(excel_bytes)

    df_read = pl.read_excel(excel_bytes, engine=engine)
    assert_frame_equal(df, df_read)

    # also confirm consistent behaviour when 'infer_schema_length=0'
    df_read = pl.read_excel(excel_bytes, engine=engine, infer_schema_length=0)
    expected = pl.DataFrame({"colx": ["1.5", "-2", "0"], "coly": ["a", None, "c"]})
    assert_frame_equal(expected, df_read)


def test_schema_overrides(path_xlsx: Path, path_xlsb: Path, path_ods: Path) -> None:
    df1 = pl.read_excel(
        path_xlsx,
        sheet_name="test4",
        schema_overrides={"cardinality": pl.UInt16},
    ).drop_nulls()

    assert df1.schema["cardinality"] == pl.UInt16
    assert df1.schema["rows_by_key"] == pl.Float64
    assert df1.schema["iter_groups"] == pl.Float64

    df2 = pl.read_excel(
        path_xlsx,
        sheet_name="test4",
        engine="xlsx2csv",
        read_options={"schema_overrides": {"cardinality": pl.UInt16}},
    ).drop_nulls()

    assert df2.schema["cardinality"] == pl.UInt16
    assert df2.schema["rows_by_key"] == pl.Float64
    assert df2.schema["iter_groups"] == pl.Float64

    df3 = pl.read_excel(
        path_xlsx,
        sheet_name="test4",
        engine="xlsx2csv",
        schema_overrides={"cardinality": pl.UInt16},
        read_options={
            "schema_overrides": {
                "rows_by_key": pl.Float32,
                "iter_groups": pl.Float32,
            },
        },
    ).drop_nulls()

    assert df3.schema["cardinality"] == pl.UInt16
    assert df3.schema["rows_by_key"] == pl.Float32
    assert df3.schema["iter_groups"] == pl.Float32

    for workbook_path in (path_xlsx, path_xlsb, path_ods):
        read_spreadsheet = (
            pl.read_ods if workbook_path.suffix == ".ods" else pl.read_excel
        )
        df4 = read_spreadsheet(  # type: ignore[operator]
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
        # cannot specify 'cardinality' in both schema_overrides and read_options
        pl.read_excel(
            path_xlsx,
            sheet_name="test4",
            engine="xlsx2csv",
            schema_overrides={"cardinality": pl.UInt16},
            read_options={"schema_overrides": {"cardinality": pl.Int32}},
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
    for col, dtype in overrides.items():
        assert df["test4"].schema[col] == dtype


@pytest.mark.parametrize(
    ("engine", "read_opts_param"),
    [
        ("xlsx2csv", "infer_schema_length"),
        ("calamine", "schema_sample_rows"),
    ],
)
def test_invalid_parameter_combinations_infer_schema_len(
    path_xlsx: Path, engine: str, read_opts_param: str
) -> None:
    with pytest.raises(
        ParameterCollisionError,
        match=f"cannot specify both `infer_schema_length`.*{read_opts_param}",
    ):
        pl.read_excel(  # type: ignore[call-overload]
            path_xlsx,
            sheet_id=1,
            engine=engine,
            read_options={read_opts_param: 512},
            infer_schema_length=1024,
        )


@pytest.mark.parametrize(
    ("engine", "read_opts_param"),
    [
        ("xlsx2csv", "columns"),
        ("calamine", "use_columns"),
    ],
)
def test_invalid_parameter_combinations_columns(
    path_xlsx: Path, engine: str, read_opts_param: str
) -> None:
    with pytest.raises(
        ParameterCollisionError,
        match=f"cannot specify both `columns`.*{read_opts_param}",
    ):
        pl.read_excel(  # type: ignore[call-overload]
            path_xlsx,
            sheet_id=1,
            engine=engine,
            read_options={read_opts_param: ["B", "C", "D"]},
            columns=["A", "B", "C"],
        )


def test_unsupported_engine() -> None:
    with pytest.raises(NotImplementedError):
        pl.read_excel(None, engine="foo")  # type: ignore[call-overload]


def test_unsupported_binary_workbook(path_xlsb: Path) -> None:
    with pytest.raises(Exception, match="does not support binary format"):
        pl.read_excel(path_xlsb, engine="openpyxl")


@pytest.mark.parametrize("engine", ["xlsx2csv", "openpyxl", "calamine"])
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
            "table_style": "Table Style Dark 2",
            "column_totals": True,
            "float_precision": 0,
        },
        # slightly customized formatting, with some formulas
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
        # heavily customized formatting/definition
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
                frozenset(
                    FLOAT_DTYPES
                ): '_(£* #,##0.00_);_(£* (#,##0.00);_(£* "-"??_);_(@_)',
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

    engine: ExcelSpreadsheetEngine
    for engine in ("calamine", "xlsx2csv"):
        read_options, has_header = (
            ({}, True)
            if write_params.get("include_header", True)
            else (
                {"new_columns": ["dtm", "str", "val"]}
                if engine == "xlsx2csv"
                else {"column_names": ["dtm", "str", "val"]},
                False,
            )
        )

        fmt_strptime = "%Y-%m-%d"
        if write_params.get("dtype_formats", {}).get(pl.Date) == "dd-mm-yyyy":
            fmt_strptime = "%d-%m-%Y"

        # write to xlsx using various parameters...
        xls = BytesIO()
        _wb = df.write_excel(workbook=xls, worksheet="data", **write_params)

        # ...and read it back again:
        xldf = pl.read_excel(
            xls,
            sheet_name="data",
            engine=engine,
            read_options=read_options,
            has_header=has_header,
        )[:3].select(df.columns[:3])

        if engine == "xlsx2csv":
            xldf = xldf.with_columns(pl.col("dtm").str.strptime(pl.Date, fmt_strptime))

        assert_frame_equal(df, xldf)


@pytest.mark.parametrize("engine", ["xlsx2csv", "calamine"])
def test_excel_write_column_and_row_totals(engine: ExcelSpreadsheetEngine) -> None:
    df = pl.DataFrame(
        {
            "id": ["aaa", "bbb", "ccc", "ddd", "eee"],
            # float cols
            "q1": [100.0, 55.5, -20.0, 0.5, 35.0],
            "q2": [30.5, -10.25, 15.0, 60.0, 20.5],
            # int cols
            "q3": [-50, 0, 40, 80, 80],
            "q4": [75, 55, 25, -10, -55],
        }
    )
    for fn_sum in (True, "sum", "SUM"):
        xls = BytesIO()
        df.write_excel(
            xls,
            worksheet="misc",
            sparklines={"trend": ["q1", "q2", "q3", "q4"]},
            row_totals={
                # add semiannual row total columns
                "h1": ("q1", "q2"),
                "h2": ("q3", "q4"),
            },
            column_totals=fn_sum,
        )

        # note that the totals are written as formulae, so we
        # won't have the calculated values in the dataframe
        xldf = pl.read_excel(xls, sheet_name="misc", engine=engine)
        assert xldf.columns == ["id", "q1", "q2", "q3", "q4", "trend", "h1", "h2"]
        assert xldf.row(-1) == (None, 0.0, 0.0, 0, 0, None, 0.0, 0)


@pytest.mark.parametrize("engine", ["xlsx2csv", "openpyxl", "calamine"])
def test_excel_write_compound_types(engine: ExcelSpreadsheetEngine) -> None:
    df = pl.DataFrame(
        {"x": [[1, 2], [3, 4], [5, 6]], "y": ["a", "b", "c"], "z": [9, 8, 7]}
    ).select("x", pl.struct(["y", "z"]))

    xls = BytesIO()
    df.write_excel(xls, worksheet="data")

    # expect string conversion (only scalar values are supported)
    xldf = pl.read_excel(xls, sheet_name="data", engine=engine)
    assert xldf.rows() == [
        ("[1, 2]", "{'y': 'a', 'z': 9}"),
        ("[3, 4]", "{'y': 'b', 'z': 8}"),
        ("[5, 6]", "{'y': 'c', 'z': 7}"),
    ]


@pytest.mark.parametrize("engine", ["xlsx2csv", "openpyxl", "calamine"])
def test_excel_read_no_headers(engine: ExcelSpreadsheetEngine) -> None:
    df = pl.DataFrame(
        {"colx": [1, 2, 3], "coly": ["aaa", "bbb", "ccc"], "colz": [0.5, 0.0, -1.0]}
    )
    xls = BytesIO()
    df.write_excel(xls, worksheet="data", include_header=False)

    xldf = pl.read_excel(xls, engine=engine, has_header=False)
    expected = xldf.rename({"column_1": "colx", "column_2": "coly", "column_3": "colz"})
    assert_frame_equal(df, expected)


@pytest.mark.parametrize("engine", ["xlsx2csv", "openpyxl", "calamine"])
def test_excel_write_sparklines(engine: ExcelSpreadsheetEngine) -> None:
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
    ).cast(dtypes={pl.Int64: pl.Float64})

    # also: confirm that we can use a Workbook directly with "write_excel"
    xls = BytesIO()
    with Workbook(xls) as wb:
        df.write_excel(
            workbook=wb,
            worksheet="frame_data",
            table_style="Table Style Light 2",
            dtype_formats={frozenset(NUMERIC_DTYPES): "#,##0_);(#,##0)"},
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
        # ignore an openpyxl user warning about sparklines
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
        assert set(xldf[sparkline_col]) in ({None}, {""})

    assert xldf.columns == ["id", "+/-", "q1", "q2", "q3", "q4", "trend", "h1", "h2"]
    assert_frame_equal(
        df, xldf.drop("+/-", "trend", "h1", "h2").cast(dtypes={pl.Int64: pl.Float64})
    )


def test_excel_write_multiple_tables() -> None:
    from xlsxwriter import Workbook

    # note: also checks that empty tables don't error on write
    df = pl.DataFrame(schema={"colx": pl.Date, "coly": pl.String, "colz": pl.Float64})

    # write multiple frames to multiple worksheets
    xls = BytesIO()
    with Workbook(xls) as wb:
        df.write_excel(workbook=wb, worksheet="sheet1", position="A1")
        df.write_excel(workbook=wb, worksheet="sheet1", position="A6")
        df.write_excel(workbook=wb, worksheet="sheet2", position="A1")

        # validate integration of externally-added formats
        fmt = wb.add_format({"bg_color": "#ffff00"})
        df.write_excel(
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


def test_excel_write_worksheet_object() -> None:
    # write to worksheet object
    from xlsxwriter import Workbook

    df = pl.DataFrame({"colx": ["aaa", "bbb", "ccc"], "coly": [-1234, 0, 5678]})

    with Workbook(xls := BytesIO()) as wb:
        ws = wb.add_worksheet("frame_data")
        df.write_excel(wb, worksheet=ws)
        ws.hide_zero()

    assert_frame_equal(df, pl.read_excel(xls, sheet_name="frame_data"))

    with pytest.raises(  # noqa: SIM117
        ValueError,
        match="the given workbook object .* is not the parent of worksheet 'frame_data'",
    ):
        with Workbook(BytesIO()) as wb:
            df.write_excel(wb, worksheet=ws)

    with pytest.raises(  # noqa: SIM117
        TypeError,
        match="worksheet object requires the parent workbook object; found workbook=None",
    ):
        with Workbook(BytesIO()) as wb:
            df.write_excel(None, worksheet=ws)


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
        (pl.read_excel, "path_xls_empty"),
        (pl.read_ods, "path_ods_empty"),
    ],
)
def test_excel_empty_sheet(
    read_spreadsheet: Callable[..., pl.DataFrame],
    source: str,
    request: pytest.FixtureRequest,
) -> None:
    ods = (empty_spreadsheet_path := request.getfixturevalue(source)).suffix == ".ods"
    read_spreadsheet = pl.read_ods if ods else pl.read_excel  # type: ignore[assignment]

    with pytest.raises(NoDataError, match="empty Excel sheet"):
        read_spreadsheet(empty_spreadsheet_path)

    engine_params = [{}] if ods else [{"engine": "calamine"}]
    for params in engine_params:
        df = read_spreadsheet(
            empty_spreadsheet_path,
            sheet_name="no_data",
            raise_if_empty=False,
            **params,
        )
        expected = pl.DataFrame()
        assert_frame_equal(df, expected)

        df = read_spreadsheet(
            empty_spreadsheet_path,
            sheet_name="no_rows",
            raise_if_empty=False,
            **params,
        )
        expected = pl.DataFrame(schema={f"col{c}": pl.String for c in ("x", "y", "z")})
        assert_frame_equal(df, expected)


@pytest.mark.parametrize(
    ("engine", "hidden_columns"),
    [
        ("xlsx2csv", ["a"]),
        ("openpyxl", ["a", "b"]),
        ("calamine", ["a", "b"]),
        ("xlsx2csv", cs.numeric()),
        ("openpyxl", cs.last()),
    ],
)
def test_excel_hidden_columns(
    hidden_columns: list[str] | SelectorType,
    engine: ExcelSpreadsheetEngine,
) -> None:
    df = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})

    xls = BytesIO()
    df.write_excel(xls, hidden_columns=hidden_columns)

    read_df = pl.read_excel(xls)
    assert_frame_equal(df, read_df)


def test_excel_mixed_calamine_float_data(io_files_path: Path) -> None:
    df = pl.read_excel(io_files_path / "nan_test.xlsx", engine="calamine")
    nan = float("nan")
    assert_frame_equal(
        pl.DataFrame({"float_col": [nan, nan, nan, 100.0, 200.0, 300.0]}),
        df,
    )


@pytest.mark.parametrize("engine", ["xlsx2csv", "openpyxl", "calamine"])
def test_excel_type_inference_with_nulls(engine: ExcelSpreadsheetEngine) -> None:
    df = pl.DataFrame(
        {
            "a": [1, 2, None],
            "b": [1.0, None, 3.5],
            "c": ["x", None, "z"],
            "d": [True, False, None],
            "e": [date(2023, 1, 1), None, date(2023, 1, 4)],
            "f": [
                datetime(2023, 1, 1),
                datetime(2000, 10, 10, 10, 10),
                None,
            ],
        }
    )
    xls = BytesIO()
    df.write_excel(xls)

    reversed_cols = list(reversed(df.columns))
    read_cols: Sequence[str] | Sequence[int]
    for read_cols in (
        reversed_cols,
        [5, 4, 3, 2, 1, 0],
    ):
        read_df = pl.read_excel(
            xls,
            engine=engine,
            columns=read_cols,
            schema_overrides={
                "e": pl.Date,
                "f": pl.Datetime("us"),
            },
        )
        assert_frame_equal(df.select(reversed_cols), read_df)


@pytest.mark.parametrize(
    ("path", "file_type"),
    [
        ("path_xls", "xls"),
        ("path_xlsx", "xlsx"),
        ("path_xlsb", "xlsb"),
    ],
)
def test_identify_workbook(
    path: str, file_type: str, request: pytest.FixtureRequest
) -> None:
    # identify from file path
    spreadsheet_path = request.getfixturevalue(path)
    assert _identify_workbook(spreadsheet_path) == file_type

    # note that we can't distinguish between xlsx and xlsb
    # from the magic bytes block alone (so we default to xlsx)
    if file_type == "xlsb":
        file_type = "xlsx"

    # identify from IO[bytes]
    with Path.open(spreadsheet_path, "rb") as f:
        assert _identify_workbook(f) == file_type
        assert isinstance(pl.read_excel(f, engine="calamine"), pl.DataFrame)

    # identify from bytes
    with Path.open(spreadsheet_path, "rb") as f:
        raw_data = f.read()
        assert _identify_workbook(raw_data) == file_type
        assert isinstance(pl.read_excel(raw_data, engine="calamine"), pl.DataFrame)

    # identify from BytesIO
    with Path.open(spreadsheet_path, "rb") as f:
        bytesio_data = BytesIO(f.read())
        assert _identify_workbook(bytesio_data) == file_type
        assert isinstance(pl.read_excel(bytesio_data, engine="calamine"), pl.DataFrame)
