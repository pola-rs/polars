from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Sequence,
)

from polars.datatypes import (
    FLOAT_DTYPES,
    INTEGER_DTYPES,
    NUMERIC_DTYPES,
    Date,
    Datetime,
    Time,
)

if TYPE_CHECKING:
    from xlsxwriter import Workbook
    from xlsxwriter.worksheet import Worksheet

    import polars.internals as pli
    from polars.datatypes import OneOrMoreDataTypes, PolarsDataType


_XL_DEFAULT_FLOAT_FORMAT_ = "#,##0.{zeros};[Red]-#,##0.{zeros}"
_XL_DEFAULT_INTEGER_FORMAT_ = "#,##0;[Red]-#,##0"
_XL_DEFAULT_DTYPE_FORMATS_: dict[PolarsDataType, str] = {
    Datetime: "yyyy-mm-dd hh:mm:ss",
    Date: "yyyy-mm-dd",
    Time: "hh:mm:ss",
}
for tp in INTEGER_DTYPES:
    _XL_DEFAULT_DTYPE_FORMATS_[tp] = _XL_DEFAULT_INTEGER_FORMAT_


def _xl_setup_workbook(
    workbook: Workbook | BytesIO | Path | str | None, worksheet: str | None = None
) -> tuple[Workbook, Worksheet, bool]:
    """Establish the target excel workbook and worksheet."""
    from xlsxwriter import Workbook

    if isinstance(workbook, Workbook):
        wb, can_close = workbook, False
        ws = wb.get_worksheet_by_name(name=worksheet)
    else:
        workbook_options = {
            "nan_inf_to_errors": True,
            "strings_to_formulas": False,
            "default_date_format": _XL_DEFAULT_DTYPE_FORMATS_[Date],
        }
        if isinstance(workbook, BytesIO):
            wb, ws, can_close = Workbook(workbook, workbook_options), None, True
        else:
            file = Path("dataframe.xlsx" if workbook is None else workbook)
            wb = Workbook(
                (file if file.suffix else file.with_suffix(".xlsx"))
                .expanduser()
                .resolve(strict=False),
                workbook_options,
            )
            ws, can_close = None, True

    if ws is None:
        ws = wb.add_worksheet(name=worksheet)
    return wb, ws, can_close


def _xl_setup_table_columns(
    df: pli.DataFrame,
    wb: Workbook,
    column_formats: dict[str, str] | None = None,
    column_totals: dict[str, str] | Sequence[str] | bool | None = None,
    dtype_formats: dict[OneOrMoreDataTypes, str] | None = None,
    float_precision: int = 3,
) -> list[dict[str, Any]]:
    """Setup and unify all column-related formatting/defaults."""
    total_funcs = (
        {col: "sum" for col in column_totals}
        if isinstance(column_totals, Sequence)
        else (column_totals.copy() if isinstance(column_totals, dict) else {})
    )
    column_formats = (column_formats or {}).copy()
    dtype_formats = (dtype_formats or {}).copy()
    for tp, _fmt in list(dtype_formats.items()):
        if isinstance(tp, (tuple, frozenset)):
            dtype_formats.update(dict.fromkeys(tp, dtype_formats.pop(tp)))

    # default float format
    zeros = "0" * float_precision
    fmt_float = (
        _XL_DEFAULT_INTEGER_FORMAT_
        if not zeros
        else _XL_DEFAULT_FLOAT_FORMAT_.format(zeros=zeros)
    )

    # assign default dtype formats
    for tp, fmt in _XL_DEFAULT_DTYPE_FORMATS_.items():
        dtype_formats.setdefault(tp, fmt)
    for tp in FLOAT_DTYPES:
        dtype_formats.setdefault(tp, fmt_float)
    for tp, fmt in dtype_formats.items():
        dtype_formats[tp] = fmt

    # associate formats/functions with specific columns
    for col, tp in df.schema.items():
        base_type = tp.base_type()
        if base_type in dtype_formats:
            fmt = dtype_formats.get(tp, dtype_formats[base_type])
            column_formats.setdefault(col, fmt)
        if base_type in NUMERIC_DTYPES:
            if column_totals is True:
                total_funcs.setdefault(col, "sum")

    # ensure externally supplied formats are made available
    for col, fmt in column_formats.items():
        if isinstance(fmt, str):
            column_formats[col] = wb.add_format({"num_format": fmt})
        elif isinstance(fmt, dict):
            if "num_format" not in fmt:
                tp = df.schema.get(col)
                if tp in dtype_formats:
                    fmt["num_format"] = dtype_formats[tp]
            column_formats[col] = wb.add_format(fmt)

    return [
        {
            "header": col,
            "format": column_formats.get(col),
            "total_function": total_funcs.get(col),
        }
        for col in df.columns
    ]


def _xl_setup_table_options(
    table_style: dict[str, Any] | str | None
) -> tuple[dict[str, Any] | str | None, dict[str, Any]]:
    """Setup table options, distinguishing style name from other formatting."""
    if isinstance(table_style, dict):
        valid_options = (
            "style",
            "banded_columns",
            "banded_rows",
            "first_column",
            "last_column",
        )
        for key in table_style:
            if key not in valid_options:
                raise ValueError(f"Invalid table style key:{key}")

        table_options = table_style.copy()
        table_style = table_options.pop("style", None)
    else:
        table_options = {}

    return table_style, table_options


def _xl_unique_table_name(ws: Worksheet) -> str:
    """Establish a unique (per-worksheet) incrementing table object name."""
    default_name = "PolarsFrameTable"
    n_polars_tables = sum(1 for t in ws.tables if t["name"].startswith(default_name))
    return f"{default_name}{n_polars_tables}"


def _xl_column_range(
    df: pli.DataFrame, table_start: tuple[int, int], col: str, has_header: bool
) -> tuple[int, int, int, int]:
    """Return the excel sheet range of a named column, accounting for all offsets."""
    col_start = (
        table_start[0] + int(has_header),
        table_start[1] + df.find_idx_by_name(col),
    )
    col_finish = (col_start[0] + len(df) - 1, col_start[1])
    return col_start + col_finish
