from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    Sequence,
    Tuple,
    Union,
    overload,
)

from polars import internals as pli
from polars.datatypes import (
    FLOAT_DTYPES,
    INTEGER_DTYPES,
    NUMERIC_DTYPES,
    Date,
    Datetime,
    Time,
)
from polars.exceptions import DuplicateError


def _cluster(iterable: Iterable[Any], n: int = 2) -> Iterable[Any]:
    return zip(*[iter(iterable)] * n)


if TYPE_CHECKING:
    import sys

    from xlsxwriter import Workbook
    from xlsxwriter.worksheet import Worksheet

    from polars.dataframe.frame import DataFrame
    from polars.type_aliases import OneOrMoreDataTypes, PolarsDataType

    if sys.version_info >= (3, 10):
        from typing import Literal, TypeAlias
    else:
        from typing_extensions import Literal, TypeAlias


_XL_DEFAULT_FLOAT_FORMAT_ = "#,##0.{zeros};[Red]-#,##0.{zeros}"
_XL_DEFAULT_INTEGER_FORMAT_ = "#,##0;[Red]-#,##0"
_XL_DEFAULT_DTYPE_FORMATS_: dict[PolarsDataType, str] = {
    Datetime: "yyyy-mm-dd hh:mm:ss",
    Date: "yyyy-mm-dd;@",
    Time: "hh:mm:ss;@",
}
for tp in INTEGER_DTYPES:
    _XL_DEFAULT_DTYPE_FORMATS_[tp] = _XL_DEFAULT_INTEGER_FORMAT_


ColumnTotalsDefinition: TypeAlias = Union[
    # dict of colname(s) to str, a sequence of str, or a boolean
    Dict[Union[str, Tuple[str, ...]], str],
    Sequence[str],
    bool,
]
ConditionalFormatDict: TypeAlias = Dict[
    # dict of colname(s) to str, dict, or sequence of str/dict
    Union[str, Tuple[str, ...]],
    Union[str, Union[Dict[str, Any], Sequence[Union[str, Dict[str, Any]]]]],
]


def _adjacent_cols(df: DataFrame, cols: Iterable[str]) -> bool:
    """Indicate if the given columns are all adjacent to one another."""
    idxs = sorted(df.find_idx_by_name(col) for col in cols)
    return idxs == sorted(range(min(idxs), max(idxs) + 1))


def _unpack_multi_column_dict(
    d: dict[str | Sequence[str], Any] | Any
) -> dict[str, Any] | Any:
    """Unpack multi-col dictionary into equivalent single-col definitions."""
    if not isinstance(d, dict):
        return d
    unpacked: dict[str, Any] = {}
    for key, value in d.items():
        if isinstance(key, str) or not isinstance(key, Sequence):
            key = (key,)
        for k in key:
            unpacked[k] = value
    return unpacked


def _xl_apply_conditional_formats(
    df: DataFrame,
    workbook: Workbook,
    worksheet: Worksheet,
    conditional_formats: ConditionalFormatDict,
    table_start: tuple[int, int],
    has_header: bool,
) -> None:
    """Take all conditional formatting options and apply them to the table/range."""
    from xlsxwriter.format import Format

    for cols, formats in conditional_formats.items():
        if not isinstance(cols, str) and len(cols) == 1:
            cols = cols[0]
        if isinstance(formats, (str, dict)):
            formats = [formats]

        for fmt in formats:
            if not isinstance(fmt, dict):
                fmt = {"type": fmt}
            if isinstance(cols, str):
                col_range = _xl_column_range(df, table_start, cols, has_header)
            else:
                col_range = _xl_column_multi_range(df, table_start, cols, has_header)
                if " " in col_range:
                    col = cols[0]
                    fmt["multi_range"] = col_range
                    col_range = _xl_column_range(df, table_start, col, has_header)

            if "format" in fmt:
                f = fmt["format"]
                fmt["format"] = (
                    f  # already registered
                    if isinstance(f, Format)
                    else workbook.add_format(
                        {"num_format": f} if isinstance(f, str) else f
                    )
                )
            worksheet.conditional_format(col_range, fmt)


@overload
def _xl_column_range(  # type: ignore[misc]
    df: DataFrame,
    table_start: tuple[int, int],
    col: str | tuple[int, int],
    has_header: bool,
    as_range: Literal[True] = ...,
) -> str:
    ...


@overload
def _xl_column_range(
    df: DataFrame,
    table_start: tuple[int, int],
    col: str | tuple[int, int],
    has_header: bool,
    as_range: Literal[False] = ...,
) -> tuple[int, int, int, int]:
    ...


def _xl_column_range(
    df: DataFrame,
    table_start: tuple[int, int],
    col: str | tuple[int, int],
    has_header: bool,
    as_range: bool = True,
) -> tuple[int, int, int, int] | str:
    """Return the excel sheet range of a named column, accounting for all offsets."""
    col_start = (
        table_start[0] + int(has_header),
        table_start[1] + df.find_idx_by_name(col) if isinstance(col, str) else col[0],
    )
    col_finish = (
        col_start[0] + len(df) - 1,
        col_start[1] + 0 if isinstance(col, str) else (col[1] - col[0]),
    )
    if as_range:
        return "".join(_xl_rowcols_to_range(*col_start, *col_finish))
    else:
        return col_start + col_finish


def _xl_column_multi_range(
    df: DataFrame,
    table_start: tuple[int, int],
    cols: Iterable[str],
    has_header: bool,
) -> str:
    """Return column ranges as an xlsxwriter 'multi_range' string, or spanning range."""
    if _adjacent_cols(df, cols):
        col_idxs = sorted(df.find_idx_by_name(c) for c in cols)
        return _xl_column_range(
            df, table_start, (col_idxs[0], col_idxs[-1]), has_header
        )
    return " ".join(_xl_column_range(df, table_start, col, has_header) for col in cols)


def _xl_inject_dummy_table_columns(
    df: DataFrame, options: dict[str, Sequence[str] | dict[str, Any]]
) -> DataFrame:
    """Insert dummy frame columns in order to create empty/named table columns."""
    df_original_columns = set(df.columns)
    df_select_cols = df.columns.copy()

    for col, definition in options.items():
        if col in df_original_columns:
            raise DuplicateError(f"Cannot create a second {col!r} column")
        elif not isinstance(definition, dict):
            df_select_cols.append(col)
        else:
            insert_after = definition.get("insert_after")
            insert_before = definition.get("insert_before")
            if insert_after is None and insert_before is None:
                df_select_cols.append(col)
            else:
                insert_idx = (
                    df_select_cols.index(insert_after) + 1  # type: ignore[arg-type]
                    if insert_before is None
                    else df_select_cols.index(insert_before)
                )
                df_select_cols.insert(insert_idx, col)

    df = df.select(
        [
            (col if col in df_original_columns else pli.lit(None).alias(col))
            for col in df_select_cols
        ]
    )
    return df


def _xl_inject_sparklines(
    ws: Worksheet,
    df: DataFrame,
    table_start: tuple[int, int],
    col: str,
    has_header: bool,
    params: Sequence[str] | dict[str, Any],
) -> None:
    """Inject sparklines into (previously-created) empty table columns."""
    from xlsxwriter.utility import xl_rowcol_to_cell

    data_cols = params.get("columns") if isinstance(params, dict) else params
    if not data_cols:
        raise ValueError("Supplying 'columns' param value is mandatory for sparklines")
    if not _adjacent_cols(df, data_cols):
        raise RuntimeError("sparkline data range/cols must all be adjacent")

    spk_row, spk_col, _, _ = _xl_column_range(
        df, table_start, col, has_header, as_range=False
    )
    data_idxs = sorted(df.find_idx_by_name(col) for col in data_cols)
    data_start_col = table_start[1] + data_idxs[0]
    data_end_col = table_start[1] + data_idxs[-1]

    if not isinstance(params, dict):
        options = {}
    else:
        # strip polars-specific params before passing to xlsxwriter
        options = {
            name: val
            for name, val in params.items()
            if name not in ("columns", "insert_after", "insert_before")
        }
        if "negative_points" not in options:
            options["negative_points"] = options.get("type") in ("column", "win_loss")

    for _ in range(len(df)):
        data_start = xl_rowcol_to_cell(spk_row, data_start_col)
        data_end = xl_rowcol_to_cell(spk_row, data_end_col)
        options["range"] = f"{data_start}:{data_end}"
        ws.add_sparkline(spk_row, spk_col, options)
        spk_row += 1


def _xl_rowcols_to_range(*row_col_pairs: int) -> list[str]:
    """Return list of "A1:B2" range refs from pairs of row/col indexes."""
    from xlsxwriter.utility import xl_rowcol_to_cell

    cell_refs = (xl_rowcol_to_cell(row, col) for row, col in _cluster(row_col_pairs))
    return [f"{cell_start}:{cell_end}" for cell_start, cell_end in _cluster(cell_refs)]


def _xl_setup_table_columns(
    df: DataFrame,
    wb: Workbook,
    column_totals: ColumnTotalsDefinition | None = None,
    column_formats: dict[str | tuple[str, ...], str] | None = None,
    dtype_formats: dict[OneOrMoreDataTypes, str] | None = None,
    sparklines: dict[str, Sequence[str] | dict[str, Any]] | None = None,
    float_precision: int = 3,
) -> tuple[list[dict[str, Any]], DataFrame]:
    """Setup and unify all column-related formatting/defaults."""
    column_totals = _unpack_multi_column_dict(column_totals)  # type: ignore[assignment]
    column_formats = _unpack_multi_column_dict(column_formats)  # type: ignore[assignment]

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

    # inject sparkline placeholder(s)
    if sparklines:
        df = _xl_inject_dummy_table_columns(df, sparklines)

    # default float/fallback format
    fmt_default = wb.add_format({"valign": "vcenter"})
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
                total_funcs.setdefault(col, "sum")  # type: ignore[attr-defined]

    # ensure externally supplied formats are made available
    for col, fmt in column_formats.items():  # type: ignore[assignment]
        if isinstance(fmt, str):
            column_formats[col] = wb.add_format(
                {"num_format": fmt, "valign": "vcenter"}
            )
        elif isinstance(fmt, dict):
            if "num_format" not in fmt:
                tp = df.schema.get(col)
                if tp in dtype_formats:
                    fmt["num_format"] = dtype_formats[tp]
            if "valign" not in fmt:
                fmt["valign"] = "vcenter"
            column_formats[col] = wb.add_format(fmt)

    # assemble table columns
    table_columns = [
        {
            "header": col,
            "format": column_formats.get(col, fmt_default),
            "total_function": total_funcs.get(col),  # type: ignore[attr-defined]
        }
        for col in df.columns
    ]
    return table_columns, df


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


def _xl_unique_table_name(wb: Workbook) -> str:
    """Establish a unique (per-workbook) table object name."""
    table_prefix = "PolarsFrameTable"
    polars_tables: set[str] = set()
    for ws in wb.worksheets():
        polars_tables.update(
            tbl["name"] for tbl in ws.tables if tbl["name"].startswith(table_prefix)
        )
    n = len(polars_tables)
    table_name = f"{table_prefix}{n}"
    while table_name in polars_tables:
        n += 1
        table_name = f"{table_prefix}{n}"
    return table_name
