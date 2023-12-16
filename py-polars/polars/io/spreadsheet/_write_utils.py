from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Sequence, overload

from polars import functions as F
from polars.datatypes import (
    FLOAT_DTYPES,
    INTEGER_DTYPES,
    Date,
    Datetime,
    Float64,
    List,
    Object,
    Struct,
    Time,
)
from polars.dependencies import json
from polars.exceptions import DuplicateError
from polars.selectors import _expand_selector_dicts, _expand_selectors

if TYPE_CHECKING:
    from typing import Literal

    from xlsxwriter import Workbook
    from xlsxwriter.format import Format
    from xlsxwriter.worksheet import Worksheet

    from polars import DataFrame, Series
    from polars.type_aliases import (
        ColumnFormatDict,
        ColumnTotalsDefinition,
        ConditionalFormatDict,
        OneOrMoreDataTypes,
        PolarsDataType,
        RowTotalsDefinition,
    )


def _cluster(iterable: Iterable[Any], n: int = 2) -> Iterable[Any]:
    return zip(*[iter(iterable)] * n)


_XL_DEFAULT_FLOAT_FORMAT_ = "#,##0.000;[Red]-#,##0.000"
_XL_DEFAULT_INTEGER_FORMAT_ = "#,##0;[Red]-#,##0"
_XL_DEFAULT_DTYPE_FORMATS_: dict[PolarsDataType, str] = {
    Datetime: "yyyy-mm-dd hh:mm:ss",
    Date: "yyyy-mm-dd;@",
    Time: "hh:mm:ss;@",
}
for tp in INTEGER_DTYPES:
    _XL_DEFAULT_DTYPE_FORMATS_[tp] = _XL_DEFAULT_INTEGER_FORMAT_


class _XLFormatCache:
    """Create/cache only one Format object per distinct set of format options."""

    def __init__(self, wb: Workbook):
        self._cache: dict[str, Format] = {}
        self.wb = wb

    @staticmethod
    def _key(fmt: dict[str, Any]) -> str:
        return json.dumps(fmt, sort_keys=True, default=str)

    def get(self, fmt: dict[str, Any] | Format) -> Format:
        if not isinstance(fmt, dict):
            wbfmt = fmt
        else:
            key = self._key(fmt)
            wbfmt = self._cache.get(key)
            if wbfmt is None:
                wbfmt = self.wb.add_format(fmt)
                self._cache[key] = wbfmt
        return wbfmt


def _adjacent_cols(df: DataFrame, cols: Iterable[str], min_max: dict[str, Any]) -> bool:
    """Indicate if the given columns are all adjacent to one another."""
    idxs = sorted(df.get_column_index(col) for col in cols)
    if idxs != sorted(range(min(idxs), max(idxs) + 1)):
        return False
    else:
        columns = df.columns
        min_max["min"] = {"idx": idxs[0], "name": columns[idxs[0]]}
        min_max["max"] = {"idx": idxs[-1], "name": columns[idxs[-1]]}
        return True


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
    ws: Worksheet,
    *,
    conditional_formats: ConditionalFormatDict,
    table_start: tuple[int, int],
    include_header: bool,
    format_cache: _XLFormatCache,
) -> None:
    """Take all conditional formatting options and apply them to the table/range."""
    from xlsxwriter.format import Format

    for cols, formats in _expand_selector_dicts(
        df, conditional_formats, expand_keys=True, expand_values=False, tuple_keys=True
    ).items():
        if not isinstance(cols, str) and len(cols) == 1:
            cols = next(iter(cols))
        if isinstance(formats, (str, dict)):
            formats = [formats]

        for fmt in formats:
            if not isinstance(fmt, dict):
                fmt = {"type": fmt}
            if isinstance(cols, str):
                col_range = _xl_column_range(
                    df, table_start, cols, include_header=include_header
                )
            else:
                col_range = _xl_column_multi_range(
                    df, table_start, cols, include_header=include_header
                )
                if " " in col_range:
                    col = next(iter(cols))
                    fmt["multi_range"] = col_range
                    col_range = _xl_column_range(
                        df, table_start, col, include_header=include_header
                    )

            if "format" in fmt:
                f = fmt["format"]
                fmt["format"] = (
                    f  # already registered
                    if isinstance(f, Format)
                    else format_cache.get(
                        {"num_format": f} if isinstance(f, str) else f
                    )
                )
            ws.conditional_format(col_range, fmt)


@overload
def _xl_column_range(
    df: DataFrame,
    table_start: tuple[int, int],
    col: str | tuple[int, int],
    *,
    include_header: bool,
    as_range: Literal[True] = ...,
) -> str:
    ...


@overload
def _xl_column_range(
    df: DataFrame,
    table_start: tuple[int, int],
    col: str | tuple[int, int],
    *,
    include_header: bool,
    as_range: Literal[False],
) -> tuple[int, int, int, int]:
    ...


def _xl_column_range(
    df: DataFrame,
    table_start: tuple[int, int],
    col: str | tuple[int, int],
    *,
    include_header: bool,
    as_range: bool = True,
) -> tuple[int, int, int, int] | str:
    """Return the excel sheet range of a named column, accounting for all offsets."""
    col_start = (
        table_start[0] + int(include_header),
        table_start[1] + df.get_column_index(col) if isinstance(col, str) else col[0],
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
    *,
    include_header: bool,
) -> str:
    """Return column ranges as an xlsxwriter 'multi_range' string, or spanning range."""
    m: dict[str, Any] = {}
    if _adjacent_cols(df, cols, min_max=m):
        return _xl_column_range(
            df,
            table_start,
            (m["min"]["idx"], m["max"]["idx"]),
            include_header=include_header,
        )
    return " ".join(
        _xl_column_range(df, table_start, col, include_header=include_header)
        for col in cols
    )


def _xl_inject_dummy_table_columns(
    df: DataFrame, options: dict[str, Any], dtype: PolarsDataType | None = None
) -> DataFrame:
    """Insert dummy frame columns in order to create empty/named table columns."""
    df_original_columns = set(df.columns)
    df_select_cols = df.columns.copy()
    cast_lookup = {}

    for col, definition in options.items():
        if col in df_original_columns:
            raise DuplicateError(f"cannot create a second {col!r} column")
        elif not isinstance(definition, dict):
            df_select_cols.append(col)
        else:
            cast_lookup[col] = definition.get("return_dtype")
            insert_before = definition.get("insert_before")
            insert_after = definition.get("insert_after")

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
            (
                col
                if col in df_original_columns
                else (
                    F.lit(None).cast(
                        cast_lookup.get(col, dtype)  # type:ignore[arg-type]
                    )
                    if dtype or (col in cast_lookup and cast_lookup[col] is not None)
                    else F.lit(None)
                ).alias(col)
            )
            for col in df_select_cols
        ]
    )
    return df


def _xl_inject_sparklines(
    ws: Worksheet,
    df: DataFrame,
    table_start: tuple[int, int],
    col: str,
    *,
    include_header: bool,
    params: Sequence[str] | dict[str, Any],
) -> None:
    """Inject sparklines into (previously-created) empty table columns."""
    from xlsxwriter.utility import xl_rowcol_to_cell

    m: dict[str, Any] = {}
    data_cols = params.get("columns") if isinstance(params, dict) else params
    if not data_cols:
        raise ValueError("supplying 'columns' param value is mandatory for sparklines")
    elif not _adjacent_cols(df, data_cols, min_max=m):
        raise RuntimeError("sparkline data range/cols must all be adjacent")

    spk_row, spk_col, _, _ = _xl_column_range(
        df, table_start, col, include_header=include_header, as_range=False
    )
    data_start_col = table_start[1] + m["min"]["idx"]
    data_end_col = table_start[1] + m["max"]["idx"]

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
    format_cache: _XLFormatCache,
    column_totals: ColumnTotalsDefinition | None = None,
    column_formats: ColumnFormatDict | None = None,
    dtype_formats: dict[OneOrMoreDataTypes, str] | None = None,
    header_format: dict[str, Any] | None = None,
    sparklines: dict[str, Sequence[str] | dict[str, Any]] | None = None,
    formulas: dict[str, str | dict[str, str]] | None = None,
    row_totals: RowTotalsDefinition | None = None,
    float_precision: int = 3,
) -> tuple[list[dict[str, Any]], dict[str | tuple[str, ...], str], DataFrame]:
    """Setup and unify all column-related formatting/defaults."""

    # no excel support for compound types; cast to their simple string representation
    def _map_str(s: Series) -> Series:
        return s.__class__(s.name, [str(v) for v in s.to_list()])

    cast_cols = [
        F.col(col).map_batches(_map_str).alias(col)
        for col, tp in df.schema.items()
        if tp in (List, Struct, Object)
    ]
    if cast_cols:
        df = df.with_columns(cast_cols)

    column_totals = _unpack_multi_column_dict(  # type: ignore[assignment]
        _expand_selector_dicts(df, column_totals, expand_keys=True, expand_values=False)
        if isinstance(column_totals, dict)
        else _expand_selectors(df, column_totals)
    )
    column_formats = _unpack_multi_column_dict(  # type: ignore[assignment]
        _expand_selector_dicts(
            df, column_formats, expand_keys=True, expand_values=False, tuple_keys=True
        )
    )

    # normalise column totals
    column_total_funcs = (
        {col: "sum" for col in column_totals}
        if isinstance(column_totals, Sequence)
        else (column_totals.copy() if isinstance(column_totals, dict) else {})
    )

    # normalise row totals
    if not row_totals:
        row_total_funcs = {}
    else:
        numeric_cols = {col for col, tp in df.schema.items() if tp.is_numeric()}
        if not isinstance(row_totals, dict):
            sum_cols = (
                numeric_cols
                if row_totals is True
                else (
                    {row_totals}
                    if isinstance(row_totals, str)
                    else set(_expand_selectors(df, row_totals))
                )
            )
            n_ucase = sum((c[0] if c else "").isupper() for c in df.columns)
            total = f"{'T' if (n_ucase > len(df.columns) // 2) else 't'}otal"
            row_total_funcs = {total: _xl_table_formula(df, sum_cols, "sum")}
        else:
            row_totals = _expand_selector_dicts(
                df, row_totals, expand_keys=False, expand_values=True
            )
            row_total_funcs = {
                name: _xl_table_formula(
                    df, numeric_cols if cols is True else cols, "sum"
                )
                for name, cols in row_totals.items()
            }

    # normalise formulas
    column_formulas = {
        col: {"formula": options} if isinstance(options, str) else options
        for col, options in (formulas or {}).items()
    }

    # normalise formats
    column_formats = dict(column_formats or {})
    dtype_formats = dict(dtype_formats or {})

    for tp in list(dtype_formats):
        if isinstance(tp, (tuple, frozenset)):
            dtype_formats.update(dict.fromkeys(tp, dtype_formats.pop(tp)))
    for fmt in dtype_formats.values():
        if not isinstance(fmt, str):
            raise TypeError(
                f"invalid dtype_format value: {fmt!r} (expected format string, got {type(fmt).__name__!r})"
            )

    # inject sparkline/row-total placeholder(s)
    if sparklines:
        df = _xl_inject_dummy_table_columns(df, sparklines)
    if column_formulas:
        df = _xl_inject_dummy_table_columns(df, column_formulas)
    if row_totals:
        df = _xl_inject_dummy_table_columns(df, row_total_funcs, dtype=Float64)

    # seed format cache with default fallback format
    fmt_default = format_cache.get({"valign": "vcenter"})

    # default float format
    zeros = "0" * float_precision
    fmt_float = (
        _XL_DEFAULT_INTEGER_FORMAT_
        if not zeros
        else _XL_DEFAULT_FLOAT_FORMAT_.replace(".000", f".{zeros}")
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
        if base_type.is_numeric():
            if column_totals is True:
                column_total_funcs.setdefault(col, "sum")
            elif isinstance(column_totals, str):
                column_total_funcs.setdefault(col, column_totals.lower())
        if col not in column_formats:
            column_formats[col] = fmt_default

    # ensure externally supplied formats are made available
    for col, fmt in column_formats.items():  # type: ignore[assignment]
        if isinstance(fmt, str):
            column_formats[col] = format_cache.get(
                {"num_format": fmt, "valign": "vcenter"}
            )
        elif isinstance(fmt, dict):
            if "num_format" not in fmt:
                tp = df.schema.get(col)
                if tp in dtype_formats:
                    fmt["num_format"] = dtype_formats[tp]
            if "valign" not in fmt:
                fmt["valign"] = "vcenter"
            column_formats[col] = format_cache.get(fmt)

    # optional custom header format
    col_header_format = format_cache.get(header_format) if header_format else None

    # assemble table columns
    table_columns = [
        {
            k: v
            for k, v in {
                "header": col,
                "format": column_formats[col],
                "header_format": col_header_format,
                "total_function": column_total_funcs.get(col),
                "formula": (
                    row_total_funcs.get(col)
                    or column_formulas.get(col, {}).get("formula")
                ),
            }.items()
            if v is not None
        }
        for col in df.columns
    ]
    return table_columns, column_formats, df  # type: ignore[return-value]


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
                raise ValueError(f"invalid table style key: {key!r}")

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


def _xl_table_formula(df: DataFrame, cols: Iterable[str], func: str) -> str:
    """Return a formula using structured references to columns in a named table."""
    m: dict[str, Any] = {}
    if isinstance(cols, str):
        cols = [cols]
    if _adjacent_cols(df, cols, min_max=m):
        return f"={func.upper()}([@[{m['min']['name']}]:[{m['max']['name']}]])"
    else:
        colrefs = ",".join(f"[@[{c}]]" for c in cols)
        return f"={func.upper()}({colrefs})"


def _xl_unique_table_name(wb: Workbook) -> str:
    """Establish a unique (per-workbook) table object name."""
    table_prefix = "Frame"
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
