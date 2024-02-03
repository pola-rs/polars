from __future__ import annotations

import re
from contextlib import nullcontext
from datetime import time
from io import BufferedReader, BytesIO, StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO, Callable, NoReturn, Sequence, overload

import polars._reexport as pl
from polars import functions as F
from polars.datatypes import (
    FLOAT_DTYPES,
    NUMERIC_DTYPES,
    Date,
    Datetime,
    Int64,
    Null,
    String,
)
from polars.dependencies import import_optional
from polars.exceptions import NoDataError, ParameterCollisionError
from polars.io._utils import PortableTemporaryFile, _looks_like_url, _process_file_url
from polars.io.csv.functions import read_csv
from polars.utils.deprecation import deprecate_renamed_parameter
from polars.utils.various import normalize_filepath

if TYPE_CHECKING:
    from typing import Literal

    from polars.type_aliases import ExcelSpreadsheetEngine, SchemaDict


@overload
def read_excel(
    source: str | BytesIO | Path | BinaryIO | bytes,
    *,
    sheet_id: None = ...,
    sheet_name: str,
    engine: ExcelSpreadsheetEngine | None = ...,
    engine_options: dict[str, Any] | None = ...,
    read_options: dict[str, Any] | None = ...,
    schema_overrides: SchemaDict | None = ...,
    raise_if_empty: bool = ...,
) -> pl.DataFrame:
    ...


@overload
def read_excel(
    source: str | BytesIO | Path | BinaryIO | bytes,
    *,
    sheet_id: None = ...,
    sheet_name: None = ...,
    engine: ExcelSpreadsheetEngine | None = ...,
    engine_options: dict[str, Any] | None = ...,
    read_options: dict[str, Any] | None = ...,
    schema_overrides: SchemaDict | None = ...,
    raise_if_empty: bool = ...,
) -> pl.DataFrame:
    ...


@overload
def read_excel(
    source: str | BytesIO | Path | BinaryIO | bytes,
    *,
    sheet_id: int,
    sheet_name: str,
    engine: ExcelSpreadsheetEngine | None = ...,
    engine_options: dict[str, Any] | None = ...,
    read_options: dict[str, Any] | None = ...,
    schema_overrides: SchemaDict | None = ...,
    raise_if_empty: bool = ...,
) -> NoReturn:
    ...


# note: 'ignore' required as mypy thinks that the return value for
# Literal[0] overlaps with the return value for other integers
@overload  # type: ignore[overload-overlap]
def read_excel(
    source: str | BytesIO | Path | BinaryIO | bytes,
    *,
    sheet_id: Literal[0] | Sequence[int],
    sheet_name: None = ...,
    engine: ExcelSpreadsheetEngine | None = ...,
    engine_options: dict[str, Any] | None = ...,
    read_options: dict[str, Any] | None = ...,
    schema_overrides: SchemaDict | None = ...,
    raise_if_empty: bool = ...,
) -> dict[str, pl.DataFrame]:
    ...


@overload
def read_excel(
    source: str | BytesIO | Path | BinaryIO | bytes,
    *,
    sheet_id: int,
    sheet_name: None = ...,
    engine: ExcelSpreadsheetEngine | None = ...,
    engine_options: dict[str, Any] | None = ...,
    read_options: dict[str, Any] | None = ...,
    schema_overrides: SchemaDict | None = ...,
    raise_if_empty: bool = ...,
) -> pl.DataFrame:
    ...


@overload
def read_excel(
    source: str | BytesIO | Path | BinaryIO | bytes,
    *,
    sheet_id: None,
    sheet_name: list[str] | tuple[str],
    engine: ExcelSpreadsheetEngine | None = ...,
    engine_options: dict[str, Any] | None = ...,
    read_options: dict[str, Any] | None = ...,
    schema_overrides: SchemaDict | None = ...,
    raise_if_empty: bool = ...,
) -> dict[str, pl.DataFrame]:
    ...


@deprecate_renamed_parameter("xlsx2csv_options", "engine_options", version="0.20.6")
@deprecate_renamed_parameter("read_csv_options", "read_options", version="0.20.7")
def read_excel(
    source: str | BytesIO | Path | BinaryIO | bytes,
    *,
    sheet_id: int | Sequence[int] | None = None,
    sheet_name: str | list[str] | tuple[str] | None = None,
    engine: ExcelSpreadsheetEngine | None = None,
    engine_options: dict[str, Any] | None = None,
    read_options: dict[str, Any] | None = None,
    schema_overrides: SchemaDict | None = None,
    raise_if_empty: bool = True,
) -> pl.DataFrame | dict[str, pl.DataFrame]:
    """
    Read Excel spreadsheet data into a DataFrame.

    .. versionadded:: 0.20.6
        Added "calamine" fastexcel engine for Excel Workbooks (.xlsx, .xlsb, .xls).
    .. versionadded:: 0.19.4
        Added "pyxlsb" engine for Excel Binary Workbooks (.xlsb).
    .. versionadded:: 0.19.3
        Added "openpyxl" engine, and added `schema_overrides` parameter.

    Parameters
    ----------
    source
        Path to a file or a file-like object (by file-like object, we refer to objects
        that have a `read()` method, such as a file handler (e.g. via builtin `open`
        function) or `BytesIO`).
    sheet_id
        Sheet number(s) to convert (set `0` to load all sheets as DataFrames) and
        return a `{sheetname:frame,}` dict. (Defaults to `1` if neither this nor
        `sheet_name` are specified). Can also take a sequence of sheet numbers.
    sheet_name
        Sheet name(s) to convert; cannot be used in conjunction with `sheet_id`. If more
        than one is given then a `{sheetname:frame,}` dict is returned.
    engine
        Library used to parse the spreadsheet file; currently defaults to "xlsx2csv"
        if not explicitly set.

        * "xlsx2csv": converts the data to an in-memory CSV before using the native
          polars `read_csv` method to parse the result. You can pass `engine_options`
          and `read_options` to refine the conversion.
        * "openpyxl": this engine is significantly slower than `xlsx2csv` but supports
          additional automatic type inference; potentially useful if you are otherwise
          unable to parse your sheet with the (default) `xlsx2csv` engine in
          conjunction with the `schema_overrides` parameter.
        * "pyxlsb": this engine is used for Excel Binary Workbooks (`.xlsb` files).
          Note that you have to use `schema_overrides` to correctly load date/datetime
          columns (or these will be read as floats representing offset Julian values).
        * "calamine": this engine can be used for reading all major types of Excel
          Workbook (`.xlsx`, `.xlsb`, `.xls`) and is *dramatically* faster than the
          other options, using the `fastexcel` module to bind calamine.

    engine_options
        Additional options passed to the underlying engine's primary parsing
        constructor (given below), if supported:

        * "xlsx2csv": `Xlsx2csv`
        * "openpyxl": `load_workbook`
        * "pyxlsb": `open_workbook`
        * "calamine": `n/a`

    read_options
        Extra options passed to the function that reads the sheet data (for example,
        the `read_csv` method if using the "xlsx2csv" engine, to which you could
        pass ``{"infer_schema_length": None}``, or the `load_sheet_by_name` method
        if using the "calamine" engine.
    schema_overrides
        Support type specification or override of one or more columns.
    raise_if_empty
        When there is no data in the sheet,`NoDataError` is raised. If this parameter
        is set to False, an empty DataFrame (with no columns) is returned instead.

    Notes
    -----
    When using the default `xlsx2csv` engine the target Excel sheet is first converted
    to CSV using `xlsx2csv.Xlsx2csv(source).convert()` and then parsed with Polars'
    :func:`read_csv` function. You can pass additional options to `read_options`
    to influence this part of the parsing pipeline.

    Returns
    -------
    DataFrame
        If reading a single sheet.
    dict
        If reading multiple sheets, a "{sheetname: DataFrame, ...}" dict is returned.

    Examples
    --------
    Read the "data" worksheet from an Excel file into a DataFrame.

    >>> pl.read_excel(
    ...     source="test.xlsx",
    ...     sheet_name="data",
    ... )  # doctest: +SKIP

    Read table data from sheet 3 in an Excel workbook as a DataFrame while skipping
    empty lines in the sheet. As sheet 3 does not have a header row and the default
    engine is `xlsx2csv` you can pass the necessary additional settings for this
    to the "read_options" parameter; these will be passed to :func:`read_csv`.

    >>> pl.read_excel(
    ...     source="test.xlsx",
    ...     sheet_id=3,
    ...     engine_options={"skip_empty_lines": True},
    ...     read_options={"has_header": False, "new_columns": ["a", "b", "c"]},
    ... )  # doctest: +SKIP

    If the correct datatypes can't be determined you can use `schema_overrides` and/or
    some of the :func:`read_csv` documentation to see which options you can pass to fix
    this issue. For example `"infer_schema_length": None` can be used to read the
    data twice, once to infer the correct output types and once more to then read the
    data with those types. If the types are known in advance then `schema_overrides`
    is the more efficient option.

    >>> pl.read_excel(
    ...     source="test.xlsx",
    ...     read_options={"infer_schema_length": 1000},
    ...     schema_overrides={"dt": pl.Date},
    ... )  # doctest: +SKIP

    The `openpyxl` package can also be used to parse Excel data; it has slightly
    better default type detection, but is slower than `xlsx2csv`. If you have a sheet
    that is better read using this package you can set the engine as "openpyxl" (if you
    use this engine then `read_options` cannot be set).

    >>> pl.read_excel(
    ...     source="test.xlsx",
    ...     engine="openpyxl",
    ...     schema_overrides={"dt": pl.Datetime, "value": pl.Int32},
    ... )  # doctest: +SKIP
    """
    return _read_spreadsheet(
        sheet_id,
        sheet_name,
        source=source,
        engine=engine,
        engine_options=engine_options,
        read_options=read_options,
        schema_overrides=schema_overrides,
        raise_if_empty=raise_if_empty,
    )


@overload
def read_ods(
    source: str | BytesIO | Path | BinaryIO | bytes,
    *,
    sheet_id: None = ...,
    sheet_name: str,
    schema_overrides: SchemaDict | None = None,
    raise_if_empty: bool = ...,
) -> pl.DataFrame:
    ...


@overload
def read_ods(
    source: str | BytesIO | Path | BinaryIO | bytes,
    *,
    sheet_id: None = ...,
    sheet_name: None = ...,
    schema_overrides: SchemaDict | None = None,
    raise_if_empty: bool = ...,
) -> pl.DataFrame:
    ...


@overload
def read_ods(
    source: str | BytesIO | Path | BinaryIO | bytes,
    *,
    sheet_id: int,
    sheet_name: str,
    schema_overrides: SchemaDict | None = None,
    raise_if_empty: bool = ...,
) -> NoReturn:
    ...


@overload  # type: ignore[overload-overlap]
def read_ods(
    source: str | BytesIO | Path | BinaryIO | bytes,
    *,
    sheet_id: Literal[0] | Sequence[int],
    sheet_name: None = ...,
    schema_overrides: SchemaDict | None = None,
    raise_if_empty: bool = ...,
) -> dict[str, pl.DataFrame]:
    ...


@overload
def read_ods(
    source: str | BytesIO | Path | BinaryIO | bytes,
    *,
    sheet_id: int,
    sheet_name: None = ...,
    schema_overrides: SchemaDict | None = None,
    raise_if_empty: bool = ...,
) -> pl.DataFrame:
    ...


@overload
def read_ods(
    source: str | BytesIO | Path | BinaryIO | bytes,
    *,
    sheet_id: None,
    sheet_name: list[str] | tuple[str],
    schema_overrides: SchemaDict | None = None,
    raise_if_empty: bool = ...,
) -> dict[str, pl.DataFrame]:
    ...


def read_ods(
    source: str | BytesIO | Path | BinaryIO | bytes,
    *,
    sheet_id: int | Sequence[int] | None = None,
    sheet_name: str | list[str] | tuple[str] | None = None,
    schema_overrides: SchemaDict | None = None,
    raise_if_empty: bool = True,
) -> pl.DataFrame | dict[str, pl.DataFrame]:
    """
    Read OpenOffice (ODS) spreadsheet data into a DataFrame.

    Parameters
    ----------
    source
        Path to a file or a file-like object (by file-like object, we refer to objects
        that have a `read()` method, such as a file handler (e.g. via builtin `open`
        function) or `BytesIO`).
    sheet_id
        Sheet number(s) to convert, starting from 1 (set `0` to load *all* worksheets
        as DataFrames) and return a `{sheetname:frame,}` dict. (Defaults to `1` if
        neither this nor `sheet_name` are specified). Can also take a sequence of sheet
        numbers.
    sheet_name
        Sheet name(s) to convert; cannot be used in conjunction with `sheet_id`. If
        more than one is given then a `{sheetname:frame,}` dict is returned.
    schema_overrides
        Support type specification or override of one or more columns.
    raise_if_empty
        When there is no data in the sheet,`NoDataError` is raised. If this parameter
        is set to False, an empty DataFrame (with no columns) is returned instead.

    Returns
    -------
    DataFrame, or a `{sheetname: DataFrame, ...}` dict if reading multiple sheets.

    Examples
    --------
    Read the "data" worksheet from an OpenOffice spreadsheet file into a DataFrame.

    >>> pl.read_ods(
    ...     source="test.ods",
    ...     sheet_name="data",
    ... )  # doctest: +SKIP

    If the correct dtypes can't be determined, use the `schema_overrides` parameter
    to specify them.

    >>> pl.read_ods(
    ...     source="test.ods",
    ...     sheet_id=3,
    ...     schema_overrides={"dt": pl.Date},
    ...     raise_if_empty=False,
    ... )  # doctest: +SKIP
    """
    return _read_spreadsheet(
        sheet_id,
        sheet_name,
        source=source,
        engine="ods",
        engine_options={},
        read_options={},
        schema_overrides=schema_overrides,
        raise_if_empty=raise_if_empty,
    )


def _identify_from_magic_bytes(data: bytes | BinaryIO | BytesIO) -> str | None:
    if isinstance(data, bytes):
        data = BytesIO(data)

    xls_bytes = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"  # excel 97-2004
    xlsx_bytes = b"PK\x03\x04"  # xlsx/openoffice

    initial_position = data.tell()
    try:
        magic_bytes = data.read(8)
        if magic_bytes == xls_bytes:
            return "xls"
        elif magic_bytes[:4] == xlsx_bytes:
            return "xlsx"
        return None
    finally:
        data.seek(initial_position)


def _identify_workbook(wb: str | bytes | Path | BinaryIO | BytesIO) -> str | None:
    """Use file extension (and magic bytes) to identify Workbook type."""
    if not isinstance(wb, (str, Path)):
        # raw binary data (bytesio, etc)
        return _identify_from_magic_bytes(wb)
    else:
        p = Path(wb)
        ext = p.suffix[1:].lower()

        # unambiguous file extensions
        if ext in ("xlsx", "xlsm", "xlsb"):
            return ext
        elif ext[:2] == "od":
            return "ods"

        # check magic bytes to resolve ambiguity (eg: xls/xlsx, or no extension)
        with p.open("rb") as f:
            magic_bytes = BytesIO(f.read(8))
            return _identify_from_magic_bytes(magic_bytes)


def _read_spreadsheet(
    sheet_id: int | Sequence[int] | None,
    sheet_name: str | list[str] | tuple[str] | None,
    source: str | BytesIO | Path | BinaryIO | bytes,
    engine: ExcelSpreadsheetEngine | Literal["ods"] | None,
    engine_options: dict[str, Any] | None = None,
    read_options: dict[str, Any] | None = None,
    schema_overrides: SchemaDict | None = None,
    *,
    raise_if_empty: bool = True,
) -> pl.DataFrame | dict[str, pl.DataFrame]:
    if is_file := isinstance(source, (str, Path)):
        source = normalize_filepath(source)
        if _looks_like_url(source):
            source = _process_file_url(source)

    if engine is None:
        if is_file and str(source).lower().endswith(".ods"):
            # note: engine cannot be 'None' here (if called from read_ods)
            msg = "OpenDocumentSpreadsheet files require use of `read_ods`, not `read_excel`"
            raise ValueError(msg)

        # note: eventually want 'calamine' to be the default for all extensions
        file_type = _identify_workbook(source)
        if file_type == "xlsb":
            engine = "pyxlsb"
        elif file_type == "xls":
            engine = "calamine"
        else:
            engine = "xlsx2csv"

    # establish the reading function, parser, and available worksheets
    reader_fn, parser, worksheets = _initialise_spreadsheet_parser(
        engine, source, engine_options or {}
    )
    try:
        # parse data from the indicated sheet(s)
        sheet_names, return_multi = _get_sheet_names(sheet_id, sheet_name, worksheets)
        parsed_sheets = {
            name: reader_fn(
                parser=parser,
                sheet_name=name,
                schema_overrides=schema_overrides,
                read_options=(read_options or {}),
                raise_if_empty=raise_if_empty,
            )
            for name in sheet_names
        }
    finally:
        if hasattr(parser, "close"):
            parser.close()

    if not parsed_sheets:
        param, value = ("id", sheet_id) if sheet_name is None else ("name", sheet_name)
        msg = f"no matching sheets found when `sheet_{param}` is {value!r}"
        raise ValueError(msg)

    if return_multi:
        return parsed_sheets
    return next(iter(parsed_sheets.values()))


def _get_sheet_names(
    sheet_id: int | Sequence[int] | None,
    sheet_name: str | list[str] | tuple[str] | None,
    worksheets: list[dict[str, Any]],
) -> tuple[list[str], bool]:
    """Establish sheets to read; indicate if we are returning a dict frames."""
    if sheet_id is not None and sheet_name is not None:
        msg = f"cannot specify both `sheet_name` ({sheet_name!r}) and `sheet_id` ({sheet_id!r})"
        raise ValueError(msg)

    sheet_names = []
    if sheet_id is None and sheet_name is None:
        sheet_names.append(worksheets[0]["name"])
        return_multi = False
    elif sheet_id == 0:
        sheet_names.extend(ws["name"] for ws in worksheets)
        return_multi = True
    else:
        return_multi = (
            (isinstance(sheet_name, Sequence) and not isinstance(sheet_name, str))
            or isinstance(sheet_id, Sequence)
            or sheet_id == 0
        )
        if names := (
            (sheet_name,) if isinstance(sheet_name, str) else sheet_name or ()
        ):
            known_sheet_names = {ws["name"] for ws in worksheets}
            for name in names:
                if name not in known_sheet_names:
                    msg = f"no matching sheet found when `sheet_name` is {name!r}"
                    raise ValueError(msg)
                sheet_names.append(name)
        else:
            ids = (sheet_id,) if isinstance(sheet_id, int) else sheet_id or ()
            sheet_names_by_idx = {
                idx: ws["name"]
                for idx, ws in enumerate(worksheets, start=1)
                if (sheet_id == 0 or ws["index"] in ids or ws["name"] in names)
            }
            for idx in ids:
                if (name := sheet_names_by_idx.get(idx)) is None:  # type: ignore[assignment]
                    msg = f"no matching sheet found when `sheet_id` is {idx}"
                    raise ValueError(msg)
                sheet_names.append(name)
    return sheet_names, return_multi


def _initialise_spreadsheet_parser(
    engine: str | None,
    source: str | BytesIO | Path | BinaryIO | bytes,
    engine_options: dict[str, Any],
) -> tuple[Callable[..., pl.DataFrame], Any, list[dict[str, Any]]]:
    """Instantiate the indicated spreadsheet parser and establish related properties."""
    if isinstance(source, (str, Path)) and not Path(source).exists():
        raise FileNotFoundError(source)

    if engine == "xlsx2csv":  # default
        xlsx2csv = import_optional("xlsx2csv")

        # establish sensible defaults for unset options
        for option, value in {
            "exclude_hidden_sheets": False,
            "skip_empty_lines": False,
            "skip_hidden_rows": False,
            "floatformat": "%f",
        }.items():
            engine_options.setdefault(option, value)

        parser = xlsx2csv.Xlsx2csv(source, **engine_options)
        sheets = parser.workbook.sheets
        return _read_spreadsheet_xlsx2csv, parser, sheets

    elif engine == "openpyxl":
        openpyxl = import_optional("openpyxl")
        parser = openpyxl.load_workbook(source, data_only=True, **engine_options)
        sheets = [{"index": i + 1, "name": ws.title} for i, ws in enumerate(parser)]
        return _read_spreadsheet_openpyxl, parser, sheets

    elif engine == "calamine":
        # note: can't read directly from bytes (yet) so
        read_buffered = False
        if read_bytesio := isinstance(source, BytesIO) or (
            read_buffered := isinstance(source, BufferedReader)
        ):
            temp_data = PortableTemporaryFile(delete=True)

        with temp_data if (read_bytesio or read_buffered) else nullcontext() as tmp:
            if read_bytesio and tmp is not None:
                tmp.write(source.read() if read_buffered else source.getvalue())  # type: ignore[union-attr]
                source = tmp.name
                tmp.close()

            fxl = import_optional("fastexcel", min_version="0.7.0")
            parser = fxl.read_excel(source, **engine_options)
            sheets = [
                {"index": i + 1, "name": nm} for i, nm in enumerate(parser.sheet_names)
            ]
            return _read_spreadsheet_calamine, parser, sheets

    elif engine == "pyxlsb":
        pyxlsb = import_optional("pyxlsb")
        try:
            parser = pyxlsb.open_workbook(source, **engine_options)
        except KeyError as err:
            if "no item named 'xl/_rels/workbook.bin.rels'" in str(err):
                msg = f"invalid Excel Binary Workbook: {source!r}"
                raise TypeError(msg) from None
            raise
        sheets = [
            {"index": i + 1, "name": name} for i, name in enumerate(parser.sheets)
        ]
        return _read_spreadsheet_pyxlsb, parser, sheets

    elif engine == "ods":
        ezodf = import_optional("ezodf")
        parser = ezodf.opendoc(source, **engine_options)
        sheets = [
            {"index": i + 1, "name": ws.name} for i, ws in enumerate(parser.sheets)
        ]
        return _read_spreadsheet_ods, parser, sheets

    msg = f"unrecognized engine: {engine!r}"
    raise NotImplementedError(msg)


def _csv_buffer_to_frame(
    csv: StringIO,
    separator: str,
    read_options: dict[str, Any],
    schema_overrides: SchemaDict | None,
    *,
    raise_if_empty: bool,
) -> pl.DataFrame:
    """Translate StringIO buffer containing delimited data as a DataFrame."""
    # handle (completely) empty sheet data
    if csv.tell() == 0:
        if raise_if_empty:
            msg = (
                "empty Excel sheet"
                "\n\nIf you want to read this as an empty DataFrame, set `raise_if_empty=False`."
            )
            raise NoDataError(msg)
        return pl.DataFrame()

    if read_options is None:
        read_options = {}
    if schema_overrides:
        if (csv_dtypes := read_options.get("dtypes", {})) and set(
            csv_dtypes
        ).intersection(schema_overrides):
            msg = "cannot specify columns in both `schema_overrides` and `read_options['dtypes']`"
            raise ParameterCollisionError(msg)
        read_options = read_options.copy()
        read_options["dtypes"] = {**csv_dtypes, **schema_overrides}

    # otherwise rewind the buffer and parse as csv
    csv.seek(0)
    df = read_csv(
        csv,
        separator=separator,
        **read_options,
    )
    return _drop_null_data(df, raise_if_empty=raise_if_empty)


def _drop_null_data(df: pl.DataFrame, *, raise_if_empty: bool) -> pl.DataFrame:
    """If DataFrame contains columns/rows that contain only nulls, drop them."""
    null_cols = []
    for col_name in df.columns:
        # note that if multiple unnamed columns are found then all but the first one
        # will be named as "_duplicated_{n}" (or "__UNNAMED__{n}" from calamine)
        if col_name == "" or re.match(r"(_duplicated_|__UNNAMED__)\d+$", col_name):
            col = df[col_name]
            if (
                col.dtype == Null
                or col.null_count() == len(df)
                or (
                    col.dtype in NUMERIC_DTYPES
                    and col.replace(0, None).null_count() == len(df)
                )
            ):
                null_cols.append(col_name)
    if null_cols:
        df = df.drop(*null_cols)

    if len(df) == 0 and len(df.columns) == 0:
        if not raise_if_empty:
            return df
        else:
            msg = (
                "empty Excel sheet"
                "\n\nIf you want to read this as an empty DataFrame, set `raise_if_empty=False`."
            )
            raise NoDataError(msg)

    return df.filter(~F.all_horizontal(F.all().is_null()))


def _read_spreadsheet_ods(
    parser: Any,
    sheet_name: str | None,
    read_options: dict[str, Any],
    schema_overrides: SchemaDict | None,
    *,
    raise_if_empty: bool,
) -> pl.DataFrame:
    """Use the 'ezodf' library to read data from the given worksheet."""
    sheets = parser.sheets
    if sheet_name is not None:
        ws = next((s for s in sheets if s.name == sheet_name), None)
        if ws is None:
            msg = f"sheet {sheet_name!r} not found"
            raise ValueError(msg)
    else:
        ws = sheets[0]

    row_data = []
    found_row_data = False
    for row in ws.rows():
        row_values = [c.value for c in row]
        if found_row_data or (found_row_data := any(v is not None for v in row_values)):
            row_data.append(row_values)

    overrides = {}
    strptime_cols = {}
    headers: list[str] = []

    if not row_data:
        df = pl.DataFrame()
    else:
        for idx, name in enumerate(row_data[0]):
            headers.append(name or (f"_duplicated_{idx}" if headers else ""))

        trailing_null_row = all(v is None for v in row_data[-1])
        row_data = row_data[1 : -1 if trailing_null_row else None]

        if schema_overrides:
            for nm, dtype in schema_overrides.items():
                if dtype in (Datetime, Date):
                    strptime_cols[nm] = dtype
                else:
                    overrides[nm] = dtype

        df = pl.DataFrame(
            row_data,
            orient="row",
            schema=headers,
            schema_overrides=overrides,
        )

    if strptime_cols:
        df = df.with_columns(
            (
                F.col(nm).str.replace("[T ]00:00:00$", "")
                if dtype == Date
                else F.col(nm)
            ).str.strptime(
                dtype  # type: ignore[arg-type]
            )
            for nm, dtype in strptime_cols.items()
        )

    df.columns = headers
    return _drop_null_data(df, raise_if_empty=raise_if_empty)


def _read_spreadsheet_openpyxl(
    parser: Any,
    sheet_name: str | None,
    read_options: dict[str, Any],
    schema_overrides: SchemaDict | None,
    *,
    raise_if_empty: bool,
) -> pl.DataFrame:
    """Use the 'openpyxl' library to read data from the given worksheet."""
    ws = parser[sheet_name]

    # prefer detection of actual table objects; otherwise read
    # data in the used worksheet range, dropping null columns
    header: list[str | None] = []
    if tables := getattr(ws, "tables", None):
        table = next(iter(tables.values()))
        rows = list(ws[table.ref])
        header.extend(cell.value for cell in rows.pop(0))
        if table.totalsRowCount:
            rows = rows[: -table.totalsRowCount]
        rows_iter = iter(rows)
    else:
        rows_iter = ws.iter_rows()
        for row in rows_iter:
            row_values = [cell.value for cell in row]
            if any(v is not None for v in row_values):
                header.extend(row_values)
                break

    series_data = []
    for name, column_data in zip(header, zip(*rows_iter)):
        if name:
            values = [cell.value for cell in column_data]
            if (dtype := (schema_overrides or {}).get(name)) == String:
                # note: if we init series with mixed-type data (eg: str/int)
                # the non-strings will become null, so we handle the cast here
                values = [str(v) if (v is not None) else v for v in values]

            s = pl.Series(name, values, dtype=dtype)
            series_data.append(s)

    df = pl.DataFrame(
        {s.name: s for s in series_data},
        schema_overrides=schema_overrides,
    )
    return _drop_null_data(df, raise_if_empty=raise_if_empty)


def _read_spreadsheet_calamine(
    parser: Any,
    sheet_name: str | None,
    read_options: dict[str, Any],
    schema_overrides: SchemaDict | None,
    *,
    raise_if_empty: bool,
) -> pl.DataFrame:
    ws = parser.load_sheet_by_name(sheet_name, **read_options)
    df = ws.to_polars()

    if schema_overrides:
        df = df.cast(dtypes=schema_overrides)

    df = _drop_null_data(df, raise_if_empty=raise_if_empty)

    # refine dtypes
    type_checks = []
    for c, dtype in df.schema.items():
        # may read integer data as float; cast back to int where possible.
        if dtype in FLOAT_DTYPES:
            check_cast = [F.col(c).floor().eq(F.col(c)), F.col(c).cast(Int64)]
            type_checks.append(check_cast)
        # do a similar check for datetime columns that have only 00:00:00 times.
        elif dtype == Datetime:
            check_cast = [
                F.col(c).dt.time().eq(time(0, 0, 0)),
                F.col(c).cast(Date),
            ]
            type_checks.append(check_cast)

    if type_checks:
        apply_cast = df.select(
            [d[0].all(ignore_nulls=True) for d in type_checks],
        ).row(0)
        if downcast := [
            cast for apply, (_, cast) in zip(apply_cast, type_checks) if apply
        ]:
            df = df.with_columns(*downcast)

    return df


def _read_spreadsheet_pyxlsb(
    parser: Any,
    sheet_name: str | None,
    read_options: dict[str, Any],
    schema_overrides: SchemaDict | None,
    *,
    raise_if_empty: bool,
) -> pl.DataFrame:
    from pyxlsb import convert_date

    ws = parser.get_sheet(sheet_name)
    try:
        # establish header/data rows
        header: list[str | None] = []
        rows_iter = ws.rows()
        for row in rows_iter:
            row_values = [cell.v for cell in row]
            if any(v is not None for v in row_values):
                header.extend(row_values)
                break

        # load data rows as series
        series_data = []
        for name, column_data in zip(header, zip(*rows_iter)):
            if name:
                values = [cell.v for cell in column_data]
                if (dtype := (schema_overrides or {}).get(name)) == String:
                    # note: if we init series with mixed-type data (eg: str/int)
                    # the non-strings will become null, so we handle the cast here
                    values = [
                        str(int(v) if isinstance(v, float) and v.is_integer() else v)
                        if (v is not None)
                        else v
                        for v in values
                    ]
                elif dtype in (Datetime, Date):
                    dtype = None

                s = pl.Series(name, values, dtype=dtype)
                series_data.append(s)
    finally:
        ws.close()

    if schema_overrides:
        for idx, s in enumerate(series_data):
            if schema_overrides.get(s.name) in (Datetime, Date):
                series_data[idx] = s.map_elements(convert_date)

    df = pl.DataFrame(
        {s.name: s for s in series_data},
        schema_overrides=schema_overrides,
    )
    return _drop_null_data(df, raise_if_empty=raise_if_empty)


def _read_spreadsheet_xlsx2csv(
    parser: Any,
    sheet_name: str | None,
    read_options: dict[str, Any],
    schema_overrides: SchemaDict | None,
    *,
    raise_if_empty: bool,
) -> pl.DataFrame:
    """Use the 'xlsx2csv' library to read data from the given worksheet."""
    csv_buffer = StringIO()
    parser.convert(
        outfile=csv_buffer,
        sheetname=sheet_name,
    )
    if read_options is None:
        read_options = {}
    read_options.setdefault("truncate_ragged_lines", True)

    return _csv_buffer_to_frame(
        csv_buffer,
        separator=",",
        read_options=read_options,
        schema_overrides=schema_overrides,
        raise_if_empty=raise_if_empty,
    )
