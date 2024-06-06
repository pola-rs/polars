from __future__ import annotations

import re
from datetime import time
from io import BufferedReader, BytesIO, StringIO, TextIOWrapper
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Callable, NoReturn, Sequence, overload

import polars._reexport as pl
from polars import functions as F
from polars._utils.deprecation import (
    deprecate_renamed_parameter,
    issue_deprecation_warning,
)
from polars._utils.various import normalize_filepath, parse_version
from polars.datatypes import (
    FLOAT_DTYPES,
    INTEGER_DTYPES,
    N_INFER_DEFAULT,
    NUMERIC_DTYPES,
    Boolean,
    Date,
    Datetime,
    Duration,
    Int64,
    Null,
    String,
)
from polars.dependencies import import_optional
from polars.exceptions import (
    ModuleUpgradeRequired,
    NoDataError,
    ParameterCollisionError,
)
from polars.io._utils import looks_like_url, process_file_url
from polars.io.csv.functions import read_csv

if TYPE_CHECKING:
    from typing import Literal

    from polars.type_aliases import ExcelSpreadsheetEngine, SchemaDict


@overload
def read_excel(
    source: str | Path | IO[bytes] | bytes,
    *,
    sheet_id: None = ...,
    sheet_name: str,
    engine: ExcelSpreadsheetEngine | None = ...,
    engine_options: dict[str, Any] | None = ...,
    read_options: dict[str, Any] | None = ...,
    schema_overrides: SchemaDict | None = ...,
    infer_schema_length: int | None = ...,
    raise_if_empty: bool = ...,
) -> pl.DataFrame: ...


@overload
def read_excel(
    source: str | Path | IO[bytes] | bytes,
    *,
    sheet_id: None = ...,
    sheet_name: None = ...,
    engine: ExcelSpreadsheetEngine | None = ...,
    engine_options: dict[str, Any] | None = ...,
    read_options: dict[str, Any] | None = ...,
    schema_overrides: SchemaDict | None = ...,
    infer_schema_length: int | None = ...,
    raise_if_empty: bool = ...,
) -> pl.DataFrame: ...


@overload
def read_excel(
    source: str | Path | IO[bytes] | bytes,
    *,
    sheet_id: int,
    sheet_name: str,
    engine: ExcelSpreadsheetEngine | None = ...,
    engine_options: dict[str, Any] | None = ...,
    read_options: dict[str, Any] | None = ...,
    schema_overrides: SchemaDict | None = ...,
    infer_schema_length: int | None = ...,
    raise_if_empty: bool = ...,
) -> NoReturn: ...


# note: 'ignore' required as mypy thinks that the return value for
# Literal[0] overlaps with the return value for other integers
@overload  # type: ignore[overload-overlap]
def read_excel(
    source: str | Path | IO[bytes] | bytes,
    *,
    sheet_id: Literal[0] | Sequence[int],
    sheet_name: None = ...,
    engine: ExcelSpreadsheetEngine | None = ...,
    engine_options: dict[str, Any] | None = ...,
    read_options: dict[str, Any] | None = ...,
    schema_overrides: SchemaDict | None = ...,
    infer_schema_length: int | None = ...,
    raise_if_empty: bool = ...,
) -> dict[str, pl.DataFrame]: ...


@overload
def read_excel(
    source: str | Path | IO[bytes] | bytes,
    *,
    sheet_id: int,
    sheet_name: None = ...,
    engine: ExcelSpreadsheetEngine | None = ...,
    engine_options: dict[str, Any] | None = ...,
    read_options: dict[str, Any] | None = ...,
    schema_overrides: SchemaDict | None = ...,
    infer_schema_length: int | None = ...,
    raise_if_empty: bool = ...,
) -> pl.DataFrame: ...


@overload
def read_excel(
    source: str | Path | IO[bytes] | bytes,
    *,
    sheet_id: None,
    sheet_name: list[str] | tuple[str],
    engine: ExcelSpreadsheetEngine | None = ...,
    engine_options: dict[str, Any] | None = ...,
    read_options: dict[str, Any] | None = ...,
    schema_overrides: SchemaDict | None = ...,
    infer_schema_length: int | None = ...,
    raise_if_empty: bool = ...,
) -> dict[str, pl.DataFrame]: ...


@deprecate_renamed_parameter("xlsx2csv_options", "engine_options", version="0.20.6")
@deprecate_renamed_parameter("read_csv_options", "read_options", version="0.20.7")
def read_excel(
    source: str | Path | IO[bytes] | bytes,
    *,
    sheet_id: int | Sequence[int] | None = None,
    sheet_name: str | list[str] | tuple[str] | None = None,
    engine: ExcelSpreadsheetEngine | None = None,
    engine_options: dict[str, Any] | None = None,
    read_options: dict[str, Any] | None = None,
    schema_overrides: SchemaDict | None = None,
    infer_schema_length: int | None = N_INFER_DEFAULT,
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
        Path to a file or a file-like object (by "file-like object" we refer to objects
        that have a `read()` method, such as a file handler like the builtin `open`
        function, or a `BytesIO` instance).
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
        * "calamine": this engine can be used for reading all major types of Excel
          Workbook (`.xlsx`, `.xlsb`, `.xls`) and is *dramatically* faster than the
          other options, using the `fastexcel` module to bind the calamine reader.
        * "openpyxl": this engine is significantly slower than `xlsx2csv` but supports
          additional automatic type inference; potentially useful if you are otherwise
          unable to parse your sheet with the (default) `xlsx2csv` engine in
          conjunction with the `schema_overrides` parameter.
        * "pyxlsb": this engine can be used for Excel Binary Workbooks (`.xlsb` files).
          Note that you have to use `schema_overrides` to correctly load date/datetime
          columns (or these will be read as floats representing offset Julian values).
          You should now prefer the "calamine" engine for this Workbook type.
    engine_options
        Additional options passed to the underlying engine's primary parsing
        constructor (given below), if supported:

        * "xlsx2csv": `Xlsx2csv`
        * "calamine": n/a (can only provide `read_options`)
        * "openpyxl": `load_workbook`
        * "pyxlsb": `open_workbook`
    read_options
        Options passed to the underlying engine method that reads the sheet data.
        Where supported, this allows for additional control over parsing. The
        specific read methods associated with each engine are:

        * "xlsx2csv": `pl.read_csv`
        * "calamine": `ExcelReader.load_sheet_by_name`
        * "openpyxl": n/a (can only provide `engine_options`)
        * "pyxlsb":  n/a (can only provide `engine_options`)
    schema_overrides
        Support type specification or override of one or more columns.
    infer_schema_length
        The maximum number of rows to scan for schema inference. If set to `None`, the
        entire dataset is scanned to determine the dtypes, which can slow parsing for
        large workbooks. Note that only the "calamine" and "xlsx2csv" engines support
        this parameter; for all others it is a no-op.
    raise_if_empty
        When there is no data in the sheet,`NoDataError` is raised. If this parameter
        is set to False, an empty DataFrame (with no columns) is returned instead.

    Notes
    -----
    * When using the default `xlsx2csv` engine the target Excel sheet is first converted
      to CSV using `xlsx2csv.Xlsx2csv(source).convert()` and then parsed with Polars'
      :func:`read_csv` function. You can pass additional options to `read_options`
      to influence this part of the parsing pipeline.
    * Where possible, prefer the "calamine" engine for reading Excel Workbooks, as it is
      significantly faster than the other options, and is intended to become the default
      engine for all Excel file types in a future release.
    * If you want to read multiple sheets and set *different* options (`read_options`,
      `schema_overrides`, etc), you should make separate calls as the options are set
      globally, not on a per-sheet basis.

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
    this issue. For example, if using `xlsx2csv` or `calamine` the "infer_schema_length"
    parameter can be set to `None` to force reading the entire dataset to infer the
    best dtypes. If column types are known in advance, and there is no ambiguity in the
    parsing, `schema_overrides` is typically the more efficient option.

    >>> pl.read_excel(
    ...     source="test.xlsx",
    ...     schema_overrides={"dt": pl.Date},
    ...     infer_schema_length=None,
    ...     engine="calamine",
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
        infer_schema_length=infer_schema_length,
        raise_if_empty=raise_if_empty,
    )


@overload
def read_ods(
    source: str | Path | IO[bytes] | bytes,
    *,
    sheet_id: None = ...,
    sheet_name: str,
    schema_overrides: SchemaDict | None = None,
    infer_schema_length: int | None = ...,
    raise_if_empty: bool = ...,
) -> pl.DataFrame: ...


@overload
def read_ods(
    source: str | Path | IO[bytes] | bytes,
    *,
    sheet_id: None = ...,
    sheet_name: None = ...,
    schema_overrides: SchemaDict | None = None,
    infer_schema_length: int | None = ...,
    raise_if_empty: bool = ...,
) -> pl.DataFrame: ...


@overload
def read_ods(
    source: str | Path | IO[bytes] | bytes,
    *,
    sheet_id: int,
    sheet_name: str,
    schema_overrides: SchemaDict | None = None,
    infer_schema_length: int | None = ...,
    raise_if_empty: bool = ...,
) -> NoReturn: ...


@overload  # type: ignore[overload-overlap]
def read_ods(
    source: str | Path | IO[bytes] | bytes,
    *,
    sheet_id: Literal[0] | Sequence[int],
    sheet_name: None = ...,
    schema_overrides: SchemaDict | None = None,
    infer_schema_length: int | None = ...,
    raise_if_empty: bool = ...,
) -> dict[str, pl.DataFrame]: ...


@overload
def read_ods(
    source: str | Path | IO[bytes] | bytes,
    *,
    sheet_id: int,
    sheet_name: None = ...,
    schema_overrides: SchemaDict | None = None,
    infer_schema_length: int | None = ...,
    raise_if_empty: bool = ...,
) -> pl.DataFrame: ...


@overload
def read_ods(
    source: str | Path | IO[bytes] | bytes,
    *,
    sheet_id: None,
    sheet_name: list[str] | tuple[str],
    schema_overrides: SchemaDict | None = None,
    infer_schema_length: int | None = ...,
    raise_if_empty: bool = ...,
) -> dict[str, pl.DataFrame]: ...


def read_ods(
    source: str | Path | IO[bytes] | bytes,
    *,
    sheet_id: int | Sequence[int] | None = None,
    sheet_name: str | list[str] | tuple[str] | None = None,
    schema_overrides: SchemaDict | None = None,
    infer_schema_length: int | None = N_INFER_DEFAULT,
    raise_if_empty: bool = True,
) -> pl.DataFrame | dict[str, pl.DataFrame]:
    """
    Read OpenOffice (ODS) spreadsheet data into a DataFrame.

    Parameters
    ----------
    source
        Path to a file or a file-like object (by "file-like object" we refer to objects
        that have a `read()` method, such as a file handler like the builtin `open`
        function, or a `BytesIO` instance).
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
    infer_schema_length
        The maximum number of rows to scan for schema inference. If set to `None`, the
        entire dataset is scanned to determine the dtypes, which can slow parsing for
        large workbooks.
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
    to specify them, or increase the inference length with `infer_schema_length`.

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
        engine="calamine",
        engine_options={},
        read_options=None,
        schema_overrides=schema_overrides,
        infer_schema_length=infer_schema_length,
        raise_if_empty=raise_if_empty,
    )


def _identify_from_magic_bytes(data: IO[bytes] | bytes) -> str | None:
    if isinstance(data, bytes):
        data = BytesIO(data)

    xls_bytes = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"  # excel 97-2004
    xlsx_bytes = b"PK\x03\x04"  # xlsx/openoffice (zipped xml)

    initial_position = data.tell()
    try:
        magic_bytes = data.read(8)
        if magic_bytes == xls_bytes:
            return "xls"
        elif magic_bytes[:4] == xlsx_bytes:
            return "xlsx"
    except UnicodeDecodeError:
        pass
    finally:
        data.seek(initial_position)
    return None


def _identify_workbook(wb: str | Path | IO[bytes] | bytes) -> str | None:
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
    source: str | Path | IO[bytes] | bytes,
    engine: ExcelSpreadsheetEngine | Literal["ods"] | None,
    engine_options: dict[str, Any] | None = None,
    read_options: dict[str, Any] | None = None,
    schema_overrides: SchemaDict | None = None,
    infer_schema_length: int | None = N_INFER_DEFAULT,
    *,
    raise_if_empty: bool = True,
) -> pl.DataFrame | dict[str, pl.DataFrame]:
    if is_file := isinstance(source, (str, Path)):
        source = normalize_filepath(source)
        if looks_like_url(source):
            source = process_file_url(source)

    if engine is None:
        if is_file and str(source).lower().endswith(".ods"):
            # note: if called from "read_ods" the engine cannot be 'None', hence
            # this check is only triggered when called from "read_excel"
            msg = "OpenDocumentSpreadsheet files require use of `read_ods`, not `read_excel`"
            raise ValueError(msg)

        # note: eventually want 'calamine' to be the default for all extensions
        file_type = _identify_workbook(source)
        engine = "calamine" if file_type in ("xlsb", "xls") else "xlsx2csv"

    read_options = (read_options or {}).copy()
    engine_options = (engine_options or {}).copy()

    # normalise some top-level parameters to 'read_options' entries
    if engine == "calamine":
        if ("schema_sample_rows" in read_options) and (
            infer_schema_length != N_INFER_DEFAULT
        ):
            msg = 'cannot specify both `infer_schema_length` and `read_options["schema_sample_rows"]`'
            raise ParameterCollisionError(msg)
        read_options["schema_sample_rows"] = infer_schema_length

    elif engine == "xlsx2csv":
        if ("infer_schema_length" in read_options) and (
            infer_schema_length != N_INFER_DEFAULT
        ):
            msg = 'cannot specify both `infer_schema_length` and `read_options["infer_schema_length"]`'
            raise ParameterCollisionError(msg)
        read_options["infer_schema_length"] = infer_schema_length
    else:
        read_options["infer_schema_length"] = infer_schema_length

    # establish the reading function, parser, and available worksheets
    reader_fn, parser, worksheets = _initialise_spreadsheet_parser(
        engine, source, engine_options
    )
    try:
        # parse data from the indicated sheet(s)
        sheet_names, return_multi = _get_sheet_names(sheet_id, sheet_name, worksheets)
        parsed_sheets = {
            name: reader_fn(
                parser=parser,
                sheet_name=name,
                schema_overrides=schema_overrides,
                read_options=read_options,
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
    source: str | Path | IO[bytes] | bytes,
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
        fastexcel = import_optional("fastexcel", min_version="0.7.0")
        reading_bytesio, reading_bytes = (
            isinstance(source, BytesIO),
            isinstance(source, bytes),
        )
        if (reading_bytesio or reading_bytes) and parse_version(
            module_version := fastexcel.__version__
        ) < (0, 10):
            msg = f"`fastexcel` >= 0.10 is required to read bytes; found {module_version})"
            raise ModuleUpgradeRequired(msg)

        if reading_bytesio:
            source = source.getbuffer().tobytes()  # type: ignore[union-attr]
        elif isinstance(source, (BufferedReader, TextIOWrapper)):
            if "b" not in source.mode:
                msg = f"file {source.name!r} must be opened in binary mode"
                raise OSError(msg)
            elif (filename := source.name) and Path(filename).exists():
                source = filename
            else:
                source = source.read()

        parser = fastexcel.read_excel(source, **engine_options)
        sheets = [
            {"index": i + 1, "name": nm} for i, nm in enumerate(parser.sheet_names)
        ]
        return _read_spreadsheet_calamine, parser, sheets

    elif engine == "pyxlsb":
        issue_deprecation_warning(
            "the 'pyxlsb' engine is deprecated and should be replaced with 'calamine'",
            version="0.20.22",
        )
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
        csv_dtypes = read_options.get("dtypes", {})
        if csv_dtypes:
            issue_deprecation_warning(
                "The `dtypes` parameter for `read_csv` is deprecated. It has been renamed to `schema_overrides`.",
                version="0.20.31",
            )
        csv_schema_overrides = read_options.get("schema_overrides", csv_dtypes)

        if csv_schema_overrides and set(csv_schema_overrides).intersection(
            schema_overrides
        ):
            msg = "cannot specify columns in both `schema_overrides` and `read_options['dtypes']`"
            raise ParameterCollisionError(msg)

        read_options = read_options.copy()
        read_options["schema_overrides"] = {**csv_schema_overrides, **schema_overrides}

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


def _read_spreadsheet_openpyxl(
    parser: Any,
    sheet_name: str | None,
    read_options: dict[str, Any],
    schema_overrides: SchemaDict | None,
    *,
    raise_if_empty: bool,
) -> pl.DataFrame:
    """Use the 'openpyxl' library to read data from the given worksheet."""
    infer_schema_length = read_options.pop("infer_schema_length", None)
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
        infer_schema_length=infer_schema_length,
        strict=False,
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
    # if we have 'schema_overrides' and a more recent version of `fastexcel`
    # we can pass translated dtypes to the engine to refine the initial parse
    fastexcel = import_optional("fastexcel")
    fastexcel_version = parse_version(fastexcel.__version__)
    if fastexcel_version < (0, 9) and "schema_sample_rows" in read_options:
        msg = f"a more recent version of `fastexcel` is required (>= 0.9; found {fastexcel.__version__})"
        raise ModuleUpgradeRequired(msg)

    if (schema_overrides := (schema_overrides or {})) and fastexcel_version >= (0, 10):
        parser_dtypes = read_options.get("dtypes", {})
        for name, dtype in schema_overrides.items():
            if name not in parser_dtypes:
                if (base_dtype := dtype.base_type()) in INTEGER_DTYPES:
                    parser_dtypes[name] = "int"
                elif base_dtype in FLOAT_DTYPES:
                    parser_dtypes[name] = "float"
                elif base_dtype == String:
                    parser_dtypes[name] = "string"
                elif base_dtype == Datetime:
                    parser_dtypes[name] = "datetime"
                elif base_dtype == Date:
                    parser_dtypes[name] = "date"
                elif base_dtype == Duration:
                    parser_dtypes[name] = "duration"
                elif base_dtype == Boolean:
                    parser_dtypes[name] = "bool"
        read_options["dtypes"] = parser_dtypes

    ws = parser.load_sheet_by_name(name=sheet_name, **read_options)
    df = ws.to_polars()

    # note: even if we applied parser dtypes we still re-apply schema_overrides
    # natively as we can refine integer/float types, temporal precision, etc.
    if schema_overrides:
        df = df.cast(dtypes=schema_overrides)

    df = _drop_null_data(df, raise_if_empty=raise_if_empty)

    # further refine dtypes
    type_checks = []
    for c, dtype in df.schema.items():
        if c not in schema_overrides:
            # may read integer data as float; cast back to int where possible.
            if dtype in FLOAT_DTYPES:
                check_cast = [
                    F.col(c).floor().eq_missing(F.col(c)) & F.col(c).is_not_nan(),
                    F.col(c).cast(Int64),
                ]
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

    infer_schema_length = read_options.pop("infer_schema_length", None)
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
                series_data[idx] = s.map_elements(convert_date, return_dtype=Datetime)

    df = pl.DataFrame(
        {s.name: s for s in series_data},
        schema_overrides=schema_overrides,
        infer_schema_length=infer_schema_length,
        strict=False,
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

    parser.convert(outfile=csv_buffer, sheetname=sheet_name)
    read_options.setdefault("truncate_ragged_lines", True)

    return _csv_buffer_to_frame(
        csv_buffer,
        separator=",",
        read_options=read_options,
        schema_overrides=schema_overrides,
        raise_if_empty=raise_if_empty,
    )
