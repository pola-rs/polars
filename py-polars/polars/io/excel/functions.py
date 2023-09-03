from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO, NoReturn, overload

import polars._reexport as pl
from polars.exceptions import NoDataError
from polars.io.csv.functions import read_csv
from polars.utils.various import normalise_filepath

if TYPE_CHECKING:
    from io import BytesIO
    from typing import Literal


@overload
def read_excel(
    source: str | BytesIO | Path | BinaryIO | bytes,
    *,
    sheet_id: None = ...,
    sheet_name: str,
    xlsx2csv_options: dict[str, Any] | None = ...,
    read_csv_options: dict[str, Any] | None = ...,
    engine: Literal["xlsx2csv", "openpyxl"] | None = ...,
    raise_if_empty: bool = ...,
) -> pl.DataFrame:
    ...


@overload
def read_excel(
    source: str | BytesIO | Path | BinaryIO | bytes,
    *,
    sheet_id: None = ...,
    sheet_name: None = ...,
    xlsx2csv_options: dict[str, Any] | None = ...,
    read_csv_options: dict[str, Any] | None = ...,
    engine: Literal["xlsx2csv", "openpyxl"] | None = ...,
    raise_if_empty: bool = ...,
) -> pl.DataFrame:
    ...


@overload
def read_excel(
    source: str | BytesIO | Path | BinaryIO | bytes,
    *,
    sheet_id: int,
    sheet_name: str,
    xlsx2csv_options: dict[str, Any] | None = ...,
    read_csv_options: dict[str, Any] | None = ...,
    engine: Literal["xlsx2csv", "openpyxl"] | None = ...,
    raise_if_empty: bool = ...,
) -> NoReturn:
    ...


# mypy wants the return value for Literal[0] to
# overlap with the return value for other integers.
@overload  # type: ignore[misc]
def read_excel(
    source: str | BytesIO | Path | BinaryIO | bytes,
    *,
    sheet_id: Literal[0],
    sheet_name: None = ...,
    xlsx2csv_options: dict[str, Any] | None = ...,
    read_csv_options: dict[str, Any] | None = ...,
    engine: Literal["xlsx2csv", "openpyxl"] | None = ...,
    raise_if_empty: bool = ...,
) -> dict[str, pl.DataFrame]:
    ...


@overload
def read_excel(
    source: str | BytesIO | Path | BinaryIO | bytes,
    *,
    sheet_id: int,
    sheet_name: None = ...,
    xlsx2csv_options: dict[str, Any] | None = ...,
    read_csv_options: dict[str, Any] | None = ...,
    engine: Literal["xlsx2csv", "openpyxl"] | None = ...,
    raise_if_empty: bool = ...,
) -> pl.DataFrame:
    ...


def read_excel(
    source: str | BytesIO | Path | BinaryIO | bytes,
    *,
    sheet_id: int | None = None,
    sheet_name: str | None = None,
    xlsx2csv_options: dict[str, Any] | None = None,
    read_csv_options: dict[str, Any] | None = None,
    engine: Literal["xlsx2csv", "openpyxl"] | None = None,
    raise_if_empty: bool = True,
) -> pl.DataFrame | dict[str, pl.DataFrame]:
    """
    Read Excel (XLSX) sheet into a DataFrame.

    If using the ``xlsx2csv`` engine, converts an Excel sheet with
    ``xlsx2csv.Xlsx2csv().convert()`` to CSV and parses the CSV output with
    :func:`read_csv`.

    When using the ``openpyxl`` engine, reads an Excel sheet with
    ``openpyxl.load_workbook(source)``.

    Parameters
    ----------
    source
        Path to a file or a file-like object (by file-like object, we refer to objects
        that have a ``read()`` method, such as a file handler (e.g. via builtin ``open``
        function) or ``BytesIO``).
    sheet_id
        Sheet number to convert (set ``0`` to load all sheets as DataFrames) and return
        a ``{sheetname:frame,}`` dict. (Defaults to `1` if neither this nor `sheet_name`
        are specified).
    sheet_name
        Sheet name to convert; cannot be used in conjunction with `sheet_id`.
    xlsx2csv_options
        Extra options passed to ``xlsx2csv.Xlsx2csv()``,
        e.g. ``{"skip_empty_lines": True}``
    read_csv_options
        Extra options passed to :func:`read_csv` for parsing the CSV file returned by
        ``xlsx2csv.Xlsx2csv().convert()``
        e.g.: ``{"has_header": False, "new_columns": ["a", "b", "c"],
        "infer_schema_length": None}``
    engine
        Library used to parse Excel, either openpyxl or xlsx2csv (default is xlsx2csv).
        Please note that xlsx2csv converts first to csv, making type inference worse
        than openpyxl. To remedy that, you can use the extra options defined on
        `xlsx2csv_options` and `read_csv_options`
    raise_if_empty
        When there is no data in the sheet,``NoDataError`` is raised. If this parameter
        is set to False, an empty DataFrame (with no columns) is returned instead.

    Returns
    -------
    DataFrame, or a sheetname to DataFrame dict when ``sheet_id == 0``.

    Examples
    --------
    Read the "data" worksheet from an Excel file into a DataFrame.

    >>> pl.read_excel(
    ...     source="test.xlsx",
    ...     sheet_name="data",
    ... )  # doctest: +SKIP

    Read sheet 3 from Excel sheet file to a DataFrame while skipping empty lines in the
    sheet. As sheet 3 does not have header row, pass the needed settings to
    :func:`read_csv`.

    >>> pl.read_excel(
    ...     source="test.xlsx",
    ...     sheet_id=3,
    ...     xlsx2csv_options={"skip_empty_lines": True},
    ...     read_csv_options={"has_header": False, "new_columns": ["a", "b", "c"]},
    ... )  # doctest: +SKIP

    If the correct datatypes can't be determined by polars, look at the :func:`read_csv`
    documentation to see which options you can pass to fix this issue. For example
    ``"infer_schema_length": None`` can be used to read the data twice, once to infer
    the correct output types and once to  convert the input to the correct types.
    When `"infer_schema_length": 1000``, only the first 1000 lines are read twice.

    >>> pl.read_excel(
    ...     source="test.xlsx",
    ...     read_csv_options={"infer_schema_length": None},
    ... )  # doctest: +SKIP

    The ``openpyxl`` engine can also be used to provide automatic type inference.
    To do so, specify the right engine (`xlsx2csv_options` and `read_csv_options`
    will be ignored):

    >>> pl.read_excel(
    ...     source="test.xlsx",
    ...     engine="openpyxl",
    ... )  # doctest: +SKIP

    If :func:`read_excel` does not work or you need to read other types of
    spreadsheet files, you can try pandas ``pd.read_excel()``
    (supports `xls`, `xlsx`, `xlsm`, `xlsb`, `odf`, `ods` and `odt`).

    >>> pl.from_pandas(pd.read_excel("test.xlsx"))  # doctest: +SKIP

    """
    if sheet_id is not None and sheet_name is not None:
        raise ValueError(
            f"cannot specify both `sheet_name` ({sheet_name!r}) and `sheet_id` ({sheet_id!r})"
        )

    if isinstance(source, (str, Path)):
        source = normalise_filepath(source)

    if xlsx2csv_options is None:
        xlsx2csv_options = {}
    if read_csv_options is None:
        read_csv_options = {}

    reader_fn: Any  # make mypy happy
    # do conditions imports
    if engine == "openpyxl":
        try:
            import openpyxl
        except ImportError:
            raise ImportError(
                "openpyxl is not installed\n\nPlease run `pip install openpyxl`"
            ) from None
        parser = openpyxl.load_workbook(source)
        sheets = [
            {"index": i + 1, "name": sheet.title} for i, sheet in enumerate(parser)
        ]
        reader_fn = _read_excel_sheet_openpyxl
    elif engine == "xlsx2csv" or engine is None:  # default
        try:
            import xlsx2csv
        except ImportError:
            raise ModuleNotFoundError(
                "xlsx2csv is not installed\n\nPlease run: `pip install xlsx2csv`"
            ) from None
        # convert sheets to csv
        parser = xlsx2csv.Xlsx2csv(source, **xlsx2csv_options)
        sheets = parser.workbook.sheets
        reader_fn = _read_excel_sheet_xlsx2csv
    else:
        raise NotImplementedError(f"Cannot find the engine `{engine}`")

    if sheet_id == 0:
        return {
            sheet["name"]: reader_fn(
                parser=parser,
                sheet_id=sheet["index"],
                sheet_name=None,
                read_csv_options=read_csv_options,
                raise_if_empty=raise_if_empty,
            )
            for sheet in sheets
        }
    else:
        # read a specific sheet by id or name
        if sheet_name is None:
            sheet_id = sheet_id or 1
        return reader_fn(
            parser=parser,
            sheet_id=sheet_id,
            sheet_name=sheet_name,
            read_csv_options=read_csv_options,
            raise_if_empty=raise_if_empty,
        )


def _read_excel_sheet_openpyxl(
    parser: Any,
    sheet_id: int | None,
    sheet_name: str | None,
    read_csv_options: dict[str, Any] | None,
    raise_if_empty: bool,
) -> pl.DataFrame:
    # read requested sheet if provided on kwargs, otherwise read active sheet
    if sheet_name is not None:
        ws = parser[sheet_name]
    elif sheet_id is not None:
        ws = parser.worksheets[sheet_id - 1]
    else:
        ws = parser.active

    rows_iter = iter(ws.rows)

    # check whether to include or omit the header
    header = [str(cell.value) for cell in next(rows_iter)]

    df = pl.DataFrame(
        {key: cell.value for key, cell in zip(header, row)} for row in rows_iter
    )
    if raise_if_empty and len(df) == 0:
        raise NoDataError(
            "Empty Excel sheet; if you want to read this as "
            "an empty DataFrame, set `raise_if_empty=False`"
        )
    return df


def _read_excel_sheet_xlsx2csv(
    parser: Any,
    sheet_id: int | None,
    sheet_name: str | None,
    read_csv_options: dict[str, Any],
    raise_if_empty: bool,
) -> pl.DataFrame:
    # parse sheet data into the given buffer
    csv_buffer = StringIO()
    parser.convert(outfile=csv_buffer, sheetid=sheet_id, sheetname=sheet_name)

    # handle (completely) empty sheet data
    if csv_buffer.tell() == 0:
        if raise_if_empty:
            raise NoDataError(
                "Empty Excel sheet; if you want to read this as "
                "an empty DataFrame, set `raise_if_empty=False`"
            )
        return pl.DataFrame()

    # otherwise rewind the buffer and parse as csv
    csv_buffer.seek(0)
    return read_csv(csv_buffer, **read_csv_options)
