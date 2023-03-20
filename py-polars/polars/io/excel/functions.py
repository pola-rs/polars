from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO, overload

from polars.internals import DataFrame
from polars.io.csv.functions import read_csv
from polars.utils.decorators import deprecate_nonkeyword_arguments, deprecated_alias
from polars.utils.various import normalise_filepath

if TYPE_CHECKING:
    import sys
    from io import BytesIO

    if sys.version_info >= (3, 8):
        from typing import Literal
    else:
        from typing_extensions import Literal


@overload
def read_excel(
    source: str | BytesIO | Path | BinaryIO | bytes,
    sheet_id: Literal[None],
    sheet_name: Literal[None],
    xlsx2csv_options: dict[str, Any] | None,
    read_csv_options: dict[str, Any] | None,
    engine: Literal["xlsx2csv", "openpyxl"] | None = None,
) -> dict[str, DataFrame]:
    ...


@overload
def read_excel(
    source: str | BytesIO | Path | BinaryIO | bytes,
    sheet_id: Literal[None],
    sheet_name: str,
    xlsx2csv_options: dict[str, Any] | None = None,
    read_csv_options: dict[str, Any] | None = None,
    engine: Literal["xlsx2csv", "openpyxl"] | None = None,
) -> DataFrame:
    ...


@overload
def read_excel(
    source: str | BytesIO | Path | BinaryIO | bytes,
    sheet_id: int,
    sheet_name: Literal[None],
    xlsx2csv_options: dict[str, Any] | None = None,
    read_csv_options: dict[str, Any] | None = None,
    engine: Literal["xlsx2csv", "openpyxl"] | None = None,
) -> DataFrame:
    ...


@deprecate_nonkeyword_arguments()
@deprecated_alias(file="source")
def read_excel(
    source: str | BytesIO | Path | BinaryIO | bytes,
    sheet_id: int | None = 0,
    sheet_name: str | None = None,
    xlsx2csv_options: dict[str, Any] | None = None,
    read_csv_options: dict[str, Any] | None = None,
    engine: Literal["xlsx2csv", "openpyxl"] | None = None,
) -> DataFrame | dict[str, DataFrame]:
    """
    Read Excel (XLSX) sheet into a DataFrame.

    Converts an Excel sheet with ``xlsx2csv.Xlsx2csv().convert()`` to CSV and parses the
    CSV output with :func:`read_csv`.

    Parameters
    ----------
    source
        Path to a file or a file-like object.
        By file-like object, we refer to objects with a ``read()`` method, such as a
        file handler (e.g. via builtin ``open`` function) or ``BytesIO``.
    sheet_id
        Sheet number to convert (0 for all sheets).
    sheet_name
        Sheet name to convert.
    xlsx2csv_options
        Extra options passed to ``xlsx2csv.Xlsx2csv()``.
        e.g.: ``{"skip_empty_lines": True}``
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
    Returns
    -------
    DataFrame | dict[str, DataFrame]


    Examples
    --------
    Read "My Datasheet" sheet from Excel sheet file to a DataFrame.

    >>> pl.read_excel(
    ...     "test.xlsx",
    ...     sheet_name="My Datasheet",
    ... )  # doctest: +SKIP

    Read sheet 3 from Excel sheet file to a DataFrame while skipping empty lines in the
    sheet. As sheet 3 does not have header row, pass the needed settings to
    :func:`read_csv`.

    >>> pl.read_excel(
    ...     "test.xlsx",
    ...     sheet_id=3,
    ...     xlsx2csv_options={"skip_empty_lines": True},
    ...     read_csv_options={"has_header": False, "new_columns": ["a", "b", "c"]},
    ... )  # doctest: +SKIP

    If the correct datatypes can't be determined by polars, look at :func:`read_csv`
    documentation to see which options you can pass to fix this issue. For example
    ``"infer_schema_length": None`` can be used to read the whole data twice, once to
    infer the correct output types and once to actually convert the input to the correct
    types. With `"infer_schema_length": 1000``, only the first 1000 lines are read
    twice.

    >>> pl.read_excel(
    ...     "test.xlsx",
    ...     read_csv_options={"infer_schema_length": None},
    ... )  # doctest: +SKIP

    If :func:`read_excel` does not work or you need to read other types of spreadsheet
    files, you can try pandas ``pd.read_excel()``
    (supports `xls`, `xlsx`, `xlsm`, `xlsb`, `odf`, `ods` and `odt`).

    >>> pl.from_pandas(pd.read_excel("test.xlsx"))  # doctest: +SKIP

    """
    if isinstance(source, (str, Path)):
        source = normalise_filepath(source)

    if not xlsx2csv_options:
        xlsx2csv_options = {}

    if not read_csv_options:
        read_csv_options = {}

    reader_fn: Any  # make mypy happy
    # do conditions imports
    if engine == "openpyxl":
        try:
            import openpyxl  # type: ignore[import]
        except ImportError:
            raise ImportError(
                "openpyxl is not installed. Please run `pip install openpyxl`."
            ) from None
        parser = openpyxl.load_workbook(source, read_only=True)
        sheets = [{"index": i, "name": sheet.title} for i, sheet in enumerate(parser)]
        reader_fn = _read_excel_sheet_openpyxl
        # setup good defaults for the sheet id
        engine_sheet_id = sheet_id
    elif engine == "xlsx2csv" or engine is None:  # default
        try:
            import xlsx2csv
        except ImportError:
            raise ImportError(
                "xlsx2csv is not installed. Please run `pip install xlsx2csv`."
            ) from None
        # Convert sheets from XSLX document to CSV.
        parser = xlsx2csv.Xlsx2csv(source, **xlsx2csv_options)
        sheets = parser.workbook.sheets
        reader_fn = _read_excel_sheet_xlsx2csv
        # setup good defaults for the sheet id
        engine_sheet_id = 1 if sheet_id == 0 else sheet_id
    else:
        raise NotImplementedError(f"Cannot find the {engine} engine")

    if sheet_id is None and sheet_name is None:
        ret_val = {
            sheet["name"]: reader_fn(parser, sheet["index"], None, read_csv_options)
            for sheet in sheets
        }
    else:
        ret_val = reader_fn(parser, engine_sheet_id, sheet_name, read_csv_options)

    if engine == "openpyxl":
        # close iterator
        parser.close()
    return ret_val


def _read_excel_sheet_openpyxl(
    parser: Any,
    sheet_id: int | None,
    sheet_name: str | None,
    _: dict[str, Any] | None,
) -> DataFrame:
    # read requested sheet if provided on kwargs, otherwise read active sheet
    if sheet_name is not None:
        ws = parser[sheet_name]
    elif sheet_id is not None:
        ws = parser.worksheets[sheet_id]
    else:
        ws = parser.active

    rows_iter = iter(ws.rows)

    # check whether to include or omit the header
    header = [str(cell.value) for cell in next(rows_iter)]

    df = DataFrame(
        {key: cell.value for key, cell in zip(header, row)} for row in rows_iter
    )
    return df


def _read_excel_sheet_xlsx2csv(
    parser: Any,
    sheet_id: int | None,
    sheet_name: str | None,
    read_csv_options: dict[str, Any],
) -> DataFrame:
    csv_buffer = StringIO()

    # Parse XLSX sheet to CSV.
    parser.convert(outfile=csv_buffer, sheetid=sheet_id, sheetname=sheet_name)

    # Rewind buffer to start.
    csv_buffer.seek(0)

    # Parse CSV output.
    return read_csv(csv_buffer, **read_csv_options)
