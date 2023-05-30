from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO, NoReturn, overload

from polars.io.csv.functions import read_csv
from polars.utils.various import normalise_filepath

if TYPE_CHECKING:
    import sys
    from io import BytesIO

    from polars import DataFrame

    if sys.version_info >= (3, 8):
        from typing import Literal
    else:
        from typing_extensions import Literal


@overload
def read_excel(
    source: str | BytesIO | Path | BinaryIO | bytes,
    *,
    sheet_id: None = ...,
    sheet_name: str,
    xlsx2csv_options: dict[str, Any] | None = ...,
    read_csv_options: dict[str, Any] | None = ...,
) -> DataFrame:
    ...


@overload
def read_excel(
    source: str | BytesIO | Path | BinaryIO | bytes,
    *,
    sheet_id: None = ...,
    sheet_name: None = ...,
    xlsx2csv_options: dict[str, Any] | None = ...,
    read_csv_options: dict[str, Any] | None = ...,
) -> DataFrame:
    ...


@overload
def read_excel(
    source: str | BytesIO | Path | BinaryIO | bytes,
    *,
    sheet_id: int,
    sheet_name: str,
    xlsx2csv_options: dict[str, Any] | None = ...,
    read_csv_options: dict[str, Any] | None = ...,
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
) -> dict[str, DataFrame]:
    ...


@overload
def read_excel(
    source: str | BytesIO | Path | BinaryIO | bytes,
    *,
    sheet_id: int,
    sheet_name: None = ...,
    xlsx2csv_options: dict[str, Any] | None = ...,
    read_csv_options: dict[str, Any] | None = ...,
) -> DataFrame:
    ...


def read_excel(
    source: str | BytesIO | Path | BinaryIO | bytes,
    *,
    sheet_id: int | None = None,
    sheet_name: str | None = None,
    xlsx2csv_options: dict[str, Any] | None = None,
    read_csv_options: dict[str, Any] | None = None,
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
        Sheet number to convert (``0`` for all sheets). Defaults to `1` if neither this
        nor `sheet_name` are specified.
    sheet_name
        Sheet name to convert. Cannot be used in conjunction with `sheet_id`.
    xlsx2csv_options
        Extra options passed to ``xlsx2csv.Xlsx2csv()``.
        e.g.: ``{"skip_empty_lines": True}``
    read_csv_options
        Extra options passed to :func:`read_csv` for parsing the CSV file returned by
        ``xlsx2csv.Xlsx2csv().convert()``
        e.g.: ``{"has_header": False, "new_columns": ["a", "b", "c"],
        "infer_schema_length": None}``

    Returns
    -------
    DataFrame

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
    try:
        import xlsx2csv
    except ImportError:
        raise ImportError(
            "xlsx2csv is not installed. Please run `pip install xlsx2csv`."
        ) from None

    if isinstance(source, (str, Path)):
        source = normalise_filepath(source)

    if not xlsx2csv_options:
        xlsx2csv_options = {}

    if not read_csv_options:
        read_csv_options = {}

    # Convert sheets from XSLX document to CSV.
    parser = xlsx2csv.Xlsx2csv(source, **xlsx2csv_options)

    if sheet_name is None and sheet_id is None:
        return _read_excel_sheet(parser, 1, None, read_csv_options)
    elif sheet_name is None and ((sheet_id is not None) and (sheet_id > 0)):
        return _read_excel_sheet(parser, sheet_id, None, read_csv_options)
    elif sheet_name is None and ((sheet_id is not None) and (sheet_id == 0)):
        return {
            sheet["name"]: _read_excel_sheet(
                parser, sheet["index"], None, read_csv_options
            )
            for sheet in parser.workbook.sheets
        }
    elif sheet_name is not None and sheet_id is None:
        return _read_excel_sheet(parser, None, sheet_name, read_csv_options)
    else:
        raise ValueError("Cannot specify both `sheet_name` and `sheet_id`")


def _read_excel_sheet(
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
