from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO, Callable, Mapping, Sequence, TextIO

import polars._reexport as pl
from polars.datatypes import N_INFER_DEFAULT, String
from polars.io._utils import _prepare_file_arg
from polars.io.csv._utils import _check_arg_is_1byte, _update_columns
from polars.io.csv.batched_reader import BatchedCsvReader
from polars.utils.deprecation import deprecate_renamed_parameter
from polars.utils.various import handle_projection_columns, normalize_filepath

if TYPE_CHECKING:
    from io import BytesIO

    from polars import DataFrame, LazyFrame
    from polars.type_aliases import CsvEncoding, PolarsDataType, SchemaDict


@deprecate_renamed_parameter(
    old_name="comment_char", new_name="comment_prefix", version="0.19.14"
)
def read_csv(
    source: str | TextIO | BytesIO | Path | BinaryIO | bytes,
    *,
    has_header: bool = True,
    columns: Sequence[int] | Sequence[str] | None = None,
    new_columns: Sequence[str] | None = None,
    separator: str = ",",
    comment_prefix: str | None = None,
    quote_char: str | None = '"',
    skip_rows: int = 0,
    dtypes: Mapping[str, PolarsDataType] | Sequence[PolarsDataType] | None = None,
    schema: SchemaDict | None = None,
    null_values: str | Sequence[str] | dict[str, str] | None = None,
    missing_utf8_is_empty_string: bool = False,
    ignore_errors: bool = False,
    try_parse_dates: bool = False,
    n_threads: int | None = None,
    infer_schema_length: int | None = N_INFER_DEFAULT,
    batch_size: int = 8192,
    n_rows: int | None = None,
    encoding: CsvEncoding | str = "utf8",
    low_memory: bool = False,
    rechunk: bool = True,
    use_pyarrow: bool = False,
    storage_options: dict[str, Any] | None = None,
    skip_rows_after_header: int = 0,
    row_count_name: str | None = None,
    row_count_offset: int = 0,
    sample_size: int = 1024,
    eol_char: str = "\n",
    raise_if_empty: bool = True,
    truncate_ragged_lines: bool = False,
) -> DataFrame:
    r"""
    Read a CSV file into a `DataFrame`.

    Parameters
    ----------
    source
        A path to a CSV file or a file-like object. By file-like object, we refer to
        objects that have a `read()` method, such as a file handler (e.g. from the
        builtin `open <https://docs.python.org/3/library/functions.html#open>`_
        function) or `BytesIO <https://docs.python.org/3/library/io.html#io.BytesIO>`_.
        If `fsspec <https://filesystem-spec.readthedocs.io>`_ is installed, it will be
        used to open remote files.
    has_header
        Whether to interpret the first row of the dataset as a header row.
        If `has_header=False`, the first column will be named `'column_1'`, the second
        `'column_2'`, and so on.
    columns
        A list of column indices (starting at zero) or column names to read.
    new_columns
        A list of column names that will overwrite the original column names after
        parsing the CSV file. Often used in combination with `has_header=False`.
        If the given list is shorter than the width of the `DataFrame`, the remaining
        columns will retain their original names.
    separator
        A single-byte character to interpret as the separator between CSV fields.
    comment_prefix
        A string of 1 to 5 characters (e.g. `#` or `//`) denoting a comment. The CSV
        reader will skip lines starting with this string.
    quote_char
        A single-byte character to interpret as the CSV quote character.
        Setting `quote_char=None` disables special handling and escaping of quotes.
    skip_rows
        The number of rows to ignore before starting to read the CSV. The header will
        be parsed at this offset.
    dtypes
        A `{colname: dtype}` dictionary of columns to read in as specific dtypes,
        instead of auto-inferring the dtypes for these columns. Alternately, a list of
        dtypes of length `N`, where `N` is less than or equal to the number of columns
        in the CSV file; the first `N` columns will be read in as these dtypes.
        The list version is especially useful when specifying `new_columns`.
    schema
        A `{colname: dtype}` dictionary of columns to read in as specific dtypes.
        This argument requires all columns to be listed, whereas `dtypes` requires only
        certain columns.
    null_values
        Values to interpret as `null` values. You can provide a:

        - `str`: Values equal to this string will be interpreted as `null`.
        - `List[str]`: Values equal to any string in the list will be interpreted as
          `null`.
        - `Dict[str, str]`: A dictionary that maps column names to `null` value strings.
    missing_utf8_is_empty_string
        Whether to read in missing values in :class:`String` columns as `""` instead of
        `null`.
    ignore_errors
        Whether to try to keep reading lines if some lines yield errors.
        Before using this option, try the following troubleshooting steps:

        - Increase the number of rows used for schema inference, e.g. with
          `infer_schema_length=10000`.
        - Override automatic dtype inference for specific columns with the `dtypes`
          option.
        - Use `infer_schema_length=0` to read all columns as :class:`String`, to check
          which values might be causing an issue.
    try_parse_dates
        Whether to try to automatically parse dates. Most ISO8601-like formats can be
        inferred, as well as a handful of others. If this does not succeed, the column
        remains of data type :class:`String`. If `use_pyarrow=True`, dates will always
        be parsed.
    n_threads
        The number of threads to use during CSV parsing. Defaults to the number of
        physical CPUs in your system.
    infer_schema_length
        The maximum number of rows to read when automatically inferring columns'
        dtypes. If dtypes are inferred wrongly (e.g. as :class:`Int64` instead of
        :class:`Float64`), try to increase `infer_schema_length` or override the
        inferred dtype for those columns with `dtypes`.
        If `infer_schema_length=0`, all columns will be read as :class:`String`.
        If `infer_schema_length=None`, the entire CSV will be scanned to infer dtypes
        (slow).
    batch_size
        The number of rows at a time to read into an intermediate buffer during CSV
        reading. Modify this to change performance.
    n_rows
        The number of rows to read from the CSV file. During multi-threaded reading,
        more than `n_rows` lines may be read, but only `n_rows` rows will appear in the
        returned `DataFrame`.
    encoding : {'utf8', 'utf8-lossy', ...}
        `'utf8-lossy'` means that invalid UTF-8 values are replaced with `�` characters.
        Encodings other than 'utf8'` and `'utf8-lossy'` are only supported when
        `use_pyarrow=True`; when using other encodings, the input is first decoded in
        memory with Python. Defaults to `'utf8'`.
    low_memory
        Whether to reduce memory usage at the expense of speed.
    rechunk
        Whether to ensure each column of the result is stored contiguously in
        memory; see :func:`rechunk` for details.
    use_pyarrow
        Whether to attempt to use the CSV parser from :mod:`pyarrow` instead of Polars.
        This is not always possible and depends on which other options are specified.
        Type inference may differ from Polars: for instance, this will always parse
        dates, even if `try_parse_dates=False`.
    storage_options
        Keyword arguments passed to `fsspec.open()
        <https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.open>`_.
        According to `fsspec`, these represent "Extra options that make sense to a
        particular storage connection, e.g. host, port, username, password, etc.".
    skip_rows_after_header
        The number of rows to ignore after the header and before starting to read
        the rest of the CSV.
    row_count_name
        If not `None`, add a row count column with this name as the first column.
    row_count_offset
        An integer offset to start the row count at; only used when `row_count_name`
        is not `None`.
    sample_size
        The number of rows to initially read from the CSV file when deciding how much
        memory to allocate when reading the full file. Increasing this may improve
        performance.
    eol_char
        A single-byte character to interpret as the end-of-line character. The default
        `"\n"` also works for files with Windows line endings (`"\r\n"`); the extra
        `"\r"` will be removed during processing.
    raise_if_empty
        Whether to raise a :class:`NoDataError` instead of returning an empty
        `DataFrame` with no columns when the source file is empty.
    truncate_ragged_lines
        Whether to truncate "ragged" lines that are longer than the first line, instead
        of raising a :class:`ComputeError`. If `schema` is not `None`, the length of the
        schema is used instead of the length of the first line.

    Returns
    -------
    DataFrame

    See Also
    --------
    scan_csv : Lazily read from a CSV file, or multiple files via glob patterns.

    Notes
    -----
    This operation defaults to a :func:`rechunk` operation at the end, meaning
    that each column will be stored continuously in memory.
    Set `rechunk=False` if you are benchmarking the csv-reader. A `rechunk` is
    an expensive operation.

    """
    _check_arg_is_1byte("separator", separator, can_be_empty=False)
    _check_arg_is_1byte("quote_char", quote_char, can_be_empty=True)
    _check_arg_is_1byte("eol_char", eol_char, can_be_empty=False)

    projection, columns = handle_projection_columns(columns)
    storage_options = storage_options or {}

    if columns and not has_header:
        for column in columns:
            if not column.startswith("column_"):
                raise ValueError(
                    "specified column names do not start with 'column_',"
                    " but autogenerated header names were requested"
                )

    if (
        use_pyarrow
        and dtypes is None
        and n_rows is None
        and n_threads is None
        and not low_memory
        and null_values is None
    ):
        include_columns: Sequence[str] | None = None
        if columns:
            if not has_header:
                # Convert 'column_1', 'column_2', ... column names to 'f0', 'f1', ...
                # column names for pyarrow, if CSV file does not contain a header.
                include_columns = [f"f{int(column[7:]) - 1}" for column in columns]
            else:
                include_columns = columns

        if not columns and projection:
            # Convert column indices from projection to 'f0', 'f1', ... column names
            # for pyarrow.
            include_columns = [f"f{column_idx}" for column_idx in projection]

        with _prepare_file_arg(
            source,
            encoding=None,
            use_pyarrow=True,
            raise_if_empty=raise_if_empty,
            storage_options=storage_options,
        ) as data:
            import pyarrow as pa
            import pyarrow.csv

            try:
                tbl = pa.csv.read_csv(
                    data,
                    pa.csv.ReadOptions(
                        skip_rows=skip_rows,
                        autogenerate_column_names=not has_header,
                        encoding=encoding,
                    ),
                    pa.csv.ParseOptions(
                        delimiter=separator,
                        quote_char=quote_char if quote_char else False,
                        double_quote=quote_char is not None and quote_char == '"',
                    ),
                    pa.csv.ConvertOptions(
                        column_types=None,
                        include_columns=include_columns,
                        include_missing_columns=ignore_errors,
                    ),
                )
            except pa.ArrowInvalid as err:
                if raise_if_empty or "Empty CSV" not in str(err):
                    raise
                return pl.DataFrame()

        if not has_header:
            # Rename 'f0', 'f1', ... columns names autogenerated by pyarrow
            # to 'column_1', 'column_2', ...
            tbl = tbl.rename_columns(
                [f"column_{int(column[1:]) + 1}" for column in tbl.column_names]
            )

        df = pl.DataFrame._from_arrow(tbl, rechunk=rechunk)
        if new_columns:
            return _update_columns(df, new_columns)
        return df

    if projection and dtypes and isinstance(dtypes, list):
        if len(projection) < len(dtypes):
            raise ValueError(
                "more dtypes overrides are specified than there are selected columns"
            )

        # Fix list of dtypes when used together with projection as polars CSV reader
        # wants a list of dtypes for the x first columns before it does the projection.
        dtypes_list: list[PolarsDataType] = [String] * (max(projection) + 1)

        for idx, column_idx in enumerate(projection):
            if idx < len(dtypes):
                dtypes_list[column_idx] = dtypes[idx]

        dtypes = dtypes_list

    if columns and dtypes and isinstance(dtypes, list):
        if len(columns) < len(dtypes):
            raise ValueError(
                "more dtypes overrides are specified than there are selected columns"
            )

        # Map list of dtypes when used together with selected columns as a dtypes dict
        # so the dtypes are applied to the correct column instead of the first x
        # columns.
        dtypes = dict(zip(columns, dtypes))

    if new_columns and dtypes and isinstance(dtypes, dict):
        current_columns = None

        # As new column names are not available yet while parsing the CSV file, rename
        # column names in dtypes to old names (if possible) so they can be used during
        # CSV parsing.
        if columns:
            if len(columns) < len(new_columns):
                raise ValueError(
                    "more new column names are specified than there are selected"
                    " columns"
                )

            # Get column names of requested columns.
            current_columns = columns[0 : len(new_columns)]
        elif not has_header:
            # When there are no header, column names are autogenerated (and known).

            if projection:
                if columns and len(columns) < len(new_columns):
                    raise ValueError(
                        "more new column names are specified than there are selected"
                        " columns"
                    )
                # Convert column indices from projection to 'column_1', 'column_2', ...
                # column names.
                current_columns = [
                    f"column_{column_idx + 1}" for column_idx in projection
                ]
            else:
                # Generate autogenerated 'column_1', 'column_2', ... column names for
                # new column names.
                current_columns = [
                    f"column_{column_idx}"
                    for column_idx in range(1, len(new_columns) + 1)
                ]
        else:
            # When a header is present, column names are not known yet.

            if len(dtypes) <= len(new_columns):
                # If dtypes dictionary contains less or same amount of values than new
                # column names a list of dtypes can be created if all listed column
                # names in dtypes dictionary appear in the first consecutive new column
                # names.
                dtype_list = [
                    dtypes[new_column_name]
                    for new_column_name in new_columns[0 : len(dtypes)]
                    if new_column_name in dtypes
                ]

                if len(dtype_list) == len(dtypes):
                    dtypes = dtype_list

        if current_columns and isinstance(dtypes, dict):
            new_to_current = dict(zip(new_columns, current_columns))
            # Change new column names to current column names in dtype.
            dtypes = {
                new_to_current.get(column_name, column_name): column_dtype
                for column_name, column_dtype in dtypes.items()
            }

    with _prepare_file_arg(
        source,
        encoding=encoding,
        use_pyarrow=False,
        raise_if_empty=raise_if_empty,
        storage_options=storage_options,
    ) as data:
        df = pl.DataFrame._read_csv(
            data,
            has_header=has_header,
            columns=columns if columns else projection,
            separator=separator,
            comment_prefix=comment_prefix,
            quote_char=quote_char,
            skip_rows=skip_rows,
            dtypes=dtypes,
            schema=schema,
            null_values=null_values,
            missing_utf8_is_empty_string=missing_utf8_is_empty_string,
            ignore_errors=ignore_errors,
            try_parse_dates=try_parse_dates,
            n_threads=n_threads,
            infer_schema_length=infer_schema_length,
            batch_size=batch_size,
            n_rows=n_rows,
            encoding=encoding if encoding == "utf8-lossy" else "utf8",
            low_memory=low_memory,
            rechunk=rechunk,
            skip_rows_after_header=skip_rows_after_header,
            row_count_name=row_count_name,
            row_count_offset=row_count_offset,
            sample_size=sample_size,
            eol_char=eol_char,
            raise_if_empty=raise_if_empty,
            truncate_ragged_lines=truncate_ragged_lines,
        )

    if new_columns:
        return _update_columns(df, new_columns)
    return df


@deprecate_renamed_parameter(
    old_name="comment_char", new_name="comment_prefix", version="0.19.14"
)
def read_csv_batched(
    source: str | Path,
    *,
    has_header: bool = True,
    columns: Sequence[int] | Sequence[str] | None = None,
    new_columns: Sequence[str] | None = None,
    separator: str = ",",
    comment_prefix: str | None = None,
    quote_char: str | None = '"',
    skip_rows: int = 0,
    dtypes: Mapping[str, PolarsDataType] | Sequence[PolarsDataType] | None = None,
    null_values: str | Sequence[str] | dict[str, str] | None = None,
    missing_utf8_is_empty_string: bool = False,
    ignore_errors: bool = False,
    try_parse_dates: bool = False,
    n_threads: int | None = None,
    infer_schema_length: int | None = N_INFER_DEFAULT,
    batch_size: int = 50_000,
    n_rows: int | None = None,
    encoding: CsvEncoding | str = "utf8",
    low_memory: bool = False,
    rechunk: bool = True,
    skip_rows_after_header: int = 0,
    row_count_name: str | None = None,
    row_count_offset: int = 0,
    sample_size: int = 1024,
    eol_char: str = "\n",
    raise_if_empty: bool = True,
) -> BatchedCsvReader:
    r"""
    Read a CSV file in batches.

    This function does not read in the data itself, but merely gathers statistics and
    determines the file chunks to read in, returning a :class:`BatchedCsvReader`.
    Calling :func:`polars.io.csv.batched_reader.BatchedCsvReader.next_batches` will
    return a list of `n` dataframes of the given `batch_size`.

    Parameters
    ----------
    source
        A path to a CSV file or a file-like object. By file-like object, we refer to
        objects that have a `read()` method, such as a file handler (e.g. from the
        builtin `open <https://docs.python.org/3/library/functions.html#open>`_
        function) or `BytesIO <https://docs.python.org/3/library/io.html#io.BytesIO>`_.
        If `fsspec <https://filesystem-spec.readthedocs.io>`_ is installed, it will be
        used to open remote files.
    has_header
        Whether to interpret the first row of the dataset as a header row.
        If `has_header=False`, the first column will be named `'column_1'`, the second
        `'column_2'`, and so on.
    columns
        A list of column indices (starting at zero) or column names to read.
    new_columns
        A list of column names that will overwrite the original column names after
        parsing the CSV file. Often used in combination with `has_header=False`.
        If the given list is shorter than the width of the `DataFrame`, the remaining
        columns will retain their original names.
    separator
        A single-byte character to interpret as the separator between CSV fields.
    comment_prefix
        A string of 1 to 5 characters (e.g. `#` or `//`) denoting a comment. The CSV
        reader will skip lines starting with this string.
    quote_char
        A single-byte character to interpret as the CSV quote character.
        Setting `quote_char=None` disables special handling and escaping of quotes.
    skip_rows
        The number of rows to ignore before starting to read the CSV. The header will
        be parsed at this offset.
    dtypes
        A `{colname: dtype}` dictionary of columns to read in as specific dtypes,
        instead of auto-inferring the dtypes for these columns. Alternately, a list of
        dtypes of length `N`, where `N` is less than or equal to the number of columns
        in the CSV file; the first `N` columns will be read in as these dtypes.
        The list version is especially useful when specifying `new_columns`.
    null_values
        Values to interpret as `null` values. You can provide a:

        - `str`: Values equal to this string will be interpreted as `null`.
        - `List[str]`: Values equal to any string in the list will be interpreted as
          `null`.
        - `Dict[str, str]`: A dictionary that maps column names to `null` value strings.
    missing_utf8_is_empty_string
        Whether to read in missing values in :class:`String` columns as `""` instead of
        `null`.
    ignore_errors
        Whether to try to keep reading lines if some lines yield errors.
        Before using this option, try the following troubleshooting steps:

        - Increase the number of rows used for schema inference, e.g. with
          `infer_schema_length=10000`.
        - Override automatic dtype inference for specific columns with the `dtypes`
          option.
        - Use `infer_schema_length=0` to read all columns as :class:`String`, to check
          which values might be causing an issue.
    try_parse_dates
        Whether to try to automatically parse dates. Most ISO8601-like formats can be
        inferred, as well as a handful of others. If this does not succeed, the column
        remains of data type :class:`String`.
    n_threads
        The number of threads to use during CSV parsing. Defaults to the number of
        physical CPUs in your system.
    infer_schema_length
        The maximum number of rows to read when automatically inferring columns'
        dtypes. If dtypes are inferred wrongly (e.g. as :class:`Int64` instead of
        :class:`Float64`), try to increase `infer_schema_length` or override the
        inferred dtype for those columns with `dtypes`.
        If `infer_schema_length=0`, all columns will be read as :class:`String`.
        If `infer_schema_length=None`, the entire CSV will be scanned to infer dtypes
        (slow).
    batch_size
        The number of rows at a time to read into an intermediate buffer during CSV
        reading. Modify this to change performance.
    n_rows
        The number of rows to read from the CSV file. During multi-threaded reading,
        more than `n_rows` rows may be read, but only `n_rows` rows will appear in the
        returned `DataFrame`.
    encoding : {'utf8', 'utf8-lossy'}
        `'utf8-lossy'` means that invalid utf8 values are replaced with `�` characters.
        Defaults to `'utf8'`.
    low_memory
        Whether to reduce memory usage at the expense of speed.
    rechunk
        Whether to ensure each column of the result is stored contiguously in
        memory; see :func:`rechunk` for details.
    skip_rows_after_header
        The number of rows to ignore after the header and before starting to read
        the rest of the CSV.
    row_count_name
        If not `None`, add a row count column with this name as the first column.
    row_count_offset
        An integer offset to start the row count at; only used when `row_count_name`
        is not `None`.
    sample_size
        The number of rows to initially read from the CSV file when deciding how much
        memory to allocate when reading the full file. Increasing this may improve
        performance.
    eol_char
        A single-byte character to interpret as the end-of-line character. The default
        `"\n"` also works for files with Windows line endings (`"\r\n"`); the extra
        `"\r"` will be removed during processing.
    raise_if_empty
        Whether to raise a :class:`NoDataError` instead of returning an empty
        `DataFrame` with no columns when the source file is empty.

    Returns
    -------
    BatchedCsvReader

    See Also
    --------
    scan_csv : Lazily read from a CSV file, or multiple files via glob patterns.

    Examples
    --------
    >>> reader = pl.read_csv_batched(
    ...     "./tpch/tables_scale_100/lineitem.tbl",
    ...     separator="|",
    ...     try_parse_dates=True,
    ... )  # doctest: +SKIP
    >>> batches = reader.next_batches(5)  # doctest: +SKIP
    >>> for df in batches:  # doctest: +SKIP
    ...     print(df)

    Read big CSV file in batches and write a CSV file for each "group" of interest.

    >>> seen_groups = set()
    >>> reader = pl.read_csv_batched("big_file.csv")  # doctest: +SKIP
    >>> batches = reader.next_batches(100)  # doctest: +SKIP

    >>> while batches:  # doctest: +SKIP
    ...     df_current_batches = pl.concat(batches)
    ...     partition_dfs = df_current_batches.partition_by("group", as_dict=True)
    ...
    ...     for group, df in partition_dfs.items():
    ...         if group in seen_groups:
    ...             with open(f"./data/{group}.csv", "a") as fh:
    ...                 fh.write(df.write_csv(file=None, include_header=False))
    ...         else:
    ...             df.write_csv(file=f"./data/{group}.csv", include_header=True)
    ...         seen_groups.add(group)
    ...
    ...     batches = reader.next_batches(100)

    """
    projection, columns = handle_projection_columns(columns)

    if columns and not has_header:
        for column in columns:
            if not column.startswith("column_"):
                raise ValueError(
                    "specified column names do not start with 'column_',"
                    " but autogenerated header names were requested"
                )

    if projection and dtypes and isinstance(dtypes, list):
        if len(projection) < len(dtypes):
            raise ValueError(
                "more dtypes overrides are specified than there are selected columns"
            )

        # Fix list of dtypes when used together with projection as polars CSV reader
        # wants a list of dtypes for the x first columns before it does the projection.
        dtypes_list: list[PolarsDataType] = [String] * (max(projection) + 1)

        for idx, column_idx in enumerate(projection):
            if idx < len(dtypes):
                dtypes_list[column_idx] = dtypes[idx]

        dtypes = dtypes_list

    if columns and dtypes and isinstance(dtypes, list):
        if len(columns) < len(dtypes):
            raise ValueError(
                "more dtypes overrides are specified than there are selected columns"
            )

        # Map list of dtypes when used together with selected columns as a dtypes dict
        # so the dtypes are applied to the correct column instead of the first x
        # columns.
        dtypes = dict(zip(columns, dtypes))

    if new_columns and dtypes and isinstance(dtypes, dict):
        current_columns = None

        # As new column names are not available yet while parsing the CSV file, rename
        # column names in dtypes to old names (if possible) so they can be used during
        # CSV parsing.
        if columns:
            if len(columns) < len(new_columns):
                raise ValueError(
                    "more new column names are specified than there are selected columns"
                )

            # Get column names of requested columns.
            current_columns = columns[0 : len(new_columns)]
        elif not has_header:
            # When there are no header, column names are autogenerated (and known).

            if projection:
                if columns and len(columns) < len(new_columns):
                    raise ValueError(
                        "more new column names are specified than there are selected columns"
                    )
                # Convert column indices from projection to 'column_1', 'column_2', ...
                # column names.
                current_columns = [
                    f"column_{column_idx + 1}" for column_idx in projection
                ]
            else:
                # Generate autogenerated 'column_1', 'column_2', ... column names for
                # new column names.
                current_columns = [
                    f"column_{column_idx}"
                    for column_idx in range(1, len(new_columns) + 1)
                ]
        else:
            # When a header is present, column names are not known yet.

            if len(dtypes) <= len(new_columns):
                # If dtypes dictionary contains less or same amount of values than new
                # column names a list of dtypes can be created if all listed column
                # names in dtypes dictionary appear in the first consecutive new column
                # names.
                dtype_list = [
                    dtypes[new_column_name]
                    for new_column_name in new_columns[0 : len(dtypes)]
                    if new_column_name in dtypes
                ]

                if len(dtype_list) == len(dtypes):
                    dtypes = dtype_list

        if current_columns and isinstance(dtypes, dict):
            new_to_current = dict(zip(new_columns, current_columns))
            # Change new column names to current column names in dtype.
            dtypes = {
                new_to_current.get(column_name, column_name): column_dtype
                for column_name, column_dtype in dtypes.items()
            }

    return BatchedCsvReader(
        source,
        has_header=has_header,
        columns=columns if columns else projection,
        separator=separator,
        comment_prefix=comment_prefix,
        quote_char=quote_char,
        skip_rows=skip_rows,
        dtypes=dtypes,
        null_values=null_values,
        missing_utf8_is_empty_string=missing_utf8_is_empty_string,
        ignore_errors=ignore_errors,
        try_parse_dates=try_parse_dates,
        n_threads=n_threads,
        infer_schema_length=infer_schema_length,
        batch_size=batch_size,
        n_rows=n_rows,
        encoding=encoding if encoding == "utf8-lossy" else "utf8",
        low_memory=low_memory,
        rechunk=rechunk,
        skip_rows_after_header=skip_rows_after_header,
        row_count_name=row_count_name,
        row_count_offset=row_count_offset,
        sample_size=sample_size,
        eol_char=eol_char,
        new_columns=new_columns,
        raise_if_empty=raise_if_empty,
    )


@deprecate_renamed_parameter(
    old_name="comment_char", new_name="comment_prefix", version="0.19.14"
)
def scan_csv(
    source: str | Path | list[str] | list[Path],
    *,
    has_header: bool = True,
    separator: str = ",",
    comment_prefix: str | None = None,
    quote_char: str | None = '"',
    skip_rows: int = 0,
    dtypes: SchemaDict | Sequence[PolarsDataType] | None = None,
    schema: SchemaDict | None = None,
    null_values: str | Sequence[str] | dict[str, str] | None = None,
    missing_utf8_is_empty_string: bool = False,
    ignore_errors: bool = False,
    cache: bool = True,
    with_column_names: Callable[[list[str]], list[str]] | None = None,
    infer_schema_length: int | None = N_INFER_DEFAULT,
    n_rows: int | None = None,
    encoding: CsvEncoding = "utf8",
    low_memory: bool = False,
    rechunk: bool = True,
    skip_rows_after_header: int = 0,
    row_count_name: str | None = None,
    row_count_offset: int = 0,
    try_parse_dates: bool = False,
    eol_char: str = "\n",
    new_columns: Sequence[str] | None = None,
    raise_if_empty: bool = True,
    truncate_ragged_lines: bool = False,
) -> LazyFrame:
    r"""
    Lazily read from a CSV file, or multiple files via glob patterns.

    This allows the query optimizer to push down predicates and projections to the scan
    level, potentially reducing memory overhead.

    Parameters
    ----------
    source
        A path to a CSV file, or a glob pattern matching multiple files.
    has_header
        Whether to interpret the first row of the dataset as a header row.
        If `has_header=False`, the first column will be named `'column_1'`, the second
        `'column_2'`, and so on.
    separator
        A single-byte character to interpret as the separator between CSV fields.
    comment_prefix
        A string of 1 to 5 characters (e.g. `#` or `//`) denoting a comment. The CSV
        reader will skip lines starting with this string.
    quote_char
        A single-byte character to interpret as the CSV quote character.
        Setting `quote_char=None` disables special handling and escaping of quotes.
    skip_rows
        The number of rows to ignore before starting to read the CSV. The header will
        be parsed at this offset.
    dtypes
        A `{colname: dtype}` dictionary of columns to read in as specific dtypes,
        instead of auto-inferring the dtypes for these columns. Alternately, a list of
        dtypes of length `N`, where `N` is less than or equal to the number of columns
        in the CSV file; the first `N` columns will be read in as these dtypes.
        The list version is especially useful when specifying `new_columns`.
    schema
        A `{colname: dtype}` dictionary of columns to read in as specific dtypes.
        This argument requires all columns to be listed, whereas `dtypes` requires only
        certain columns.
    null_values
        Values to interpret as `null` values. You can provide a:

        - `str`: Values equal to this string will be interpreted as `null`.
        - `List[str]`: Values equal to any string in the list will be interpreted as
          `null`.
        - `Dict[str, str]`: A dictionary that maps column names to `null` value strings.
    missing_utf8_is_empty_string
        Whether to read in missing values in :class:`String` columns as `""` instead of
        `null`.
    ignore_errors
        Whether to try to keep reading lines if some lines yield errors.
        Before using this option, try the following troubleshooting steps:

        - Increase the number of rows used for schema inference, e.g. with
          `infer_schema_length=10000`.
        - Override automatic dtype inference for specific columns with the `dtypes`
          option.
        - Use `infer_schema_length=0` to read all columns as :class:`String`, to check
          which values might be causing an issue.
    cache
        Whether to cache the result after reading.
    with_column_names
        Apply a function over the column names just in time (when they are determined);
        this function will receive (and should return) a list of column names.
    infer_schema_length
        The maximum number of rows to read when automatically inferring columns'
        dtypes. If dtypes are inferred wrongly (e.g. as :class:`Int64` instead of
        :class:`Float64`), try to increase `infer_schema_length` or override the
        inferred dtype for those columns with `dtypes`.
        If `infer_schema_length=0`, all columns will be read as :class:`String`.
        If `infer_schema_length=None`, the entire CSV will be scanned to infer dtypes
        (slow).
    n_rows
        The number of rows to read from the CSV file. During multi-threaded reading,
        more than `n_rows` rows may be read, but only `n_rows` rows will appear in the
        returned `DataFrame`.
    encoding : {'utf8', 'utf8-lossy'}
        `'utf8-lossy'` means that invalid UTF-8 values are replaced with `�` characters.
        Defaults to `'utf8'`.
    low_memory
        Whether to reduce memory usage at the expense of speed.
    rechunk
        Whether to ensure each column of the result is stored contiguously in
        memory; see :func:`rechunk` for details.
    skip_rows_after_header
        The number of rows to ignore after the header and before starting to read
        the rest of the CSV.
    row_count_name
        If not `None`, add a row count column with this name as the first column.
    row_count_offset
        An integer offset to start the row count at; only used when `row_count_name`
        is not `None`.
    try_parse_dates
        Whether to try to automatically parse dates. Most ISO8601-like formats can be
        inferred, as well as a handful of others. If this does not succeed, the column
        remains of data type :class:`String`.
    eol_char
        A single-byte character to interpret as the end-of-line character. The default
        `"\n"` also works for files with Windows line endings (`"\r\n"`); the extra
        `"\r"` will be removed during processing.
    new_columns
        A list of column names that will overwrite the original column names after
        parsing the CSV file. Often used in combination with `has_header=False`.
        If the given list is shorter than the width of the `DataFrame`, the remaining
        columns will retain their original names.
    raise_if_empty
        Whether to raise a :class:`NoDataError` instead of returning an empty
        `DataFrame` with no columns when the source file is empty.
    truncate_ragged_lines
        Whether to truncate "ragged" lines that are longer than the first line, instead
        of raising a :class:`ComputeError`. If `schema` is not `None`, the length of the
        schema is used instead of the length of the first line.

    Returns
    -------
    LazyFrame

    See Also
    --------
    read_csv : Read a CSV file into a DataFrame.

    Examples
    --------
    >>> import pathlib
    >>>
    >>> (
    ...     pl.scan_csv("my_long_file.csv")  # lazy, doesn't do a thing
    ...     .select(
    ...         ["a", "c"]
    ...     )  # select only 2 columns (other columns will not be read)
    ...     .filter(
    ...         pl.col("a") > 10
    ...     )  # the filter is pushed down the scan, so less data is read into memory
    ...     .fetch(100)  # pushed a limit of 100 rows to the scan level
    ... )  # doctest: +SKIP

    We can use `with_column_names` to modify the header before scanning:

    >>> df = pl.DataFrame(
    ...     {"BrEeZaH": [1, 2, 3, 4], "LaNgUaGe": ["is", "hard", "to", "read"]}
    ... )
    >>> path: pathlib.Path = dirpath / "mydf.csv"
    >>> df.write_csv(path)
    >>> pl.scan_csv(
    ...     path, with_column_names=lambda cols: [col.lower() for col in cols]
    ... ).collect()
    shape: (4, 2)
    ┌─────────┬──────────┐
    │ breezah ┆ language │
    │ ---     ┆ ---      │
    │ i64     ┆ str      │
    ╞═════════╪══════════╡
    │ 1       ┆ is       │
    │ 2       ┆ hard     │
    │ 3       ┆ to       │
    │ 4       ┆ read     │
    └─────────┴──────────┘

    You can also simply replace column names (or provide them if the file has none)
    by passing a list of new column names to the `new_columns` parameter:

    >>> df.write_csv(path)
    >>> pl.scan_csv(
    ...     path,
    ...     new_columns=["idx", "txt"],
    ...     dtypes=[pl.UInt16, pl.String],
    ... ).collect()
    shape: (4, 2)
    ┌─────┬──────┐
    │ idx ┆ txt  │
    │ --- ┆ ---  │
    │ u16 ┆ str  │
    ╞═════╪══════╡
    │ 1   ┆ is   │
    │ 2   ┆ hard │
    │ 3   ┆ to   │
    │ 4   ┆ read │
    └─────┴──────┘

    """
    if not new_columns and isinstance(dtypes, Sequence):
        raise TypeError(f"expected 'dtypes' dict, found {type(dtypes).__name__!r}")
    elif new_columns:
        if with_column_names:
            raise ValueError(
                "cannot set both `with_column_names` and `new_columns`; mutually exclusive"
            )
        if dtypes and isinstance(dtypes, Sequence):
            dtypes = dict(zip(new_columns, dtypes))

        # wrap new column names as a callable
        def with_column_names(cols: list[str]) -> list[str]:
            if len(cols) > len(new_columns):
                return new_columns + cols[len(new_columns) :]  # type: ignore[operator]
            else:
                return new_columns  # type: ignore[return-value]

    _check_arg_is_1byte("separator", separator, can_be_empty=False)
    _check_arg_is_1byte("quote_char", quote_char, can_be_empty=True)

    if isinstance(source, (str, Path)):
        source = normalize_filepath(source)
    else:
        source = [normalize_filepath(source) for source in source]

    return pl.LazyFrame._scan_csv(
        source,
        has_header=has_header,
        separator=separator,
        comment_prefix=comment_prefix,
        quote_char=quote_char,
        skip_rows=skip_rows,
        dtypes=dtypes,  # type: ignore[arg-type]
        schema=schema,
        null_values=null_values,
        missing_utf8_is_empty_string=missing_utf8_is_empty_string,
        ignore_errors=ignore_errors,
        cache=cache,
        with_column_names=with_column_names,
        infer_schema_length=infer_schema_length,
        n_rows=n_rows,
        low_memory=low_memory,
        rechunk=rechunk,
        skip_rows_after_header=skip_rows_after_header,
        encoding=encoding,
        row_count_name=row_count_name,
        row_count_offset=row_count_offset,
        try_parse_dates=try_parse_dates,
        eol_char=eol_char,
        raise_if_empty=raise_if_empty,
        truncate_ragged_lines=truncate_ragged_lines,
    )
