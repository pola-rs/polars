from __future__ import annotations

import contextlib
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from polars.datatypes import N_INFER_DEFAULT, py_type_to_dtype
from polars.io.csv._utils import _update_columns
from polars.utils._wrap import wrap_df
from polars.utils.various import (
    _prepare_row_count_args,
    _process_null_values,
    handle_projection_columns,
    normalize_filepath,
)

with contextlib.suppress(ImportError):  # Module not available when building docs
    from polars.polars import PyBatchedCsv

if TYPE_CHECKING:
    from polars import DataFrame
    from polars.type_aliases import CsvEncoding, PolarsDataType, SchemaDict


class BatchedCsvReader:
    """Read a CSV file in batches."""

    def __init__(
        self,
        source: str | Path,
        *,
        has_header: bool = True,
        columns: Sequence[int] | Sequence[str] | None = None,
        separator: str = ",",
        comment_char: str | None = None,
        quote_char: str | None = '"',
        skip_rows: int = 0,
        dtypes: None | (SchemaDict | Sequence[PolarsDataType]) = None,
        null_values: str | Sequence[str] | dict[str, str] | None = None,
        missing_utf8_is_empty_string: bool = False,
        ignore_errors: bool = False,
        try_parse_dates: bool = False,
        n_threads: int | None = None,
        infer_schema_length: int | None = N_INFER_DEFAULT,
        batch_size: int = 50_000,
        n_rows: int | None = None,
        encoding: CsvEncoding = "utf8",
        low_memory: bool = False,
        rechunk: bool = True,
        skip_rows_after_header: int = 0,
        row_count_name: str | None = None,
        row_count_offset: int = 0,
        sample_size: int = 1024,
        eol_char: str = "\n",
        new_columns: Sequence[str] | None = None,
        raise_if_empty: bool = True,
        truncate_ragged_lines: bool = False,
    ):
        path: str | None
        if isinstance(source, (str, Path)):
            path = normalize_filepath(source)

        dtype_list: Sequence[tuple[str, PolarsDataType]] | None = None
        dtype_slice: Sequence[PolarsDataType] | None = None
        if dtypes is not None:
            if isinstance(dtypes, dict):
                dtype_list = []
                for k, v in dtypes.items():
                    dtype_list.append((k, py_type_to_dtype(v)))
            elif isinstance(dtypes, Sequence):
                dtype_slice = dtypes
            else:
                raise TypeError("`dtypes` arg should be list or dict")

        processed_null_values = _process_null_values(null_values)
        projection, columns = handle_projection_columns(columns)

        self._reader = PyBatchedCsv.new(
            infer_schema_length=infer_schema_length,
            chunk_size=batch_size,
            has_header=has_header,
            ignore_errors=ignore_errors,
            n_rows=n_rows,
            skip_rows=skip_rows,
            projection=projection,
            separator=separator,
            rechunk=rechunk,
            columns=columns,
            encoding=encoding,
            n_threads=n_threads,
            path=path,
            overwrite_dtype=dtype_list,
            overwrite_dtype_slice=dtype_slice,
            low_memory=low_memory,
            comment_char=comment_char,
            quote_char=quote_char,
            null_values=processed_null_values,
            missing_utf8_is_empty_string=missing_utf8_is_empty_string,
            try_parse_dates=try_parse_dates,
            skip_rows_after_header=skip_rows_after_header,
            row_count=_prepare_row_count_args(row_count_name, row_count_offset),
            sample_size=sample_size,
            eol_char=eol_char,
            raise_if_empty=raise_if_empty,
            truncate_ragged_lines=truncate_ragged_lines,
        )
        self.new_columns = new_columns

    def next_batches(self, n: int) -> list[DataFrame] | None:
        """
        Read `n` batches from the reader.

        The `n` chunks will be parallelized over the
        available threads.

        Parameters
        ----------
        n
            Number of chunks to fetch.
            This is ideally >= number of threads

        Examples
        --------
        >>> reader = pl.read_csv_batched(
        ...     "./tpch/tables_scale_100/lineitem.tbl",
        ...     separator="|",
        ...     try_parse_dates=True,
        ... )  # doctest: +SKIP
        >>> reader.next_batches(5)  # doctest: +SKIP

        Returns
        -------
        list of DataFrames

        """
        batches = self._reader.next_batches(n)
        if batches is not None:
            if self.new_columns:
                return [
                    _update_columns(wrap_df(df), self.new_columns) for df in batches
                ]
            else:
                return [wrap_df(df) for df in batches]
        return None
