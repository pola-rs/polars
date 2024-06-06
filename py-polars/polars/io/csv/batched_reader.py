from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Sequence

from polars._utils.various import (
    _process_null_values,
    normalize_filepath,
)
from polars._utils.wrap import wrap_df
from polars.datatypes import N_INFER_DEFAULT, py_type_to_dtype
from polars.io._utils import parse_columns_arg, parse_row_index_args
from polars.io.csv._utils import _update_columns

with contextlib.suppress(ImportError):  # Module not available when building docs
    from polars.polars import PyBatchedCsv

if TYPE_CHECKING:
    from pathlib import Path

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
        comment_prefix: str | None = None,
        quote_char: str | None = '"',
        skip_rows: int = 0,
        schema_overrides: SchemaDict | Sequence[PolarsDataType] | None = None,
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
        row_index_name: str | None = None,
        row_index_offset: int = 0,
        sample_size: int = 1024,
        eol_char: str = "\n",
        new_columns: Sequence[str] | None = None,
        raise_if_empty: bool = True,
        truncate_ragged_lines: bool = False,
        decimal_comma: bool = False,
    ):
        path = normalize_filepath(source)

        dtype_list: Sequence[tuple[str, PolarsDataType]] | None = None
        dtype_slice: Sequence[PolarsDataType] | None = None
        if schema_overrides is not None:
            if isinstance(schema_overrides, dict):
                dtype_list = []
                for k, v in schema_overrides.items():
                    dtype_list.append((k, py_type_to_dtype(v)))
            elif isinstance(schema_overrides, Sequence):
                dtype_slice = schema_overrides
            else:
                msg = "`schema_overrides` arg should be list or dict"
                raise TypeError(msg)

        processed_null_values = _process_null_values(null_values)
        projection, columns = parse_columns_arg(columns)

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
            comment_prefix=comment_prefix,
            quote_char=quote_char,
            null_values=processed_null_values,
            missing_utf8_is_empty_string=missing_utf8_is_empty_string,
            try_parse_dates=try_parse_dates,
            skip_rows_after_header=skip_rows_after_header,
            row_index=parse_row_index_args(row_index_name, row_index_offset),
            sample_size=sample_size,
            eol_char=eol_char,
            raise_if_empty=raise_if_empty,
            truncate_ragged_lines=truncate_ragged_lines,
            decimal_comma=decimal_comma,
        )
        self.new_columns = new_columns

    def next_batches(self, n: int) -> list[DataFrame] | None:
        """
        Read `n` batches from the reader.

        These batches will be parallelized over the available threads.

        Parameters
        ----------
        n
            Number of chunks to fetch; ideally this is >= number of threads.

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
