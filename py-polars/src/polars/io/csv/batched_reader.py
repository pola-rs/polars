from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from polars.datatypes import N_INFER_DEFAULT

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from polars import DataFrame
    from polars._typing import CsvEncoding, PolarsDataType, SchemaDict


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
        skip_lines: int = 0,
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
        eol_char: str = "\n",
        new_columns: Sequence[str] | None = None,
        raise_if_empty: bool = True,
        truncate_ragged_lines: bool = False,
        decimal_comma: bool = False,
    ) -> None:
        q = pl.scan_csv(
            infer_schema_length=infer_schema_length,
            has_header=has_header,
            ignore_errors=ignore_errors,
            n_rows=n_rows,
            skip_rows=skip_rows,
            skip_lines=skip_lines,
            separator=separator,
            rechunk=rechunk,
            encoding=encoding,
            source=source,
            schema_overrides=schema_overrides,
            low_memory=low_memory,
            comment_prefix=comment_prefix,
            quote_char=quote_char,
            null_values=null_values,
            missing_utf8_is_empty_string=missing_utf8_is_empty_string,
            try_parse_dates=try_parse_dates,
            skip_rows_after_header=skip_rows_after_header,
            row_index_name=row_index_name,
            row_index_offset=row_index_offset,
            eol_char=eol_char,
            raise_if_empty=raise_if_empty,
            truncate_ragged_lines=truncate_ragged_lines,
            decimal_comma=decimal_comma,
            new_columns=new_columns,
        )

        if columns is not None:
            q = q.select(columns)

        # Trigger empty data.
        if raise_if_empty:
            q.collect_schema()
        self._reader = q.collect_batches(chunk_size=batch_size)

    def next_batches(self, n: int) -> list[DataFrame] | None:
        """
        Read `n` batches from the reader.

        Parameters
        ----------
        n
            Number of chunks to fetch.

        Examples
        --------
        >>> reader = pl.read_csv_batched(
        ...     "./pdsh/tables_scale_100/lineitem.tbl",
        ...     separator="|",
        ...     try_parse_dates=True,
        ... )  # doctest: +SKIP
        >>> reader.next_batches(5)  # doctest: +SKIP

        Returns
        -------
        list of DataFrames
        """
        chunks = []

        for _ in range(n):
            try:
                chunk = self._reader.__next__()
                if chunk is not None:
                    chunks.append(chunk)
            except StopIteration:  # noqa: PERF203
                break

        if len(chunks) > 0:
            return chunks
        return None
