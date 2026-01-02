from __future__ import annotations

import contextlib
from pathlib import Path
from typing import IO, TYPE_CHECKING

from polars._utils.various import normalize_filepath
from polars._utils.wrap import wrap_df
from polars.io.plugins import register_io_source

with contextlib.suppress(ImportError):  # Module not available when building docs
    from polars._plr import PyPcapReader

if TYPE_CHECKING:
    from collections.abc import Iterator

    import polars._reexport as pl
    from polars import DataFrame, LazyFrame


def scan_pcap(
    source: str | Path | IO[bytes] | bytes,
    *,
    n_rows: int | None = None,
) -> LazyFrame:
    """
    Lazily read from PCAP format using an IO plugin.

    Parameters
    ----------
    source
        Path to a file or a file-like object (by "file-like object" we refer to objects
        that have a `read()` method, such as a file handler like the builtin `open`
        function, or a `BytesIO` instance).
    n_rows
        Stop reading from PCAP file after reading `n_rows`.

    Returns
    -------
    LazyFrame
    """
    if isinstance(source, (str, Path)):
        source = normalize_filepath(source)

    src = PyPcapReader(source, n_rows)

    def source_generator(
        with_columns: list[str] | None,
        predicate: pl.Expr | None,
        n_rows: int | None,
        batch_size: int | None,
    ) -> Iterator[DataFrame]:
        # TODO: Implement projection and predicate pushdown in the reader itself
        # for better performance.
        while True:
            batch = src.next_batch(batch_size or 10000)
            if batch is None:
                break

            df = wrap_df(batch)
            if with_columns is not None:
                df = df.select(with_columns)
            if predicate is not None:
                df = df.filter(predicate)
            if n_rows is not None:
                # This is a bit naive but works for a plugin.
                df = df.head(n_rows)
                n_rows -= len(df)

            yield df

            if n_rows is not None and n_rows <= 0:
                break

    return register_io_source(io_source=source_generator, schema=src.schema())


def read_pcap(
    source: str | Path | IO[bytes] | bytes,
    *,
    n_rows: int | None = None,
) -> DataFrame:
    """
    Read into a DataFrame from PCAP format.

    Parameters
    ----------
    source
        Path to a file or a file-like object (by "file-like object" we refer to objects
        that have a `read()` method, such as a file handler like the builtin `open`
        function, or a `BytesIO` instance).
    n_rows
        Stop reading from PCAP file after reading `n_rows`.

    Returns
    -------
    DataFrame
    """
    return scan_pcap(source, n_rows=n_rows).collect()
