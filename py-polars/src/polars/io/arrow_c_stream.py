"""Functions for scanning Arrow C Stream sources."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars._reexport as pl
from polars._utils.unstable import unstable
from polars.io.plugins import register_io_source

if TYPE_CHECKING:
    from collections.abc import Iterator

    from polars import DataFrame, Expr
    from polars.lazyframe import LazyFrame

__all__ = ["scan_arrow_c_stream"]


@unstable()
def scan_arrow_c_stream(source: Any) -> LazyFrame:
    """
    Scan a source that implements the Arrow PyCapsule Interface.

    This creates a lazy scan over an object that exposes a `__arrow_c_stream__`
    method (e.g. a `pyarrow.RecordBatchReader`, or a `nanoarrow` stream). The
    source is consumed, batch by batch, during query execution.

    .. warning::
        This functionality is considered **unstable**. It may be changed
        at any point without it being considered a breaking change.

    Parameters
    ----------
    source
        Any object implementing the Arrow PyCapsule Interface
        (i.e. it has a `__arrow_c_stream__` method).

    Returns
    -------
    LazyFrame

    Notes
    -----
    - The source must produce Arrow RecordBatches with a struct-typed schema.
      See: https://arrow.apache.org/docs/format/CStreamInterface.html.
    - Once the stream is consumed, the same source cannot be scanned again.

    Examples
    --------
    Scan from a PyArrow RecordBatchReader:

    >>> import pyarrow as pa
    >>> schema = pa.schema([("a", pa.int64()), ("b", pa.string())])
    >>> batches = [
    ...     pa.record_batch([[1, 2, 3], ["x", "y", "z"]], schema=schema),
    ...     pa.record_batch([[4, 5], ["a", "b"]], schema=schema),
    ... ]
    >>> reader = pa.RecordBatchReader.from_batches(schema, batches)
    >>> pl.scan_arrow_c_stream(reader).collect()
    shape: (5, 2)
    ┌─────┬─────┐
    │ a   ┆ b   │
    │ --- ┆ --- │
    │ i64 ┆ str │
    ╞═════╪═════╡
    │ 1   ┆ x   │
    │ 2   ┆ y   │
    │ 3   ┆ z   │
    │ 4   ┆ a   │
    │ 5   ┆ b   │
    └─────┴─────┘
    """
    if not hasattr(source, "__arrow_c_stream__"):
        msg = "source must implement the Arrow PyCapsule Interface (__arrow_c_stream__)"
        raise TypeError(msg)

    import polars._plr as plr

    reader = plr.PyArrowCStreamReader(source)

    def io_source(
        with_columns: list[str] | None,
        predicate: Expr | None,
        n_rows: int | None,
        batch_size: int | None,  # noqa: ARG001
    ) -> Iterator[DataFrame]:
        remaining = n_rows
        while (batch := reader.next_batch(with_columns)) is not None:
            df = pl.DataFrame._from_pydf(batch)
            if predicate is not None:
                df = df.filter(predicate)
            if remaining is not None:
                df = df.head(remaining)
                remaining -= df.height
            yield df
            if remaining is not None and remaining <= 0:
                return

    return register_io_source(io_source=io_source, schema=reader.schema)
