"""Functions for scanning Arrow C Stream sources."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from polars._utils.unstable import unstable

if TYPE_CHECKING:
    from polars._typing import SchemaDict
    from polars.lazyframe import LazyFrame

__all__ = ["scan_arrow_c_stream"]


@unstable()
def scan_arrow_c_stream(
    source: Any,
    *,
    schema: SchemaDict | None = None,
) -> LazyFrame:
    """
    Scan an Arrow C Stream source.

    This function creates a lazy scan over a source that implements the
    Arrow PyCapsule Interface (`__arrow_c_stream__` protocol). The source
    is consumed during streaming execution.

    .. warning::
        This functionality is considered **unstable**. It may be changed
        at any point without it being considered a breaking change.

    Parameters
    ----------
    source
        Any object implementing the Arrow PyCapsule Interface
        (i.e., has a `__arrow_c_stream__` method).
    schema
        Optional schema to enforce on the data. If not provided,
        the schema will be inferred from the stream.

    Returns
    -------
    LazyFrame

    Notes
    -----
    - This function supports streaming execution via the new streaming engine.
    - The source must produce Arrow RecordBatches with a struct schema.
    - Once the stream is consumed, the same source cannot be reused.

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
    >>> lf = pl.scan_arrow_c_stream(reader)  # doctest: +SKIP
    >>> lf.collect()  # doctest: +SKIP
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
    from polars.lazyframe import LazyFrame

    # schema is passed directly to Rust (FromPyObject handles conversion)
    pylf = plr.PyLazyFrame.scan_arrow_c_stream(source, schema)
    return LazyFrame._from_pyldf(pylf)
