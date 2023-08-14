from __future__ import annotations

from typing import TYPE_CHECKING

from polars.io.pyarrow_dataset.anonymous_scan import _scan_pyarrow_dataset
from polars.utils.deprecation import deprecate_renamed_function

if TYPE_CHECKING:
    from polars import LazyFrame
    from polars.dependencies import pyarrow as pa


def scan_pyarrow_dataset(
    source: pa.dataset.Dataset,
    *,
    allow_pyarrow_filter: bool = True,
    batch_size: int | None = None,
) -> LazyFrame:
    """
    Scan a pyarrow dataset.

    This can be useful to connect to cloud or partitioned datasets.

    Parameters
    ----------
    source
        Pyarrow dataset to scan.
    allow_pyarrow_filter
        Allow predicates to be pushed down to pyarrow. This can lead to different
        results if comparisons are done with null values as pyarrow handles this
        different than polars does.
    batch_size
        The maximum row count for scanned pyarrow record batches.

    Warnings
    --------
    This API is experimental and may change without it being considered a breaking
    change.

    Notes
    -----
    When using partitioning, the appropriate ``partitioning`` option must be set on
    ``pyarrow.dataset.dataset`` before passing to Polars or the partitioned-on column(s)
    may not get passed to Polars.

    Examples
    --------
    >>> import pyarrow.dataset as ds
    >>> dset = ds.dataset("s3://my-partitioned-folder/", format="ipc")  # doctest: +SKIP
    >>> (
    ...     pl.scan_pyarrow_dataset(dset)
    ...     .filter("bools")
    ...     .select(["bools", "floats", "date"])
    ...     .collect()
    ... )  # doctest: +SKIP
    shape: (1, 3)
    ┌───────┬────────┬────────────┐
    │ bools ┆ floats ┆ date       │
    │ ---   ┆ ---    ┆ ---        │
    │ bool  ┆ f64    ┆ date       │
    ╞═══════╪════════╪════════════╡
    │ true  ┆ 2.0    ┆ 1970-05-04 │
    └───────┴────────┴────────────┘

    """
    return _scan_pyarrow_dataset(
        source,
        allow_pyarrow_filter=allow_pyarrow_filter,
        batch_size=batch_size,
    )


@deprecate_renamed_function(new_name="scan_pyarrow_dataset", version="0.16.10")
def scan_ds(ds: pa.dataset.Dataset, *, allow_pyarrow_filter: bool = True) -> LazyFrame:
    """
    Scan a pyarrow dataset.

    This can be useful to connect to cloud or partitioned datasets.

    Parameters
    ----------
    ds
        Pyarrow dataset to scan.
    allow_pyarrow_filter
        Allow predicates to be pushed down to pyarrow. This can lead to different
        results if comparisons are done with null values as pyarrow handles this
        different than polars does.

    Warnings
    --------
    This API is experimental and may change without it being considered a breaking
    change.

    Examples
    --------
    >>> import pyarrow.dataset as ds
    >>> dset = ds.dataset("s3://my-partitioned-folder/", format="ipc")  # doctest: +SKIP
    >>> out = (
    ...     pl.scan_ds(dset)
    ...     .filter("bools")
    ...     .select(["bools", "floats", "date"])
    ...     .collect()
    ... )  # doctest: +SKIP
    shape: (1, 3)
    ┌───────┬────────┬────────────┐
    │ bools ┆ floats ┆ date       │
    │ ---   ┆ ---    ┆ ---        │
    │ bool  ┆ f64    ┆ date       │
    ╞═══════╪════════╪════════════╡
    │ true  ┆ 2.0    ┆ 1970-05-04 │
    └───────┴────────┴────────────┘

    """
    return scan_pyarrow_dataset(ds, allow_pyarrow_filter=allow_pyarrow_filter)
