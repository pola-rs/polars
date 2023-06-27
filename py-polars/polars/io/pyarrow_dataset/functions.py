from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from polars.io.pyarrow_dataset.anonymous_scan import _scan_pyarrow_dataset
from polars.utils.various import find_stacklevel

if TYPE_CHECKING:
    from polars import LazyFrame
    from polars.dependencies import pyarrow as pa


def scan_pyarrow_dataset(
    source: pa.dataset.Dataset, *, allow_pyarrow_filter: bool = True
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

    Warnings
    --------
    This API is experimental and may change without it being considered a breaking
    change.

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
    return _scan_pyarrow_dataset(source, allow_pyarrow_filter=allow_pyarrow_filter)


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
    warnings.warn(
        "`scan_ds` has been renamed; this"
        " redirect is temporary, please use `scan_pyarrow_dataset` instead",
        category=DeprecationWarning,
        stacklevel=find_stacklevel(),
    )
    return scan_pyarrow_dataset(ds, allow_pyarrow_filter=allow_pyarrow_filter)
