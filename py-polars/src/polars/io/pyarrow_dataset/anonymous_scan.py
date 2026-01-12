from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any, Literal, overload

import polars._reexport as pl
from polars._dependencies import pyarrow as pa

if TYPE_CHECKING:
    from collections.abc import Iterator

    from polars import DataFrame, LazyFrame


def _scan_pyarrow_dataset(
    ds: pa.dataset.Dataset,
    *,
    allow_pyarrow_filter: bool = True,
    batch_size: int | None = None,
) -> LazyFrame:
    """
    Pickle the partially applied function `_scan_pyarrow_dataset_impl`.

    The bytes are then sent to the polars logical plan. It can be deserialized once
    executed and ran.

    Parameters
    ----------
    ds
        pyarrow dataset
    allow_pyarrow_filter
        Allow predicates to be pushed down to pyarrow. This can lead to different
        results if comparisons are done with null values as pyarrow handles this
        different than polars does.
    batch_size
        The maximum row count for scanned pyarrow record batches.
    """
    # when `allow_pyarrow_filter=False`, the Rust side passes `batch_size`
    # positionally, so we set as `user_batch_size` to avoid collision
    batch_size_key = "batch_size" if allow_pyarrow_filter else "user_batch_size"
    func = partial(
        _scan_pyarrow_dataset_impl,
        ds,
        allow_pyarrow_filter=allow_pyarrow_filter,
        **{batch_size_key: batch_size},
    )
    return pl.LazyFrame._scan_python_function(
        ds.schema, func, pyarrow=allow_pyarrow_filter
    )


@overload
def _scan_pyarrow_dataset_impl(
    ds: pa.dataset.Dataset,
    with_columns: list[str] | None,
    predicate: str | bytes | None,
    n_rows: int | None,
    batch_size: int | None = ...,
    *,
    allow_pyarrow_filter: Literal[True] = ...,
    user_batch_size: int | None = ...,
) -> DataFrame: ...


@overload
def _scan_pyarrow_dataset_impl(
    ds: pa.dataset.Dataset,
    with_columns: list[str] | None,
    predicate: str | bytes | None,
    n_rows: int | None,
    batch_size: int | None = ...,
    *,
    allow_pyarrow_filter: Literal[False],
    user_batch_size: int | None = ...,
) -> tuple[Iterator[DataFrame], bool]: ...


def _scan_pyarrow_dataset_impl(
    ds: pa.dataset.Dataset,
    with_columns: list[str] | None,
    predicate: str | bytes | None,
    n_rows: int | None,
    batch_size: int | None = None,
    *,
    allow_pyarrow_filter: bool = True,
    user_batch_size: int | None = None,
) -> DataFrame | tuple[Iterator[DataFrame], bool]:
    """
    Take the projected columns and materialize an arrow table.

    Parameters
    ----------
    ds
        pyarrow dataset.
    with_columns
        Columns that are projected.
    predicate
        pyarrow expression string (when `allow_pyarrow_filter=True`) or
        serialized Polars predicate bytes (when `allow_pyarrow_filter=False`).
    n_rows:
        Materialize only `n` rows from the arrow dataset.
    batch_size
        The maximum row count for scanned pyarrow record batches.
    allow_pyarrow_filter
        If True, evaluate predicate and return DataFrame directly.
        If False, return `(generator, False)` tuple for IOPlugin path.
    user_batch_size
        User-specified `batch_size` (takes precedence over Rust-provided `batch_size`).

    Warnings
    --------
    Don't use this if you accept untrusted user inputs. Predicates will be evaluated
    with python 'eval'. There is sanitation in place, but it is a possible attack
    vector.

    Returns
    -------
    DataFrame or tuple[Iterator[DataFrame], bool]
    """
    from polars import from_arrow

    _filter = None

    if allow_pyarrow_filter and predicate:
        from polars._utils.convert import (
            to_py_date,
            to_py_datetime,
            to_py_time,
            to_py_timedelta,
        )
        from polars.datatypes import Date, Datetime, Duration

        _filter = eval(
            predicate,
            {
                "pa": pa,
                "Date": Date,
                "Datetime": Datetime,
                "Duration": Duration,
                "to_py_date": to_py_date,
                "to_py_datetime": to_py_datetime,
                "to_py_time": to_py_time,
                "to_py_timedelta": to_py_timedelta,
            },
        )

    common_params: dict[str, Any] = {"columns": with_columns, "filter": _filter}
    batch_size = user_batch_size if user_batch_size is not None else batch_size
    if batch_size is not None:
        common_params["batch_size"] = batch_size

    if allow_pyarrow_filter:
        if n_rows:
            return from_arrow(ds.head(n_rows, **common_params))  # type: ignore[return-value]
        return from_arrow(ds.to_table(**common_params))  # type: ignore[return-value]
    else:

        def frames() -> Iterator[DataFrame]:
            if n_rows:
                tbl = ds.head(n_rows, **common_params)
            else:
                tbl = ds.to_table(**common_params)
            yield from_arrow(tbl)  # type: ignore[misc]

        return frames(), False
