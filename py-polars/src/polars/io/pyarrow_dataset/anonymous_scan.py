from __future__ import annotations

import json
from functools import partial
from typing import TYPE_CHECKING, Any

import polars._reexport as pl
from polars._dependencies import pyarrow as pa

if TYPE_CHECKING:
    from collections.abc import Iterator

    from polars import DataFrame, LazyFrame


_BINOP_DISPATCH = {
    "eq": "__eq__",
    "neq": "__ne__",
    "lt": "__lt__",
    "lte": "__le__",
    "gt": "__gt__",
    "gte": "__ge__",
    "and": "__and__",
    "or": "__or__",
}


def _build_pyarrow_expr(node: list[Any]) -> Any:
    """Recursively build pyarrow expr from json."""
    tag: str = node[0]  # first position will always be the tag

    if tag == "field":
        name: str = node[1]
        return pa.compute.field(name)
    if tag == "binop":
        binop: str = node[1]
        left = _build_pyarrow_expr(node[2])
        right = _build_pyarrow_expr(node[3])
        return getattr(left, _BINOP_DISPATCH[binop])(right)
    if tag == "is_null":
        return _build_pyarrow_expr(node[1]).is_null()
    if tag == "is_not_null":
        return _build_pyarrow_expr(node[1]).is_valid()
    if tag == "lit_str":
        return pa.compute.scalar(node[1])
    if tag == "lit_bool":
        return pa.compute.scalar(node[1])
    if tag == "lit_i64" or tag == "lit_f64":
        return pa.compute.scalar(node[1])
    if tag == "lit_null":
        return pa.compute.scalar(None)

    msg = f"unknown predicate node tag: {tag!r}"
    raise ValueError(msg)


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
    func = partial(
        _scan_pyarrow_dataset_impl,
        ds,
        allow_pyarrow_filter=allow_pyarrow_filter,
        user_batch_size=batch_size,
    )
    return pl.LazyFrame._scan_python_function(
        ds.schema, func, pyarrow=allow_pyarrow_filter
    )


def _scan_pyarrow_dataset_impl(
    ds: pa.dataset.Dataset,
    with_columns: list[str] | None,
    predicate: str | bytes | None,
    n_rows: int | None,
    batch_size: int | None = None,
    *,
    allow_pyarrow_filter: bool = True,
    user_batch_size: int | None = None,
) -> tuple[Iterator[DataFrame], bool]:
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

    Returns
    -------
    tuple[Iterator[DataFrame], bool]
    A generator over the DataFrames and a boolean indicating if the
    predicates could be parsed.
    This boolean is always `False` as there might be some predicates
    that could not be converted
    to pyarrow and need to be applied as post-predicate.
    """
    filter_ = None
    filter_post_slice_ = None

    if allow_pyarrow_filter and predicate is not None:
        _filter = _build_pyarrow_expr(json.loads(predicate))

        if n_rows is None:
            filter_ = _filter
        else:
            filter_post_slice_ = _filter

    common_params: dict[str, Any] = {"columns": with_columns, "filter": filter_}
    batch_size = user_batch_size if user_batch_size is not None else batch_size
    if batch_size is not None:
        common_params["batch_size"] = batch_size

    def frames() -> Iterator[DataFrame]:
        yield pl.DataFrame(
            (
                ds.head(n_rows, **common_params).filter(filter_post_slice_)
                if filter_post_slice_ is not None
                else ds.head(n_rows, **common_params)
            )
            if n_rows is not None
            else ds.to_table(**common_params)
        )

    return frames(), False
