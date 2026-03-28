from __future__ import annotations

import ast
from functools import partial
from typing import TYPE_CHECKING, Any, Literal, overload

import polars._reexport as pl
from polars._dependencies import pyarrow as pa

if TYPE_CHECKING:
    from collections.abc import Iterator

    from polars import DataFrame, LazyFrame

# Allowed AST node types for predicate validation. The Rust function `predicate_to_pa`
# in `crates/polars-plan/src/plans/python/pyarrow.rs` generates the expression strings
# that are parsed here. If new node types are added on the Rust side, this whitelist
#  must be updated to match.
_ALLOWED_PREDICATE_NODES: set[type[ast.AST]] = {
    ast.Expression,
    ast.Constant,
    ast.List,
    ast.Tuple,
    ast.Name,
    ast.Attribute,
    ast.Call,
    ast.Load,
    ast.Compare,
    ast.BinOp,
    ast.UnaryOp,
    ast.BoolOp,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.BitAnd,
    ast.BitOr,
    ast.Invert,
    ast.And,
    ast.Or,
}

# Only these top-level names may appear in predicate expressions.
# This blocks calls to builtins like exec, eval, __import__, etc.
# This is only a guard, there's only so much you can do with python's
# dynamism.
_ALLOWED_PREDICATE_NAMES: set[str] = {
    "pa",
    "Date",
    "Datetime",
    "Duration",
    "to_py_date",
    "to_py_datetime",
    "to_py_time",
    "to_py_timedelta",
    "True",
    "False",
    "None",
}


def _validate_predicate_ast(tree: ast.AST) -> None:
    """Validate that a predicate AST only contains allowed node types and names."""
    for node in ast.walk(tree):
        if type(node) not in _ALLOWED_PREDICATE_NODES:
            msg = f"disallowed node type in predicate: {type(node).__name__}"
            raise ValueError(msg)
        if isinstance(node, ast.Name) and node.id not in _ALLOWED_PREDICATE_NAMES:
            msg = f"disallowed name in predicate: {node.id!r}"
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
    Predicates are sanitized in multiple places using multiple methods, but a
    attacker could probably circumvent them all. Avoid using this path if untrusted
    user inputs could land here.

    Returns
    -------
    DataFrame or tuple[Iterator[DataFrame], bool]
    """
    filter_ = None
    filter_post_slice_ = None

    if allow_pyarrow_filter and predicate is not None:
        from polars._utils.convert import (
            to_py_date,
            to_py_datetime,
            to_py_time,
            to_py_timedelta,
        )
        from polars.datatypes import Date, Datetime, Duration

        tree = ast.parse(predicate, mode="eval")
        _validate_predicate_ast(tree)
        _filter = eval(
            compile(tree, "<predicate>", "eval"),
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

    if allow_pyarrow_filter:
        [x] = frames()
        return x

    else:
        return frames(), False
