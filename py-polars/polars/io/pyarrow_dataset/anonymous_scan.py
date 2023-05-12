from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import polars._reexport as pl
from polars.dependencies import pickle
from polars.dependencies import pyarrow as pa  # noqa: TCH001

if TYPE_CHECKING:
    from polars import DataFrame, LazyFrame


def _scan_pyarrow_dataset(
    ds: pa.dataset.Dataset, allow_pyarrow_filter: bool = True
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

    """
    func = partial(_scan_pyarrow_dataset_impl, ds)
    func_serialized = pickle.dumps(func)
    return pl.LazyFrame._scan_python_function(
        ds.schema, func_serialized, allow_pyarrow_filter
    )


def _scan_pyarrow_dataset_impl(
    ds: pa.dataset.Dataset,
    with_columns: list[str] | None,
    predicate: str | None,
    n_rows: int | None,
) -> DataFrame:
    """
    Take the projected columns and materialize an arrow table.

    Parameters
    ----------
    ds
        pyarrow dataset
    with_columns
        Columns that are projected
    predicate
        pyarrow expression that can be evaluated with eval
    n_rows:
        Materialize only n rows from the arrow dataset

    Returns
    -------
    DataFrame

    """
    from polars import from_arrow

    _filter = None
    if predicate:
        # imports are used by inline python evaluated by `eval`
        from polars.datatypes import Date, Datetime, Duration  # noqa: F401
        from polars.utils.convert import (
            _to_python_datetime,  # noqa: F401
            _to_python_time,  # noqa: F401
            _to_python_timedelta,  # noqa: F401
        )

        _filter = eval(predicate)
    if n_rows:
        return from_arrow(ds.head(n_rows, columns=with_columns, filter=_filter))  # type: ignore[return-value]

    return from_arrow(ds.to_table(columns=with_columns, filter=_filter))  # type: ignore[return-value]
