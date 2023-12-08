from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars._reexport as pl
from polars.convert import from_arrow
from polars.dependencies import _PYARROW_AVAILABLE
from polars.dependencies import pyarrow as pa
from polars.interchange.dataframe import PolarsDataFrame
from polars.utils.various import parse_version

if TYPE_CHECKING:
    from polars import DataFrame
    from polars.interchange.protocol import SupportsInterchange


def from_dataframe(df: SupportsInterchange, *, allow_copy: bool = True) -> DataFrame:
    """
    Build a Polars DataFrame from any dataframe supporting the interchange protocol.

    Parameters
    ----------
    df
        Object supporting the dataframe interchange protocol, i.e. must have implemented
        the `__dataframe__` method.
    allow_copy
        Allow memory to be copied to perform the conversion. If set to False, causes
        conversions that are not zero-copy to fail.
    """
    if isinstance(df, pl.DataFrame):
        return df
    elif isinstance(df, PolarsDataFrame):
        return df._df

    if not hasattr(df, "__dataframe__"):
        raise TypeError(
            f"`df` of type {type(df).__name__!r} does not support the dataframe interchange protocol"
        )

    pa_table = _df_to_pyarrow_table(df, allow_copy=allow_copy)
    return from_arrow(pa_table, rechunk=allow_copy)  # type: ignore[return-value]


def _df_to_pyarrow_table(df: Any, *, allow_copy: bool = False) -> pa.Table:
    if not _PYARROW_AVAILABLE or parse_version(pa.__version__) < (11, 0):
        raise ImportError(
            "pyarrow>=11.0.0 is required for converting a dataframe interchange object"
            " to a Polars dataframe"
        )

    import pyarrow.interchange  # noqa: F401

    if not allow_copy:
        return _df_to_pyarrow_table_zero_copy(df)

    return pa.interchange.from_dataframe(df, allow_copy=True)


def _df_to_pyarrow_table_zero_copy(df: Any) -> pa.Table:
    dfi = df.__dataframe__(allow_copy=False)
    if _dfi_contains_categorical_data(dfi):
        raise TypeError(
            "Polars can not currently guarantee zero-copy conversion from Arrow for categorical columns"
            "\n\nSet `allow_copy=True` or cast categorical columns to string first."
        )

    if isinstance(df, pa.Table):
        return df
    elif isinstance(df, pa.RecordBatch):
        return pa.Table.from_batches([df])
    else:
        return pa.interchange.from_dataframe(dfi, allow_copy=False)


def _dfi_contains_categorical_data(dfi: Any) -> bool:
    CATEGORICAL_DTYPE = 23
    return any(c.dtype[0] == CATEGORICAL_DTYPE for c in dfi.get_columns())
