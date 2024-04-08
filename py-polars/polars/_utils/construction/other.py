from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars._reexport as pl
from polars._utils.construction.utils import get_first_non_none
from polars.datatypes import UInt32
from polars.dependencies import numpy as np
from polars.dependencies import pyarrow as pa
from polars.meta import get_index_type

if TYPE_CHECKING:
    from polars import Series
    from polars.dependencies import pandas as pd


def numpy_to_idxs(idxs: np.ndarray[Any, Any], size: int) -> Series:
    # Unsigned or signed Numpy array (ordered from fastest to slowest).
    #   - np.uint32 (polars) or np.uint64 (polars_u64_idx) numpy array
    #     indexes.
    #   - Other unsigned numpy array indexes are converted to pl.UInt32
    #     (polars) or pl.UInt64 (polars_u64_idx).
    #   - Signed numpy array indexes are converted pl.UInt32 (polars) or
    #     pl.UInt64 (polars_u64_idx) after negative indexes are converted
    #     to absolute indexes.
    if idxs.ndim != 1:
        msg = "only 1D numpy array is supported as index"
        raise ValueError(msg)

    idx_type = get_index_type()

    if len(idxs) == 0:
        return pl.Series("", [], dtype=idx_type)

    # Numpy array with signed or unsigned integers.
    if idxs.dtype.kind not in ("i", "u"):
        msg = "unsupported idxs datatype"
        raise NotImplementedError(msg)

    if idx_type == UInt32:
        if idxs.dtype in {np.int64, np.uint64} and idxs.max() >= 2**32:
            msg = "index positions should be smaller than 2^32"
            raise ValueError(msg)
        if idxs.dtype == np.int64 and idxs.min() < -(2**32):
            msg = "index positions should be bigger than -2^32 + 1"
            raise ValueError(msg)

    if idxs.dtype.kind == "i" and idxs.min() < 0:
        if idx_type == UInt32:
            if idxs.dtype in (np.int8, np.int16):
                idxs = idxs.astype(np.int32)
        else:
            if idxs.dtype in (np.int8, np.int16, np.int32):
                idxs = idxs.astype(np.int64)

        # Update negative indexes to absolute indexes.
        idxs = np.where(idxs < 0, size + idxs, idxs)

    # numpy conversion is much faster
    idxs = idxs.astype(np.uint32) if idx_type == UInt32 else idxs.astype(np.uint64)

    return pl.Series("", idxs, dtype=idx_type)


def pandas_series_to_arrow(
    values: pd.Series[Any] | pd.Index[Any],
    *,
    length: int | None = None,
    nan_to_null: bool = True,
) -> pa.Array:
    """
    Convert a pandas Series to an Arrow Array.

    Parameters
    ----------
    values : :class:`pandas.Series` or :class:`pandas.Index`.
        Series to convert to arrow
    nan_to_null : bool, default = True
        Interpret `NaN` as missing values.
    length : int, optional
        in case all values are null, create a null array of this length.
        if unset, length is inferred from values.

    Returns
    -------
    :class:`pyarrow.Array`
    """
    dtype = getattr(values, "dtype", None)
    if dtype == "object":
        first_non_none = get_first_non_none(values.values)  # type: ignore[arg-type]
        if isinstance(first_non_none, str):
            return pa.array(values, pa.large_utf8(), from_pandas=nan_to_null)
        elif first_non_none is None:
            return pa.nulls(length or len(values), pa.large_utf8())
        return pa.array(values, from_pandas=nan_to_null)
    elif dtype:
        return pa.array(values, from_pandas=nan_to_null)
    else:
        # Pandas Series is actually a Pandas DataFrame when the original DataFrame
        # contains duplicated columns and a duplicated column is requested with df["a"].
        msg = "duplicate column names found: "
        raise ValueError(
            msg,
            f"{values.columns.tolist()!s}",  # type: ignore[union-attr]
        )


def coerce_arrow(array: pa.Array) -> pa.Array:
    """..."""
    import pyarrow.compute as pc

    if hasattr(array, "num_chunks") and array.num_chunks > 1:
        # small integer keys can often not be combined, so let's already cast
        # to the uint32 used by polars
        if pa.types.is_dictionary(array.type) and (
            pa.types.is_int8(array.type.index_type)
            or pa.types.is_uint8(array.type.index_type)
            or pa.types.is_int16(array.type.index_type)
            or pa.types.is_uint16(array.type.index_type)
            or pa.types.is_int32(array.type.index_type)
        ):
            array = pc.cast(
                array, pa.dictionary(pa.uint32(), pa.large_string())
            ).combine_chunks()
    return array
