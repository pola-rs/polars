from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from polars._utils.construction.utils import get_first_non_none
from polars.dependencies import pyarrow as pa

if TYPE_CHECKING:
    from polars.dependencies import pandas as pd


def pandas_series_to_arrow(
    values: pd.Series[Any] | pd.Index[Any],
    *,
    length: int | None = None,
    nan_to_null: bool = True,
) -> pa.Array[Any]:
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


def coerce_arrow(array: pa.Array[Any] | pa.ChunkedArray[Any]) -> pa.Array[Any]:
    """..."""
    if isinstance(array, pa.ChunkedArray):
        # TODO: [pyarrow] remove explicit cast when combine_chunks is fixed
        array = cast(pa.Array[Any], array.combine_chunks())
    if pa.types.is_dictionary(array.type):
        array_type = cast(pa.DictionaryType[Any, Any], array.type)
        if (
            pa.types.is_int8(array_type.index_type)
            or pa.types.is_uint8(array_type.index_type)
            or pa.types.is_int16(array_type.index_type)
            or pa.types.is_uint16(array_type.index_type)
            or pa.types.is_int32(array_type.index_type)
        ):
            array = array.cast(pa.dictionary(pa.uint32(), pa.large_string()))
    return array
