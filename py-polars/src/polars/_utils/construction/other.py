from __future__ import annotations

from typing import TYPE_CHECKING, Any

from polars._dependencies import pyarrow as pa
from polars._utils.construction.utils import get_first_non_none

if TYPE_CHECKING:
    from polars._dependencies import pandas as pd


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
        # Widen keys before combining chunks to avoid dictionary-index overflow.
        if pa.types.is_dictionary(array.type) and (
            pa.types.is_int8(array.type.index_type)
            or pa.types.is_uint8(array.type.index_type)
            or pa.types.is_int16(array.type.index_type)
            or pa.types.is_uint16(array.type.index_type)
            or pa.types.is_int32(array.type.index_type)
        ):
            value_type = array.type.value_type
            # Dictionary unification can merge positive and negative zero.
            if pa.types.is_float32(value_type) or pa.types.is_float64(value_type):
                return array
            if (
                pa.types.is_timestamp(value_type) or pa.types.is_duration(value_type)
            ) and value_type.unit == "s":
                target = (
                    pa.timestamp("ms", value_type.tz)
                    if pa.types.is_timestamp(value_type)
                    else pa.duration("ms")
                )
                # Decode first so only referenced values are checked for overflow.
                return pa.chunked_array(
                    [
                        pc.cast(chunk.dictionary_decode(), target, safe=True)
                        for chunk in array.iterchunks()
                    ],
                    type=target,
                )
            if pa.types.is_binary(value_type) or _is_pa_type(value_type, "binary_view"):
                value_type = pa.large_binary()
            elif not _keeps_dictionary_value_type(value_type):
                # Strings, and value types Polars can't import as dictionary values
                # (e.g. nested dictionaries), become a string dictionary.
                value_type = pa.large_string()
            array = pc.cast(
                array, pa.dictionary(pa.uint32(), value_type)
            ).combine_chunks()
    return array


def _is_pa_type(dtype: pa.DataType, name: str) -> bool:
    # Some type checks, e.g. `is_binary_view`, don't exist in older pyarrow versions.
    check = getattr(pa.types, f"is_{name}", None)
    return check is not None and check(dtype)


def _keeps_dictionary_value_type(dtype: pa.DataType) -> bool:
    """Whether a dictionary of this value type is combined without a string cast."""
    return (
        pa.types.is_integer(dtype)
        or pa.types.is_boolean(dtype)
        or (pa.types.is_decimal128(dtype) and 0 <= dtype.scale <= dtype.precision)
        or pa.types.is_date(dtype)
        or pa.types.is_time(dtype)
        or pa.types.is_timestamp(dtype)
        or pa.types.is_duration(dtype)
        or pa.types.is_large_binary(dtype)
        or pa.types.is_fixed_size_binary(dtype)
    )
