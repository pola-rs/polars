from typing import Any, Optional, List, Union, Dict, Tuple
import warnings

import pyarrow as pa


def coerce_arrow(array: pa.Array) -> pa.Array:
    # also coerces timezone to naive representation
    # units are accounted for by pyarrow
    if "timestamp" in str(array.type):
        warnings.warn(
            "Conversion of (potentially) timezone aware to naive datetimes. TZ information may be lost",
        )
        ts_ms = pa.compute.cast(array, pa.timestamp("ms"), safe=False)
        ms = pa.compute.cast(ts_ms, pa.int64())
        del ts_ms
        array = pa.compute.cast(ms, pa.date64())
        del ms
    # note: Decimal256 could not be cast to float
    elif isinstance(array.type, pa.Decimal128Type):
        array = pa.compute.cast(array, pa.float64())

    # simplest solution is to cast to (large)-string arrays
    # this is copy and expensive
    elif isinstance(array, pa.DictionaryArray):
        if array.dictionary.type == pa.string():
            array = pa.compute.cast(pa.compute.cast(array, pa.utf8()), pa.large_utf8())
        else:
            raise ValueError(
                "polars does not support dictionary encoded types other than strings"
            )

    if hasattr(array, "num_chunks") and array.num_chunks > 1:
        if pa.types.is_string(array.type):
            array = pa.compute.cast(array, pa.large_utf8())
        elif pa.types.is_list(array.type):
            array = pa.compute.cast(array, pa.large_list())
        array = array.combine_chunks()
    return array


def _is_expr(arg: Any) -> bool:
    return hasattr(arg, "_pyexpr")


def _process_null_values(
    null_values: Optional[Union[str, List[str], Dict[str, str]]] = None,
) -> Optional[Union[str, List[str], List[Tuple[str, str]]]]:  # type: ignore
    processed_null_values = null_values  # type: ignore
    if null_values is not None and isinstance(null_values, dict):
        processed_null_values = []
        for k, v in null_values.items():
            processed_null_values.append((k, v))  # type: ignore
    return processed_null_values  # type: ignore
