import pyarrow as pa


def coerce_arrow(array: "pa.Array") -> "pa.Array":
    if array.type == pa.timestamp("s"):
        array = pa.compute.cast(
            pa.compute.multiply(pa.compute.cast(array, pa.int64()), 1000),
            pa.date64(),
        )
    elif array.type == pa.timestamp("ms"):
        array = pa.compute.cast(pa.compute.cast(array, pa.int64()), pa.date64())
    elif array.type == pa.timestamp("us"):
        array = pa.compute.cast(
            pa.compute.divide(pa.compute.cast(array, pa.int64()), 1000),
            pa.date64(),
        )
    elif array.type == pa.timestamp("ns"):
        array = pa.compute.cast(
            pa.compute.divide(pa.compute.cast(array, pa.int64()), 1000000),
            pa.date64(),
        )
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
        array = array.combine_chunks()
    return array


def _is_expr(arg) -> bool:
    return hasattr(arg, "_pyexpr")
