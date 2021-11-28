from typing import Any

from polars.datatypes import (
    Boolean,
    Float32,
    Float64,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Utf8,
    dtype_to_py_type,
)
from polars.internals import Series

_NUMERIC_COL_TYPES = (
    Int16,
    Int32,
    Int64,
    UInt16,
    UInt32,
    UInt64,
    UInt8,
    Utf8,
    Float32,
    Float64,
    Boolean,
)


def assert_series_equal(
    left: Series,
    right: Series,
    check_dtype: bool = True,
    check_names: bool = True,
    check_exact: bool = False,
    rtol: float = 1.0e-5,
    atol: float = 1.0e-8,
) -> None:
    obj = "Series"
    try:
        can_be_subtracted = hasattr(dtype_to_py_type(left.dtype), "__sub__")
    except NotImplementedError:
        can_be_subtracted = False

    check_exact = check_exact or not can_be_subtracted or left.dtype == Boolean
    if not (isinstance(left, Series) and isinstance(right, Series)):
        raise_assert_detail(obj, "Type mismatch", type(left), type(right))

    if left.shape != right.shape:
        raise_assert_detail(obj, "Shape mismatch", left.shape, right.shape)

    if check_dtype:
        if left.dtype != right.dtype:
            raise_assert_detail(obj, "Dtype mismatch", left.dtype, right.dtype)

    if check_names:
        if left.name != right.name:
            raise_assert_detail(obj, "Name mismatch", left.name, right.name)

    if check_exact:
        if (left != right).sum() != 0:
            raise_assert_detail(
                obj, "Exact value mismatch", left=list(left), right=list(right)
            )
    else:
        if ((left - right).abs() > (atol + rtol * right.abs())).sum() != 0:
            raise_assert_detail(
                obj, "Value mismatch", left=list(left), right=list(right)
            )


def raise_assert_detail(
    obj: str,
    message: str,
    left: Any,
    right: Any,
) -> None:
    __tracebackhide__ = True

    msg = f"""{obj} are different

{message}"""

    msg += f"""
[left]:  {left}
[right]: {right}"""

    raise AssertionError(msg)
