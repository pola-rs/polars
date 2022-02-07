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
from polars.internals import DataFrame, Series

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


def assert_frame_equal(
    left: DataFrame,
    right: DataFrame,
    check_dtype: bool = True,
    check_exact: bool = False,
    rtol: float = 1.0e-5,
    atol: float = 1.0e-8,
) -> None:
    """
    Raise detailed AssertionError if `left` does not equal `right`.


    Parameters
    ----------
    left
        the dataframe to compare
    right
        the dataframe to compare with
    check_dtype
        if True, data types need to match exactly
    check_exact
        if False, test if values are within tolerance of each other (see `rtol` & `atol`)
    rtol
        relative tolerance for inexact checking. Fraction of values in `right`
    atol
        absolute tolerance for inexact checking.

    Returns
    -------

    Examples
    --------

    >>> df1 = pl.DataFrame({"a": [1, 2, 3]})
    >>> df2 = pl.DataFrame({"a": [2, 3, 4]})
    >>> pl.testing.assert_frame_equal(df1, df2)  # doctest: +SKIP
    """

    obj = "DataFrame"
    check_column_order = True

    if not (isinstance(left, DataFrame) and isinstance(right, DataFrame)):
        raise_assert_detail(obj, "Type mismatch", type(left), type(right))

    if left.shape[0] != right.shape[0]:
        raise_assert_detail(obj, "Length mismatch", left.shape, right.shape)

    # this assumes we want it in the same order
    union_cols = list(set(left.columns).union(set(right.columns)))
    for c in union_cols:
        if c not in right.columns:
            raise AssertionError(
                f"column {c} in left dataframe, but not in right dataframe"
            )
        if c not in left.columns:
            raise AssertionError(
                f"column {c} in right dataframe, but not in left dataframe"
            )

    if check_column_order:
        if left.columns != right.columns:
            raise AssertionError("Columns are not in same order")

    # this does not assume a particular order
    for col in left.columns:
        _assert_series_inner(
            left[col], right[col], check_dtype, check_exact, atol, rtol, obj
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
    """
    Raise detailed AssertionError if `left` does not equal `right`.

    Parameters
    ----------
    left
        the series to compare
    right
        the series to compare with
    check_dtype
        if True, data types need to match exactly
    check_names
        if True, names need to match
    check_exact
        if False, test if values are within tolerance of each other (see `rtol` & `atol`)
    rtol
        relative tolerance for inexact checking. Fraction of values in `right`
    atol
        absolute tolerance for inexact checking.

    Returns
    -------

    Examples
    --------

    >>> s1 = pl.Series([1, 2, 3])
    >>> s2 = pl.Series([2, 3, 4])
    >>> pl.testing.assert_series_equal(s1, s2)  # doctest: +SKIP
    """
    obj = "Series"

    if not (isinstance(left, Series) and isinstance(right, Series)):
        raise_assert_detail(obj, "Type mismatch", type(left), type(right))

    if left.shape != right.shape:
        raise_assert_detail(obj, "Shape mismatch", left.shape, right.shape)

    if check_names:
        if left.name != right.name:
            raise_assert_detail(obj, "Name mismatch", left.name, right.name)

    _assert_series_inner(left, right, check_dtype, check_exact, atol, rtol, obj)


def _assert_series_inner(
    left: Series,
    right: Series,
    check_dtype: bool,
    check_exact: bool,
    atol: float,
    rtol: float,
    obj: str,
) -> None:
    """
    Compares Series dtype + values
    """
    try:
        can_be_subtracted = hasattr(dtype_to_py_type(left.dtype), "__sub__")
    except NotImplementedError:
        can_be_subtracted = False

    check_exact = check_exact or not can_be_subtracted or left.dtype == Boolean

    if check_dtype:
        if left.dtype != right.dtype:
            raise_assert_detail(obj, "Dtype mismatch", left.dtype, right.dtype)

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
