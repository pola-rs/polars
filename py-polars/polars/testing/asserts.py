from __future__ import annotations

import textwrap
from typing import Any

from polars import functions as F
from polars.dataframe import DataFrame
from polars.datatypes import (
    FLOAT_DTYPES,
    UNSIGNED_INTEGER_DTYPES,
    Categorical,
    DataTypeClass,
    List,
    Struct,
    dtype_to_py_type,
    unpack_dtypes,
)
from polars.exceptions import ComputeError, InvalidAssert
from polars.lazyframe import LazyFrame
from polars.series import Series


def assert_frame_equal(
    left: DataFrame | LazyFrame,
    right: DataFrame | LazyFrame,
    *,
    check_dtype: bool = True,
    check_exact: bool = False,
    rtol: float = 1.0e-5,
    atol: float = 1.0e-8,
    nans_compare_equal: bool = True,
    check_column_order: bool = True,
    check_row_order: bool = True,
) -> None:
    """
    Raise detailed AssertionError if `left` does NOT equal `right`.

    Parameters
    ----------
    left
        the dataframe to compare.
    right
        the dataframe to compare with.
    check_dtype
        if True, data types need to match exactly.
    check_exact
        if False, test if values are within tolerance of each other
        (see `rtol` & `atol`).
    rtol
        relative tolerance for inexact checking. Fraction of values in `right`.
    atol
        absolute tolerance for inexact checking.
    nans_compare_equal
        if your assert/test requires float NaN != NaN, set this to False.
    check_column_order
        if False, frames will compare equal if the required columns are present,
        irrespective of the order in which they appear.
    check_row_order
        if False, frames will compare equal if the required rows are present,
        irrespective of the order in which they appear; as this requires
        sorting, you cannot set on frames that contain unsortable columns.

    Examples
    --------
    >>> from polars.testing import assert_frame_equal
    >>> df1 = pl.DataFrame({"a": [1, 2, 3]})
    >>> df2 = pl.DataFrame({"a": [2, 3, 4]})
    >>> assert_frame_equal(df1, df2)  # doctest: +SKIP
    AssertionError: Values for column 'a' are different.
    """
    if isinstance(left, LazyFrame) and isinstance(right, LazyFrame):
        left, right = left.collect(), right.collect()
        obj = "LazyFrames"
    elif isinstance(left, DataFrame) and isinstance(right, DataFrame):
        obj = "DataFrames"
    else:
        raise_assert_detail("Inputs", "Unexpected input types", type(left), type(right))

    if left.shape[0] != right.shape[0]:  # type: ignore[union-attr]
        raise_assert_detail(obj, "Length mismatch", left.shape, right.shape)  # type: ignore[union-attr]

    left_not_right = [c for c in left.columns if c not in right.columns]
    if left_not_right:
        raise AssertionError(
            f"Columns {left_not_right} in left frame, but not in right."
        )
    right_not_left = [c for c in right.columns if c not in left.columns]
    if right_not_left:
        raise AssertionError(
            f"Columns {right_not_left} in right frame, but not in left."
        )

    if check_column_order and left.columns != right.columns:
        raise AssertionError(
            f"Columns are not in the same order:\n{left.columns!r}\n{right.columns!r}"
        )

    if not check_row_order:
        try:
            left = left.sort(by=left.columns)
            right = right.sort(by=left.columns)
        except ComputeError as exc:
            raise InvalidAssert(
                "Cannot set 'check_row_order=False' on frame with unsortable columns."
            ) from exc

    # note: does not assume a particular column order
    for c in left.columns:
        try:
            _assert_series_inner(
                left[c],  # type: ignore[arg-type, index]
                right[c],  # type: ignore[arg-type, index]
                check_dtype,
                check_exact,
                nans_compare_equal,
                atol,
                rtol,
            )
        except AssertionError as exc:
            msg = f"Values for column {c!r} are different."
            raise AssertionError(msg) from exc


def assert_frame_not_equal(
    left: DataFrame | LazyFrame,
    right: DataFrame | LazyFrame,
    *,
    check_dtype: bool = True,
    check_exact: bool = False,
    rtol: float = 1.0e-5,
    atol: float = 1.0e-8,
    nans_compare_equal: bool = True,
    check_column_order: bool = True,
    check_row_order: bool = True,
) -> None:
    """
    Raise AssertionError if `left` DOES equal `right`.

    Parameters
    ----------
    left
        the dataframe to compare.
    right
        the dataframe to compare with.
    check_dtype
        if True, data types need to match exactly.
    check_exact
        if False, test if values are within tolerance of each other
        (see `rtol` & `atol`).
    rtol
        relative tolerance for inexact checking. Fraction of values in `right`.
    atol
        absolute tolerance for inexact checking.
    nans_compare_equal
        if your assert/test requires float NaN != NaN, set this to False.
    check_column_order
        if False, frames will compare equal if the required columns are present,
        irrespective of the order in which they appear.
    check_row_order
        if False, frames will compare equal if the required rows are present,
        irrespective of the order in which they appear; as this requires
        sorting, you cannot set on frames that contain unsortable columns.

    Examples
    --------
    >>> from polars.testing import assert_frame_not_equal
    >>> df1 = pl.DataFrame({"a": [1, 2, 3]})
    >>> df2 = pl.DataFrame({"a": [2, 3, 4]})
    >>> assert_frame_not_equal(df1, df2)

    """
    try:
        assert_frame_equal(
            left=left,
            right=right,
            check_dtype=check_dtype,
            check_exact=check_exact,
            rtol=rtol,
            atol=atol,
            nans_compare_equal=nans_compare_equal,
            check_column_order=check_column_order,
            check_row_order=check_row_order,
        )
    except AssertionError:
        return
    else:
        raise AssertionError("Expected the input frames to be unequal.")


def assert_series_equal(
    left: Series,
    right: Series,
    *,
    check_dtype: bool = True,
    check_names: bool = True,
    check_exact: bool = False,
    rtol: float = 1.0e-5,
    atol: float = 1.0e-8,
    nans_compare_equal: bool = True,
) -> None:
    """
    Raise detailed AssertionError if `left` does NOT equal `right`.

    Parameters
    ----------
    left
        the series to compare.
    right
        the series to compare with.
    check_dtype
        if True, data types need to match exactly.
    check_names
        if True, names need to match.
    check_exact
        if False, test if values are within tolerance of each other
        (see `rtol` & `atol`).
    rtol
        relative tolerance for inexact checking. Fraction of values in `right`.
    atol
        absolute tolerance for inexact checking.
    nans_compare_equal
        if your assert/test requires float NaN != NaN, set this to False.

    Examples
    --------
    >>> from polars.testing import assert_series_equal
    >>> s1 = pl.Series([1, 2, 3])
    >>> s2 = pl.Series([2, 3, 4])
    >>> assert_series_equal(s1, s2)  # doctest: +SKIP

    """
    if not (
        isinstance(left, Series)  # type: ignore[redundant-expr]
        and isinstance(right, Series)
    ):
        raise_assert_detail("Inputs", "Unexpected input types", type(left), type(right))

    if len(left) != len(right):
        raise_assert_detail("Series", "Length mismatch", len(left), len(right))

    if check_names and left.name != right.name:
        raise_assert_detail("Series", "Name mismatch", left.name, right.name)

    _assert_series_inner(
        left, right, check_dtype, check_exact, nans_compare_equal, atol, rtol
    )


def assert_series_not_equal(
    left: Series,
    right: Series,
    *,
    check_dtype: bool = True,
    check_names: bool = True,
    check_exact: bool = False,
    rtol: float = 1.0e-5,
    atol: float = 1.0e-8,
    nans_compare_equal: bool = True,
) -> None:
    """
    Raise AssertionError if `left` DOES equal `right`.

    Parameters
    ----------
    left
        the series to compare.
    right
        the series to compare with.
    check_dtype
        if True, data types need to match exactly.
    check_names
        if True, names need to match.
    check_exact
        if False, test if values are within tolerance of each other
        (see `rtol` & `atol`).
    rtol
        relative tolerance for inexact checking. Fraction of values in `right`.
    atol
        absolute tolerance for inexact checking.
    nans_compare_equal
        if your assert/test requires float NaN != NaN, set this to False.

    Examples
    --------
    >>> from polars.testing import assert_series_not_equal
    >>> s1 = pl.Series([1, 2, 3])
    >>> s2 = pl.Series([2, 3, 4])
    >>> assert_series_not_equal(s1, s2)

    """
    try:
        assert_series_equal(
            left=left,
            right=right,
            check_dtype=check_dtype,
            check_names=check_names,
            check_exact=check_exact,
            rtol=rtol,
            atol=atol,
            nans_compare_equal=nans_compare_equal,
        )
    except AssertionError:
        return
    else:
        raise AssertionError("Expected the input Series to be unequal.")


def _assert_series_inner(
    left: Series,
    right: Series,
    check_dtype: bool,
    check_exact: bool,
    nans_compare_equal: bool,
    atol: float,
    rtol: float,
) -> None:
    """Compare Series dtype + values."""
    try:
        can_be_subtracted = hasattr(dtype_to_py_type(left.dtype), "__sub__")
    except NotImplementedError:
        can_be_subtracted = False

    check_exact = (
        check_exact or not can_be_subtracted or left.is_boolean() or left.is_temporal()
    )
    if check_dtype and left.dtype != right.dtype:
        raise_assert_detail("Series", "Dtype mismatch", left.dtype, right.dtype)

    if left.null_count() != right.null_count():
        raise_assert_detail(
            "Series", "null_count is not equal", left.null_count(), right.null_count()
        )

    # create mask of which (if any) values are unequal
    unequal = left.ne_missing(right)

    # handle NaN values (which compare unequal to themselves)
    comparing_float_dtypes = left.dtype in FLOAT_DTYPES and right.dtype in FLOAT_DTYPES
    if unequal.any() and nans_compare_equal:
        # when both dtypes are scalar floats
        if comparing_float_dtypes:
            unequal = unequal & ~(
                (left.is_nan() & right.is_nan()).fill_null(F.lit(False))
            )

        # account for float values in nested dtypes
        elif left.dtype.is_nested or right.dtype.is_nested:
            if _assert_series_nested_nans_equal(
                left=left.filter(unequal),
                right=right.filter(unequal),
                check_dtype=check_dtype,
                check_exact=check_exact,
                atol=atol,
                rtol=rtol,
            ):
                return

    # assert exact, or with tolerance
    if unequal.any():
        if check_exact:
            raise_assert_detail(
                "Series",
                "Exact value mismatch",
                left=list(left),
                right=list(right),
            )
        else:
            # apply check with tolerance (to the known-unequal matches).
            left, right = left.filter(unequal), right.filter(unequal)

            if all(tp in UNSIGNED_INTEGER_DTYPES for tp in (left.dtype, right.dtype)):
                # avoid potential "subtract-with-overflow" panic on uint math
                s_diff = Series("diff", [abs(v1 - v2) for v1, v2 in zip(left, right)])
            else:
                s_diff = (left - right).abs()

            mismatch, nan_info = False, ""
            if ((s_diff > (atol + rtol * right.abs())).sum() != 0) or (
                left.is_null() != right.is_null()
            ).any():
                mismatch = True
            elif comparing_float_dtypes:
                # note: take special care with NaN values.
                if not nans_compare_equal and (left.is_nan() == right.is_nan()).any():
                    nan_info = " (nans_compare_equal=False)"
                    mismatch = True
                elif (left.is_nan() != right.is_nan()).any():
                    nan_info = f" (nans_compare_equal={nans_compare_equal})"
                    mismatch = True

            if mismatch:
                raise_assert_detail(
                    "Series",
                    f"Value mismatch{nan_info}",
                    left=list(left),
                    right=list(right),
                )


def _assert_series_nested_nans_equal(
    left: Series,
    right: Series,
    check_dtype: bool,
    check_exact: bool,
    atol: float,
    rtol: float,
) -> bool:
    # check that float values exist at _some_ level of nesting
    if not any(tp in FLOAT_DTYPES for tp in unpack_dtypes(left.dtype, right.dtype)):
        return False

    # compare nested lists element-wise
    elif left.dtype == List == right.dtype:
        for s1, s2 in zip(left, right):
            if s1 is None and s2 is None:
                continue
            elif (s1 is None and s2 is not None) or (s2 is None and s1 is not None):
                raise_assert_detail("Series", "Nested value mismatch", s1, s2)
            elif len(s1) != len(s2):
                raise_assert_detail(
                    "Series", "Nested list length mismatch", len(s1), len(s2)
                )
            _assert_series_inner(
                s1,
                s2,
                check_dtype=check_dtype,
                check_exact=check_exact,
                nans_compare_equal=True,
                atol=atol,
                rtol=rtol,
            )
        return True

    # unnest structs as series and compare
    elif left.dtype == Struct == right.dtype:
        ls, rs = left.struct.unnest(), right.struct.unnest()
        if len(ls.columns) != len(rs.columns):
            raise_assert_detail(
                "Series",
                "Nested struct fields mismatch",
                len(ls.columns),
                len(rs.columns),
            )
        elif len(ls) != len(rs):
            raise_assert_detail(
                "Series", "Nested struct length mismatch", len(ls), len(rs)
            )
        for s1, s2 in zip(ls, rs):
            _assert_series_inner(
                s1,
                s2,
                check_dtype=check_dtype,
                check_exact=check_exact,
                nans_compare_equal=True,
                atol=atol,
                rtol=rtol,
            )
        return True
    else:
        # fall-back to outer codepath (if mismatched dtypes we would expect
        # the equality check to fail - unless ALL series values are null)
        return False


def raise_assert_detail(
    obj: str,
    detail: str,
    left: Any,
    right: Any,
    exc: AssertionError | None = None,
) -> None:
    """Raise a detailed assertion error."""
    __tracebackhide__ = True

    error_msg = textwrap.dedent(
        f"""\
        {obj} are different.

        {detail}
        [left]:  {left}
        [right]: {right}\
        """
    )

    raise AssertionError(error_msg) from exc


def is_categorical_dtype(data_type: Any) -> bool:
    """Check if the input is a polars Categorical dtype."""
    return (
        type(data_type) is DataTypeClass
        and issubclass(data_type, Categorical)
        or isinstance(data_type, Categorical)
    )


def assert_frame_equal_local_categoricals(df_a: DataFrame, df_b: DataFrame) -> None:
    """Assert frame equal for frames containing categoricals."""
    for (a_name, a_value), (b_name, b_value) in zip(
        df_a.schema.items(), df_b.schema.items()
    ):
        if a_name != b_name:
            print(f"{a_name} != {b_name}")
            raise AssertionError
        if a_value != b_value:
            print(f"{a_value} != {b_value}")
            raise AssertionError

    cat_to_str = F.col(Categorical).cast(str)
    assert_frame_equal(df_a.with_columns(cat_to_str), df_b.with_columns(cat_to_str))
    cat_to_phys = F.col(Categorical).to_physical()
    assert_frame_equal(df_a.with_columns(cat_to_phys), df_b.with_columns(cat_to_phys))
