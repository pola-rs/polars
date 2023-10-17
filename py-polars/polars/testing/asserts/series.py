from __future__ import annotations

from polars import functions as F
from polars.datatypes import (
    FLOAT_DTYPES,
    UNSIGNED_INTEGER_DTYPES,
    Categorical,
    List,
    Struct,
    UInt64,
    Utf8,
    dtype_to_py_type,
    unpack_dtypes,
)
from polars.series import Series
from polars.testing.asserts.utils import raise_assertion_error


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
    categorical_as_str: bool = False,
) -> None:
    """
    Assert that the left and right Series are equal.

    Raises a detailed ``AssertionError`` if the Series differ.
    This function is intended for use in unit tests.

    Parameters
    ----------
    left
        The first Series to compare.
    right
        The second Series to compare.
    check_dtype
        Require data types to match.
    check_names
        Require names to match.
    check_exact
        Require data values to match exactly. If set to ``False``, values are considered
        equal when within tolerance of each other (see ``rtol`` and ``atol``).
    rtol
        Relative tolerance for inexact checking. Fraction of values in ``right``.
    atol
        Absolute tolerance for inexact checking.
    nans_compare_equal
        Consider NaN values to be equal.
    categorical_as_str
        Cast categorical columns to string before comparing. Enabling this helps
        compare columns that do not share the same string cache.

    See Also
    --------
    assert_frame_equal
    assert_series_not_equal

    Examples
    --------
    >>> from polars.testing import assert_series_equal
    >>> s1 = pl.Series([1, 2, 3])
    >>> s2 = pl.Series([1, 5, 3])
    >>> assert_series_equal(s1, s2)  # doctest: +SKIP
    Traceback (most recent call last):
    ...
    AssertionError: Series are different (value mismatch)
    [left]:  [1, 2, 3]
    [right]: [1, 5, 3]

    """
    _assert_correct_input_type(left, right)

    if left.len() != right.len():
        raise_assertion_error("Series", "length mismatch", left.len(), right.len())

    if check_names and left.name != right.name:
        raise_assertion_error("Series", "name mismatch", left.name, right.name)

    if check_dtype and left.dtype != right.dtype:
        raise_assertion_error("Series", "dtype mismatch", left.dtype, right.dtype)

    _assert_series_values_equal(
        left,
        right,
        check_exact=check_exact,
        atol=atol,
        rtol=rtol,
        nans_compare_equal=nans_compare_equal,
        categorical_as_str=categorical_as_str,
    )


def _assert_correct_input_type(left: Series, right: Series) -> None:
    if not (isinstance(left, Series) and isinstance(right, Series)):  # type: ignore[redundant-expr]
        raise_assertion_error(
            "inputs",
            "unexpected input types",
            type(left).__name__,
            type(right).__name__,
        )


def _assert_series_values_equal(
    left: Series,
    right: Series,
    *,
    check_exact: bool,
    atol: float,
    rtol: float,
    nans_compare_equal: bool,
    categorical_as_str: bool,
) -> None:
    """Assert that the values in both Series are equal."""
    if categorical_as_str and left.dtype == Categorical:
        left = left.cast(Utf8)
        right = right.cast(Utf8)

    # create mask of which (if any) values are unequal
    unequal = left.ne_missing(right)

    # handle NaN values (which compare unequal to themselves)
    comparing_floats = left.dtype in FLOAT_DTYPES and right.dtype in FLOAT_DTYPES
    if unequal.any() and nans_compare_equal:
        # when both dtypes are scalar floats
        if comparing_floats:
            unequal = unequal & ~(
                (left.is_nan() & right.is_nan()).fill_null(F.lit(False))
            )
    if comparing_floats and not nans_compare_equal:
        unequal = unequal | left.is_nan() | right.is_nan()

    # check nested dtypes in separate function
    if left.dtype.is_nested or right.dtype.is_nested:
        if _assert_series_nested(
            left=left.filter(unequal),
            right=right.filter(unequal),
            check_exact=check_exact,
            atol=atol,
            rtol=rtol,
            nans_compare_equal=nans_compare_equal,
            categorical_as_str=categorical_as_str,
        ):
            return

    try:
        can_be_subtracted = hasattr(dtype_to_py_type(left.dtype), "__sub__")
    except NotImplementedError:
        can_be_subtracted = False

    check_exact = (
        check_exact or not can_be_subtracted or left.is_boolean() or left.is_temporal()
    )

    # assert exact, or with tolerance
    if unequal.any():
        if check_exact:
            raise_assertion_error(
                "Series",
                "exact value mismatch",
                left=left.to_list(),
                right=right.to_list(),
            )
        else:
            equal, nan_info = _check_series_equal_inexact(
                left,
                right,
                unequal,
                atol=atol,
                rtol=rtol,
                nans_compare_equal=nans_compare_equal,
                comparing_floats=comparing_floats,
            )

            if not equal:
                raise_assertion_error(
                    "Series",
                    f"value mismatch{nan_info}",
                    left=left.to_list(),
                    right=right.to_list(),
                )


def _check_series_equal_inexact(
    left: Series,
    right: Series,
    unequal: Series,
    atol: float,
    rtol: float,
    *,
    nans_compare_equal: bool,
    comparing_floats: bool,
) -> tuple[bool, str]:
    # apply check with tolerance (to the known-unequal matches).
    left, right = left.filter(unequal), right.filter(unequal)

    if all(tp in UNSIGNED_INTEGER_DTYPES for tp in (left.dtype, right.dtype)):
        # avoid potential "subtract-with-overflow" panic on uint math
        s_diff = Series(
            "diff", [abs(v1 - v2) for v1, v2 in zip(left, right)], dtype=UInt64
        )
    else:
        s_diff = (left - right).abs()

    equal, nan_info = True, ""
    if ((s_diff > (atol + rtol * right.abs())).sum() != 0) or (
        left.is_null() != right.is_null()
    ).any():
        equal = False

    elif comparing_floats:
        # note: take special care with NaN values.
        # if NaNs don't compare as equal, any NaN in the left Series is
        # sufficient for a mismatch because the if condition above already
        # compares the null values.
        if not nans_compare_equal and left.is_nan().any():
            equal = False
            nan_info = " (nans_compare_equal=False)"
        elif (left.is_nan() != right.is_nan()).any():
            equal = False
            nan_info = f" (nans_compare_equal={nans_compare_equal})"

    return equal, nan_info


def _assert_series_nested(
    left: Series,
    right: Series,
    *,
    check_exact: bool,
    atol: float,
    rtol: float,
    nans_compare_equal: bool,
    categorical_as_str: bool,
) -> bool:
    # check that float values exist at _some_ level of nesting
    if not any(tp in FLOAT_DTYPES for tp in unpack_dtypes(left.dtype, right.dtype)):
        return False

    # compare nested lists element-wise
    elif left.dtype == List == right.dtype:
        for s1, s2 in zip(left, right):
            if s1 is None and s2 is None:
                if nans_compare_equal:
                    continue
                else:
                    raise_assertion_error(
                        "Series",
                        f"Nested value mismatch (nans_compare_equal={nans_compare_equal})",
                        s1,
                        s2,
                    )
            elif (s1 is None and s2 is not None) or (s2 is None and s1 is not None):
                raise_assertion_error("Series", "nested value mismatch", s1, s2)
            elif len(s1) != len(s2):
                raise_assertion_error(
                    "Series", "nested list length mismatch", len(s1), len(s2)
                )

            _assert_series_values_equal(
                s1,
                s2,
                check_exact=check_exact,
                atol=atol,
                rtol=rtol,
                nans_compare_equal=nans_compare_equal,
                categorical_as_str=categorical_as_str,
            )
        return True

    # unnest structs as series and compare
    elif left.dtype == Struct == right.dtype:
        ls, rs = left.struct.unnest(), right.struct.unnest()
        if len(ls.columns) != len(rs.columns):
            raise_assertion_error(
                "Series",
                "nested struct fields mismatch",
                len(ls.columns),
                len(rs.columns),
            )
        elif len(ls) != len(rs):
            raise_assertion_error(
                "Series", "nested struct length mismatch", len(ls), len(rs)
            )
        for s1, s2 in zip(ls, rs):
            _assert_series_values_equal(
                s1,
                s2,
                check_exact=check_exact,
                atol=atol,
                rtol=rtol,
                nans_compare_equal=nans_compare_equal,
                categorical_as_str=categorical_as_str,
            )
        return True
    else:
        # fall-back to outer codepath (if mismatched dtypes we would expect
        # the equality check to fail - unless ALL series values are null)
        return False


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
    categorical_as_str: bool = False,
) -> None:
    """
    Assert that the left and right Series are **not** equal.

    This function is intended for use in unit tests.

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
    categorical_as_str
        Cast categorical columns to string before comparing. Enabling this helps
        compare DataFrames that do not share the same string cache.

    See Also
    --------
    assert_series_equal
    assert_frame_not_equal

    Examples
    --------
    >>> from polars.testing import assert_series_not_equal
    >>> s1 = pl.Series([1, 2, 3])
    >>> s2 = pl.Series([1, 2, 3])
    >>> assert_series_not_equal(s1, s2)  # doctest: +SKIP
    Traceback (most recent call last):
    ...
    AssertionError: Series are equal

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
            categorical_as_str=categorical_as_str,
        )
    except AssertionError:
        return
    else:
        msg = "Series are equal"
        raise AssertionError(msg)
