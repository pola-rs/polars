from __future__ import annotations

from polars.datatypes import (
    FLOAT_DTYPES,
    NESTED_DTYPES,
    NUMERIC_DTYPES,
    UNSIGNED_INTEGER_DTYPES,
    Categorical,
    Int64,
    List,
    Struct,
    UInt64,
    Utf8,
    unpack_dtypes,
)
from polars.exceptions import ComputeError
from polars.series import Series
from polars.testing.asserts.utils import raise_assertion_error


def assert_series_equal(
    left: Series,
    right: Series,
    *,
    check_dtype: bool = True,
    check_names: bool = True,
    check_exact: bool = False,
    rtol: float = 1e-5,
    atol: float = 1e-8,
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

    Notes
    -----
    When using pytest, it may be worthwhile to shorten Python traceback printing
    by passing ``--tb=short``. The default mode tends to be unhelpfully verbose.
    More information in the
    `pytest docs <https://docs.pytest.org/en/latest/how-to/output.html#modifying-python-traceback-printing>`_.

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
    if not (isinstance(left, Series) and isinstance(right, Series)):  # type: ignore[redundant-expr]
        raise_assertion_error(
            "inputs",
            "unexpected input types",
            type(left).__name__,
            type(right).__name__,
        )

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
        rtol=rtol,
        atol=atol,
        nans_compare_equal=nans_compare_equal,
        categorical_as_str=categorical_as_str,
    )


def _assert_series_values_equal(
    left: Series,
    right: Series,
    *,
    check_exact: bool,
    rtol: float,
    atol: float,
    nans_compare_equal: bool,
    categorical_as_str: bool,
) -> None:
    """Assert that the values in both Series are equal."""
    _assert_series_null_values_match(left, right)
    _assert_series_nan_values_match(left, right, nans_compare_equal=nans_compare_equal)

    if categorical_as_str and left.dtype == Categorical:
        left, right = left.cast(Utf8), right.cast(Utf8)

    # Start out simple - regular comparison where None == None
    unequal = left.ne_missing(right)

    # Handle NaN values (which compare unequal to themselves)
    comparing_floats = left.dtype in FLOAT_DTYPES and right.dtype in FLOAT_DTYPES
    if comparing_floats and nans_compare_equal:
        both_nan = left.is_nan().fill_null(False)
        unequal = unequal & ~both_nan

    # check nested dtypes in separate function
    if left.dtype in NESTED_DTYPES or right.dtype in NESTED_DTYPES:
        if _assert_series_nested(
            left=left.filter(unequal),
            right=right.filter(unequal),
            check_exact=check_exact,
            rtol=rtol,
            atol=atol,
            nans_compare_equal=nans_compare_equal,
            categorical_as_str=categorical_as_str,
        ):
            return

    # If no differences found during exact checking, we're done
    if not unequal.any():
        return

    # Only do inexact checking for numeric types
    check_exact = (
        check_exact
        or left.dtype not in NUMERIC_DTYPES
        or right.dtype not in NUMERIC_DTYPES
    )
    if check_exact:
        raise_assertion_error(
            "Series",
            "exact value mismatch",
            left=left.to_list(),
            right=right.to_list(),
        )

    _assert_series_values_within_tolerance(
        left,
        right,
        unequal,
        rtol=rtol,
        atol=atol,
    )


def _assert_series_null_values_match(left: Series, right: Series) -> None:
    null_value_mismatch = left.is_null() != right.is_null()
    if null_value_mismatch.any():
        raise_assertion_error(
            "Series", "null value mismatch", left.to_list(), right.to_list()
        )


def _assert_series_nan_values_match(
    left: Series, right: Series, *, nans_compare_equal: bool
) -> None:
    if not (left.dtype in FLOAT_DTYPES and right.dtype in FLOAT_DTYPES):
        return

    if nans_compare_equal:
        nan_value_mismatch = left.is_nan() != right.is_nan()
        if nan_value_mismatch.any():
            raise_assertion_error(
                "Series",
                "nan value mismatch - nans compare equal",
                left.to_list(),
                right.to_list(),
            )

    elif left.is_nan().any() or right.is_nan().any():
        raise_assertion_error(
            "Series",
            "nan value mismatch - nans compare unequal",
            left.to_list(),
            right.to_list(),
        )


def _assert_series_nested(
    left: Series,
    right: Series,
    *,
    check_exact: bool,
    rtol: float,
    atol: float,
    nans_compare_equal: bool,
    categorical_as_str: bool,
) -> bool:
    # check that float values exist at _some_ level of nesting
    if not any(tp in FLOAT_DTYPES for tp in unpack_dtypes(left.dtype, right.dtype)):
        return False

    # compare nested lists element-wise
    elif left.dtype == List == right.dtype:
        for s1, s2 in zip(left, right):
            if (s1 is None and s2 is not None) or (s2 is None and s1 is not None):
                raise_assertion_error("Series", "nested value mismatch", s1, s2)
            elif s1.len() != s2.len():
                raise_assertion_error(
                    "Series", "nested list length mismatch", len(s1), len(s2)
                )

            _assert_series_values_equal(
                s1,
                s2,
                check_exact=check_exact,
                rtol=rtol,
                atol=atol,
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
                rtol=rtol,
                atol=atol,
                nans_compare_equal=nans_compare_equal,
                categorical_as_str=categorical_as_str,
            )
        return True
    else:
        # fall-back to outer codepath (if mismatched dtypes we would expect
        # the equality check to fail - unless ALL series values are null)
        return False


def _assert_series_values_within_tolerance(
    left: Series,
    right: Series,
    unequal: Series,
    *,
    rtol: float,
    atol: float,
) -> None:
    left_unequal, right_unequal = left.filter(unequal), right.filter(unequal)

    difference = _calc_absolute_diff(left_unequal, right_unequal)
    tolerance = atol + rtol * right.abs()
    exceeds_tolerance = difference > tolerance

    if exceeds_tolerance.any():
        raise_assertion_error(
            "Series",
            "value mismatch",
            left.to_list(),
            right.to_list(),
        )


def _calc_absolute_diff(left: Series, right: Series) -> Series:
    if left.dtype in UNSIGNED_INTEGER_DTYPES and right.dtype in UNSIGNED_INTEGER_DTYPES:
        try:
            left = left.cast(Int64)
            right = right.cast(Int64)
        except ComputeError:
            # Handle big UInt64 values through conversion to Python
            diff = [abs(v1 - v2) for v1, v2 in zip(left, right)]
            return Series(diff, dtype=UInt64)

    return (left - right).abs()


def assert_series_not_equal(
    left: Series,
    right: Series,
    *,
    check_dtype: bool = True,
    check_names: bool = True,
    check_exact: bool = False,
    rtol: float = 1e-5,
    atol: float = 1e-8,
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
