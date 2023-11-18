from __future__ import annotations

from typing import TYPE_CHECKING

from polars.datatypes import (
    NUMERIC_DTYPES,
    Array,
    Categorical,
    Decimal,
    Float64,
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
from polars.utils.deprecation import issue_deprecation_warning

if TYPE_CHECKING:
    from polars.type_aliases import PolarsDataType


def assert_series_equal(
    left: Series,
    right: Series,
    *,
    check_dtype: bool = True,
    check_names: bool = True,
    check_exact: bool = False,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    categorical_as_str: bool = False,
    nans_compare_equal: bool | None = None,
) -> None:
    """
    Assert that the left and right Series are equal.

    Raises a detailed `AssertionError` if the Series differ.
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
        Require data values to match exactly. If set to `False`, values are considered
        equal when within tolerance of each other (see `rtol` and `atol`).
        Logical types like dates are always checked exactly.
    rtol
        Relative tolerance for inexact checking, given as a fraction of the values in
        `right`.
    atol
        Absolute tolerance for inexact checking.
    categorical_as_str
        Cast categorical columns to string before comparing. Enabling this helps
        compare columns that do not share the same string cache.
    nans_compare_equal
        Consider NaN values to be equal.

        .. deprecated: 0.19.12
            This parameter will be removed. Default behaviour will remain as though it
            were set to `True`.

    See Also
    --------
    assert_frame_equal
    assert_series_not_equal

    Notes
    -----
    When using pytest, it may be worthwhile to shorten Python traceback printing
    by passing `--tb=short`. The default mode tends to be unhelpfully verbose.
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
    if nans_compare_equal is not None:
        issue_deprecation_warning(
            "The `nans_compare_equal` parameter for `assert_series_equal` is deprecated."
            " Default behaviour will remain as though it were set to `True`.",
            version="0.19.12",
        )
    else:
        nans_compare_equal = True

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
    # Handle categoricals
    if categorical_as_str:
        if left.dtype == Categorical:
            left = left.cast(Utf8)
        if right.dtype == Categorical:
            right = right.cast(Utf8)

    # Handle decimals
    # TODO: Delete this branch when Decimal equality is implemented
    # https://github.com/pola-rs/polars/issues/12118
    if left.dtype == Decimal:
        left = left.cast(Float64)
    if right.dtype == Decimal:
        right = right.cast(Float64)

    # Determine unequal elements
    try:
        unequal = left.ne_missing(right)
    except ComputeError as exc:
        raise_assertion_error(
            "Series",
            "incompatible data types",
            left=left.dtype,
            right=right.dtype,
            cause=exc,
        )

    # Handle NaN values (which compare unequal to themselves)
    if nans_compare_equal and _comparing_floats(left.dtype, right.dtype):
        both_nan = (left.is_nan() & right.is_nan()).fill_null(False)
        unequal = unequal & ~both_nan

    # Check nested dtypes in separate function
    if _comparing_nested_numerics(left.dtype, right.dtype):
        try:
            _assert_series_nested_values_equal(
                left=left.filter(unequal),
                right=right.filter(unequal),
                check_exact=check_exact,
                rtol=rtol,
                atol=atol,
                nans_compare_equal=nans_compare_equal,
                categorical_as_str=categorical_as_str,
            )
        except AssertionError as exc:
            raise_assertion_error(
                "Series",
                "nested value mismatch",
                left=left.to_list(),
                right=right.to_list(),
                cause=exc,
            )
        else:  # All nested values match
            return

    # If no differences found during exact checking, we're done
    if not unequal.any():
        return

    # Only do inexact checking for numeric types
    if check_exact or not left.dtype.is_numeric() or not right.dtype.is_numeric():
        raise_assertion_error(
            "Series", "exact value mismatch", left=left.to_list(), right=right.to_list()
        )

    _assert_series_null_values_match(left, right)
    _assert_series_nan_values_match(left, right, nans_compare_equal=nans_compare_equal)
    _assert_series_values_within_tolerance(
        left,
        right,
        unequal,
        rtol=rtol,
        atol=atol,
    )


def _assert_series_nested_values_equal(
    left: Series,
    right: Series,
    *,
    check_exact: bool,
    rtol: float,
    atol: float,
    nans_compare_equal: bool,
    categorical_as_str: bool,
) -> None:
    # compare nested lists element-wise
    if _comparing_lists(left.dtype, right.dtype):
        for s1, s2 in zip(left, right):
            if s1 is None or s2 is None:
                raise_assertion_error("Series", "nested value mismatch", s1, s2)

            _assert_series_values_equal(
                s1,
                s2,
                check_exact=check_exact,
                rtol=rtol,
                atol=atol,
                nans_compare_equal=nans_compare_equal,
                categorical_as_str=categorical_as_str,
            )

    # unnest structs as series and compare
    else:
        ls, rs = left.struct.unnest(), right.struct.unnest()
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


def _assert_series_null_values_match(left: Series, right: Series) -> None:
    null_value_mismatch = left.is_null() != right.is_null()
    if null_value_mismatch.any():
        raise_assertion_error(
            "Series", "null value mismatch", left.to_list(), right.to_list()
        )


def _assert_series_nan_values_match(
    left: Series, right: Series, *, nans_compare_equal: bool
) -> None:
    if not _comparing_floats(left.dtype, right.dtype):
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


def _comparing_floats(left: PolarsDataType, right: PolarsDataType) -> bool:
    return left.is_float() and right.is_float()


def _comparing_lists(left: PolarsDataType, right: PolarsDataType) -> bool:
    return left in (List, Array) and right in (List, Array)


def _comparing_structs(left: PolarsDataType, right: PolarsDataType) -> bool:
    return left == Struct and right == Struct


def _comparing_nested_numerics(left: PolarsDataType, right: PolarsDataType) -> bool:
    if not (_comparing_lists(left, right) or _comparing_structs(left, right)):
        return False

    return bool(NUMERIC_DTYPES & unpack_dtypes(left, right))


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
    tolerance = atol + rtol * right_unequal.abs()
    exceeds_tolerance = difference > tolerance

    if exceeds_tolerance.any():
        raise_assertion_error(
            "Series",
            "value mismatch",
            left.to_list(),
            right.to_list(),
        )


def _calc_absolute_diff(left: Series, right: Series) -> Series:
    if left.dtype.is_unsigned_integer() and right.dtype.is_unsigned_integer():
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
    categorical_as_str: bool = False,
    nans_compare_equal: bool | None = None,
) -> None:
    """
    Assert that the left and right Series are **not** equal.

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
        Require data values to match exactly. If set to `False`, values are considered
        equal when within tolerance of each other (see `rtol` and `atol`).
        Logical types like dates are always checked exactly.
    rtol
        Relative tolerance for inexact checking, given as a fraction of the values in
        `right`.
    atol
        Absolute tolerance for inexact checking.
    categorical_as_str
        Cast categorical columns to string before comparing. Enabling this helps
        compare columns that do not share the same string cache.
    nans_compare_equal
        Consider NaN values to be equal.

        .. deprecated: 0.19.12
            This parameter will be removed. Default behaviour will remain as though it
            were set to `True`.

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
