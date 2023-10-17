from __future__ import annotations

from polars.dataframe import DataFrame
from polars.exceptions import ComputeError, InvalidAssert
from polars.lazyframe import LazyFrame
from polars.testing.asserts.series import _assert_series_inner
from polars.testing.asserts.utils import raise_assertion_error


def assert_frame_equal(
    left: DataFrame | LazyFrame,
    right: DataFrame | LazyFrame,
    *,
    check_row_order: bool = True,
    check_column_order: bool = True,
    check_dtype: bool = True,
    check_exact: bool = False,
    rtol: float = 1.0e-5,
    atol: float = 1.0e-8,
    nans_compare_equal: bool = True,
    categorical_as_str: bool = False,
) -> None:
    """
    Assert that the left and right frame are equal.

    Raises a detailed ``AssertionError`` if the frames differ.
    This function is intended for use in unit tests.

    Parameters
    ----------
    left
        The first DataFrame or LazyFrame to compare.
    right
        The second DataFrame or LazyFrame to compare.
    check_row_order
        Require row order to match.

        .. note::
            Setting this to ``False`` requires sorting the data, which will fail on
            frames that contain unsortable columns.
    check_column_order
        Require column order to match.
    check_dtype
        Require data types to match.
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
    assert_series_equal
    assert_frame_not_equal

    Examples
    --------
    >>> from polars.testing import assert_frame_equal
    >>> df1 = pl.DataFrame({"a": [1, 2, 3]})
    >>> df2 = pl.DataFrame({"a": [1, 5, 3]})
    >>> assert_frame_equal(df1, df2)  # doctest: +SKIP
    Traceback (most recent call last):
    ...
    AssertionError: Series are different (value mismatch)
    [left]:  [1, 2, 3]
    [right]: [1, 5, 3]

    The above exception was the direct cause of the following exception:

    Traceback (most recent call last):
    ...
    AssertionError: values for column 'a' are different

    """
    lazy = _assert_frame_type_equal(left, right)
    objs = "LazyFrames" if lazy else "DataFrames"

    _assert_frame_schema_equal(
        left, right, check_column_order=check_column_order, check_dtype=check_dtype
    )

    if lazy:
        left, right = left.collect(), right.collect()  # type: ignore[union-attr]

    if left.height != right.height:  # type: ignore[union-attr]
        raise_assertion_error(objs, "length mismatch", left.height, right.height)  # type: ignore[union-attr]

    if not check_row_order:
        try:
            left = left.sort(by=left.columns)
            right = right.sort(by=left.columns)
        except ComputeError as exc:
            msg = "cannot set `check_row_order=False` on frame with unsortable columns"
            raise InvalidAssert(msg) from exc

    # note: does not assume a particular column order
    for c in left.columns:
        try:
            _assert_series_inner(
                left[c],  # type: ignore[arg-type, index]
                right[c],  # type: ignore[arg-type, index]
                check_dtype=False,  # already checked
                check_exact=check_exact,
                atol=atol,
                rtol=rtol,
                nans_compare_equal=nans_compare_equal,
                categorical_as_str=categorical_as_str,
            )
        except AssertionError as exc:
            msg = f"values for column {c!r} are different"
            raise AssertionError(msg) from exc


def _assert_frame_type_equal(
    left: DataFrame | LazyFrame, right: DataFrame | LazyFrame
) -> bool:
    if isinstance(left, LazyFrame) and isinstance(right, LazyFrame):
        return True
    elif isinstance(left, DataFrame) and isinstance(right, DataFrame):
        return False
    else:
        raise_assertion_error(
            "Inputs",
            "unexpected input types",
            type(left).__name__,
            type(right).__name__,
        )


def _assert_frame_schema_equal(
    left: DataFrame | LazyFrame,
    right: DataFrame | LazyFrame,
    *,
    check_dtype: bool = True,
    check_column_order: bool = True,
) -> None:
    left_schema, right_schema = left.schema, right.schema

    # Fast path for equal frames
    if left_schema == right_schema:
        return

    # We know schemas do not match...

    if check_column_order and check_dtype:
        # ... so we can raise here
        msg = "x"
        raise AssertionError(msg)

    elif not check_column_order and check_dtype:
        msg = "y"
        raise AssertionError(msg)

    # Assert that column names match in any order
    if left_schema.keys() != right_schema.keys():
        if left_not_right := [c for c in left_schema if c not in right_schema]:
            msg = f"columns {left_not_right!r} in left frame, but not in right"
            raise AssertionError(msg)
        if right_not_left := [c for c in right_schema if c not in left_schema]:
            msg = f"columns {right_not_left!r} in right frame, but not in left"
            raise AssertionError(msg)

    # Schemas don't match, but column names are known to match...
    # Either dtypes are wrong, or column order is wrong, or both

    if check_dtype:
        if dict(left_schema) != dict(right_schema):
            msg = "dtypes do not match"
            raise AssertionError(msg)

    # Check for column order
    if check_column_order:
        left_columns, right_columns = list(left_schema), list(right_schema)
        if left_columns != right_columns:
            # TODO: Use raise_assertion_error to get nice formatting
            msg = f"columns are not in the same order:\n{left_columns!r}\n{right_columns!r}"
            raise AssertionError(msg)

        # Here we know the order matches. So dtypes must not match!
        if check_dtype:
            msg = "dtypes do not match"
            raise AssertionError(msg)

    else:
        # Here we know that we are not checking for column order
        # Either dtypes are wrong, or column order is wrong but it doesn't matter, or both

        # We are checking for dtypes and don't care about column order
        if check_dtype and dict(left_schema) != dict(right_schema):
            msg = "dtypes do not match"
            raise AssertionError(msg)


def assert_frame_not_equal(
    left: DataFrame | LazyFrame,
    right: DataFrame | LazyFrame,
    *,
    check_row_order: bool = True,
    check_column_order: bool = True,
    check_dtype: bool = True,
    check_exact: bool = False,
    rtol: float = 1.0e-5,
    atol: float = 1.0e-8,
    nans_compare_equal: bool = True,
    categorical_as_str: bool = False,
) -> None:
    """
    Assert that the left and right frame are **not** equal.

    This function is intended for use in unit tests.

    Parameters
    ----------
    left
        The first DataFrame or LazyFrame to compare.
    right
        The second DataFrame or LazyFrame to compare.
    check_row_order
        Require row order to match.

        .. note::
            Setting this to ``False`` requires sorting the data, which will fail on
            frames that contain unsortable columns.
    check_column_order
        Require column order to match.
    check_dtype
        Require data types to match.
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
    >>> from polars.testing import assert_frame_not_equal
    >>> df1 = pl.DataFrame({"a": [1, 2, 3]})
    >>> df2 = pl.DataFrame({"a": [1, 2, 3]})
    >>> assert_frame_not_equal(df1, df2)  # doctest: +SKIP
    Traceback (most recent call last):
    ...
    AssertionError: frames are equal

    """
    try:
        assert_frame_equal(
            left=left,
            right=right,
            check_column_order=check_column_order,
            check_row_order=check_row_order,
            check_dtype=check_dtype,
            check_exact=check_exact,
            rtol=rtol,
            atol=atol,
            nans_compare_equal=nans_compare_equal,
            categorical_as_str=categorical_as_str,
        )
    except AssertionError:
        return
    else:
        msg = "frames are equal"
        raise AssertionError(msg)
