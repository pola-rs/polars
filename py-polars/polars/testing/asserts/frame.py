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
    collect_input_frames = isinstance(left, LazyFrame) and isinstance(right, LazyFrame)
    if collect_input_frames:
        objs = "LazyFrames"
    elif isinstance(left, DataFrame) and isinstance(right, DataFrame):
        objs = "DataFrames"
    else:
        raise_assertion_error(
            "Inputs",
            "unexpected input types",
            type(left).__name__,
            type(right).__name__,
        )

    if left_not_right := [c for c in left.columns if c not in right.columns]:
        msg = f"columns {left_not_right!r} in left frame, but not in right"
        raise AssertionError(msg)

    if right_not_left := [c for c in right.columns if c not in left.columns]:
        msg = f"columns {right_not_left!r} in right frame, but not in left"
        raise AssertionError(msg)

    if check_column_order and left.columns != right.columns:
        msg = f"columns are not in the same order:\n{left.columns!r}\n{right.columns!r}"
        raise AssertionError(msg)

    if collect_input_frames:
        if check_dtype:  # check this _before_ we collect
            left_schema, right_schema = left.schema, right.schema
            if left_schema != right_schema:
                raise_assertion_error(
                    objs, "lazy schemas are not equal", left_schema, right_schema
                )
        left, right = left.collect(), right.collect()  # type: ignore[union-attr]

    if left.shape[0] != right.shape[0]:  # type: ignore[union-attr]
        raise_assertion_error(objs, "length mismatch", left.shape, right.shape)  # type: ignore[union-attr]

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
                check_dtype=check_dtype,
                check_exact=check_exact,
                atol=atol,
                rtol=rtol,
                nans_compare_equal=nans_compare_equal,
                categorical_as_str=categorical_as_str,
            )
        except AssertionError as exc:
            msg = f"values for column {c!r} are different"
            raise AssertionError(msg) from exc


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
