from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

from polars._utils.unstable import unstable

if TYPE_CHECKING:
    from polars._typing import SchemaDict

with contextlib.suppress(ImportError):  # Module not available when building docs
    from polars._plr import assert_schema_equal_py


@unstable()
def assert_schema_equal(
    left_schema: SchemaDict,
    right_schema: SchemaDict,
    *,
    check_column_order: bool = True,
    check_dtypes: bool = True,
) -> None:
    """
    Assert that the schema of the left and right frame are equal.

    Raises a detailed `AssertionError` if the schemas of the frames differ.
    This function is intended for use in unit tests.

    .. warning::
        This functionality is considered **unstable**. It may be changed
        at any point without it being considered a breaking change.

    Parameters
    ----------
    left_schema
        The first DataFrame or LazyFrame to compare.
    right_schema
        The second DataFrame or LazyFrame to compare.
    check_column_order
        Requires column order to match.
    check_dtypes
        Requires data types to match.

    Examples
    --------
    >>> import polars as pl
    >>> from polars.testing import assert_schema_equal
    >>> df1 = pl.DataFrame({"b": [3, 4], "a": [1, 2]})
    >>> df2 = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    >>> assert_schema_equal(df1.schema, df2.schema)
    Traceback (most recent call last):
    ...
    AssertionError: DataFrames are different (columns are not in the same order)
    [left]: ["b", "a"]
    [right]: ["a", "b"]
    """
    from polars import Schema

    assert_schema_equal_py(
        Schema(left_schema),
        Schema(right_schema),
        check_column_order=check_column_order,
        check_dtypes=check_dtypes,
    )
