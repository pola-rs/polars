from functools import reduce
from typing import Any

import pytest

import polars as pl
from polars import testing


@pytest.fixture
def df() -> pl.DataFrame:
    df = pl.DataFrame(
        {
            "bools": [False, True, False],
            "bools_nulls": [None, True, False],
            "int": [1, 2, 3],
            "int_nulls": [1, None, 3],
            "floats": [1.0, 2.0, 3.0],
            "floats_nulls": [1.0, None, 3.0],
            "strings": ["foo", "bar", "ham"],
            "strings_nulls": ["foo", None, "ham"],
            "date": [1324, 123, 1234],
            "datetime": [13241324, 12341256, 12341234],
            "time": [13241324, 12341256, 12341234],
        }
    )
    return df.with_columns(
        [
            pl.col("date").cast(pl.Date),
            pl.col("datetime").cast(pl.Datetime),
            pl.col("strings").cast(pl.Categorical).alias("cat"),
            pl.col("time").cast(pl.Time),
        ]
    )


@pytest.fixture
def fruits_cars() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "A": [1, 2, 3, 4, 5],
            "fruits": ["banana", "banana", "apple", "apple", "banana"],
            "B": [5, 4, 3, 2, 1],
            "cars": ["beetle", "audi", "beetle", "beetle", "beetle"],
        }
    )


def _getattr_multi(obj: object, op: str) -> Any:
    """ "
    Allows `op` to be multiple layers deep, i.e. op="str.lengths" will mean we first
    get the attribute "str", and then the attribute "lengths"
    """
    op_list = op.split(".")
    return reduce(lambda o, m: getattr(o, m), op_list, obj)


def verify_series_and_expr_api(
    input: pl.Series, expected: pl.Series, op: str, *args: Any
) -> None:
    """
    Small helper function to test element-wise functions for both the series and expressions api.

    Examples
    --------
    >>> s = pl.Series([1, 3, 2])
    >>> expected = pl.Series([1, 2, 3])
    >>> verify_series_and_expr_api(s, expected, "sort")
    """
    expr = _getattr_multi(pl.col("*"), op)(*args)
    result_expr: pl.Series = input.to_frame().select(expr)[:, 0]  # type: ignore
    result_series = _getattr_multi(input, op)(*args)
    testing.assert_series_equal(result_expr, expected)
    testing.assert_series_equal(result_series, expected)
