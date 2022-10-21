from __future__ import annotations

from typing import Any

import polars.internals as pli
from polars.testing.asserts import _getattr_multi, assert_series_equal


def verify_series_and_expr_api(
    input: pli.Series, expected: pli.Series | None, op: str, *args: Any, **kwargs: Any
) -> None:
    """
    Test element-wise functions for both the series and expressions API.

    Examples
    --------
    >>> s = pl.Series([1, 3, 2])
    >>> expected = pl.Series([1, 2, 3])
    >>> verify_series_and_expr_api(s, expected, "sort")

    """
    expr = _getattr_multi(pli.col("*"), op)(*args, **kwargs)
    result_expr = input.to_frame().select(expr)[:, 0]
    result_series = _getattr_multi(input, op)(*args, **kwargs)
    if expected is None:
        assert_series_equal(result_series, result_expr)
    else:
        assert_series_equal(result_expr, expected)
        assert_series_equal(result_series, expected)
