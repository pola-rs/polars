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


def _to_rust_syntax(df: pli.DataFrame) -> str:
    """Utility to generate the syntax that creates a polars 'DataFrame' in Rust."""
    syntax = "df![\n"

    def format_s(s: pli.Series) -> str:
        if s.null_count() == 0:
            return str(s.to_list()).replace("'", '"')
        else:
            tmp = "["
            for val in s:
                if val is None:
                    tmp += "None, "
                else:
                    if isinstance(val, str):
                        tmp += f'Some("{val}"), '
                    else:
                        tmp += f"Some({val}), "
            tmp = tmp[:-2] + "]"
            return tmp

    for s in df:
        syntax += f'    "{s.name}" => {format_s(s)},\n'
    syntax += "]"
    return syntax
