from __future__ import annotations

from functools import reduce

import polars as pl
from polars.testing import assert_frame_equal


def test_map_return_py_object() -> None:
    df = pl.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

    result = df.select([pl.all().map(lambda s: reduce(lambda a, b: a + b, s))])

    expected = pl.DataFrame({"A": [6], "B": [15]})
    assert_frame_equal(result, expected)


def test_map_no_dtype_set_8531() -> None:
    df = pl.DataFrame({"a": [1]})

    result = df.with_columns(
        pl.col("a").map(lambda x: x * 2).shift_and_fill(fill_value=0, periods=0)
    )

    expected = pl.DataFrame({"a": [2]})
    assert_frame_equal(result, expected)
