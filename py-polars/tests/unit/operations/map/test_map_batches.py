from __future__ import annotations

from functools import reduce

import numpy as np
import pytest

import polars as pl
from polars.testing import assert_frame_equal


def test_map_return_py_object() -> None:
    df = pl.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

    result = df.select([pl.all().map_batches(lambda s: reduce(lambda a, b: a + b, s))])

    expected = pl.DataFrame({"A": [6], "B": [15]})
    assert_frame_equal(result, expected)


def test_map_no_dtype_set_8531() -> None:
    df = pl.DataFrame({"a": [1]})

    result = df.with_columns(
        pl.col("a").map_batches(lambda x: x * 2).shift_and_fill(fill_value=0, periods=0)
    )

    expected = pl.DataFrame({"a": [2]})
    assert_frame_equal(result, expected)


def test_error_on_reducing_map() -> None:
    df = pl.DataFrame(
        {"id": [0, 0, 0, 1, 1, 1], "t": [2, 4, 5, 10, 11, 14], "y": [0, 1, 1, 2, 3, 4]}
    )
    with pytest.raises(
        pl.InvalidOperationError,
        match=(
            r"output length of `map` \(6\) must be equal to "
            r"the input length \(1\); consider using `apply` instead"
        ),
    ):
        df.group_by("id").agg(pl.map_batches(["t", "y"], np.trapz))

    df = pl.DataFrame({"x": [1, 2, 3, 4], "group": [1, 2, 1, 2]})

    with pytest.raises(
        pl.InvalidOperationError,
        match=(
            r"output length of `map` \(4\) must be equal to "
            r"the input length \(1\); consider using `apply` instead"
        ),
    ):
        df.select(
            pl.col("x")
            .map_batches(
                lambda x: x.cut(breaks=[1, 2, 3], include_breaks=True).struct.unnest()
            )
            .over("group")
        )


def test_map_deprecated() -> None:
    with pytest.deprecated_call():
        pl.map(["a", "b"], lambda x: x[0])
    with pytest.deprecated_call():
        pl.col("a").map(lambda x: x)
    with pytest.deprecated_call():
        pl.LazyFrame({"a": [1, 2]}).map(lambda x: x)
