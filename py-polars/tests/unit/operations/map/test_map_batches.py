from __future__ import annotations

from functools import reduce

import numpy as np
import pytest

import polars as pl
from polars.exceptions import ComputeError, InvalidOperationError
from polars.testing import assert_frame_equal


def test_map_return_py_object() -> None:
    df = pl.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

    result = df.select([pl.all().map_batches(lambda s: reduce(lambda a, b: a + b, s))])

    expected = pl.DataFrame({"A": [6], "B": [15]})
    assert_frame_equal(result, expected)


def test_map_no_dtype_set_8531() -> None:
    df = pl.DataFrame({"a": [1]})

    result = df.with_columns(
        pl.col("a").map_batches(lambda x: x * 2).shift(n=0, fill_value=0)
    )

    expected = pl.DataFrame({"a": [2]})
    assert_frame_equal(result, expected)


def test_error_on_reducing_map() -> None:
    df = pl.DataFrame(
        {"id": [0, 0, 0, 1, 1, 1], "t": [2, 4, 5, 10, 11, 14], "y": [0, 1, 1, 2, 3, 4]}
    )
    with pytest.raises(
        InvalidOperationError,
        match=(
            r"output length of `map` \(1\) must be equal to "
            r"the input length \(6\); consider using `apply` instead"
        ),
    ):
        df.group_by("id").agg(pl.map_batches(["t", "y"], np.mean))

    df = pl.DataFrame({"x": [1, 2, 3, 4], "group": [1, 2, 1, 2]})

    with pytest.raises(
        InvalidOperationError,
        match=(
            r"output length of `map` \(1\) must be equal to "
            r"the input length \(4\); consider using `apply` instead"
        ),
    ):
        df.select(
            pl.col("x")
            .map_batches(
                lambda x: x.cut(breaks=[1, 2, 3], include_breaks=True).struct.unnest(),
                is_elementwise=True,
            )
            .over("group")
        )


def test_map_batches_group() -> None:
    df = pl.DataFrame(
        {"id": [0, 0, 0, 1, 1, 1], "t": [2, 4, 5, 10, 11, 14], "y": [0, 1, 1, 2, 3, 4]}
    )
    assert df.group_by("id").agg(pl.col("t").map_batches(lambda s: s.sum())).sort(
        "id"
    ).to_dict(as_series=False) == {"id": [0, 1], "t": [[11], [35]]}
    # If returns_scalar is True, the result won't be wrapped in a list:
    assert df.group_by("id").agg(
        pl.col("t").map_batches(lambda s: s.sum(), returns_scalar=True)
    ).sort("id").to_dict(as_series=False) == {"id": [0, 1], "t": [11, 35]}


def test_ufunc_args() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [2, 4, 6]})
    result = df.select(
        z=np.add(pl.col("a"), pl.col("b"))  # type: ignore[call-overload]
    )
    expected = pl.DataFrame({"z": [3, 6, 9]})
    assert_frame_equal(result, expected)
    result = df.select(z=np.add(2, pl.col("a")))  # type: ignore[call-overload]
    expected = pl.DataFrame({"z": [3, 4, 5]})
    assert_frame_equal(result, expected)


def test_lazy_map_schema() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": ["a", "b", "c"]})

    # identity
    assert_frame_equal(df.lazy().map_batches(lambda x: x).collect(), df)

    def custom(df: pl.DataFrame) -> pl.Series:
        return df["a"]

    with pytest.raises(
        ComputeError,
        match="Expected 'LazyFrame.map' to return a 'DataFrame', got a",
    ):
        df.lazy().map_batches(custom).collect()  # type: ignore[arg-type]

    def custom2(
        df: pl.DataFrame,
    ) -> pl.DataFrame:
        # changes schema
        return df.select(pl.all().cast(pl.String))

    with pytest.raises(
        ComputeError,
        match="The output schema of 'LazyFrame.map' is incorrect. Expected",
    ):
        df.lazy().map_batches(custom2).collect()

    assert df.lazy().map_batches(
        custom2, validate_output_schema=False
    ).collect().to_dict(as_series=False) == {"a": ["1", "2", "3"], "b": ["a", "b", "c"]}
