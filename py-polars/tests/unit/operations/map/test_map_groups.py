from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pytest

import polars as pl
from polars.exceptions import ComputeError
from polars.testing import assert_frame_equal


def test_map_groups() -> None:
    df = pl.DataFrame(
        {
            "a": ["a", "b", "a", "b", "b", "c"],
            "b": [1, 2, 3, 4, 5, 6],
            "c": [6, 5, 4, 3, 2, 1],
        }
    )

    result = df.group_by("a").map_groups(lambda df: df[["c"]].sum())

    expected = pl.DataFrame({"c": [10, 10, 1]})
    assert_frame_equal(result, expected, check_row_order=False)


def test_map_groups_lazy() -> None:
    lf = pl.LazyFrame({"a": [1, 1, 3], "b": [1.0, 2.0, 3.0]})

    schema = {"a": pl.Float64, "b": pl.Float64}
    result = lf.group_by("a").map_groups(lambda df: df * 2.0, schema=schema)

    expected = pl.LazyFrame({"a": [6.0, 2.0, 2.0], "b": [6.0, 2.0, 4.0]})
    assert_frame_equal(result, expected, check_row_order=False)
    assert result.collect_schema() == expected.collect_schema()


def test_map_groups_rolling() -> None:
    df = pl.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": [1, 2, 3, 4, 5],
        }
    ).set_sorted("a")

    def function(df: pl.DataFrame) -> pl.DataFrame:
        return df.select(
            pl.col("a").min(),
            pl.col("b").max(),
        )

    result = df.rolling("a", period="2i").map_groups(function, schema=df.schema)

    expected = pl.DataFrame(
        [
            pl.Series("a", [1, 1, 2, 3, 4], dtype=pl.Int64),
            pl.Series("b", [1, 2, 3, 4, 5], dtype=pl.Int64),
        ]
    )
    assert_frame_equal(result, expected)


def test_map_groups_empty() -> None:
    df = pl.DataFrame(schema={"x": pl.Int64})
    with pytest.raises(
        ComputeError, match=r"cannot group_by \+ apply on empty 'DataFrame'"
    ):
        df.group_by("x").map_groups(lambda x: x)


def test_map_groups_none() -> None:
    df = pl.DataFrame(
        {
            "g": [1, 1, 1, 2, 2, 2, 5],
            "a": [2, 4, 5, 190, 1, 4, 1],
            "b": [1, 3, 2, 1, 43, 3, 1],
        }
    )

    out = (
        df.group_by("g", maintain_order=True).agg(
            pl.map_groups(
                exprs=["a", pl.col("b") ** 4, pl.col("a") / 4],
                function=lambda x: x[0] * x[1] + x[2].sum(),
            ).alias("multiple")
        )
    )["multiple"]
    assert out[0].to_list() == [4.75, 326.75, 82.75]
    assert out[1].to_list() == [238.75, 3418849.75, 372.75]

    out_df = df.select(pl.map_batches(exprs=["a", "b"], function=lambda s: s[0] * s[1]))
    assert out_df["a"].to_list() == (df["a"] * df["b"]).to_list()

    # check if we can return None
    def func(s: Sequence[pl.Series]) -> pl.Series | None:
        if s[0][0] == 190:
            return None
        else:
            return s[0]

    out = (
        df.group_by("g", maintain_order=True).agg(
            pl.map_groups(
                exprs=["a", pl.col("b") ** 4, pl.col("a") / 4], function=func
            ).alias("multiple")
        )
    )["multiple"]
    assert out[1] is None


def test_map_groups_object_output() -> None:
    df = pl.DataFrame(
        {
            "names": ["foo", "ham", "spam", "cheese", "egg", "foo"],
            "dates": ["1", "1", "2", "3", "3", "4"],
            "groups": ["A", "A", "B", "B", "B", "C"],
        }
    )

    class Foo:
        def __init__(self, payload: Any):
            self.payload = payload

    result = df.group_by("groups").agg(
        pl.map_groups(
            [pl.col("dates"), pl.col("names")], lambda s: Foo(dict(zip(s[0], s[1])))
        )
    )

    assert result.dtypes == [pl.String, pl.Object]


def test_map_groups_numpy_output_3057() -> None:
    df = pl.DataFrame(
        {
            "id": [0, 0, 0, 1, 1, 1],
            "t": [2.0, 4.3, 5, 10, 11, 14],
            "y": [0.0, 1, 1.3, 2, 3, 4],
        }
    )

    result = df.group_by("id", maintain_order=True).agg(
        pl.map_groups(["y", "t"], lambda lst: np.mean([lst[0], lst[1]])).alias("result")
    )

    expected = pl.DataFrame({"id": [0, 1], "result": [2.266666, 7.333333]})
    assert_frame_equal(result, expected)
