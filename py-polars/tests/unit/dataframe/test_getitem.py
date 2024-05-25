from __future__ import annotations

from typing import Any

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given

import polars as pl
from polars.testing import assert_frame_equal
from polars.testing.parametric import column, dataframes


@given(
    df=dataframes(
        max_size=10,
        cols=[
            column(
                "start",
                dtype=pl.Int8,
                allow_null=True,
                strategy=st.integers(min_value=-8, max_value=8),
            ),
            column(
                "stop",
                dtype=pl.Int8,
                allow_null=True,
                strategy=st.integers(min_value=-6, max_value=6),
            ),
            column(
                "step",
                dtype=pl.Int8,
                allow_null=True,
                strategy=st.integers(min_value=-4, max_value=4).filter(
                    lambda x: x != 0
                ),
            ),
            column("misc", dtype=pl.Int32),
        ],
    )
    # generated dataframe example -
    # ┌───────┬──────┬──────┬───────┐
    # │ start ┆ stop ┆ step ┆ misc  │
    # │ ---   ┆ ---  ┆ ---  ┆ ---   │
    # │ i8    ┆ i8   ┆ i8   ┆ i32   │
    # ╞═══════╪══════╪══════╪═══════╡
    # │ 2     ┆ -1   ┆ null ┆ -55   │
    # │ -3    ┆ 0    ┆ -2   ┆ 61582 │
    # │ null  ┆ 1    ┆ 2    ┆ 5865  │
    # └───────┴──────┴──────┴───────┘
)
def test_frame_slice(df: pl.DataFrame) -> None:
    # take strategy-generated integer values from the frame as slice bounds.
    # use these bounds to slice the same frame, and then validate the result
    # against a py-native slice of the same data using the same bounds.
    #
    # given the average number of rows in the frames, and the value of
    # max_examples, this will result in close to 5000 test permutations,
    # running in around ~1.5 secs (depending on hardware/etc).
    py_data = df.rows()

    for start, stop, step, _ in py_data:
        s = slice(start, stop, step)
        sliced_py_data = py_data[s]
        sliced_df_data = df[s].rows()

        assert (
            sliced_py_data == sliced_df_data
        ), f"slice [{start}:{stop}:{step}] failed on df w/len={len(df)}"


@pytest.mark.parametrize(
    "mask",
    [
        [True, False, True],
        pl.Series([True, False, True]),
        np.array([True, False, True]),
    ],
)
def test_df_getitem_column_boolean_mask(mask: Any) -> None:
    df = pl.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
    result = df[:, mask]
    expected = df.select("a", "c")
    assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ("selection", "match"),
    [
        (["a", 2], "'int' object cannot be converted to 'PyString'"),
        ([1, "c"], "'str' object cannot be interpreted as an integer"),
    ],
)
def test_df_getitem_column_mixed_inputs(selection: list[Any], match: str) -> None:
    df = pl.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
    with pytest.raises(TypeError, match=match):
        df[:, selection]
