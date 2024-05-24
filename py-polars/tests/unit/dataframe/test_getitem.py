from __future__ import annotations

import hypothesis.strategies as st
from hypothesis import given

import polars as pl
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
