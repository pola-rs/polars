# ----------------------------------------------------
# Validate LazyFrame behaviour with parametric tests
# ----------------------------------------------------
import hypothesis.strategies as st
from hypothesis import example, given

import polars as pl
from polars.testing.parametric import column, dataframes


@given(
    ldf=dataframes(
        max_size=10,
        lazy=True,
        cols=[
            column(
                "start",
                dtype=pl.Int8,
                allow_null=True,
                strategy=st.integers(min_value=-3, max_value=4),
            ),
            column(
                "stop",
                dtype=pl.Int8,
                allow_null=True,
                strategy=st.integers(min_value=-2, max_value=6),
            ),
            column(
                "step",
                dtype=pl.Int8,
                allow_null=True,
                strategy=st.integers(min_value=-3, max_value=3).filter(
                    lambda x: x != 0
                ),
            ),
            column("misc", dtype=pl.Int32),
        ],
    )
)
@example(
    ldf=pl.LazyFrame(
        {
            "start": [-1, None, 1, None, 1, -1],
            "stop": [None, 0, -1, -1, 2, 1],
            "step": [-1, -1, 1, None, -1, 1],
            "misc": [1, 2, 3, 4, 5, 6],
        }
    )
)
def test_lazyframe_getitem(ldf: pl.LazyFrame) -> None:
    py_data = ldf.collect().rows()

    for start, stop, step, _ in py_data:
        s = slice(start, stop, step)
        sliced_py_data = py_data[s]
        try:
            sliced_df_data = ldf[s].collect().rows()
            assert (
                sliced_py_data == sliced_df_data
            ), f"slice [{start}:{stop}:{step}] failed on lazy df w/len={len(py_data)}"

        except ValueError as exc:
            # test params will trigger some known
            # unsupported cases; filter them here.
            if "not supported" not in str(exc):
                raise
