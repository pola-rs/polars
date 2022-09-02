from hypothesis import given, settings
from hypothesis.strategies import integers

import polars as pl
from polars.testing import column, dataframes


@given(
    ldf=dataframes(
        max_size=10,
        lazy=True,
        cols=[
            column(
                "start",
                dtype=pl.Int8,
                null_probability=0.15,
                strategy=integers(min_value=-4, max_value=6),
            ),
            column(
                "stop",
                dtype=pl.Int8,
                null_probability=0.15,
                strategy=integers(min_value=-2, max_value=8),
            ),
            column(
                "step",
                dtype=pl.Int8,
                null_probability=0.15,
                strategy=integers(min_value=-3, max_value=3).filter(lambda x: x != 0),
            ),
            column("misc", dtype=pl.Int32),
        ],
    )
)
@settings(max_examples=500)
def test_lazyframe_slice(ldf: pl.LazyFrame) -> None:
    py_data = ldf.collect().rows()

    for start, stop, step, _ in py_data:
        s = slice(start, stop, step)
        sliced_py_data = py_data[s]
        try:
            sliced_df_data = ldf[s].collect().rows()
            assert (
                sliced_py_data == sliced_df_data
            ), f"slice [{start}:{stop}:{step}] failed on lazy df w/len={len(py_data)}"

        except ValueError as err:
            # test params will trigger some known
            # unsupported cases; filter them here.
            if "not supported" not in str(err):
                raise
