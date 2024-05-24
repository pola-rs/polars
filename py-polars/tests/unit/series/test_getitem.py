from __future__ import annotations

import hypothesis.strategies as st
from hypothesis import given

import polars as pl
from polars.testing import assert_series_equal
from polars.testing.parametric import series


@given(
    srs=series(max_size=10, dtype=pl.Int64),
    start=st.sampled_from([-5, -4, -3, -2, -1, None, 0, 1, 2, 3, 4, 5]),
    stop=st.sampled_from([-5, -4, -3, -2, -1, None, 0, 1, 2, 3, 4, 5]),
    step=st.sampled_from([-5, -4, -3, -2, -1, None, 1, 2, 3, 4, 5]),
)
def test_series_getitem(
    srs: pl.Series,
    start: int | None,
    stop: int | None,
    step: int | None,
) -> None:
    py_data = srs.to_list()

    s = slice(start, stop, step)
    sliced_py_data = py_data[s]
    sliced_pl_data = srs[s].to_list()

    assert sliced_py_data == sliced_pl_data, f"slice [{start}:{stop}:{step}] failed"
    assert_series_equal(srs, srs, check_exact=True)
