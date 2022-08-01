# -------------------------------------------------
# Validate Series behaviour with parametric tests
# -------------------------------------------------
from __future__ import annotations

from hypothesis import given, settings
from hypothesis.strategies import sampled_from

import polars as pl
from polars.testing import assert_series_equal, series  # , verify_series_and_expr_api

# # TODO: exclude obvious/known overflow inside the strategy before commenting back in
# @given(s=series(allowed_dtypes=_NUMERIC_COL_TYPES, name="a"))
# def test_cum_agg_extra(s: pl.Series) -> None:
#     # confirm that ops on generated Series match equivalent Expr call
#     # note: testing codepath-equivalence, not correctness.
#     for op in ("cumsum", "cummin", "cummax", "cumprod"):
#          verify_series_and_expr_api(s, None, op)


@given(
    srs=series(max_size=10, dtype=pl.Int64),
    start=sampled_from([-5, -4, -3, -2, -1, None, 0, 1, 2, 3, 4, 5]),
    stop=sampled_from([-5, -4, -3, -2, -1, None, 0, 1, 2, 3, 4, 5]),
    step=sampled_from([-5, -4, -3, -2, -1, None, 1, 2, 3, 4, 5]),
)
@settings(max_examples=500)
def test_series_slice(
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
