# -------------------------------------------------
# Validate Series behaviour with parametric tests
# -------------------------------------------------
from __future__ import annotations


import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis.strategies import booleans, floats, lists, sampled_from, SearchStrategy
from numpy.testing import assert_allclose
from pandas.core.window.ewm import ExponentialMovingWindow

import polars as pl
from polars import PolarsDataType
from polars.testing import assert_series_equal, series

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


EWM_VAR_PARAMS = [
    (
        pl.Float32,
        "float32",
        floats(width=16, allow_infinity=False, allow_nan=False),
    ),
    (
        pl.Float64,
        "float64",
        floats(width=32, allow_infinity=False, allow_nan=False),
    ),
]


@pytest.mark.parametrize("polars_dtype,pandas_dtype,strategy", EWM_VAR_PARAMS)
def test_ewm_var_no_nans_no_infs(
    polars_dtype: PolarsDataType,
    pandas_dtype: str,
    strategy: SearchStrategy[float],
) -> None:
    """Test pl.Series.ewm_var against the pandas output."""

    @given(
        data=lists(strategy, max_size=10),
        alpha=floats(
            min_value=0,
            max_value=1,
            exclude_min=True,
            exclude_max=True,
            allow_nan=False,
            allow_infinity=False,
            width=16,
        ),
        adjust=booleans(),
        bias=booleans(),
    )
    def run_hypothesis_tests(
        data: list[float | None], alpha: float, adjust: bool, bias: bool
    ) -> None:
        pl_series = pl.Series(data, dtype=polars_dtype)
        pd_series = pd.Series(data, dtype=pandas_dtype)

        pl_ewm_var = pl_series.ewm_var(alpha=alpha, adjust=adjust, bias=bias)
        pd_ewm_var = ExponentialMovingWindow(
            obj=pd_series, alpha=alpha, adjust=adjust
        ).var(bias=bias)

        # There's an inconsistency in the pandas function.
        #   bias is False -> first variance value is always NaN
        #   bias is True -> first variance value is always 0.0
        # We've chosen to show zero variance in both cases.
        if data and not bias:
            pd_ewm_var[0] = 0.0

        assert_allclose(pl_ewm_var, pd_ewm_var, rtol=1e-4)

    run_hypothesis_tests()
