from __future__ import annotations

from typing import Any

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given

import polars as pl
from polars.expr.expr import _prepare_alpha
from polars.testing import assert_series_equal
from polars.testing.parametric import series


def test_ewm_mean() -> None:
    s = pl.Series([2, 5, 3])

    expected = pl.Series([2.0, 4.0, 3.4285714285714284])
    assert_series_equal(s.ewm_mean(alpha=0.5, adjust=True, ignore_nulls=True), expected)
    assert_series_equal(
        s.ewm_mean(alpha=0.5, adjust=True, ignore_nulls=False), expected
    )

    expected = pl.Series([2.0, 3.8, 3.421053])
    assert_series_equal(s.ewm_mean(com=2.0, adjust=True, ignore_nulls=True), expected)
    assert_series_equal(s.ewm_mean(com=2.0, adjust=True, ignore_nulls=False), expected)

    expected = pl.Series([2.0, 3.5, 3.25])
    assert_series_equal(
        s.ewm_mean(alpha=0.5, adjust=False, ignore_nulls=True), expected
    )
    assert_series_equal(
        s.ewm_mean(alpha=0.5, adjust=False, ignore_nulls=False), expected
    )

    s = pl.Series([2, 3, 5, 7, 4])

    expected = pl.Series([None, 2.666667, 4.0, 5.6, 4.774194])
    assert_series_equal(
        s.ewm_mean(alpha=0.5, adjust=True, min_periods=2, ignore_nulls=True), expected
    )
    assert_series_equal(
        s.ewm_mean(alpha=0.5, adjust=True, min_periods=2, ignore_nulls=False), expected
    )

    expected = pl.Series([None, None, 4.0, 5.6, 4.774194])
    assert_series_equal(
        s.ewm_mean(alpha=0.5, adjust=True, min_periods=3, ignore_nulls=True), expected
    )
    assert_series_equal(
        s.ewm_mean(alpha=0.5, adjust=True, min_periods=3, ignore_nulls=False), expected
    )

    s = pl.Series([None, 1.0, 5.0, 7.0, None, 2.0, 5.0, 4])

    expected = pl.Series(
        [
            None,
            1.0,
            3.6666666666666665,
            5.571428571428571,
            None,
            3.6666666666666665,
            4.354838709677419,
            4.174603174603175,
        ],
    )
    assert_series_equal(s.ewm_mean(alpha=0.5, adjust=True, ignore_nulls=True), expected)
    expected = pl.Series(
        [
            None,
            1.0,
            3.666666666666667,
            5.571428571428571,
            None,
            3.08695652173913,
            4.2,
            4.092436974789916,
        ]
    )
    assert_series_equal(
        s.ewm_mean(alpha=0.5, adjust=True, ignore_nulls=False), expected
    )

    expected = pl.Series([None, 1.0, 3.0, 5.0, None, 3.5, 4.25, 4.125])
    assert_series_equal(
        s.ewm_mean(alpha=0.5, adjust=False, ignore_nulls=True), expected
    )

    expected = pl.Series([None, 1.0, 3.0, 5.0, None, 3.0, 4.0, 4.0])
    assert_series_equal(
        s.ewm_mean(alpha=0.5, adjust=False, ignore_nulls=False), expected
    )


def test_ewm_mean_leading_nulls() -> None:
    for min_periods in [1, 2, 3]:
        assert (
            pl.Series([1, 2, 3, 4])
            .ewm_mean(com=3, min_periods=min_periods, ignore_nulls=False)
            .null_count()
            == min_periods - 1
        )
    assert pl.Series([None, 1.0, 1.0, 1.0]).ewm_mean(
        alpha=0.5, min_periods=1, ignore_nulls=True
    ).to_list() == [None, 1.0, 1.0, 1.0]
    assert pl.Series([None, 1.0, 1.0, 1.0]).ewm_mean(
        alpha=0.5, min_periods=2, ignore_nulls=True
    ).to_list() == [None, None, 1.0, 1.0]


def test_ewm_mean_min_periods() -> None:
    series = pl.Series([1.0, None, None, None])

    ewm_mean = series.ewm_mean(alpha=0.5, min_periods=1, ignore_nulls=True)
    assert ewm_mean.to_list() == [1.0, None, None, None]
    ewm_mean = series.ewm_mean(alpha=0.5, min_periods=2, ignore_nulls=True)
    assert ewm_mean.to_list() == [None, None, None, None]

    series = pl.Series([1.0, None, 2.0, None, 3.0])

    ewm_mean = series.ewm_mean(alpha=0.5, min_periods=1, ignore_nulls=True)
    assert_series_equal(
        ewm_mean,
        pl.Series(
            [
                1.0,
                None,
                1.6666666666666665,
                None,
                2.4285714285714284,
            ]
        ),
    )
    ewm_mean = series.ewm_mean(alpha=0.5, min_periods=2, ignore_nulls=True)
    assert_series_equal(
        ewm_mean,
        pl.Series(
            [
                None,
                None,
                1.6666666666666665,
                None,
                2.4285714285714284,
            ]
        ),
    )


def test_ewm_std_var() -> None:
    series = pl.Series("a", [2, 5, 3])

    var = series.ewm_var(alpha=0.5, ignore_nulls=False)
    std = series.ewm_std(alpha=0.5, ignore_nulls=False)
    expected = pl.Series("a", [0.0, 4.5, 1.9285714285714288])
    assert np.allclose(var, std**2, rtol=1e-16)
    assert_series_equal(var, expected)


def test_ewm_std_var_with_nulls() -> None:
    series = pl.Series("a", [2, 5, None, 3])

    var = series.ewm_var(alpha=0.5, ignore_nulls=True)
    std = series.ewm_std(alpha=0.5, ignore_nulls=True)
    expected = pl.Series("a", [0.0, 4.5, None, 1.9285714285714288])
    assert_series_equal(var, expected)
    assert_series_equal(std**2, expected)

    var = series.ewm_var(alpha=0.5, ignore_nulls=False)
    std = series.ewm_std(alpha=0.5, ignore_nulls=False)
    expected = pl.Series("a", [0.0, 4.5, None, 1.7307692307692308])
    assert_series_equal(var, expected)
    assert_series_equal(std**2, expected)


def test_ewm_param_validation() -> None:
    s = pl.Series("values", range(10))

    with pytest.raises(ValueError, match="mutually exclusive"):
        s.ewm_std(com=0.5, alpha=0.5, ignore_nulls=False)

    with pytest.raises(ValueError, match="mutually exclusive"):
        s.ewm_mean(span=1.5, half_life=0.75, ignore_nulls=False)

    with pytest.raises(ValueError, match="mutually exclusive"):
        s.ewm_var(alpha=0.5, span=1.5, ignore_nulls=False)

    with pytest.raises(ValueError, match="require `com` >= 0"):
        s.ewm_std(com=-0.5, ignore_nulls=False)

    with pytest.raises(ValueError, match="require `span` >= 1"):
        s.ewm_mean(span=0.5, ignore_nulls=False)

    with pytest.raises(ValueError, match="require `half_life` > 0"):
        s.ewm_var(half_life=0, ignore_nulls=False)

    for alpha in (-0.5, -0.0000001, 0.0, 1.0000001, 1.5):
        with pytest.raises(ValueError, match="require 0 < `alpha` <= 1"):
            s.ewm_std(alpha=alpha, ignore_nulls=False)


# https://github.com/pola-rs/polars/issues/4951
def test_ewm_with_multiple_chunks() -> None:
    df0 = pl.DataFrame(
        data=[
            ("w", 6.0, 1.0),
            ("x", 5.0, 2.0),
            ("y", 4.0, 3.0),
            ("z", 3.0, 4.0),
        ],
        schema=["a", "b", "c"],
        orient="row",
    ).with_columns(
        pl.col(pl.Float64).log().diff().name.prefix("ld_"),
    )
    assert df0.n_chunks() == 1

    # NOTE: We aren't testing whether `select` creates two chunks;
    # we just need two chunks to properly test `ewm_mean`
    df1 = df0.select(["ld_b", "ld_c"])
    assert df1.n_chunks() == 2

    ewm_std = df1.with_columns(
        pl.all().ewm_std(com=20, ignore_nulls=False).name.prefix("ewm_"),
    )
    assert ewm_std.null_count().sum_horizontal()[0] == 4


def alpha_guard(**decay_param: float) -> bool:
    """Protects against unnecessary noise in small number regime."""
    if not next(iter(decay_param.values())):
        return True
    alpha = _prepare_alpha(**decay_param)
    return ((1 - alpha) if round(alpha) else alpha) > 1e-6


@given(
    s=series(
        min_size=4,
        dtype=pl.Float64,
        allow_null=True,
        strategy=st.floats(min_value=-1e8, max_value=1e8),
    ),
    half_life=st.floats(min_value=0, max_value=4, exclude_min=True).filter(
        lambda x: alpha_guard(half_life=x)
    ),
    com=st.floats(min_value=0, max_value=99).filter(lambda x: alpha_guard(com=x)),
    span=st.floats(min_value=1, max_value=10).filter(lambda x: alpha_guard(span=x)),
    ignore_nulls=st.booleans(),
    adjust=st.booleans(),
    bias=st.booleans(),
)
def test_ewm_methods(
    s: pl.Series,
    com: float | None,
    span: float | None,
    half_life: float | None,
    ignore_nulls: bool,
    adjust: bool,
    bias: bool,
) -> None:
    # validate a large set of varied EWM calculations
    for decay_param in [{"com": com}, {"span": span}, {"half_life": half_life}]:
        alpha = _prepare_alpha(**decay_param)

        # convert parametrically-generated series to pandas, then use that as a
        # reference implementation for comparison (after normalising NaN/None)
        p = s.to_pandas()

        # note: skip min_periods < 2, due to pandas-side inconsistency:
        # https://github.com/pola-rs/polars/issues/5006#issuecomment-1259477178
        for mp in range(2, len(s), len(s) // 3):
            # consolidate ewm parameters
            pl_params: dict[str, Any] = {
                "min_periods": mp,
                "adjust": adjust,
                "ignore_nulls": ignore_nulls,
            }
            pl_params.update(decay_param)
            pd_params = pl_params.copy()
            if "half_life" in pl_params:
                pd_params["halflife"] = pd_params.pop("half_life")
            if "ignore_nulls" in pl_params:
                pd_params["ignore_na"] = pd_params.pop("ignore_nulls")

            # mean:
            ewm_mean_pl = s.ewm_mean(**pl_params).fill_nan(None)
            ewm_mean_pd = pl.Series(p.ewm(**pd_params).mean())
            if alpha == 1:
                # apply fill-forward to nulls to match pandas
                # https://github.com/pola-rs/polars/pull/5011#issuecomment-1262318124
                ewm_mean_pl = ewm_mean_pl.fill_null(strategy="forward")

            assert_series_equal(ewm_mean_pl, ewm_mean_pd, atol=1e-07)

            # std:
            ewm_std_pl = s.ewm_std(bias=bias, **pl_params).fill_nan(None)
            ewm_std_pd = pl.Series(p.ewm(**pd_params).std(bias=bias))
            assert_series_equal(ewm_std_pl, ewm_std_pd, atol=1e-07)

            # var:
            ewm_var_pl = s.ewm_var(bias=bias, **pl_params).fill_nan(None)
            ewm_var_pd = pl.Series(p.ewm(**pd_params).var(bias=bias))
            assert_series_equal(ewm_var_pl, ewm_var_pd, atol=1e-07)
