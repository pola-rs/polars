# -------------------------------------------------
# Validate Series behaviour with parametric tests
# -------------------------------------------------
from __future__ import annotations

from decimal import Decimal
from typing import no_type_check

from hypothesis import given, settings
from hypothesis.strategies import booleans, floats, sampled_from

import polars as pl
from polars.internals.expr.expr import _prepare_alpha
from polars.testing import assert_series_equal, series


def alpha_guard(**decay_param: float) -> bool:
    """Protects against unnecessary noise in small number regime."""
    if not list(decay_param.values())[0]:
        return True
    alpha = _prepare_alpha(**decay_param)
    return ((1 - alpha) if round(alpha) else alpha) > 1e-6


@given(
    s=series(
        min_size=4,
        dtype=pl.Float64,
        null_probability=0.05,
        strategy=floats(min_value=-1e8, max_value=1e8),
    ),
    half_life=floats(min_value=0, max_value=4, exclude_min=True).filter(
        lambda x: alpha_guard(half_life=x)
    ),
    com=floats(min_value=0, max_value=99).filter(lambda x: alpha_guard(com=x)),
    span=floats(min_value=1, max_value=10).filter(lambda x: alpha_guard(span=x)),
    adjust=booleans(),
    bias=booleans(),
)
@no_type_check
def test_ewm_methods(
    s: pl.Series,
    com: float | None,
    span: float | None,
    half_life: float | None,
    adjust: bool,
    bias: bool,
) -> None:
    # validate a large set of varied EWM calculations
    for decay_param in ({"com": com}, {"span": span}, {"half_life": half_life}):
        alpha = _prepare_alpha(**decay_param)

        # convert parametrically-generated series to pandas, then use that as a
        # reference implementation for comparison (after normalising NaN/None)
        p = s.to_pandas()

        # note: skip min_periods < 2, due to pandas-side inconsistency:
        # https://github.com/pola-rs/polars/issues/5006#issuecomment-1259477178
        for mp in range(2, len(s), len(s) // 3):
            # consolidate ewm parameters
            pl_params = {"min_periods": mp, "adjust": adjust}
            pl_params.update(decay_param)

            pd_params = pl_params.copy()
            if "half_life" in pl_params:
                pd_params["halflife"] = pd_params.pop("half_life")

            # mean:
            ewm_mean_pl = s.ewm_mean(**pl_params).fill_nan(None)
            ewm_mean_pd = pl.Series(p.ewm(ignore_na=True, **pd_params).mean())
            if alpha == 1:
                # apply fill-forward to nulls to match pandas
                # https://github.com/pola-rs/polars/pull/5011#issuecomment-1262318124
                ewm_mean_pl = ewm_mean_pl.fill_null(strategy="forward")

            assert_series_equal(ewm_mean_pl, ewm_mean_pd, atol=1e-07)

            # std:
            ewm_std_pl = s.ewm_std(bias=bias, **pl_params).fill_nan(None)
            ewm_std_pd = pl.Series(p.ewm(ignore_na=True, **pd_params).std(bias=bias))
            assert_series_equal(ewm_std_pl, ewm_std_pd, atol=1e-07)

            # var:
            ewm_var_pl = s.ewm_var(bias=bias, **pl_params).fill_nan(None)
            ewm_var_pd = pl.Series(p.ewm(ignore_na=True, **pd_params).var(bias=bias))
            assert_series_equal(ewm_var_pl, ewm_var_pd, atol=1e-07)


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


@given(
    s1=series(min_size=1, max_size=10, dtype=pl.Datetime),
    s2=series(min_size=1, max_size=10, dtype=pl.Duration),
)
def test_series_timeunits(
    s1: pl.Series,
    s2: pl.Series,
) -> None:
    # datetime
    assert s1.to_list() == list(s1)
    assert list(s1.dt.millisecond()) == [v.microsecond // 1000 for v in s1]
    assert list(s1.dt.nanosecond()) == [v.microsecond * 1000 for v in s1]
    assert list(s1.dt.microsecond()) == [v.microsecond for v in s1]

    # duration
    millis = s2.dt.milliseconds().to_list()
    micros = s2.dt.microseconds().to_list()

    assert s1.to_list() == list(s1)
    assert millis == [int(Decimal(v) / 1000) for v in s2.cast(int)]
    assert micros == list(s2.cast(int))

    # special handling for ns timeunit (as we may generate a microsecs-based
    # timedelta that results in 64bit overflow on conversion to nanosecs)
    lower_bound, upper_bound = -(2**63), (2**63) - 1
    if all(
        (lower_bound <= (us * 1000) <= upper_bound)
        for us in micros
        if isinstance(us, int)
    ):
        for ns, us in zip(s2.dt.nanoseconds(), micros):
            assert ns == (us * 1000)  # type: ignore[operator]
