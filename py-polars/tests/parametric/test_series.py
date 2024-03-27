# -------------------------------------------------
# Validate Series behaviour with parametric tests
# -------------------------------------------------
from __future__ import annotations

from hypothesis import given, settings
from hypothesis.strategies import sampled_from

import polars as pl
from polars.testing import assert_series_equal
from polars.testing.parametric import series


@given(
    s=series(min_size=1, max_size=10, dtype=pl.Datetime),
)
def test_series_datetime_timeunits(
    s: pl.Series,
) -> None:
    # datetime
    assert s.to_list() == list(s)
    assert list(s.dt.millisecond()) == [v.microsecond // 1000 for v in s]
    assert list(s.dt.nanosecond()) == [v.microsecond * 1000 for v in s]
    assert list(s.dt.microsecond()) == [v.microsecond for v in s]


@given(
    s=series(min_size=1, max_size=10, dtype=pl.Duration),
)
def test_series_duration_timeunits(
    s: pl.Series,
) -> None:
    nanos = s.dt.total_nanoseconds().to_list()
    micros = s.dt.total_microseconds().to_list()
    millis = s.dt.total_milliseconds().to_list()

    scale = {
        "ns": 1,
        "us": 1_000,
        "ms": 1_000_000,
    }
    assert nanos == [v * scale[s.dtype.time_unit] for v in s.to_physical()]  # type: ignore[attr-defined]
    assert micros == [int(v / 1_000) for v in nanos]
    assert millis == [int(v / 1_000) for v in micros]

    # special handling for ns timeunit (as we may generate a microsecs-based
    # timedelta that results in 64bit overflow on conversion to nanosecs)
    micros = s.dt.total_microseconds().to_list()
    lower_bound, upper_bound = -(2**63), (2**63) - 1
    if all(
        (lower_bound <= (us * 1000) <= upper_bound)
        for us in micros
        if isinstance(us, int)
    ):
        for ns, us in zip(s.dt.total_nanoseconds(), micros):
            assert ns == (us * 1000)


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
