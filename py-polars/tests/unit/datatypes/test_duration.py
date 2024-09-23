from datetime import timedelta

import pytest

import polars as pl
from polars.exceptions import PanicException
from polars.testing import assert_frame_equal


def test_duration_cum_sum() -> None:
    df = pl.DataFrame({"A": [timedelta(days=1), timedelta(days=2)]})

    assert df.select(pl.col("A").cum_sum()).to_dict(as_series=False) == {
        "A": [timedelta(days=1), timedelta(days=3)]
    }
    assert df.schema["A"].is_(pl.Duration(time_unit="us"))
    for duration_dtype in (
        pl.Duration,
        pl.Duration(time_unit="ms"),
        pl.Duration(time_unit="ns"),
    ):
        assert df.schema["A"].is_(duration_dtype) is False


def test_duration_std_var() -> None:
    df = pl.DataFrame(
        {"duration": [1000, 5000, 3000]}, schema={"duration": pl.Duration}
    )

    result = df.select(
        pl.col("duration").var().name.suffix("_var"),
        pl.col("duration").std().name.suffix("_std"),
    )

    expected = pl.DataFrame(
        [
            pl.Series(
                "duration_var",
                [timedelta(microseconds=4000)],
                dtype=pl.Duration(time_unit="ms"),
            ),
            pl.Series(
                "duration_std",
                [timedelta(microseconds=2000)],
                dtype=pl.Duration(time_unit="us"),
            ),
        ]
    )

    assert_frame_equal(result, expected)


def test_series_duration_std_var() -> None:
    s = pl.Series([timedelta(days=1), timedelta(days=2), timedelta(days=4)])
    assert s.std() == timedelta(days=1, seconds=45578, microseconds=180014)
    assert s.var() == timedelta(days=201600000)


def test_series_duration_var_overflow() -> None:
    s = pl.Series([timedelta(days=10), timedelta(days=20), timedelta(days=40)])
    with pytest.raises(PanicException, match="OverflowError"):
        s.var()


def test_series_duration_units() -> None:
    td = timedelta

    assert_frame_equal(
        pl.DataFrame({"x": [0, 1, 2, 3]}).select(x=pl.duration(weeks=pl.col("x"))),
        pl.DataFrame({"x": [td(weeks=i) for i in range(4)]}),
    )
    assert_frame_equal(
        pl.DataFrame({"x": [0, 1, 2, 3]}).select(x=pl.duration(days=pl.col("x"))),
        pl.DataFrame({"x": [td(days=i) for i in range(4)]}),
    )
    assert_frame_equal(
        pl.DataFrame({"x": [0, 1, 2, 3]}).select(x=pl.duration(hours=pl.col("x"))),
        pl.DataFrame({"x": [td(hours=i) for i in range(4)]}),
    )
    assert_frame_equal(
        pl.DataFrame({"x": [0, 1, 2, 3]}).select(x=pl.duration(minutes=pl.col("x"))),
        pl.DataFrame({"x": [td(minutes=i) for i in range(4)]}),
    )
    assert_frame_equal(
        pl.DataFrame({"x": [0, 1, 2, 3]}).select(
            x=pl.duration(milliseconds=pl.col("x"))
        ),
        pl.DataFrame({"x": [td(milliseconds=i) for i in range(4)]}),
    )
    assert_frame_equal(
        pl.DataFrame({"x": [0, 1, 2, 3]}).select(
            x=pl.duration(microseconds=pl.col("x"))
        ),
        pl.DataFrame({"x": [td(microseconds=i) for i in range(4)]}),
    )
