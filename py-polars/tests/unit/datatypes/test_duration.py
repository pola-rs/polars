from datetime import timedelta

import pytest

import polars as pl
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


def test_duration_to_string() -> None:
    df = pl.DataFrame(
        {
            "td": [
                timedelta(days=180, seconds=56789, microseconds=987654),
                timedelta(days=0, seconds=64875, microseconds=8884),
                timedelta(days=2, hours=23, seconds=4975, milliseconds=1),
                timedelta(hours=1, seconds=1, milliseconds=1, microseconds=1),
                timedelta(seconds=-42, milliseconds=-42),
                None,
            ]
        },
        schema={"td": pl.Duration("us")},
    )

    df_str = df.select(
        td_ms=pl.col("td").cast(pl.Duration("ms")),
        td_int=pl.col("td").cast(pl.Int64),
        td_str_iso=pl.col("td").dt.to_string(),
        td_str_pl=pl.col("td").dt.to_string("polars"),
    )
    assert df_str.schema == {
        "td_ms": pl.Duration(time_unit="ms"),
        "td_int": pl.Int64,
        "td_str_iso": pl.String,
        "td_str_pl": pl.String,
    }

    expected = pl.DataFrame(
        {
            "td_ms": [
                timedelta(days=180, seconds=56789, milliseconds=987),
                timedelta(days=0, seconds=64875, milliseconds=8),
                timedelta(days=2, hours=23, seconds=4975, milliseconds=1),
                timedelta(hours=1, seconds=1, milliseconds=1),
                timedelta(seconds=-42, milliseconds=-42),
                None,
            ],
            "td_int": [
                15608789987654,
                64875008884,
                260575001000,
                3601001001,
                -42042000,
                None,
            ],
            "td_str_iso": [
                "P180DT15H46M29.987654S",
                "PT18H1M15.008884S",
                "P3DT22M55.001S",
                "PT1H1.001001S",
                "-PT42.042S",
                None,
            ],
            "td_str_pl": [
                "180d 15h 46m 29s 987654µs",
                "18h 1m 15s 8884µs",
                "3d 22m 55s 1ms",
                "1h 1s 1001µs",
                "-42s -42ms",
                None,
            ],
        },
        schema_overrides={"td_ms": pl.Duration(time_unit="ms")},
    )
    assert_frame_equal(expected, df_str)

    # individual +/- parts
    df = pl.DataFrame(
        {
            "td_ns": [
                timedelta(weeks=1),
                timedelta(days=1),
                timedelta(hours=1),
                timedelta(minutes=1),
                timedelta(seconds=1),
                timedelta(milliseconds=1),
                timedelta(microseconds=1),
                timedelta(seconds=0),
                timedelta(microseconds=-1),
                timedelta(milliseconds=-1),
                timedelta(seconds=-1),
                timedelta(minutes=-1),
                timedelta(hours=-1),
                timedelta(days=-1),
                timedelta(weeks=-1),
            ]
        },
        schema={"td_ns": pl.Duration("ns")},
    )
    df_str = df.select(pl.col("td_ns").dt.to_string("iso"))
    assert df_str["td_ns"].to_list() == [
        "P7D",
        "P1D",
        "PT1H",
        "PT1M",
        "PT1S",
        "PT0.001S",
        "PT0.000001S",
        "PT0S",
        "-PT0.000001S",
        "-PT0.001S",
        "-PT1S",
        "-PT1M",
        "-PT1H",
        "-P1D",
        "-P7D",
    ]


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
    with pytest.raises(OverflowError):
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


def test_comparison_with_string_raises_9461() -> None:
    df = pl.DataFrame({"duration": [timedelta(hours=2)]})
    with pytest.raises(pl.exceptions.InvalidOperationError):
        df.filter(pl.col("duration") > "1h")
