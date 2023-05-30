from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import TYPE_CHECKING

import pandas as pd
import pytest

import polars as pl
from polars.datatypes import DTYPE_TEMPORAL_UNITS
from polars.testing import assert_frame_equal
from polars.utils.convert import get_zoneinfo as ZoneInfo

if TYPE_CHECKING:
    from polars.type_aliases import TimeUnit


def test_arange() -> None:
    ldf = pl.LazyFrame({"a": [1, 1, 1]})
    result = ldf.filter(pl.col("a") >= pl.arange(0, 3)).collect()
    expected = pl.DataFrame({"a": [1, 1]})
    assert_frame_equal(result, expected)


def test_arange_decreasing() -> None:
    assert pl.arange(10, 1, -2, eager=True).to_list() == list(range(10, 1, -2))


def test_arange_expr() -> None:
    df = pl.DataFrame({"a": ["foobar", "barfoo"]})
    out = df.select([pl.arange(0, pl.col("a").count() * 10)])
    assert out.shape == (20, 1)
    assert out.to_series(0)[-1] == 19

    # eager arange
    out2 = pl.arange(0, 10, 2, eager=True)
    assert out2.to_list() == [0, 2, 4, 6, 8]

    out3 = pl.arange(pl.Series([0, 19]), pl.Series([3, 39]), step=2, eager=True)
    assert out3.dtype == pl.List
    assert out3[0].to_list() == [0, 2]

    df = pl.DataFrame({"start": [1, 2, 3, 5, 5, 5], "stop": [8, 3, 12, 8, 8, 8]})

    assert df.select(pl.arange(pl.lit(1), pl.col("stop") + 1).alias("test")).to_dict(
        False
    ) == {
        "test": [
            [1, 2, 3, 4, 5, 6, 7, 8],
            [1, 2, 3],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            [1, 2, 3, 4, 5, 6, 7, 8],
            [1, 2, 3, 4, 5, 6, 7, 8],
            [1, 2, 3, 4, 5, 6, 7, 8],
        ]
    }


def test_arange_name() -> None:
    expected_name = "arange"
    result_eager = pl.arange(0, 5, eager=True)
    assert result_eager.name == expected_name

    result_lazy = pl.select(pl.arange(0, 5)).to_series()
    assert result_lazy.name == expected_name


def test_date_range() -> None:
    result = pl.date_range(
        date(1985, 1, 1), date(2015, 7, 1), timedelta(days=1, hours=12), eager=True
    )
    assert len(result) == 7426
    assert result.dt[0] == datetime(1985, 1, 1)
    assert result.dt[1] == datetime(1985, 1, 2, 12, 0)
    assert result.dt[2] == datetime(1985, 1, 4, 0, 0)
    assert result.dt[-1] == datetime(2015, 6, 30, 12, 0)

    for time_unit in DTYPE_TEMPORAL_UNITS:
        rng = pl.date_range(
            datetime(2020, 1, 1),
            date(2020, 1, 2),
            "2h",
            time_unit=time_unit,
            eager=True,
        )
        with pytest.warns(
            DeprecationWarning, match="`Series.time_unit` is deprecated.*"
        ):
            assert rng.time_unit == time_unit
        assert rng.shape == (13,)
        assert rng.dt[0] == datetime(2020, 1, 1)
        assert rng.dt[-1] == datetime(2020, 1, 2)

    # if low/high are both date, range is also be date _iif_ the granularity is >= 1d
    result = pl.date_range(date(2022, 1, 1), date(2022, 3, 1), "1mo", eager=True)
    assert result.to_list() == [date(2022, 1, 1), date(2022, 2, 1), date(2022, 3, 1)]

    result = pl.date_range(date(2022, 1, 1), date(2022, 1, 2), "1h30m", eager=True)
    assert list(result) == [
        datetime(2022, 1, 1, 0, 0),
        datetime(2022, 1, 1, 1, 30),
        datetime(2022, 1, 1, 3, 0),
        datetime(2022, 1, 1, 4, 30),
        datetime(2022, 1, 1, 6, 0),
        datetime(2022, 1, 1, 7, 30),
        datetime(2022, 1, 1, 9, 0),
        datetime(2022, 1, 1, 10, 30),
        datetime(2022, 1, 1, 12, 0),
        datetime(2022, 1, 1, 13, 30),
        datetime(2022, 1, 1, 15, 0),
        datetime(2022, 1, 1, 16, 30),
        datetime(2022, 1, 1, 18, 0),
        datetime(2022, 1, 1, 19, 30),
        datetime(2022, 1, 1, 21, 0),
        datetime(2022, 1, 1, 22, 30),
        datetime(2022, 1, 2, 0, 0),
    ]

    result = pl.date_range(
        datetime(2022, 1, 1), datetime(2022, 1, 1, 0, 1), "987456321ns", eager=True
    )
    assert len(result) == 61
    assert result.dtype.time_unit == "ns"  # type: ignore[union-attr]
    assert result.dt.second()[-1] == 59
    assert result.cast(pl.Utf8)[-1] == "2022-01-01 00:00:59.247379260"


@pytest.mark.parametrize(
    ("time_unit", "expected_micros"),
    [
        ("ms", 986000),
        ("us", 986759),
        ("ns", 986759),
        (None, 986759),
    ],
)
def test_date_range_precision(time_unit: TimeUnit | None, expected_micros: int) -> None:
    micros = 986759
    start = datetime(2000, 5, 30, 1, 53, 4, micros)
    stop = datetime(2000, 5, 31, 1, 53, 4, micros)
    result = pl.date_range(start, stop, time_unit=time_unit, eager=True)
    expected_start = start.replace(microsecond=expected_micros)
    expected_stop = stop.replace(microsecond=expected_micros)
    assert result[0] == expected_start
    assert result[1] == expected_stop


def test_range_invalid_unit() -> None:
    with pytest.raises(pl.PolarsPanicError, match="'D' not supported"):
        pl.date_range(
            start=datetime(2021, 12, 16),
            end=datetime(2021, 12, 16, 3),
            interval="1D",
            eager=True,
        )


def test_date_range_lazy_with_literals() -> None:
    df = pl.DataFrame({"misc": ["x"]}).with_columns(
        pl.date_range(
            date(2000, 1, 1),
            date(2023, 8, 31),
            interval="987d",
            eager=False,
        ).alias("dts")
    )
    assert df.rows() == [
        (
            "x",
            [
                date(2000, 1, 1),
                date(2002, 9, 14),
                date(2005, 5, 28),
                date(2008, 2, 9),
                date(2010, 10, 23),
                date(2013, 7, 6),
                date(2016, 3, 19),
                date(2018, 12, 1),
                date(2021, 8, 14),
            ],
        )
    ]
    assert (
        df.rows()[0][1]
        == pd.date_range(
            date(2000, 1, 1), date(2023, 12, 31), freq="987d"
        ).date.tolist()
    )


@pytest.mark.parametrize("low", ["start", pl.col("start")])
@pytest.mark.parametrize("high", ["stop", pl.col("stop")])
def test_date_range_lazy_with_expressions(
    low: str | pl.Expr, high: str | pl.Expr
) -> None:
    ldf = (
        pl.DataFrame({"start": [date(2015, 6, 30)], "stop": [date(2022, 12, 31)]})
        .with_columns(
            pl.date_range(low, high, interval="678d", eager=False).alias("dts")
        )
        .lazy()
    )

    assert ldf.collect().rows() == [
        (
            date(2015, 6, 30),
            date(2022, 12, 31),
            [
                date(2015, 6, 30),
                date(2017, 5, 8),
                date(2019, 3, 17),
                date(2021, 1, 23),
                date(2022, 12, 2),
            ],
        )
    ]

    assert pl.DataFrame(
        {
            "start": [date(2000, 1, 1), date(2022, 6, 1)],
            "stop": [date(2000, 1, 2), date(2022, 6, 2)],
        }
    ).with_columns(
        pl.date_range(
            low,
            high,
            interval="1d",
            eager=True,
        ).alias("dts")
    ).to_dict(
        False
    ) == {
        "start": [date(2000, 1, 1), date(2022, 6, 1)],
        "stop": [date(2000, 1, 2), date(2022, 6, 2)],
        "dts": [
            [date(2000, 1, 1), date(2000, 1, 2)],
            [date(2022, 6, 1), date(2022, 6, 2)],
        ],
    }

    assert pl.DataFrame(
        {
            "start": [datetime(2000, 1, 1), datetime(2022, 6, 1)],
            "stop": [datetime(2000, 1, 2), datetime(2022, 6, 2)],
        }
    ).with_columns(
        pl.date_range(
            low,
            high,
            interval="1d",
            eager=True,
        ).alias("dts")
    ).to_dict(
        False
    ) == {
        "start": [datetime(2000, 1, 1, 0, 0), datetime(2022, 6, 1, 0, 0)],
        "stop": [datetime(2000, 1, 2, 0, 0), datetime(2022, 6, 2, 0, 0)],
        "dts": [
            [datetime(2000, 1, 1, 0, 0), datetime(2000, 1, 2, 0, 0)],
            [datetime(2022, 6, 1, 0, 0), datetime(2022, 6, 2, 0, 0)],
        ],
    }


def test_date_range_single_row_lazy_7110() -> None:
    df = pl.DataFrame(
        {
            "name": ["A"],
            "from": [date(2020, 1, 1)],
            "to": [date(2020, 1, 2)],
        }
    )
    result = df.with_columns(
        pl.date_range(
            start=pl.col("from"),
            end=pl.col("to"),
            interval="1d",
            eager=False,
        ).alias("date_range")
    )
    expected = pl.DataFrame(
        {
            "name": ["A"],
            "from": [date(2020, 1, 1)],
            "to": [date(2020, 1, 2)],
            "date_range": [[date(2020, 1, 1), date(2020, 1, 2)]],
        }
    )
    assert_frame_equal(result, expected)


def test_date_range_invalid_time_zone() -> None:
    with pytest.raises(
        pl.ComputeError, match="unable to parse time zone: 'foo'"
    ), pytest.warns(
        DeprecationWarning,
        match="time zones other than those in `zoneinfo.available_timezones",
    ):
        pl.date_range(
            datetime(2001, 1, 1),
            datetime(2001, 1, 3),
            interval="1d",
            time_zone="foo",
            eager=True,
        )


def test_timezone_aware_date_range() -> None:
    low = datetime(2022, 10, 17, 10, tzinfo=ZoneInfo("Asia/Shanghai"))
    high = datetime(2022, 11, 17, 10, tzinfo=ZoneInfo("Asia/Shanghai"))

    assert pl.date_range(
        low, high, interval=timedelta(days=5), eager=True
    ).to_list() == [
        datetime(2022, 10, 17, 10, 0, tzinfo=ZoneInfo(key="Asia/Shanghai")),
        datetime(2022, 10, 22, 10, 0, tzinfo=ZoneInfo(key="Asia/Shanghai")),
        datetime(2022, 10, 27, 10, 0, tzinfo=ZoneInfo(key="Asia/Shanghai")),
        datetime(2022, 11, 1, 10, 0, tzinfo=ZoneInfo(key="Asia/Shanghai")),
        datetime(2022, 11, 6, 10, 0, tzinfo=ZoneInfo(key="Asia/Shanghai")),
        datetime(2022, 11, 11, 10, 0, tzinfo=ZoneInfo(key="Asia/Shanghai")),
        datetime(2022, 11, 16, 10, 0, tzinfo=ZoneInfo(key="Asia/Shanghai")),
    ]

    with pytest.raises(
        ValueError,
        match="Cannot mix different timezone aware datetimes. "
        "Got: 'Asia/Shanghai' and 'None'",
    ):
        pl.date_range(
            low,
            high.replace(tzinfo=None),
            interval=timedelta(days=5),
            time_zone="UTC",
            eager=True,
        )

    with pytest.raises(
        ValueError,
        match="Given time_zone is different from that of timezone aware datetimes. "
        "Given: 'UTC', got: 'Asia/Shanghai'.",
    ):
        pl.date_range(
            low, high, interval=timedelta(days=5), time_zone="UTC", eager=True
        )


def test_tzaware_date_range_crossing_dst_hourly() -> None:
    result = pl.date_range(
        datetime(2021, 11, 7),
        datetime(2021, 11, 7, 2),
        "1h",
        time_zone="US/Central",
        eager=True,
    )
    assert result.to_list() == [
        datetime(2021, 11, 7, 0, 0, tzinfo=ZoneInfo("US/Central")),
        datetime(2021, 11, 7, 1, 0, tzinfo=ZoneInfo("US/Central")),
        datetime(2021, 11, 7, 1, 0, fold=1, tzinfo=ZoneInfo("US/Central")),
        datetime(2021, 11, 7, 2, 0, tzinfo=ZoneInfo("US/Central")),
    ]


def test_tzaware_date_range_crossing_dst_daily() -> None:
    result = pl.date_range(
        datetime(2021, 11, 7),
        datetime(2021, 11, 11),
        "2d",
        time_zone="US/Central",
        eager=True,
    )
    assert result.to_list() == [
        datetime(2021, 11, 7, 0, 0, tzinfo=ZoneInfo("US/Central")),
        datetime(2021, 11, 9, 0, 0, tzinfo=ZoneInfo("US/Central")),
        datetime(2021, 11, 11, 0, 0, tzinfo=ZoneInfo("US/Central")),
    ]


def test_tzaware_date_range_crossing_dst_weekly() -> None:
    result = pl.date_range(
        datetime(2021, 11, 7),
        datetime(2021, 11, 20),
        "1w",
        time_zone="US/Central",
        eager=True,
    )
    assert result.to_list() == [
        datetime(2021, 11, 7, 0, 0, tzinfo=ZoneInfo("US/Central")),
        datetime(2021, 11, 14, 0, 0, tzinfo=ZoneInfo("US/Central")),
    ]


def test_tzaware_date_range_crossing_dst_monthly() -> None:
    result = pl.date_range(
        datetime(2021, 11, 7),
        datetime(2021, 12, 20),
        "1mo",
        time_zone="US/Central",
        eager=True,
    )
    assert result.to_list() == [
        datetime(2021, 11, 7, 0, 0, tzinfo=ZoneInfo("US/Central")),
        datetime(2021, 12, 7, 0, 0, tzinfo=ZoneInfo("US/Central")),
    ]


def test_date_range_with_unsupported_datetimes() -> None:
    with pytest.raises(
        pl.ComputeError,
        match=r"datetime '2021-11-07 01:00:00' is ambiguous in time zone 'US/Central'",
    ):
        pl.date_range(
            datetime(2021, 11, 7, 1),
            datetime(2021, 11, 7, 2),
            "1h",
            time_zone="US/Central",
            eager=True,
        )
    with pytest.raises(
        pl.ComputeError,
        match=r"datetime '2021-03-28 02:30:00' is non-existent in time zone 'Europe/Vienna'",
    ):
        pl.date_range(
            datetime(2021, 3, 28, 2, 30),
            datetime(2021, 3, 28, 4),
            "1h",
            time_zone="Europe/Vienna",
            eager=True,
        )


def test_date_range_descending() -> None:
    with pytest.raises(pl.ComputeError, match="'start' cannot be greater than 'stop'"):
        pl.date_range(
            datetime(2000, 3, 20), datetime(2000, 3, 5), interval="1h", eager=True
        )
    with pytest.raises(pl.ComputeError, match="'interval' cannot be negative"):
        pl.date_range(
            datetime(2000, 3, 20), datetime(2000, 3, 21), interval="-1h", eager=True
        )


def test_date_range_end_of_month_5441() -> None:
    start = date(2020, 1, 31)
    stop = date(2021, 1, 31)
    with pytest.raises(
        pl.ComputeError, match=r"cannot advance '2020-01-31 00:00:00' by 1 month\(s\)"
    ):
        pl.date_range(start, stop, interval="1mo", eager=True)


def test_date_range_name() -> None:
    expected_name = "date"
    result_eager = pl.date_range(date(2020, 1, 1), date(2020, 1, 3), eager=True)
    assert result_eager.name == expected_name

    result_lazy = pl.select(
        pl.date_range(date(2020, 1, 1), date(2020, 1, 3), eager=False)
    ).to_series()
    assert result_lazy.name == expected_name


def test_time_range_lit() -> None:
    for eager in (True, False):
        tm = pl.select(
            pl.time_range(
                start=time(1, 2, 3),
                end=time(23, 59, 59),
                interval="5h45m10s333ms",
                closed="right",
                eager=eager,
            ).alias("tm")
        )
        if not eager:
            tm = tm.select(pl.col("tm").explode())
        assert tm["tm"].to_list() == [
            time(6, 47, 13, 333000),
            time(12, 32, 23, 666000),
            time(18, 17, 33, 999000),
        ]

        # validate unset start/end
        tm = pl.select(
            pl.time_range(
                interval="5h45m10s333ms",
                eager=eager,
            ).alias("tm")
        )
        if not eager:
            tm = tm.select(pl.col("tm").explode())
        assert tm["tm"].to_list() == [
            time(0, 0),
            time(5, 45, 10, 333000),
            time(11, 30, 20, 666000),
            time(17, 15, 30, 999000),
            time(23, 0, 41, 332000),
        ]

        tm = pl.select(
            pl.time_range(
                start=pl.lit(time(23, 59, 59, 999980)),
                interval="10000ns",
                eager=eager,
            ).alias("tm")
        )
        tm = tm.select(pl.col("tm").explode())
        assert tm["tm"].to_list() == [
            time(23, 59, 59, 999980),
            time(23, 59, 59, 999990),
        ]


def test_time_range_expr() -> None:
    df = pl.DataFrame(
        {
            "start": pl.time_range(interval="6h", eager=True),
            "stop": pl.time_range(start=time(2, 59), interval="5h59m", eager=True),
        }
    ).with_columns(
        intervals=pl.time_range("start", pl.col("stop"), interval="1h29m", eager=False)
    )
    # shape: (4, 3)
    # ┌──────────┬──────────┬────────────────────────────────┐
    # │ start    ┆ stop     ┆ intervals                      │
    # │ ---      ┆ ---      ┆ ---                            │
    # │ time     ┆ time     ┆ list[time]                     │
    # ╞══════════╪══════════╪════════════════════════════════╡
    # │ 00:00:00 ┆ 02:59:00 ┆ [00:00:00, 01:29:00, 02:58:00] │
    # │ 06:00:00 ┆ 08:58:00 ┆ [06:00:00, 07:29:00, 08:58:00] │
    # │ 12:00:00 ┆ 14:57:00 ┆ [12:00:00, 13:29:00]           │
    # │ 18:00:00 ┆ 20:56:00 ┆ [18:00:00, 19:29:00]           │
    # └──────────┴──────────┴────────────────────────────────┘
    assert df.rows() == [
        (time(0, 0), time(2, 59), [time(0, 0), time(1, 29), time(2, 58)]),
        (time(6, 0), time(8, 58), [time(6, 0), time(7, 29), time(8, 58)]),
        (time(12, 0), time(14, 57), [time(12, 0), time(13, 29)]),
        (time(18, 0), time(20, 56), [time(18, 0), time(19, 29)]),
    ]


def test_time_range_name() -> None:
    expected_name = "time"
    result_eager = pl.time_range(time(10), time(12), eager=True)
    assert result_eager.name == expected_name

    result_lazy = pl.select(pl.time_range(time(10), time(12), eager=False)).to_series()
    assert result_lazy.name == expected_name


def test_deprecated_name_arg() -> None:
    name = "x"
    with pytest.deprecated_call():
        result_lazy = pl.date_range(date(2023, 1, 1), date(2023, 1, 3), name=name)
        assert result_lazy.meta.output_name() == name

    with pytest.deprecated_call():
        result_eager = pl.date_range(
            date(2023, 1, 1), date(2023, 1, 3), name=name, eager=True
        )
        assert result_eager.name == name

    with pytest.deprecated_call():
        result_lazy = pl.time_range(time(10), time(12), name=name)
        assert result_lazy.meta.output_name() == name

    with pytest.deprecated_call():
        result_eager = pl.time_range(time(10), time(12), name=name, eager=True)
        assert result_eager.name == name
