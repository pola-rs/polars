from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

import polars as pl
from polars.exceptions import ComputeError, InvalidOperationError
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from zoneinfo import ZoneInfo

    from polars._typing import Label, StartBy
else:
    from polars._utils.convert import string_to_zoneinfo as ZoneInfo


@pytest.mark.parametrize(
    ("input_df", "expected_grouped_df"),
    [
        (
            (
                pl.DataFrame(
                    {
                        "dt": [
                            datetime(2021, 12, 31, 0, 0, 0),
                            datetime(2022, 1, 1, 0, 0, 1),
                            datetime(2022, 3, 31, 0, 0, 1),
                            datetime(2022, 4, 1, 0, 0, 1),
                        ]
                    }
                )
            ),
            pl.DataFrame(
                {
                    "dt": [
                        datetime(2021, 10, 1),
                        datetime(2022, 1, 1),
                        datetime(2022, 4, 1),
                    ],
                    "num_points": [1, 2, 1],
                },
                schema={"dt": pl.Datetime, "num_points": pl.UInt32},
            ).sort("dt"),
        )
    ],
)
def test_group_by_dynamic(
    input_df: pl.DataFrame, expected_grouped_df: pl.DataFrame
) -> None:
    result = (
        input_df.sort("dt")
        .group_by_dynamic("dt", every="1q")
        .agg(pl.col("dt").count().alias("num_points"))
        .sort("dt")
    )
    assert_frame_equal(result, expected_grouped_df)


@pytest.mark.parametrize(
    ("every", "offset"),
    [
        ("3d", "-1d"),
        (timedelta(days=3), timedelta(days=-1)),
    ],
)
def test_dynamic_group_by_timezone_awareness(
    every: str | timedelta, offset: str | timedelta
) -> None:
    df = pl.DataFrame(
        (
            pl.datetime_range(
                datetime(2020, 1, 1),
                datetime(2020, 1, 10),
                timedelta(days=1),
                time_unit="ns",
                eager=True,
            )
            .alias("datetime")
            .dt.replace_time_zone("UTC"),
            pl.arange(1, 11, eager=True).alias("value"),
        )
    )

    assert (
        df.group_by_dynamic(
            "datetime",
            every=every,
            offset=offset,
            closed="right",
            include_boundaries=True,
            label="datapoint",
        ).agg(pl.col("value").last())
    ).dtypes == [pl.Datetime("ns", "UTC")] * 3 + [pl.Int64]


@pytest.mark.parametrize("tzinfo", [None, ZoneInfo("UTC"), ZoneInfo("Asia/Kathmandu")])
def test_group_by_dynamic_startby_5599(tzinfo: ZoneInfo | None) -> None:
    # start by datapoint
    start = datetime(2022, 12, 16, tzinfo=tzinfo)
    stop = datetime(2022, 12, 16, hour=3, tzinfo=tzinfo)
    df = pl.DataFrame({"date": pl.datetime_range(start, stop, "30m", eager=True)})

    assert df.group_by_dynamic(
        "date",
        every="31m",
        include_boundaries=True,
        label="datapoint",
        start_by="datapoint",
    ).agg(pl.len()).to_dict(as_series=False) == {
        "_lower_boundary": [
            datetime(2022, 12, 16, 0, 0, tzinfo=tzinfo),
            datetime(2022, 12, 16, 0, 31, tzinfo=tzinfo),
            datetime(2022, 12, 16, 1, 2, tzinfo=tzinfo),
            datetime(2022, 12, 16, 1, 33, tzinfo=tzinfo),
            datetime(2022, 12, 16, 2, 4, tzinfo=tzinfo),
            datetime(2022, 12, 16, 2, 35, tzinfo=tzinfo),
        ],
        "_upper_boundary": [
            datetime(2022, 12, 16, 0, 31, tzinfo=tzinfo),
            datetime(2022, 12, 16, 1, 2, tzinfo=tzinfo),
            datetime(2022, 12, 16, 1, 33, tzinfo=tzinfo),
            datetime(2022, 12, 16, 2, 4, tzinfo=tzinfo),
            datetime(2022, 12, 16, 2, 35, tzinfo=tzinfo),
            datetime(2022, 12, 16, 3, 6, tzinfo=tzinfo),
        ],
        "date": [
            datetime(2022, 12, 16, 0, 0, tzinfo=tzinfo),
            datetime(2022, 12, 16, 1, 0, tzinfo=tzinfo),
            datetime(2022, 12, 16, 1, 30, tzinfo=tzinfo),
            datetime(2022, 12, 16, 2, 0, tzinfo=tzinfo),
            datetime(2022, 12, 16, 2, 30, tzinfo=tzinfo),
            datetime(2022, 12, 16, 3, 0, tzinfo=tzinfo),
        ],
        "len": [2, 1, 1, 1, 1, 1],
    }

    # start by monday
    start = datetime(2022, 1, 1, tzinfo=tzinfo)
    stop = datetime(2022, 1, 12, 7, tzinfo=tzinfo)

    df = pl.DataFrame(
        {"date": pl.datetime_range(start, stop, "12h", eager=True)}
    ).with_columns(pl.col("date").dt.weekday().alias("day"))

    result = df.group_by_dynamic(
        "date",
        every="1w",
        period="3d",
        include_boundaries=True,
        start_by="monday",
        label="datapoint",
    ).agg([pl.len(), pl.col("day").first().alias("data_day")])
    assert result.to_dict(as_series=False) == {
        "_lower_boundary": [
            datetime(2022, 1, 3, 0, 0, tzinfo=tzinfo),
            datetime(2022, 1, 10, 0, 0, tzinfo=tzinfo),
        ],
        "_upper_boundary": [
            datetime(2022, 1, 6, 0, 0, tzinfo=tzinfo),
            datetime(2022, 1, 13, 0, 0, tzinfo=tzinfo),
        ],
        "date": [
            datetime(2022, 1, 3, 0, 0, tzinfo=tzinfo),
            datetime(2022, 1, 10, 0, 0, tzinfo=tzinfo),
        ],
        "len": [6, 5],
        "data_day": [1, 1],
    }
    # start by saturday
    result = df.group_by_dynamic(
        "date",
        every="1w",
        period="3d",
        include_boundaries=True,
        start_by="saturday",
        label="datapoint",
    ).agg([pl.len(), pl.col("day").first().alias("data_day")])
    assert result.to_dict(as_series=False) == {
        "_lower_boundary": [
            datetime(2022, 1, 1, 0, 0, tzinfo=tzinfo),
            datetime(2022, 1, 8, 0, 0, tzinfo=tzinfo),
        ],
        "_upper_boundary": [
            datetime(2022, 1, 4, 0, 0, tzinfo=tzinfo),
            datetime(2022, 1, 11, 0, 0, tzinfo=tzinfo),
        ],
        "date": [
            datetime(2022, 1, 1, 0, 0, tzinfo=tzinfo),
            datetime(2022, 1, 8, 0, 0, tzinfo=tzinfo),
        ],
        "len": [6, 6],
        "data_day": [6, 6],
    }


def test_group_by_dynamic_by_monday_and_offset_5444() -> None:
    df = pl.DataFrame(
        {
            "date": [
                "2022-11-01",
                "2022-11-02",
                "2022-11-05",
                "2022-11-08",
                "2022-11-08",
                "2022-11-09",
                "2022-11-10",
            ],
            "label": ["a", "b", "a", "a", "b", "a", "b"],
            "value": [1, 2, 3, 4, 5, 6, 7],
        }
    ).with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d").set_sorted())

    result = df.group_by_dynamic(
        "date", every="1w", offset="1d", group_by="label", start_by="monday"
    ).agg(pl.col("value").sum())

    assert result.to_dict(as_series=False) == {
        "label": ["a", "a", "b", "b"],
        "date": [
            date(2022, 11, 1),
            date(2022, 11, 8),
            date(2022, 11, 1),
            date(2022, 11, 8),
        ],
        "value": [4, 10, 2, 12],
    }

    # test empty
    result_empty = (
        df.filter(pl.col("date") == date(1, 1, 1))
        .group_by_dynamic(
            "date", every="1w", offset="1d", group_by="label", start_by="monday"
        )
        .agg(pl.col("value").sum())
    )
    assert result_empty.schema == result.schema


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("left", [datetime(2020, 1, 1), datetime(2020, 1, 2)]),
        ("right", [datetime(2020, 1, 2), datetime(2020, 1, 3)]),
        ("datapoint", [datetime(2020, 1, 1, 1), datetime(2020, 1, 2, 3)]),
    ],
)
def test_group_by_dynamic_label(label: Label, expected: list[datetime]) -> None:
    df = pl.DataFrame(
        {
            "ts": [
                datetime(2020, 1, 1, 1),
                datetime(2020, 1, 1, 2),
                datetime(2020, 1, 2, 3),
                datetime(2020, 1, 2, 4),
            ],
            "n": [1, 2, 3, 4],
            "group": ["a", "a", "b", "b"],
        }
    ).sort("ts")
    result = (
        df.group_by_dynamic("ts", every="1d", label=label, group_by="group")
        .agg(pl.col("n"))["ts"]
        .to_list()
    )
    assert result == expected


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("left", [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)]),
        ("right", [datetime(2020, 1, 2), datetime(2020, 1, 3), datetime(2020, 1, 4)]),
        (
            "datapoint",
            [datetime(2020, 1, 1, 1), datetime(2020, 1, 2, 2), datetime(2020, 1, 3, 3)],
        ),
    ],
)
def test_group_by_dynamic_label_with_by(label: Label, expected: list[datetime]) -> None:
    df = pl.DataFrame(
        {
            "ts": [
                datetime(2020, 1, 1, 1),
                datetime(2020, 1, 2, 2),
                datetime(2020, 1, 3, 3),
            ],
            "n": [1, 2, 3],
        }
    ).sort("ts")
    result = (
        df.group_by_dynamic("ts", every="1d", label=label)
        .agg(pl.col("n"))["ts"]
        .to_list()
    )
    assert result == expected


def test_group_by_dynamic_slice_pushdown() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": ["a", "a", "b"], "c": [1, 3, 5]}).lazy()
    df = (
        df.sort("a")
        .group_by_dynamic("a", group_by="b", every="2i")
        .agg((pl.col("c") - pl.col("c").shift(fill_value=0)).sum().alias("c"))
    )
    assert df.head(2).collect().to_dict(as_series=False) == {
        "b": ["a", "a"],
        "a": [0, 2],
        "c": [1, 3],
    }


def test_rolling_kernels_group_by_dynamic_7548() -> None:
    assert pl.DataFrame(
        {"time": pl.arange(0, 4, eager=True), "value": pl.arange(0, 4, eager=True)}
    ).group_by_dynamic("time", every="1i", period="3i").agg(
        pl.col("value"),
        pl.col("value").min().alias("min_value"),
        pl.col("value").max().alias("max_value"),
        pl.col("value").sum().alias("sum_value"),
    ).to_dict(as_series=False) == {
        "time": [0, 1, 2, 3],
        "value": [[0, 1, 2], [1, 2, 3], [2, 3], [3]],
        "min_value": [0, 1, 2, 3],
        "max_value": [2, 3, 3, 3],
        "sum_value": [3, 6, 5, 3],
    }


def test_rolling_dynamic_sortedness_check() -> None:
    # when the by argument is passed, the sortedness flag
    # will be unset as the take shuffles data, so we must explicitly
    # check the sortedness
    df = pl.DataFrame(
        {
            "idx": [1, 2, -1, 2, 1, 1],
            "group": [1, 1, 1, 2, 2, 1],
        }
    )

    with pytest.raises(ComputeError, match=r"input data is not sorted"):
        df.group_by_dynamic("idx", every="2i", group_by="group").agg(
            pl.col("idx").alias("idx1")
        )

    # no `by` argument
    with pytest.raises(
        InvalidOperationError,
        match=r"argument in operation 'group_by_dynamic' is not sorted",
    ):
        df.group_by_dynamic("idx", every="2i").agg(pl.col("idx").alias("idx1"))


@pytest.mark.parametrize("time_zone", [None, "UTC", "Asia/Kathmandu"])
def test_group_by_dynamic_elementwise_following_mean_agg_6904(
    time_zone: str | None,
) -> None:
    df = (
        pl.DataFrame(
            {
                "a": [datetime(2021, 1, 1) + timedelta(seconds=2**i) for i in range(5)],
                "b": [float(i) for i in range(5)],
            }
        )
        .with_columns(pl.col("a").dt.replace_time_zone(time_zone))
        .lazy()
        .set_sorted("a")
        .group_by_dynamic("a", every="10s", period="100s")
        .agg([pl.col("b").mean().sin().alias("c")])
        .collect()
    )
    assert_frame_equal(
        df,
        pl.DataFrame(
            {
                "a": [
                    datetime(2021, 1, 1, 0, 0),
                    datetime(2021, 1, 1, 0, 0, 10),
                ],
                "c": [0.9092974268256817, -0.7568024953079282],
            }
        ).with_columns(pl.col("a").dt.replace_time_zone(time_zone)),
    )


@pytest.mark.parametrize("every", ["1h", timedelta(hours=1)])
@pytest.mark.parametrize("tzinfo", [None, ZoneInfo("UTC"), ZoneInfo("Asia/Kathmandu")])
def test_group_by_dynamic_lazy(every: str | timedelta, tzinfo: ZoneInfo | None) -> None:
    ldf = pl.LazyFrame(
        {
            "time": pl.datetime_range(
                start=datetime(2021, 12, 16, tzinfo=tzinfo),
                end=datetime(2021, 12, 16, 2, tzinfo=tzinfo),
                interval="30m",
                eager=True,
            ),
            "n": range(5),
        }
    )
    df = (
        ldf.group_by_dynamic("time", every=every, closed="right")
        .agg(
            [
                pl.col("time").min().alias("time_min"),
                pl.col("time").max().alias("time_max"),
            ]
        )
        .collect()
    )
    assert sorted(df.rows()) == [
        (
            datetime(2021, 12, 15, 23, 0, tzinfo=tzinfo),
            datetime(2021, 12, 16, 0, 0, tzinfo=tzinfo),
            datetime(2021, 12, 16, 0, 0, tzinfo=tzinfo),
        ),
        (
            datetime(2021, 12, 16, 0, 0, tzinfo=tzinfo),
            datetime(2021, 12, 16, 0, 30, tzinfo=tzinfo),
            datetime(2021, 12, 16, 1, 0, tzinfo=tzinfo),
        ),
        (
            datetime(2021, 12, 16, 1, 0, tzinfo=tzinfo),
            datetime(2021, 12, 16, 1, 30, tzinfo=tzinfo),
            datetime(2021, 12, 16, 2, 0, tzinfo=tzinfo),
        ),
    ]


def test_group_by_dynamic_validation() -> None:
    df = pl.DataFrame(
        {
            "index": [0, 0, 1, 1],
            "group": ["banana", "pear", "banana", "pear"],
            "weight": [2, 3, 5, 7],
        }
    )

    with pytest.raises(ComputeError, match="'every' argument must be positive"):
        df.group_by_dynamic("index", group_by="group", every="-1i", period="2i").agg(
            pl.col("weight")
        )


def test_no_sorted_no_error() -> None:
    df = pl.DataFrame(
        {
            "dt": [datetime(2001, 1, 1), datetime(2001, 1, 2)],
        }
    )
    result = df.group_by_dynamic("dt", every="1h").agg(pl.len().alias("count"))
    expected = pl.DataFrame(
        {
            "dt": [datetime(2001, 1, 1), datetime(2001, 1, 2)],
            "count": [1, 1],
        },
        schema_overrides={"count": pl.get_index_type()},
    )
    assert_frame_equal(result, expected)


@pytest.mark.parametrize("tzinfo", [None, ZoneInfo("UTC"), ZoneInfo("Asia/Kathmandu")])
def test_truncate_negative_offset(tzinfo: ZoneInfo | None) -> None:
    time_zone = tzinfo.key if tzinfo is not None else None
    df = pl.DataFrame(
        {
            "event_date": [
                datetime(2021, 4, 11),
                datetime(2021, 4, 29),
                datetime(2021, 5, 29),
            ],
            "adm1_code": [1, 2, 1],
        }
    ).set_sorted("event_date")
    df = df.with_columns(pl.col("event_date").dt.replace_time_zone(time_zone))
    out = df.group_by_dynamic(
        index_column="event_date",
        every="1mo",
        period="2mo",
        offset="-1mo",
        include_boundaries=True,
    ).agg(
        [
            pl.col("adm1_code"),
        ]
    )

    assert out["event_date"].to_list() == [
        datetime(2021, 3, 1, tzinfo=tzinfo),
        datetime(2021, 4, 1, tzinfo=tzinfo),
        datetime(2021, 5, 1, tzinfo=tzinfo),
    ]
    df = pl.DataFrame(
        {
            "event_date": [
                datetime(2021, 4, 11),
                datetime(2021, 4, 29),
                datetime(2021, 5, 29),
            ],
            "adm1_code": [1, 2, 1],
            "five_type": ["a", "b", "a"],
            "actor": ["a", "a", "a"],
            "admin": ["a", "a", "a"],
            "fatalities": [10, 20, 30],
        }
    ).set_sorted("event_date")
    df = df.with_columns(pl.col("event_date").dt.replace_time_zone(time_zone))

    out = df.group_by_dynamic(
        index_column="event_date",
        every="1mo",
        group_by=["admin", "five_type", "actor"],
    ).agg([pl.col("adm1_code").unique(), (pl.col("fatalities") > 0).sum()])

    assert out["event_date"].to_list() == [
        datetime(2021, 4, 1, tzinfo=tzinfo),
        datetime(2021, 5, 1, tzinfo=tzinfo),
        datetime(2021, 4, 1, tzinfo=tzinfo),
    ]

    for dt in [pl.Int32, pl.Int64]:
        df = (
            pl.DataFrame(
                {
                    "idx": np.arange(6),
                    "A": ["A", "A", "B", "B", "B", "C"],
                }
            )
            .with_columns(pl.col("idx").cast(dt))
            .set_sorted("idx")
        )

        out = df.group_by_dynamic(
            "idx", every="2i", period="3i", include_boundaries=True
        ).agg(pl.col("A"))

        assert out.shape == (3, 4)
        assert out["A"].to_list() == [
            ["A", "A", "B"],
            ["B", "B", "B"],
            ["B", "C"],
        ]


def test_groupy_by_dynamic_median_10695() -> None:
    df = pl.DataFrame(
        {
            "timestamp": pl.datetime_range(
                datetime(2023, 8, 22, 15, 44, 30),
                datetime(2023, 8, 22, 15, 48, 50),
                "20s",
                eager=True,
            ),
            "foo": [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        }
    )

    assert df.group_by_dynamic(
        index_column="timestamp",
        every="60s",
        period="3m",
    ).agg(pl.col("foo").median()).to_dict(as_series=False) == {
        "timestamp": [
            datetime(2023, 8, 22, 15, 44),
            datetime(2023, 8, 22, 15, 45),
            datetime(2023, 8, 22, 15, 46),
            datetime(2023, 8, 22, 15, 47),
            datetime(2023, 8, 22, 15, 48),
        ],
        "foo": [1.0, 1.0, 1.0, 1.0, 1.0],
    }


def test_group_by_dynamic_when_conversion_crosses_dates_7274() -> None:
    df = (
        pl.DataFrame(
            data={
                "timestamp": ["1970-01-01 00:00:00+01:00", "1970-01-01 01:00:00+01:00"],
                "value": [1, 1],
            }
        )
        .with_columns(
            pl.col("timestamp")
            .str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S%:z")
            .dt.convert_time_zone("Africa/Lagos")
            .set_sorted()
        )
        .with_columns(
            pl.col("timestamp")
            .dt.convert_time_zone("UTC")
            .alias("timestamp_utc")
            .set_sorted()
        )
    )
    result = df.group_by_dynamic(
        index_column="timestamp", every="1d", closed="left"
    ).agg(pl.col("value").count())
    expected = pl.DataFrame({"timestamp": [datetime(1970, 1, 1)], "value": [2]})
    expected = expected.with_columns(
        pl.col("timestamp").dt.replace_time_zone("Africa/Lagos"),
        pl.col("value").cast(pl.UInt32),
    )
    assert_frame_equal(result, expected)
    result = df.group_by_dynamic(
        index_column="timestamp_utc", every="1d", closed="left"
    ).agg(pl.col("value").count())
    expected = pl.DataFrame(
        {
            "timestamp_utc": [datetime(1969, 12, 31), datetime(1970, 1, 1)],
            "value": [1, 1],
        }
    )
    expected = expected.with_columns(
        pl.col("timestamp_utc").dt.replace_time_zone("UTC"),
        pl.col("value").cast(pl.UInt32),
    )
    assert_frame_equal(result, expected)


@pytest.mark.parametrize("time_zone", [None, "UTC", "Asia/Kathmandu"])
def test_default_negative_every_offset_dynamic_group_by(time_zone: str | None) -> None:
    # 2791
    dts = [
        datetime(2020, 1, 1),
        datetime(2020, 1, 2),
        datetime(2020, 2, 1),
        datetime(2020, 3, 1),
    ]
    df = pl.DataFrame({"dt": dts, "idx": range(len(dts))}).set_sorted("dt")
    df = df.with_columns(pl.col("dt").dt.replace_time_zone(time_zone))
    out = df.group_by_dynamic(index_column="dt", every="1mo", closed="right").agg(
        pl.col("idx")
    )

    expected = pl.DataFrame(
        {
            "dt": [
                datetime(2019, 12, 1, 0, 0),
                datetime(2020, 1, 1, 0, 0),
                datetime(2020, 2, 1, 0, 0),
            ],
            "idx": [[0], [1, 2], [3]],
        }
    )
    expected = expected.with_columns(pl.col("dt").dt.replace_time_zone(time_zone))
    assert_frame_equal(out, expected)


@pytest.mark.parametrize(
    ("rule", "offset"),
    [
        ("1h", timedelta(hours=2)),
        ("1d", timedelta(days=2)),
        ("1w", timedelta(weeks=2)),
    ],
)
def test_group_by_dynamic_crossing_dst(rule: str, offset: timedelta) -> None:
    start_dt = datetime(2021, 11, 7)
    end_dt = start_dt + offset
    date_range = pl.datetime_range(
        start_dt, end_dt, rule, time_zone="US/Central", eager=True
    )
    df = pl.DataFrame({"time": date_range, "value": range(len(date_range))})
    result = df.group_by_dynamic("time", every=rule, start_by="datapoint").agg(
        pl.col("value").mean()
    )
    expected = pl.DataFrame(
        {"time": date_range, "value": range(len(date_range))},
        schema_overrides={"value": pl.Float64},
    )
    assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ("start_by", "expected_time", "expected_value"),
    [
        (
            "monday",
            [
                datetime(2021, 11, 1),
                datetime(2021, 11, 8),
            ],
            [0.0, 4.0],
        ),
        (
            "tuesday",
            [
                datetime(2021, 11, 2),
                datetime(2021, 11, 9),
            ],
            [0.5, 4.5],
        ),
        (
            "wednesday",
            [
                datetime(2021, 11, 3),
                datetime(2021, 11, 10),
            ],
            [1.0, 5.0],
        ),
        (
            "thursday",
            [
                datetime(2021, 11, 4),
                datetime(2021, 11, 11),
            ],
            [1.5, 5.5],
        ),
        (
            "friday",
            [
                datetime(2021, 11, 5),
                datetime(2021, 11, 12),
            ],
            [2.0, 6.0],
        ),
        (
            "saturday",
            [
                datetime(2021, 11, 6),
                datetime(2021, 11, 13),
            ],
            [2.5, 6.5],
        ),
        (
            "sunday",
            [
                datetime(2021, 11, 7),
                datetime(2021, 11, 14),
            ],
            [3.0, 7.0],
        ),
    ],
)
def test_group_by_dynamic_startby_monday_crossing_dst(
    start_by: StartBy, expected_time: list[datetime], expected_value: list[float]
) -> None:
    start_dt = datetime(2021, 11, 7)
    end_dt = datetime(2021, 11, 14)
    date_range = pl.datetime_range(
        start_dt, end_dt, "1d", time_zone="US/Central", eager=True
    )
    df = pl.DataFrame({"time": date_range, "value": range(len(date_range))})
    result = df.group_by_dynamic("time", every="1w", start_by=start_by).agg(
        pl.col("value").mean()
    )
    expected = pl.DataFrame(
        {"time": expected_time, "value": expected_value},
    )
    expected = expected.with_columns(pl.col("time").dt.replace_time_zone("US/Central"))
    assert_frame_equal(result, expected)


def test_group_by_dynamic_startby_monday_dst_8737() -> None:
    start_dt = datetime(2021, 11, 6, 20)
    stop_dt = datetime(2021, 11, 7, 20)
    date_range = pl.datetime_range(
        start_dt, stop_dt, "1d", time_zone="US/Central", eager=True
    )
    df = pl.DataFrame({"time": date_range, "value": range(len(date_range))})
    result = df.group_by_dynamic("time", every="1w", start_by="monday").agg(
        pl.col("value").mean()
    )
    expected = pl.DataFrame(
        {
            "time": [
                datetime(2021, 11, 1),
            ],
            "value": [0.5],
        },
    )
    expected = expected.with_columns(pl.col("time").dt.replace_time_zone("US/Central"))
    assert_frame_equal(result, expected)


def test_group_by_dynamic_monthly_crossing_dst() -> None:
    start_dt = datetime(2021, 11, 1)
    end_dt = datetime(2021, 12, 1)
    date_range = pl.datetime_range(
        start_dt, end_dt, "1mo", time_zone="US/Central", eager=True
    )
    df = pl.DataFrame({"time": date_range, "value": range(len(date_range))})
    result = df.group_by_dynamic("time", every="1mo").agg(pl.col("value").mean())
    expected = pl.DataFrame(
        {"time": date_range, "value": range(len(date_range))},
        schema_overrides={"value": pl.Float64},
    )
    assert_frame_equal(result, expected)


def test_group_by_dynamic_2d_9333() -> None:
    df = pl.DataFrame({"ts": [datetime(2000, 1, 1, 3)], "values": [10.0]})
    df = df.with_columns(pl.col("ts").set_sorted())
    result = df.group_by_dynamic("ts", every="2d").agg(pl.col("values"))
    expected = pl.DataFrame({"ts": [datetime(1999, 12, 31, 0)], "values": [[10.0]]})
    assert_frame_equal(result, expected)


@pytest.mark.parametrize("every", ["1h", timedelta(hours=1)])
@pytest.mark.parametrize("tzinfo", [None, ZoneInfo("UTC"), ZoneInfo("Asia/Kathmandu")])
def test_group_by_dynamic_iter(every: str | timedelta, tzinfo: ZoneInfo | None) -> None:
    time_zone = tzinfo.key if tzinfo is not None else None
    df = pl.DataFrame(
        {
            "datetime": [
                datetime(2020, 1, 1, 10, 0),
                datetime(2020, 1, 1, 10, 50),
                datetime(2020, 1, 1, 11, 10),
            ],
            "a": [1, 2, 2],
            "b": [4, 5, 6],
        }
    ).set_sorted("datetime")
    df = df.with_columns(pl.col("datetime").dt.replace_time_zone(time_zone))

    # Without 'by' argument
    result1 = [
        (name, data.shape)
        for name, data in df.group_by_dynamic("datetime", every=every, closed="left")
    ]
    expected1 = [
        ((datetime(2020, 1, 1, 10, tzinfo=tzinfo),), (2, 3)),
        ((datetime(2020, 1, 1, 11, tzinfo=tzinfo),), (1, 3)),
    ]
    assert result1 == expected1

    # With 'by' argument
    result2 = [
        (name, data.shape)
        for name, data in df.group_by_dynamic(
            "datetime", every=every, closed="left", group_by="a"
        )
    ]
    expected2 = [
        ((1, datetime(2020, 1, 1, 10, tzinfo=tzinfo)), (1, 3)),
        ((2, datetime(2020, 1, 1, 10, tzinfo=tzinfo)), (1, 3)),
        ((2, datetime(2020, 1, 1, 11, tzinfo=tzinfo)), (1, 3)),
    ]
    assert result2 == expected2


# https://github.com/pola-rs/polars/issues/11339
@pytest.mark.parametrize("include_boundaries", [True, False])
def test_group_by_dynamic_lazy_schema(include_boundaries: bool) -> None:
    lf = pl.LazyFrame(
        {
            "dt": pl.datetime_range(
                start=datetime(2022, 2, 10),
                end=datetime(2022, 2, 12),
                eager=True,
            ),
            "n": range(3),
        }
    )

    result = lf.group_by_dynamic(
        "dt", every="2d", closed="right", include_boundaries=include_boundaries
    ).agg(pl.col("dt").min().alias("dt_min"))

    assert result.collect_schema() == result.collect().schema


def test_group_by_dynamic_12414() -> None:
    df = pl.DataFrame(
        {
            "today": [
                date(2023, 3, 3),
                date(2023, 8, 31),
                date(2023, 9, 1),
                date(2023, 9, 4),
            ],
            "b": [1, 2, 3, 4],
        }
    ).sort("today")
    assert df.group_by_dynamic(
        "today",
        every="6mo",
        period="3d",
        closed="left",
        start_by="datapoint",
        include_boundaries=True,
    ).agg(
        gt_min_count=(pl.col.b >= (pl.col.b.min())).sum(),
    ).to_dict(as_series=False) == {
        "_lower_boundary": [datetime(2023, 3, 3, 0, 0), datetime(2023, 9, 3, 0, 0)],
        "_upper_boundary": [datetime(2023, 3, 6, 0, 0), datetime(2023, 9, 6, 0, 0)],
        "today": [date(2023, 3, 3), date(2023, 9, 3)],
        "gt_min_count": [1, 1],
    }


@pytest.mark.parametrize("input", [[pl.col("b").sum()], pl.col("b").sum()])
def test_group_by_dynamic_agg_input_types(input: Any) -> None:
    df = pl.LazyFrame({"index_column": [0, 1, 2, 3], "b": [1, 3, 1, 2]}).set_sorted(
        "index_column"
    )
    result = df.group_by_dynamic(
        index_column="index_column", every="2i", closed="right"
    ).agg(input)

    expected = pl.LazyFrame({"index_column": [-2, 0, 2], "b": [1, 4, 2]})
    assert_frame_equal(result, expected)


@pytest.mark.parametrize("input", [str, "b".join])
def test_group_by_dynamic_agg_bad_input_types(input: Any) -> None:
    df = pl.LazyFrame({"index_column": [0, 1, 2, 3], "b": [1, 3, 1, 2]}).set_sorted(
        "index_column"
    )
    with pytest.raises(TypeError):
        df.group_by_dynamic(
            index_column="index_column", every="2i", closed="right"
        ).agg(input)


def test_group_by_dynamic_15225() -> None:
    df = pl.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3)],
            "c": [1, 1, 2],
        }
    )
    result = df.group_by_dynamic("b", every="2d").agg(pl.sum("a"))
    expected = pl.DataFrame({"b": [date(2020, 1, 1), date(2020, 1, 3)], "a": [3, 3]})
    assert_frame_equal(result, expected)
    result = df.group_by_dynamic("b", every="2d", group_by="c").agg(pl.sum("a"))
    expected = pl.DataFrame(
        {"c": [1, 2], "b": [date(2020, 1, 1), date(2020, 1, 3)], "a": [3, 3]}
    )
    assert_frame_equal(result, expected)


@pytest.mark.parametrize("start_by", ["window", "friday"])
def test_earliest_point_included_when_offset_is_set_15241(start_by: StartBy) -> None:
    df = pl.DataFrame(
        data={
            "t": pl.Series(
                [
                    datetime(2024, 3, 22, 3, 0, tzinfo=timezone.utc),
                    datetime(2024, 3, 22, 4, 0, tzinfo=timezone.utc),
                    datetime(2024, 3, 22, 5, 0, tzinfo=timezone.utc),
                    datetime(2024, 3, 22, 6, 0, tzinfo=timezone.utc),
                ]
            ),
            "v": [1, 10, 100, 1000],
        }
    ).set_sorted("t")
    result = df.group_by_dynamic(
        index_column="t",
        every="1d",
        offset=timedelta(hours=5),
        start_by=start_by,
    ).agg("v")
    expected = pl.DataFrame(
        {
            "t": [
                datetime(2024, 3, 21, 5, 0, tzinfo=timezone.utc),
                datetime(2024, 3, 22, 5, 0, tzinfo=timezone.utc),
            ],
            "v": [[1, 10], [100, 1000]],
        }
    )
    assert_frame_equal(result, expected)


def test_group_by_dynamic_invalid() -> None:
    df = pl.DataFrame(
        {
            "values": [1, 4],
            "times": [datetime(2020, 1, 3), datetime(2020, 1, 1)],
        },
    )
    with pytest.raises(
        InvalidOperationError, match="duration may not be a parsed integer"
    ):
        (
            df.sort("times")
            .group_by_dynamic("times", every="3000i")
            .agg(pl.col("values").sum().alias("sum"))
        )
    with pytest.raises(
        InvalidOperationError, match="duration must be a parsed integer"
    ):
        (
            df.with_row_index()
            .group_by_dynamic("index", every="3000d")
            .agg(pl.col("values").sum().alias("sum"))
        )


def test_group_by_dynamic_get() -> None:
    df = pl.DataFrame(
        {
            "time": pl.date_range(pl.date(2021, 1, 1), pl.date(2021, 1, 8), eager=True),
            "data": pl.arange(8, eager=True),
        }
    )

    assert df.group_by_dynamic(
        index_column="time",
        every="2d",
        period="3d",
        start_by="datapoint",
    ).agg(
        get=pl.col("data").get(1),
    ).to_dict(as_series=False) == {
        "time": [
            date(2021, 1, 1),
            date(2021, 1, 3),
            date(2021, 1, 5),
            date(2021, 1, 7),
        ],
        "get": [1, 3, 5, 7],
    }


def test_group_by_dynamic_exclude_index_from_expansion_17075() -> None:
    lf = pl.LazyFrame(
        {
            "time": pl.datetime_range(
                start=datetime(2021, 12, 16),
                end=datetime(2021, 12, 16, 3),
                interval="30m",
                eager=True,
            ),
            "n": range(7),
            "m": range(7),
        }
    )

    assert lf.group_by_dynamic(
        "time", every="1h", closed="right"
    ).last().collect().to_dict(as_series=False) == {
        "time": [
            datetime(2021, 12, 15, 23, 0),
            datetime(2021, 12, 16, 0, 0),
            datetime(2021, 12, 16, 1, 0),
            datetime(2021, 12, 16, 2, 0),
        ],
        "n": [0, 2, 4, 6],
        "m": [0, 2, 4, 6],
    }
