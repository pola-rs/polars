from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

import polars as pl
from polars.exceptions import ComputeError, InvalidOperationError
from polars.testing import assert_frame_equal, assert_series_equal

if TYPE_CHECKING:
    from polars._typing import ClosedInterval, PolarsIntegerType
    from tests.conftest import PlMonkeyPatch


def test_rolling() -> None:
    dates = [
        "2020-01-01 13:45:48",
        "2020-01-01 16:42:13",
        "2020-01-01 16:45:09",
        "2020-01-02 18:12:48",
        "2020-01-03 19:45:32",
        "2020-01-08 23:16:43",
    ]

    df = (
        pl.DataFrame({"dt": dates, "a": [3, 7, 5, 9, 2, 1]})
        .with_columns(pl.col("dt").str.strptime(pl.Datetime))
        .set_sorted("dt")
    )

    period: str | timedelta
    for period in ("2d", timedelta(days=2)):
        out = df.rolling(index_column="dt", period=period).agg(
            [
                pl.sum("a").alias("sum_a"),
                pl.min("a").alias("min_a"),
                pl.max("a").alias("max_a"),
            ]
        )
        assert out["sum_a"].to_list() == [3, 10, 15, 24, 11, 1]
        assert out["max_a"].to_list() == [3, 7, 7, 9, 9, 1]
        assert out["min_a"].to_list() == [3, 3, 3, 3, 2, 1]


@pytest.mark.parametrize("dtype", [pl.UInt32, pl.UInt64, pl.Int32, pl.Int64])
def test_rolling_group_by_overlapping_groups(dtype: PolarsIntegerType) -> None:
    # this first aggregates overlapping groups so they cannot be naively flattened
    df = pl.DataFrame({"a": [41, 60, 37, 51, 52, 39, 40]})

    assert_series_equal(
        (
            df.with_row_index()
            .with_columns(pl.col("index").cast(dtype))
            .rolling(index_column="index", period="5i")
            .agg(
                # trigger the apply on the expression engine
                pl.col("a")
                .map_elements(lambda x: x, return_dtype=pl.self_dtype())
                .sum()
            )
        )["a"],
        df["a"].rolling_sum(window_size=5, min_samples=1),
    )


# TODO: This test requires the environment variable to be set prior to starting
# the thread pool, which implies prior to import. The test is only valid when
# run in isolation, and invalid otherwise because of xdist import caching.
# See GH issue #22070
def test_rolling_group_by_overlapping_groups_21859_a(
    plmonkeypatch: PlMonkeyPatch,
) -> None:
    plmonkeypatch.setenv("POLARS_MAX_THREADS", "1")
    # assert pl.thread_pool_size() == 1 # pending resolution, see TODO
    df = pl.select(
        pl.date_range(pl.date(2023, 1, 1), pl.date(2023, 1, 5))
    ).with_row_index()

    out = df.rolling(index_column="date", period="1y").agg(
        a1=pl.when(pl.col("date") >= pl.col("date"))
        .then(pl.col("index").cast(pl.Int64).cum_sum())
        .last(),
        a2=pl.when(pl.col("date") >= pl.col("date"))
        .then(pl.col("index").cast(pl.Int64).cum_sum())
        .last(),
    )["a1", "a2"]
    expected = pl.DataFrame({"a1": [0, 1, 3, 6, 10], "a2": [0, 1, 3, 6, 10]})
    assert_frame_equal(out, expected)


# TODO: This test requires the environment variable to be set prior to starting
# the thread pool, which implies prior to import. The test is only valid when
# run in isolation, and invalid otherwise because of xdist import caching.
# See GH issue #22070
def test_rolling_group_by_overlapping_groups_21859_b(
    plmonkeypatch: PlMonkeyPatch,
) -> None:
    plmonkeypatch.setenv("POLARS_MAX_THREADS", "1")
    # assert pl.thread_pool_size() == 1 # pending resolution, see TODO
    df = pl.DataFrame({"a": [20, 30, 40]})
    out = (
        df.with_row_index()
        .with_columns(pl.col("index"))
        .cast(pl.Int64)
        .rolling(index_column="index", period="3i")
        .agg(
            # trigger the apply on the expression engine
            pl.col("a")
            .map_elements(lambda x: x, return_dtype=pl.self_dtype())
            .sum()
            .alias("a1"),
            pl.col("a")
            .map_elements(lambda x: x, return_dtype=pl.self_dtype())
            .sum()
            .alias("a2"),
        )["a1", "a2"]
    )
    expected = pl.DataFrame({"a1": [20, 50, 90], "a2": [20, 50, 90]})
    assert_frame_equal(out, expected)


@pytest.mark.parametrize("input", [[pl.col("b").sum()], pl.col("b").sum()])
@pytest.mark.parametrize("dtype", [pl.UInt32, pl.UInt64, pl.Int32, pl.Int64])
def test_rolling_agg_input_types(input: Any, dtype: PolarsIntegerType) -> None:
    df = pl.LazyFrame(
        {"index_column": [0, 1, 2, 3], "b": [1, 3, 1, 2]},
        schema_overrides={"index_column": dtype},
    ).set_sorted("index_column")
    result = df.rolling(index_column="index_column", period="2i").agg(input)
    expected = pl.LazyFrame(
        {"index_column": [0, 1, 2, 3], "b": [1, 4, 4, 3]},
        schema_overrides={"index_column": dtype},
    )
    assert_frame_equal(result, expected)


@pytest.mark.parametrize("input", [str, "b".join])
def test_rolling_agg_bad_input_types(input: Any) -> None:
    df = pl.LazyFrame({"index_column": [0, 1, 2, 3], "b": [1, 3, 1, 2]}).set_sorted(
        "index_column"
    )
    with pytest.raises(TypeError):
        df.rolling(index_column="index_column", period="2i").agg(input)


def test_rolling_negative_offset_3914() -> None:
    df = pl.DataFrame(
        {
            "datetime": pl.datetime_range(
                datetime(2020, 1, 1), datetime(2020, 1, 5), "1d", eager=True
            ),
        }
    )
    result = df.rolling(index_column="datetime", period="2d", offset="-4d").agg(
        pl.len()
    )
    assert result["len"].to_list() == [0, 0, 1, 2, 2]

    df = pl.DataFrame({"ints": range(20)})

    result = df.rolling(index_column="ints", period="2i", offset="-5i").agg(
        pl.col("ints").alias("matches")
    )
    expected = [
        [],
        [],
        [],
        [0],
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 8],
        [8, 9],
        [9, 10],
        [10, 11],
        [11, 12],
        [12, 13],
        [13, 14],
        [14, 15],
        [15, 16],
    ]
    assert result["matches"].to_list() == expected


@pytest.mark.parametrize("time_zone", [None, "America/Chicago"])
def test_rolling_negative_offset_crossing_dst(time_zone: str | None) -> None:
    df = pl.DataFrame(
        {
            "datetime": pl.datetime_range(
                datetime(2021, 11, 6),
                datetime(2021, 11, 9),
                "1d",
                time_zone=time_zone,
                eager=True,
            ),
            "value": [1, 4, 9, 155],
        }
    )
    result = df.rolling(index_column="datetime", period="2d", offset="-1d").agg(
        pl.col("value")
    )
    expected = pl.DataFrame(
        {
            "datetime": pl.datetime_range(
                datetime(2021, 11, 6),
                datetime(2021, 11, 9),
                "1d",
                time_zone=time_zone,
                eager=True,
            ),
            "value": [[1, 4], [4, 9], [9, 155], [155]],
        }
    )
    assert_frame_equal(result, expected)


@pytest.mark.parametrize("time_zone", [None, "America/Chicago"])
@pytest.mark.parametrize(
    ("offset", "closed", "expected_values"),
    [
        ("0d", "left", [[1, 4], [4, 9], [9, 155], [155]]),
        ("0d", "right", [[4, 9], [9, 155], [155], []]),
        ("0d", "both", [[1, 4, 9], [4, 9, 155], [9, 155], [155]]),
        ("0d", "none", [[4], [9], [155], []]),
        ("1d", "left", [[4, 9], [9, 155], [155], []]),
        ("1d", "right", [[9, 155], [155], [], []]),
        ("1d", "both", [[4, 9, 155], [9, 155], [155], []]),
        ("1d", "none", [[9], [155], [], []]),
    ],
)
def test_rolling_non_negative_offset_9077(
    time_zone: str | None,
    offset: str,
    closed: ClosedInterval,
    expected_values: list[list[int]],
) -> None:
    df = pl.DataFrame(
        {
            "datetime": pl.datetime_range(
                datetime(2021, 11, 6),
                datetime(2021, 11, 9),
                "1d",
                time_zone=time_zone,
                eager=True,
            ),
            "value": [1, 4, 9, 155],
        }
    )
    result = df.rolling(
        index_column="datetime", period="2d", offset=offset, closed=closed
    ).agg(pl.col("value"))
    expected = pl.DataFrame(
        {
            "datetime": pl.datetime_range(
                datetime(2021, 11, 6),
                datetime(2021, 11, 9),
                "1d",
                time_zone=time_zone,
                eager=True,
            ),
            "value": expected_values,
        }
    )
    assert_frame_equal(result, expected)


@pytest.mark.may_fail_auto_streaming
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
        df.rolling("idx", period="2i", group_by="group").agg(
            pl.col("idx").alias("idx1")
        )

    # no `group_by` argument
    with pytest.raises(
        InvalidOperationError,
        match="argument in operation 'rolling' is not sorted",
    ):
        df.rolling("idx", period="2i").agg(pl.col("idx").alias("idx1"))


def test_rolling_empty_groups_9973() -> None:
    dt1 = date(2001, 1, 1)
    dt2 = date(2001, 1, 2)

    data = pl.DataFrame(
        {
            "id": ["A", "A", "B", "B", "C", "C"],
            "date": [dt1, dt2, dt1, dt2, dt1, dt2],
            "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    ).sort(by=["id", "date"])

    expected = pl.DataFrame(
        {
            "id": ["A", "A", "B", "B", "C", "C"],
            "date": [
                date(2001, 1, 1),
                date(2001, 1, 2),
                date(2001, 1, 1),
                date(2001, 1, 2),
                date(2001, 1, 1),
                date(2001, 1, 2),
            ],
            "value": [[2.0], [], [4.0], [], [6.0], []],
        }
    )

    out = data.rolling(
        index_column="date",
        group_by="id",
        period="2d",
        offset="1d",
        closed="left",
    ).agg(pl.col("value"))

    assert_frame_equal(out, expected)


def test_rolling_duplicates_11281() -> None:
    df = pl.DataFrame(
        {
            "ts": [
                datetime(2020, 1, 1),
                datetime(2020, 1, 2),
                datetime(2020, 1, 2),
                datetime(2020, 1, 2),
                datetime(2020, 1, 3),
                datetime(2020, 1, 4),
            ],
            "val": [1, 2, 2, 2, 3, 4],
        }
    ).sort("ts")
    result = df.rolling("ts", period="1d", closed="left").agg(pl.col("val"))
    expected = df.with_columns(val=pl.Series([[], [1], [1], [1], [2, 2, 2], [3]]))
    assert_frame_equal(result, expected)


def test_rolling_15225() -> None:
    # https://github.com/pola-rs/polars/issues/15225

    df = pl.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3)],
            "c": [1, 1, 2],
        }
    )
    result = df.rolling("b", period="2d").agg(pl.sum("a"))
    expected = pl.DataFrame(
        {"b": [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3)], "a": [1, 3, 5]}
    )
    assert_frame_equal(result, expected)
    result = df.rolling("b", period="2d", group_by="c").agg(pl.sum("a"))
    expected = pl.DataFrame(
        {
            "c": [1, 1, 2],
            "b": [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3)],
            "a": [1, 3, 3],
        }
    )
    assert_frame_equal(result, expected)


def test_multiple_rolling_in_single_expression() -> None:
    df = pl.DataFrame(
        {
            "timestamp": pl.datetime_range(
                datetime(2024, 1, 12),
                datetime(2024, 1, 12, 0, 0, 0, 150_000),
                "10ms",
                eager=True,
                closed="left",
            ),
            "price": [0] * 15,
        }
    )

    front_count = (
        pl.col("price")
        .count()
        .rolling("timestamp", period=timedelta(milliseconds=100))
        .cast(pl.Int64)
    )
    back_count = (
        pl.col("price")
        .count()
        .rolling("timestamp", period=timedelta(milliseconds=200))
        .cast(pl.Int64)
    )
    assert df.with_columns(
        back_count.alias("back"),
        front_count.alias("front"),
        (back_count - front_count).alias("back - front"),
    )["back - front"].to_list() == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5]


def test_negative_zero_offset_16168() -> None:
    df = pl.DataFrame({"foo": [1] * 3}).sort("foo").with_row_index()
    result = df.rolling(index_column="foo", period="1i", offset="0i").agg("index")
    expected = pl.DataFrame(
        {"foo": [1, 1, 1], "index": [[], [], []]},
        schema_overrides={"index": pl.List(pl.get_index_type())},
    )
    assert_frame_equal(result, expected)
    result = df.rolling(index_column="foo", period="1i", offset="-0i").agg("index")
    assert_frame_equal(result, expected)


def test_rolling_sorted_empty_groups_16145() -> None:
    df = pl.DataFrame(
        {
            "id": [1, 2],
            "time": [
                datetime(year=1989, month=12, day=1, hour=12, minute=3),
                datetime(year=1989, month=12, day=1, hour=13, minute=14),
            ],
        }
    )

    assert (
        df.sort("id")
        .rolling(
            index_column="time",
            group_by="id",
            period="1d",
            offset="0d",
            closed="right",
        )
        .agg()
        .select("id")
    )["id"].to_list() == [1, 2]


def test_rolling_by_() -> None:
    df = pl.DataFrame({"group": pl.arange(0, 3, eager=True)}).join(
        pl.DataFrame(
            {
                "datetime": pl.datetime_range(
                    datetime(2020, 1, 1), datetime(2020, 1, 5), "1d", eager=True
                ),
            }
        ),
        how="cross",
    )
    out = (
        df.sort("datetime")
        .rolling(index_column="datetime", group_by="group", period=timedelta(days=3))
        .agg([pl.len().alias("count")])
    )

    expected = (
        df.sort(["group", "datetime"])
        .rolling(index_column="datetime", group_by="group", period="3d")
        .agg([pl.len().alias("count")])
    )
    assert_frame_equal(out.sort(["group", "datetime"]), expected)
    assert out.to_dict(as_series=False) == {
        "group": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
        "datetime": [
            datetime(2020, 1, 1, 0, 0),
            datetime(2020, 1, 2, 0, 0),
            datetime(2020, 1, 3, 0, 0),
            datetime(2020, 1, 4, 0, 0),
            datetime(2020, 1, 5, 0, 0),
            datetime(2020, 1, 1, 0, 0),
            datetime(2020, 1, 2, 0, 0),
            datetime(2020, 1, 3, 0, 0),
            datetime(2020, 1, 4, 0, 0),
            datetime(2020, 1, 5, 0, 0),
            datetime(2020, 1, 1, 0, 0),
            datetime(2020, 1, 2, 0, 0),
            datetime(2020, 1, 3, 0, 0),
            datetime(2020, 1, 4, 0, 0),
            datetime(2020, 1, 5, 0, 0),
        ],
        "count": [1, 2, 3, 3, 3, 1, 2, 3, 3, 3, 1, 2, 3, 3, 3],
    }


def test_rolling_group_by_empty_groups_by_take_6330() -> None:
    df1 = pl.DataFrame({"Event": ["Rain", "Sun"]})
    df2 = pl.DataFrame({"Date": [1, 2, 3, 4]}).set_sorted("Date")
    df = df1.join(df2, how="cross")

    result = df.rolling(
        index_column="Date", period="2i", offset="-2i", group_by="Event", closed="left"
    ).agg(pl.len())

    assert_frame_equal(
        result,
        pl.DataFrame(
            {
                "Event": ["Sun", "Sun", "Sun", "Sun", "Rain", "Rain", "Rain", "Rain"],
                "Date": [1, 2, 3, 4, 1, 2, 3, 4],
                "len": [0, 1, 2, 2, 0, 1, 2, 2],
            },
            schema_overrides={"len": pl.get_index_type()},
        ),
        check_row_order=False,
    )


def test_rolling_duplicates() -> None:
    df = pl.DataFrame(
        {
            "ts": [datetime(2000, 1, 1, 0, 0), datetime(2000, 1, 1, 0, 0)],
            "value": [0, 1],
        }
    )
    assert df.sort("ts").with_columns(pl.col("value").rolling_max_by("ts", "1d"))[
        "value"
    ].to_list() == [1, 1]


def test_rolling_group_by_by_argument() -> None:
    df = pl.DataFrame({"times": range(10), "groups": [1] * 4 + [2] * 6})

    out = df.rolling("times", period="5i", group_by=["groups"]).agg(
        pl.col("times").alias("agg_list")
    )

    expected = pl.DataFrame(
        {
            "groups": [1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
            "times": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "agg_list": [
                [0],
                [0, 1],
                [0, 1, 2],
                [0, 1, 2, 3],
                [4],
                [4, 5],
                [4, 5, 6],
                [4, 5, 6, 7],
                [4, 5, 6, 7, 8],
                [5, 6, 7, 8, 9],
            ],
        }
    )

    assert_frame_equal(out, expected)


def test_rolling_by_ordering() -> None:
    # we must check that the keys still match the time labels after the rolling window
    # with a `by` argument.
    df = pl.DataFrame(
        {
            "dt": [
                datetime(2022, 1, 1, 0, 1),
                datetime(2022, 1, 1, 0, 2),
                datetime(2022, 1, 1, 0, 3),
                datetime(2022, 1, 1, 0, 4),
                datetime(2022, 1, 1, 0, 5),
                datetime(2022, 1, 1, 0, 6),
                datetime(2022, 1, 1, 0, 7),
            ],
            "key": ["A", "A", "B", "B", "A", "B", "A"],
            "val": [1, 1, 1, 1, 1, 1, 1],
        }
    ).set_sorted("dt")

    assert df.rolling(
        index_column="dt",
        period="2m",
        closed="both",
        offset="-1m",
        group_by="key",
    ).agg(
        [
            pl.col("val").sum().alias("sum val"),
        ]
    ).to_dict(as_series=False) == {
        "key": ["A", "A", "A", "A", "B", "B", "B"],
        "dt": [
            datetime(2022, 1, 1, 0, 1),
            datetime(2022, 1, 1, 0, 2),
            datetime(2022, 1, 1, 0, 5),
            datetime(2022, 1, 1, 0, 7),
            datetime(2022, 1, 1, 0, 3),
            datetime(2022, 1, 1, 0, 4),
            datetime(2022, 1, 1, 0, 6),
        ],
        "sum val": [2, 2, 1, 1, 2, 2, 1],
    }


def test_rolling_bool() -> None:
    dates = [
        "2020-01-01 13:45:48",
        "2020-01-01 16:42:13",
        "2020-01-01 16:45:09",
        "2020-01-02 18:12:48",
        "2020-01-03 19:45:32",
        "2020-01-08 23:16:43",
    ]

    df = (
        pl.DataFrame({"dt": dates, "a": [True, False, None, None, True, False]})
        .with_columns(pl.col("dt").str.strptime(pl.Datetime))
        .set_sorted("dt")
    )

    period: str | timedelta
    for period in ("2d", timedelta(days=2)):
        out = df.rolling(index_column="dt", period=period).agg(
            sum_a=pl.col.a.sum(),
            min_a=pl.col.a.min(),
            max_a=pl.col.a.max(),
            sum_a_ref=pl.col.a.cast(pl.Int32).sum(),
            min_a_ref=pl.col.a.cast(pl.Int32).min().cast(pl.Boolean),
            max_a_ref=pl.col.a.cast(pl.Int32).max().cast(pl.Boolean),
        )
        assert out["sum_a"].to_list() == out["sum_a_ref"].to_list()
        assert out["max_a"].to_list() == out["max_a_ref"].to_list()
        assert out["min_a"].to_list() == out["min_a_ref"].to_list()


def test_rolling_var_zero_weight() -> None:
    assert_series_equal(
        pl.Series([1.0, None, 1.0, 2.0]).rolling_var(2),
        pl.Series([None, None, None, 0.5]),
    )


def test_rolling_unsupported_22065() -> None:
    with pytest.raises(pl.exceptions.InvalidOperationError):
        pl.Series("a", [[]]).rolling_sum(10)
    with pytest.raises(pl.exceptions.InvalidOperationError):
        pl.Series("a", ["1.0"], pl.Decimal(10, 2)).rolling_min(1)
    with pytest.raises(pl.exceptions.InvalidOperationError):
        pl.Series("a", [None]).rolling_sum(10)
    with pytest.raises(pl.exceptions.InvalidOperationError):
        pl.Series("a", []).rolling_sum(10)
    with pytest.raises(pl.exceptions.InvalidOperationError):
        pl.Series("a", [[None]], pl.List(pl.Null)).rolling_sum(10)


def test_rolling_mean_f32_22936() -> None:
    arr = np.array(
        [
            4.17571609e-05,
            4.27760388e-05,
            5.72538265e-05,
            5.85808011e-05,
            5.80585256e-05,
            5.66820236e-05,
            5.63966605e-05,
            5.97858889e-05,
            5.84967784e-05,
            9.24392344e04,
            5.20393951e-05,
            5.19272326e-05,
            4.18911623e-05,
            4.23079109e-05,
            4.28866042e-05,
            4.07778753e-05,
            4.04103557e-05,
            4.25533253e-05,
            5.24330462e-05,
            6.08061091e-05,
            5.93549412e-05,
            5.76712700e-05,
            6.57564160e-05,
            6.62090970e-05,
            6.46697372e-05,
            6.40037397e-05,
            6.18191480e-05,
            6.33935779e-05,
            6.13316370e-05,
            5.91840580e-05,
            5.85238740e-05,
            5.38484855e-05,
            5.27409211e-05,
            5.15455504e-05,
            5.23890667e-05,
            5.40723668e-05,
            5.63136491e-05,
            5.61193119e-05,
            5.61807392e-05,
            5.93001459e-05,
            6.08127375e-05,
            6.04183369e-05,
            6.24700697e-05,
            6.20444407e-05,
            5.98985389e-05,
            6.08591145e-05,
            5.87234099e-05,
            5.92241740e-05,
            5.97595426e-05,
            5.95900237e-05,
            5.63832436e-05,
        ],
        dtype=np.float32,
    )

    expected = pl.Series(
        [
            6.1009144701529294e-05,
            6.1128826928325e-05,
            6.113809649832547e-05,
            6.079911327105947e-05,
            6.014993414282799e-05,
            5.9692956710932776e-05,
            5.9631252952385694e-05,
            5.873607733519748e-05,
            5.873924237675965e-05,
            5.857759970240295e-05,
        ],
        dtype=pl.Float32,
    )
    out = (
        pl.Series(arr).rolling_mean(window_size=5, min_samples=1, center=True).tail(10)
    )

    assert_series_equal(expected, out)


def test_rolling_max_23066() -> None:
    df = pl.DataFrame(
        {"data": [3.0, None, 14.0, 40.0, 5.0, 10.0, 0.0, 0.0, 30.0, None]}
    )
    result = df.select(pl.col.data.rolling_max(window_size=4, min_samples=4))
    assert_frame_equal(
        result,
        pl.DataFrame(
            {"data": [None, None, None, None, None, 40.0, 40.0, 10.0, 30.0, None]}
        ),
    )


def test_rolling_non_aggregation_24012() -> None:
    df = pl.DataFrame({"idx": [1, 2], "value": ["a", "b"]})
    q = df.lazy().select(pl.col("value").rolling("idx", period="2i"))

    assert q.collect_schema() == q.collect().schema


def test_rolling_on_expressions() -> None:
    df = pl.DataFrame({"a": [None, 1, 2, 3]}).with_row_index()

    df = df.select(
        df_ri=pl.col.a.mean().rolling("index", period="3i"),
        in_ri=pl.col.a.mean().rolling(pl.row_index(), period="3i"),
    )

    assert_series_equal(df["df_ri"], df["in_ri"], check_names=False)


def test_rolling_in_group_by() -> None:
    q = pl.LazyFrame({"b": [1, 1, 2], "a": [1, 2, 3]})

    a_sum = pl.col.a.sum()
    assert_frame_equal(
        q.select(a_sum.rolling(pl.row_index(), period="2i")).collect(),
        pl.Series("a", [1, 3, 5]).to_frame(),
    )
    assert_frame_equal(
        q.group_by("b")
        .agg(a_sum.rolling(pl.row_index(), period="2i") + pl.col.a.first())
        .collect(),
        pl.DataFrame(
            {
                "b": [1, 2],
                "a": [[2, 4], [6]],
            }
        ),
        check_row_order=False,
    )
    assert_frame_equal(
        q.select(pl.col.a.implode())
        .select(
            pl.col.a.list.eval(pl.element().sum().rolling(pl.row_index(), period="2i"))
        )
        .collect(),
        pl.Series("a", [[1, 3, 5]]).to_frame(),
    )
    assert_frame_equal(
        q.group_by(pl.lit(1))
        .agg(
            a_sum.rolling(pl.row_index(), period="2i").rolling(
                pl.row_index(), period="2i"
            )
        )
        .drop("literal")
        .collect(),
        pl.Series("a", [[[1], [1, 3], [2, 5]]]).to_frame(),
    )

    a_uniq = pl.col.a.unique()
    assert_frame_equal(
        q.select(a_uniq.rolling(pl.row_index(), period="2i")).collect(),
        pl.Series("a", [[1], [1, 2], [2, 3]]).to_frame(),
    )
    assert_frame_equal(
        q.group_by("b")
        .agg(a_uniq.rolling(pl.row_index(), period="2i") + pl.col.a.first())
        .collect(),
        pl.DataFrame(
            {
                "b": [1, 2],
                "a": [[[2], [2, 3]], [[6]]],
            }
        ),
        check_row_order=False,
    )
    assert_frame_equal(
        q.select(pl.col.a.implode())
        .select(
            pl.col.a.list.eval(
                pl.element().unique().rolling(pl.row_index(), period="2i")
            )
        )
        .collect(),
        pl.Series("a", [[[1], [1, 2], [2, 3]]]).to_frame(),
    )
    assert_frame_equal(
        q.group_by(pl.lit(1))
        .agg(
            a_uniq.rolling(pl.row_index(), period="2i").rolling(
                pl.row_index(), period="2i"
            )
        )
        .drop("literal")
        .collect(),
        pl.Series("a", [[[[1]], [[1], [1, 2]], [[2], [2, 3]]]]).to_frame(),
    )


def test_rolling_in_over_25280() -> None:
    dates = [
        "2020-01-01",
        "2020-01-02",
    ]

    df = pl.DataFrame(
        {"dt": dates, "train_line": ["a", "b"], "num_passengers": [3, 7]}
    ).with_columns(pl.col("dt").str.to_date())

    result = df.with_columns(
        pl.col("num_passengers")
        .sum()
        .rolling(index_column="dt", period="1d")
        .over("train_line")
    )
    assert_frame_equal(df, result)


def test_rolling_with_slice() -> None:
    lf = (
        pl.LazyFrame({"a": [0, 5, 2, 1, 3]})
        .with_row_index()
        .rolling("index", period="2i")
        .agg(pl.col.a.sum())
    )

    expected = pl.DataFrame(
        [
            pl.Series("index", [0, 1, 2, 3, 4], pl.get_index_type()),
            pl.Series("a", [0, 5, 7, 3, 4]),
        ]
    )

    assert_frame_equal(lf.head(2).collect(), expected.head(2))
    assert_frame_equal(lf.slice(1, 3).collect(), expected.slice(1, 3))
    assert_frame_equal(lf.tail(2).collect(), expected.tail(2))
    assert_frame_equal(lf.slice(5, 1).collect(), expected.slice(5, 1))
    assert_frame_equal(lf.slice(5, 0).collect(), expected.slice(5, 0))
    assert_frame_equal(lf.slice(2, 1).collect(), expected.slice(2, 1))
