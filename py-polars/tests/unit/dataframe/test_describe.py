from __future__ import annotations

from datetime import date, datetime, time

import pytest

import polars as pl
from polars.testing import assert_frame_equal


@pytest.mark.parametrize("lazy", [False, True])
def test_df_describe(lazy: bool) -> None:
    df = pl.DataFrame(
        {
            "a": [1.0, 2.8, 3.0],
            "b": [4, 5, None],
            "c": [True, False, True],
            "d": [None, "b", "c"],
            "e": ["usd", "eur", None],
            "f": [
                datetime(2020, 1, 1, 10, 30),
                datetime(2021, 7, 5, 15, 0),
                datetime(2022, 12, 31, 20, 30),
            ],
            "g": [date(2020, 1, 1), date(2021, 7, 5), date(2022, 12, 31)],
            "h": [time(10, 30), time(15, 0), time(20, 30)],
            "i": [1_000_000, 2_000_000, 3_000_000],
        },
        schema_overrides={"e": pl.Categorical, "i": pl.Duration},
    )

    frame: pl.DataFrame | pl.LazyFrame = df.lazy() if lazy else df
    result = frame.describe()
    print(result)

    expected = pl.DataFrame(
        {
            "statistic": [
                "count",
                "null_count",
                "mean",
                "std",
                "min",
                "25%",
                "50%",
                "75%",
                "max",
            ],
            "a": [
                3.0,
                0.0,
                2.2666666666666666,
                1.1015141094572205,
                1.0,
                2.8,
                2.8,
                3.0,
                3.0,
            ],
            "b": [2.0, 1.0, 4.5, 0.7071067811865476, 4.0, 4.0, 5.0, 5.0, 5.0],
            "c": [3.0, 0.0, 2 / 3, None, False, None, None, None, True],
            "d": ["2", "1", None, None, "b", None, None, None, "c"],
            "e": ["2", "1", None, None, None, None, None, None, None],
            "f": [
                "3",
                "0",
                "2021-07-03 07:20:00",
                None,
                "2020-01-01 10:30:00",
                "2021-07-05 15:00:00",
                "2021-07-05 15:00:00",
                "2022-12-31 20:30:00",
                "2022-12-31 20:30:00",
            ],
            "g": [
                "3",
                "0",
                "2021-07-02 16:00:00",
                None,
                "2020-01-01",
                "2021-07-05",
                "2021-07-05",
                "2022-12-31",
                "2022-12-31",
            ],
            "h": [
                "3",
                "0",
                "15:20:00",
                None,
                "10:30:00",
                "15:00:00",
                "15:00:00",
                "20:30:00",
                "20:30:00",
            ],
            "i": [
                "3",
                "0",
                "0:00:02",
                None,
                "0:00:01",
                "0:00:02",
                "0:00:02",
                "0:00:03",
                "0:00:03",
            ],
        }
    )
    assert_frame_equal(result, expected)


def test_df_describe_nested() -> None:
    df = pl.DataFrame(
        {
            "struct": [{"x": 1, "y": 2}, {"x": 3, "y": 4}, {"x": 1, "y": 2}, None],
            "list": [[1, 2], [3, 4], [1, 2], None],
        }
    )
    result = df.describe()
    expected = pl.DataFrame(
        [
            ("count", 3, 3),
            ("null_count", 1, 1),
            ("mean", None, None),
            ("std", None, None),
            ("min", None, None),
            ("25%", None, None),
            ("50%", None, None),
            ("75%", None, None),
            ("max", None, None),
        ],
        schema=["statistic"] + df.columns,
        schema_overrides={"struct": pl.Float64, "list": pl.Float64},
        orient="row",
    )
    assert_frame_equal(result, expected)


def test_df_describe_custom_percentiles() -> None:
    df = pl.DataFrame({"numeric": [1, 2, 1, None]})
    result = df.describe(percentiles=(0.2, 0.4, 0.5, 0.6, 0.8))
    expected = pl.DataFrame(
        [
            ("count", 3.0),
            ("null_count", 1.0),
            ("mean", 1.3333333333333333),
            ("std", 0.5773502691896257),
            ("min", 1.0),
            ("20%", 1.0),
            ("40%", 1.0),
            ("50%", 1.0),
            ("60%", 1.0),
            ("80%", 2.0),
            ("max", 2.0),
        ],
        schema=["statistic"] + df.columns,
        orient="row",
    )
    assert_frame_equal(result, expected)


@pytest.mark.parametrize("pcts", [None, []])
def test_df_describe_no_percentiles(pcts: list[float] | None) -> None:
    df = pl.DataFrame({"numeric": [1, 2, 1, None]})
    result = df.describe(percentiles=pcts)
    expected = pl.DataFrame(
        [
            ("count", 3.0),
            ("null_count", 1.0),
            ("mean", 1.3333333333333333),
            ("std", 0.5773502691896257),
            ("min", 1.0),
            ("max", 2.0),
        ],
        schema=["statistic"] + df.columns,
        orient="row",
    )
    assert_frame_equal(result, expected)


def test_df_describe_empty_column() -> None:
    df = pl.DataFrame(schema={"a": pl.Int64})
    result = df.describe()
    expected = pl.DataFrame(
        [
            ("count", 0.0),
            ("null_count", 0.0),
            ("mean", None),
            ("std", None),
            ("min", None),
            ("25%", None),
            ("50%", None),
            ("75%", None),
            ("max", None),
        ],
        schema=["statistic"] + df.columns,
        orient="row",
    )
    assert_frame_equal(result, expected)


@pytest.mark.parametrize("lazy", [False, True])
def test_df_describe_empty(lazy: bool) -> None:
    frame: pl.DataFrame | pl.LazyFrame = pl.LazyFrame() if lazy else pl.DataFrame()
    cls_name = "LazyFrame" if lazy else "DataFrame"
    with pytest.raises(
        TypeError, match=f"cannot describe a {cls_name} that has no columns"
    ):
        frame.describe()


def test_df_describe_quantile_precision() -> None:
    df = pl.DataFrame({"a": range(10)})
    result = df.describe(percentiles=[0.99, 0.999, 0.9999])
    result_metrics = result.get_column("statistic").to_list()
    expected_metrics = ["99%", "99.9%", "99.99%"]
    for m in expected_metrics:
        assert m in result_metrics


# https://github.com/pola-rs/polars/issues/9830
def test_df_describe_object() -> None:
    df = pl.Series(
        "object",
        [{"a": 1, "b": 2}, {"a": 3, "b": 4}, {"a": 5, "b": 6}],
        dtype=pl.Object,
    ).to_frame()

    result = df.describe(percentiles=(0.05, 0.25, 0.5, 0.75, 0.95))

    expected = pl.DataFrame(
        {"statistic": ["count", "null_count"], "object": ["3", "0"]}
    )
    assert_frame_equal(result.head(2), expected)
