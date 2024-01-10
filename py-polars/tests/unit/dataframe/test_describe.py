from __future__ import annotations

from datetime import date

import pytest

import polars as pl
from polars.testing import assert_frame_equal


def test_df_describe() -> None:
    df = pl.DataFrame(
        {
            "a": [1.0, 2.8, 3.0],
            "b": [4, 5, None],
            "c": [True, False, True],
            "d": [None, "b", "c"],
            "e": ["usd", "eur", None],
            "f": [date(2020, 1, 1), date(2021, 1, 1), date(2022, 1, 1)],
        },
        schema_overrides={"e": pl.Categorical},
    )

    result = df.describe()
    print(result)
    expected = pl.DataFrame(
        {
            "describe": [
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
            "c": ["3", "0", None, None, "False", None, None, None, "True"],
            "d": ["2", "1", None, None, "b", None, None, None, "c"],
            "e": ["2", "1", None, None, None, None, None, None, None],
            "f": ["3", "0", None, None, "2020-01-01", None, None, None, "2022-01-01"],
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
        schema=["describe"] + df.columns,
        schema_overrides={"struct": pl.String, "list": pl.String},
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
        schema=["describe"] + df.columns,
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
        schema=["describe"] + df.columns,
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
        schema=["describe"] + df.columns,
    )
    assert_frame_equal(result, expected)


def test_df_describe_empty() -> None:
    df = pl.DataFrame()
    with pytest.raises(
        TypeError, match="cannot describe a DataFrame without any columns"
    ):
        df.describe()


def test_df_describe_quantile_precision() -> None:
    df = pl.DataFrame({"a": range(10)})
    result = df.describe(percentiles=[0.99, 0.999, 0.9999])
    result_metrics = result.get_column("describe").to_list()
    expected_metrics = ["99%", "99.9%", "99.99%"]
    for m in expected_metrics:
        assert m in result_metrics
