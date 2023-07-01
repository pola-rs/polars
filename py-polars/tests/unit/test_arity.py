from datetime import datetime

import polars as pl
from polars.testing import assert_frame_equal


def test_nested_when_then_and_wildcard_expansion_6284() -> None:
    df = pl.DataFrame(
        {
            "1": ["a", "b"],
            "2": ["c", "d"],
        }
    )

    out0 = df.with_columns(
        pl.when(pl.any(pl.all() == "a"))
        .then("a")
        .otherwise(pl.when(pl.any(pl.all() == "d")).then("d").otherwise(None))
        .alias("result")
    )

    out1 = df.with_columns(
        pl.when(pl.any(pl.all() == "a"))
        .then("a")
        .when(pl.any(pl.all() == "d"))
        .then("d")
        .otherwise(None)
        .alias("result")
    )

    assert_frame_equal(out0, out1)
    assert out0.to_dict(False) == {
        "1": ["a", "b"],
        "2": ["c", "d"],
        "result": ["a", "d"],
    }


def test_expression_literal_series_order() -> None:
    s = pl.Series([1, 2, 3])
    df = pl.DataFrame({"a": [1, 2, 3]})

    assert df.select(pl.col("a") + s).to_dict(False) == {"a": [2, 4, 6]}
    assert df.select(pl.lit(s) + pl.col("a")).to_dict(False) == {"": [2, 4, 6]}


def test_list_zip_with_logical_type() -> None:
    df = pl.DataFrame(
        {
            "start": [datetime(2023, 1, 1, 1, 1, 1), datetime(2023, 1, 1, 1, 1, 1)],
            "stop": [datetime(2023, 1, 1, 1, 3, 1), datetime(2023, 1, 1, 1, 4, 1)],
            "use": [1, 0],
        }
    )

    df = df.with_columns(
        pl.date_range(
            pl.col("start"), pl.col("stop"), interval="1h", eager=False, closed="left"
        ).alias("interval_1"),
        pl.date_range(
            pl.col("start"), pl.col("stop"), interval="1h", eager=False, closed="left"
        ).alias("interval_2"),
    )

    out = df.select(
        pl.when(pl.col("use") == 1)
        .then(pl.col("interval_2"))
        .otherwise(pl.col("interval_1"))
        .alias("interval_new")
    )
    assert out.dtypes == [pl.List(pl.Datetime(time_unit="us", time_zone=None))]
