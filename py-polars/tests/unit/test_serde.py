from __future__ import annotations

import pickle
from datetime import datetime, timedelta

import polars as pl
from polars.testing import assert_series_equal


def test_pickling_simple_expression() -> None:
    e = pl.col("foo").sum()
    buf = pickle.dumps(e)
    assert str(pickle.loads(buf)) == str(e)


def serde_lazy_frame_lp() -> None:
    lf = pl.DataFrame({"a": [1, 2, 3], "b": ["a", "b", "c"]}).lazy().select(pl.col("a"))
    json = lf.write_json(to_string=True)

    assert (
        pl.LazyFrame.from_json(json)
        .collect()
        .to_series()
        .series_equal(pl.Series("a", [1, 2, 3]))
    )


def test_serde_time_unit() -> None:
    assert pickle.loads(
        pickle.dumps(
            pl.Series(
                [datetime(2022, 1, 1) + timedelta(days=1) for _ in range(3)]
            ).cast(pl.Datetime("ns"))
        )
    ).dtype == pl.Datetime("ns")


def test_serde_duration() -> None:
    df = (
        pl.DataFrame(
            {
                "a": [
                    datetime(2021, 2, 1, 9, 20),
                    datetime(2021, 2, 2, 9, 20),
                ],
                "b": [4, 5],
            }
        )
        .with_columns([pl.col("a").cast(pl.Datetime("ns")).alias("a")])
        .select(pl.all())
    )
    df = df.with_columns([pl.col("a").diff(n=1).alias("a_td")])
    serde_df = pickle.loads(pickle.dumps(df))
    assert serde_df["a_td"].dtype == pl.Duration("ns")
    assert_series_equal(
        serde_df["a_td"],
        pl.Series("a_td", [None, timedelta(days=1)], dtype=pl.Duration("ns")),
    )


def test_serde_expression_5461() -> None:
    e = pl.col("a").sqrt() / pl.col("b").alias("c")
    assert pickle.loads(pickle.dumps(e)).meta == e.meta
