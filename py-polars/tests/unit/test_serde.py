from __future__ import annotations

import pickle
from datetime import datetime, timedelta

import polars as pl
from polars.testing import assert_series_equal


def test_pickling_simple_expression() -> None:
    e = pl.col("foo").sum()
    buf = pickle.dumps(e)
    assert str(pickle.loads(buf)) == str(e)


def test_serde_lazy_frame_lp() -> None:
    lf = pl.DataFrame({"a": [1, 2, 3], "b": ["a", "b", "c"]}).lazy().select(pl.col("a"))
    json = lf.write_json()

    result = pl.LazyFrame.from_json(json).collect().to_series()
    assert_series_equal(result, pl.Series("a", [1, 2, 3]))


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


def test_serde_binary() -> None:
    data = pl.Series(
        "binary_data",
        [
            b"\xba\x9b\xca\xd3y\xcb\xc9#",
            b"9\x04\xab\xe2\x11\xf3\x85",
            b"\xb8\xcb\xc9^\\\xa9-\x94\xe0H\x9d ",
            b"S\xbc:\xcb\xf0\xf5r\xfe\x18\xfeH",
            b",\xf5)y\x00\xe5\xf7",
            b"\xfd\xf6\xf1\xc2X\x0cn\xb9#",
            b"\x06\xef\xa6\xa2\xb7",
            b"@\xff\x95\xda\xff\xd2\x18",
        ],
    )
    assert_series_equal(
        data,
        pickle.loads(pickle.dumps(data)),
    )
