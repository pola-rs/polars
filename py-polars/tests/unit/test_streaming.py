from datetime import date

import numpy as np

import polars as pl


def test_streaming_groupby_types() -> None:
    df = pl.DataFrame(
        {
            "person_id": [1, 1],
            "year": [1995, 1995],
            "person_name": ["bob", "foo"],
            "bool": [True, False],
            "date": [date(2022, 1, 1), date(2022, 1, 1)],
        }
    )

    for by in ["person_id", "year", "date", ["person_id", "year"]]:
        out = (
            (
                df.lazy()
                .groupby(by)
                .agg(
                    [
                        pl.col("person_name").first().alias("str_first"),
                        pl.col("person_name").last().alias("str_last"),
                        pl.col("person_name").mean().alias("str_mean"),
                        pl.col("person_name").sum().alias("str_sum"),
                        pl.col("bool").first().alias("bool_first"),
                        pl.col("bool").last().alias("bool_last"),
                        pl.col("bool").mean().alias("bool_mean"),
                        pl.col("bool").sum().alias("bool_sum"),
                        pl.col("date").sum().alias("date_sum"),
                        pl.col("date").mean().alias("date_mean"),
                        pl.col("date").first().alias("date_first"),
                        pl.col("date").last().alias("date_last"),
                    ]
                )
            )
            .select(pl.all().exclude(by))
            .collect(streaming=True)
        )
        assert out.schema == {
            "str_first": pl.Utf8,
            "str_last": pl.Utf8,
            "str_mean": pl.Utf8,
            "str_sum": pl.Utf8,
            "bool_first": pl.Boolean,
            "bool_last": pl.Boolean,
            "bool_mean": pl.Boolean,
            "bool_sum": pl.UInt32,
            "date_sum": pl.Date,
            "date_mean": pl.Date,
            "date_first": pl.Date,
            "date_last": pl.Date,
        }

        assert out.to_dict(False) == {
            "str_first": ["bob"],
            "str_last": ["foo"],
            "str_mean": [None],
            "str_sum": [None],
            "bool_first": [True],
            "bool_last": [False],
            "bool_mean": [None],
            "bool_sum": [1],
            "date_sum": [date(2074, 1, 1)],
            "date_mean": [date(2022, 1, 1)],
            "date_first": [date(2022, 1, 1)],
            "date_last": [date(2022, 1, 1)],
        }


def test_streaming_non_streaming_gb() -> None:
    n = 100
    df = pl.DataFrame({"a": np.random.randint(0, 20, n)})
    q = df.lazy().groupby("a").agg(pl.count()).sort("a")
    assert q.collect(streaming=True).frame_equal(q.collect())

    q = df.lazy().with_column(pl.col("a").cast(pl.Utf8))
    q = q.groupby("a").agg(pl.count()).sort("a")
    assert q.collect(streaming=True).frame_equal(q.collect())
    q = df.lazy().with_column(pl.col("a").alias("b"))
    q = q.groupby(["a", "b"]).agg(pl.count()).sort("a")
    assert q.collect(streaming=True).frame_equal(q.collect())
