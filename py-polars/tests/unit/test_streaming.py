from datetime import date

import numpy as np
import pytest

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
                        pl.col("date").min().alias("date_min"),
                        pl.col("date").max().alias("date_max"),
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
            "date_min": pl.Date,
            "date_max": pl.Date,
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
            "date_min": [date(2022, 1, 1)],
            "date_max": [date(2022, 1, 1)],
        }

    with pytest.raises(pl.DuplicateError):
        (
            df.lazy()
            .groupby("person_id")
            .agg(
                [
                    pl.col("person_name").first().alias("str_first"),
                    pl.col("person_name").last().alias("str_last"),
                    pl.col("person_name").mean().alias("str_mean"),
                    pl.col("person_name").sum().alias("str_sum"),
                    pl.col("bool").first().alias("bool_first"),
                    pl.col("bool").last().alias("bool_first"),
                ]
            )
            .select(pl.all().exclude("person_id"))
            .collect(streaming=True)
        )


def test_streaming_groupby_min_max() -> None:
    df = pl.DataFrame(
        {
            "person_id": [1, 2, 3, 4, 5, 6],
            "year": [1995, 1995, 1995, 2, 2, 2],
        }
    )
    out = (
        df.lazy()
        .groupby("year")
        .agg([pl.min("person_id").alias("min"), pl.max("person_id").alias("max")])
        .collect()
        .sort("year")
    )
    assert out["min"].to_list() == [4, 1]
    assert out["max"].to_list() == [6, 3]


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


def test_streaming_groupby_sorted_fast_path() -> None:
    a = np.random.randint(0, 20, 80)
    df = pl.DataFrame(
        {
            # test on int8 as that also tests proper conversions
            "a": pl.Series(np.sort(a), dtype=pl.Int8)
        }
    ).with_row_count()

    df_sorted = df.with_column(pl.col("a").set_sorted())

    for streaming in [True, False]:
        results = []
        for df_ in [df, df_sorted]:
            out = (
                df_.lazy()
                .groupby("a")
                .agg(
                    [
                        pl.first("a").alias("first"),
                        pl.last("a").alias("last"),
                        pl.sum("a").alias("sum"),
                        pl.mean("a").alias("mean"),
                        pl.count("a").alias("count"),
                        pl.min("a").alias("min"),
                        pl.max("a").alias("max"),
                    ]
                )
                .sort("a")
                .collect(streaming=streaming)
            )
            results.append(out)

        assert results[0].frame_equal(results[1])


def test_streaming_categoricals_5921() -> None:
    with pl.StringCache():
        out_lazy = (
            pl.DataFrame({"X": ["a", "a", "a", "b", "b"], "Y": [2, 2, 2, 1, 1]})
            .lazy()
            .with_column(pl.col("X").cast(pl.Categorical))
            .groupby("X")
            .agg(pl.col("Y").min())
            .sort("X")
            .collect(streaming=True)
        )

        out_eager = (
            pl.DataFrame({"X": ["a", "a", "a", "b", "b"], "Y": [2, 2, 2, 1, 1]})
            .with_column(pl.col("X").cast(pl.Categorical))
            .groupby("X")
            .agg(pl.col("Y").min())
            .sort("X")
        )

    for out in [out_eager, out_lazy]:
        assert out.dtypes == [pl.Categorical, pl.Int64]
        assert out.to_dict(False) == {"X": ["a", "b"], "Y": [2, 1]}


def test_streaming_block_on_literals_6054() -> None:
    df = pl.DataFrame({"col_1": [0] * 5 + [1] * 5})
    s = pl.Series("col_2", list(range(10)))

    assert df.lazy().with_column(s).groupby("col_1").agg(pl.all().first()).collect(
        streaming=True
    ).sort("col_1").to_dict(False) == {"col_1": [0, 1], "col_2": [0, 5]}
