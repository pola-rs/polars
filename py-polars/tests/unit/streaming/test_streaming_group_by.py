from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pytest

import polars as pl
from polars.testing import assert_frame_equal

pytestmark = pytest.mark.xdist_group("streaming")


@pytest.mark.slow()
def test_streaming_group_by_sorted_fast_path_nulls_10273() -> None:
    df = pl.Series(
        name="x",
        values=(
            *(i for i in range(4) for _ in range(100)),
            *(None for _ in range(100)),
        ),
    ).to_frame()

    assert (
        df.set_sorted("x")
        .lazy()
        .group_by("x")
        .agg(pl.count())
        .collect(streaming=True)
        .sort("x")
    ).to_dict(as_series=False) == {
        "x": [None, 0, 1, 2, 3],
        "count": [100, 100, 100, 100, 100],
    }


def test_streaming_group_by_types() -> None:
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
                .group_by(by)
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

        assert out.to_dict(as_series=False) == {
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
            .group_by("person_id")
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


def test_streaming_group_by_min_max() -> None:
    df = pl.DataFrame(
        {
            "person_id": [1, 2, 3, 4, 5, 6],
            "year": [1995, 1995, 1995, 2, 2, 2],
        }
    )
    out = (
        df.lazy()
        .group_by("year")
        .agg([pl.min("person_id").alias("min"), pl.max("person_id").alias("max")])
        .collect()
        .sort("year")
    )
    assert out["min"].to_list() == [4, 1]
    assert out["max"].to_list() == [6, 3]


def test_streaming_non_streaming_gb() -> None:
    n = 100
    df = pl.DataFrame({"a": np.random.randint(0, 20, n)})
    q = df.lazy().group_by("a").agg(pl.count()).sort("a")
    assert_frame_equal(q.collect(streaming=True), q.collect())

    q = df.lazy().with_columns(pl.col("a").cast(pl.Utf8))
    q = q.group_by("a").agg(pl.count()).sort("a")
    assert_frame_equal(q.collect(streaming=True), q.collect())
    q = df.lazy().with_columns(pl.col("a").alias("b"))
    q = (
        q.group_by(["a", "b"])
        .agg(pl.count(), pl.col("a").sum().alias("sum_a"))
        .sort("a")
    )
    assert_frame_equal(q.collect(streaming=True), q.collect())


def test_streaming_group_by_sorted_fast_path() -> None:
    a = np.random.randint(0, 20, 80)
    df = pl.DataFrame(
        {
            # test on int8 as that also tests proper conversions
            "a": pl.Series(np.sort(a), dtype=pl.Int8)
        }
    ).with_row_count()

    df_sorted = df.with_columns(pl.col("a").set_sorted())

    for streaming in [True, False]:
        results = []
        for df_ in [df, df_sorted]:
            out = (
                df_.lazy()
                .group_by("a")
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

        assert_frame_equal(results[0], results[1])


@pytest.fixture(scope="module")
def random_integers() -> pl.Series:
    np.random.seed(1)
    return pl.Series("a", np.random.randint(0, 10, 100), dtype=pl.Int64)


@pytest.mark.write_disk()
def test_streaming_group_by_ooc_q1(
    monkeypatch: Any, random_integers: pl.Series
) -> None:
    s = random_integers
    monkeypatch.setenv("POLARS_FORCE_OOC", "1")

    result = (
        s.to_frame()
        .lazy()
        .group_by("a")
        .agg(pl.first("a").alias("a_first"), pl.last("a").alias("a_last"))
        .sort("a")
        .collect(streaming=True)
    )

    expected = pl.DataFrame(
        {
            "a": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "a_first": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "a_last": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        }
    )
    assert_frame_equal(result, expected)


@pytest.mark.write_disk()
def test_streaming_group_by_ooc_q2(
    monkeypatch: Any, random_integers: pl.Series
) -> None:
    s = random_integers
    monkeypatch.setenv("POLARS_FORCE_OOC", "1")

    result = (
        s.cast(str)
        .to_frame()
        .lazy()
        .group_by("a")
        .agg(pl.first("a").alias("a_first"), pl.last("a").alias("a_last"))
        .sort("a")
        .collect(streaming=True)
    )

    expected = pl.DataFrame(
        {
            "a": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
            "a_first": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
            "a_last": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
        }
    )
    assert_frame_equal(result, expected)


@pytest.mark.write_disk()
def test_streaming_group_by_ooc_q3(
    monkeypatch: Any, random_integers: pl.Series
) -> None:
    s = random_integers
    monkeypatch.setenv("POLARS_FORCE_OOC", "1")

    result = (
        pl.DataFrame({"a": s, "b": s})
        .lazy()
        .group_by(["a", "b"])
        .agg(pl.first("a").alias("a_first"), pl.last("a").alias("a_last"))
        .sort("a")
        .collect(streaming=True)
    )

    expected = pl.DataFrame(
        {
            "a": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "b": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "a_first": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "a_last": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        }
    )
    assert_frame_equal(result, expected)


def test_streaming_group_by_struct_key() -> None:
    df = pl.DataFrame(
        {"A": [1, 2, 3, 2], "B": ["google", "ms", "apple", "ms"], "C": [2, 3, 4, 3]}
    )
    df1 = df.lazy().with_columns(pl.struct(["A", "C"]).alias("tuples"))
    assert df1.group_by("tuples").agg(pl.count(), pl.col("B").first()).sort(
        "B"
    ).collect(streaming=True).to_dict(as_series=False) == {
        "tuples": [{"A": 3, "C": 4}, {"A": 1, "C": 2}, {"A": 2, "C": 3}],
        "count": [1, 1, 2],
        "B": ["apple", "google", "ms"],
    }


@pytest.mark.slow()
def test_streaming_group_by_all_numeric_types_stability_8570() -> None:
    m = 1000
    n = 1000

    rng = np.random.default_rng(seed=0)
    dfa = pl.DataFrame({"x": pl.arange(start=0, end=n, eager=True)})
    dfb = pl.DataFrame(
        {
            "y": rng.integers(low=0, high=10, size=m),
            "z": rng.integers(low=0, high=2, size=m),
        }
    )
    dfc = dfa.join(dfb, how="cross")

    for keys in [["x", "y"], "z"]:
        for dtype in [pl.Boolean, *pl.INTEGER_DTYPES]:
            # the alias checks if the schema is correctly handled
            dfd = (
                dfc.lazy()
                .with_columns(pl.col("z").cast(dtype))
                .group_by(keys)
                .agg(pl.col("z").sum().alias("z_sum"))
                .collect(streaming=True)
            )
            assert dfd["z_sum"].sum() == dfc["z"].sum()


def test_streaming_group_by_categorical_aggregate() -> None:
    with pl.StringCache():
        out = (
            pl.LazyFrame(
                {
                    "a": pl.Series(
                        ["a", "a", "b", "b", "c", "c", None, None], dtype=pl.Categorical
                    ),
                    "b": pl.Series(
                        pl.date_range(
                            date(2023, 4, 28),
                            date(2023, 5, 5),
                            eager=True,
                        ).to_list(),
                        dtype=pl.Date,
                    ),
                }
            )
            .group_by(["a", "b"])
            .agg([pl.col("a").first().alias("sum")])
            .collect(streaming=True)
        )

    assert out.sort("b").to_dict(as_series=False) == {
        "a": ["a", "a", "b", "b", "c", "c", None, None],
        "b": [
            date(2023, 4, 28),
            date(2023, 4, 29),
            date(2023, 4, 30),
            date(2023, 5, 1),
            date(2023, 5, 2),
            date(2023, 5, 3),
            date(2023, 5, 4),
            date(2023, 5, 5),
        ],
        "sum": ["a", "a", "b", "b", "c", "c", None, None],
    }


def test_streaming_group_by_list_9758() -> None:
    payload = {"a": [[1, 2]]}
    assert (
        pl.LazyFrame(payload)
        .group_by("a")
        .first()
        .collect(streaming=True)
        .to_dict(as_series=False)
        == payload
    )


def test_streaming_restart_non_streamable_group_by() -> None:
    df = pl.DataFrame({"id": [1], "id2": [1], "id3": [1], "value": [1]})
    res = (
        df.lazy()
        .join(df.lazy(), on=["id", "id2"], how="left")
        .filter(
            (pl.col("id3") > pl.col("id3_right"))
            & (pl.col("id3") - pl.col("id3_right") < 30)
        )
        .group_by(["id2", "id3", "id3_right"])
        .agg(
            pl.col("value").map_elements(lambda x: x).sum() * pl.col("value").sum()
        )  # non-streamable UDF + nested_agg
    )

    assert """--- PIPELINE""" in res.explain(streaming=True)


def test_group_by_min_max_string_type() -> None:
    table = pl.from_dict({"a": [1, 1, 2, 2, 2], "b": ["a", "b", "c", "d", None]})

    expected = {"a": [1, 2], "min": ["a", "c"], "max": ["b", "d"]}

    for streaming in [True, False]:
        assert (
            table.lazy()
            .group_by("a")
            .agg([pl.min("b").alias("min"), pl.max("b").alias("max")])
            .collect(streaming=streaming)
            .sort("a")
            .to_dict(as_series=False)
            == expected
        )


@pytest.mark.parametrize("literal", [True, "foo", 1])
def test_streaming_group_by_literal(literal: Any) -> None:
    df = pl.LazyFrame({"a": range(20)})

    assert df.group_by(pl.lit(literal)).agg(
        [
            pl.col("a").count().alias("a_count"),
            pl.col("a").sum().alias("a_sum"),
        ]
    ).collect(streaming=True).to_dict(as_series=False) == {
        "literal": [literal],
        "a_count": [20],
        "a_sum": [190],
    }
