from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

import polars as pl
from polars.exceptions import DuplicateError
from polars.testing import assert_frame_equal
from tests.unit.conftest import INTEGER_DTYPES

if TYPE_CHECKING:
    from pathlib import Path

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
        .agg(pl.len())
        .collect(streaming=True)
        .sort("x")
    ).to_dict(as_series=False) == {
        "x": [None, 0, 1, 2, 3],
        "len": [100, 100, 100, 100, 100],
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
                        # Date streaming mean/median has been temporarily disabled
                        # pl.col("date").mean().alias("date_mean"),
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
            "str_first": pl.String,
            "str_last": pl.String,
            "str_mean": pl.String,
            "str_sum": pl.String,
            "bool_first": pl.Boolean,
            "bool_last": pl.Boolean,
            "bool_mean": pl.Float64,
            "bool_sum": pl.UInt32,
            "date_sum": pl.Date,
            # "date_mean": pl.Date,
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
            "bool_mean": [0.5],
            "bool_sum": [1],
            "date_sum": [None],
            # Date streaming mean/median has been temporarily disabled
            # "date_mean": [date(2022, 1, 1)],
            "date_first": [date(2022, 1, 1)],
            "date_last": [date(2022, 1, 1)],
            "date_min": [date(2022, 1, 1)],
            "date_max": [date(2022, 1, 1)],
        }

    with pytest.raises(DuplicateError):
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
    q = df.lazy().group_by("a").agg(pl.len()).sort("a")
    assert_frame_equal(q.collect(streaming=True), q.collect())

    q = df.lazy().with_columns(pl.col("a").cast(pl.String))
    q = q.group_by("a").agg(pl.len()).sort("a")
    assert_frame_equal(q.collect(streaming=True), q.collect())
    q = df.lazy().with_columns(pl.col("a").alias("b"))
    q = q.group_by(["a", "b"]).agg(pl.len(), pl.col("a").sum().alias("sum_a")).sort("a")
    assert_frame_equal(q.collect(streaming=True), q.collect())


def test_streaming_group_by_sorted_fast_path() -> None:
    a = np.random.randint(0, 20, 80)
    df = pl.DataFrame(
        {
            # test on int8 as that also tests proper conversions
            "a": pl.Series(np.sort(a), dtype=pl.Int8)
        }
    ).with_row_index()

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
    random_integers: pl.Series,
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    tmp_path.mkdir(exist_ok=True)
    monkeypatch.setenv("POLARS_TEMP_DIR", str(tmp_path))
    monkeypatch.setenv("POLARS_FORCE_OOC", "1")

    lf = random_integers.to_frame().lazy()
    result = (
        lf.group_by("a")
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
    random_integers: pl.Series,
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    tmp_path.mkdir(exist_ok=True)
    monkeypatch.setenv("POLARS_TEMP_DIR", str(tmp_path))
    monkeypatch.setenv("POLARS_FORCE_OOC", "1")

    lf = random_integers.cast(str).to_frame().lazy()
    result = (
        lf.group_by("a")
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
    random_integers: pl.Series,
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    tmp_path.mkdir(exist_ok=True)
    monkeypatch.setenv("POLARS_TEMP_DIR", str(tmp_path))
    monkeypatch.setenv("POLARS_FORCE_OOC", "1")

    lf = pl.LazyFrame({"a": random_integers, "b": random_integers})
    result = (
        lf.group_by("a", "b")
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
    assert df1.group_by("tuples").agg(pl.len(), pl.col("B").first()).sort("B").collect(
        streaming=True
    ).to_dict(as_series=False) == {
        "tuples": [{"A": 3, "C": 4}, {"A": 1, "C": 2}, {"A": 2, "C": 3}],
        "len": [1, 1, 2],
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
        for dtype in [*INTEGER_DTYPES, pl.Boolean]:
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

    assert "STREAMING" in res.explain(streaming=True)


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


@pytest.mark.parametrize("streaming", [True, False])
def test_group_by_multiple_keys_one_literal(streaming: bool) -> None:
    df = pl.DataFrame({"a": [1, 1, 2], "b": [4, 5, 6]})

    expected = {"a": [1, 2], "literal": [1, 1], "b": [5, 6]}
    assert (
        df.lazy()
        .group_by("a", pl.lit(1))
        .agg(pl.col("b").max())
        .sort(["a", "b"])
        .collect(streaming=streaming)
        .to_dict(as_series=False)
        == expected
    )


def test_streaming_group_null_count() -> None:
    df = pl.DataFrame({"g": [1] * 6, "a": ["yes", None] * 3}).lazy()
    assert df.group_by("g").agg(pl.col("a").count()).collect(streaming=True).to_dict(
        as_series=False
    ) == {"g": [1], "a": [3]}


def test_streaming_group_by_binary_15116() -> None:
    assert (
        pl.LazyFrame(
            {
                "str": [
                    "A",
                    "A",
                    "BB",
                    "BB",
                    "CCCC",
                    "CCCC",
                    "DDDDDDDD",
                    "DDDDDDDD",
                    "EEEEEEEEEEEEEEEE",
                    "A",
                ]
            }
        )
        .select([pl.col("str").cast(pl.Binary)])
        .group_by(["str"])
        .agg([pl.len().alias("count")])
    ).sort("str").collect(streaming=True).to_dict(as_series=False) == {
        "str": [b"A", b"BB", b"CCCC", b"DDDDDDDD", b"EEEEEEEEEEEEEEEE"],
        "count": [3, 2, 2, 2, 1],
    }


def test_streaming_group_by_convert_15380(partition_limit: int) -> None:
    assert (
        pl.DataFrame({"a": [1] * partition_limit}).group_by(b="a").len()["len"].item()
        == partition_limit
    )


@pytest.mark.parametrize("streaming", [True, False])
@pytest.mark.parametrize("n_rows_limit_offset", [-1, +3])
def test_streaming_group_by_boolean_mean_15610(
    n_rows_limit_offset: int, streaming: bool, partition_limit: int
) -> None:
    n_rows = partition_limit + n_rows_limit_offset

    # Also test non-streaming because it sometimes dispatched to streaming agg.
    expect = pl.DataFrame({"a": [False, True], "c": [0.0, 0.5]})

    n_repeats = n_rows // 3
    assert n_repeats > 0

    out = (
        pl.select(
            a=pl.repeat([True, False, True], n_repeats).explode(),
            b=pl.repeat([True, False, False], n_repeats).explode(),
        )
        .lazy()
        .group_by("a")
        .agg(c=pl.mean("b"))
        .sort("a")
        .collect(streaming=streaming)
    )

    assert_frame_equal(out, expect)
