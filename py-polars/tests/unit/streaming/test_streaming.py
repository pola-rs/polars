import time
import typing
from datetime import date
from typing import Any

import numpy as np
import pytest

import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal


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
    assert_frame_equal(q.collect(streaming=True), q.collect())

    q = df.lazy().with_columns(pl.col("a").cast(pl.Utf8))
    q = q.groupby("a").agg(pl.count()).sort("a")
    assert_frame_equal(q.collect(streaming=True), q.collect())
    q = df.lazy().with_columns(pl.col("a").alias("b"))
    q = (
        q.groupby(["a", "b"])
        .agg(pl.count(), pl.col("a").sum().alias("sum_a"))
        .sort("a")
    )
    assert_frame_equal(q.collect(streaming=True), q.collect())


def test_streaming_groupby_sorted_fast_path() -> None:
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

        assert_frame_equal(results[0], results[1])


def test_streaming_categoricals_5921() -> None:
    with pl.StringCache():
        out_lazy = (
            pl.DataFrame({"X": ["a", "a", "a", "b", "b"], "Y": [2, 2, 2, 1, 1]})
            .lazy()
            .with_columns(pl.col("X").cast(pl.Categorical))
            .groupby("X")
            .agg(pl.col("Y").min())
            .sort("Y", descending=True)
            .collect(streaming=True)
        )

        out_eager = (
            pl.DataFrame({"X": ["a", "a", "a", "b", "b"], "Y": [2, 2, 2, 1, 1]})
            .with_columns(pl.col("X").cast(pl.Categorical))
            .groupby("X")
            .agg(pl.col("Y").min())
            .sort("Y", descending=True)
        )

    for out in [out_eager, out_lazy]:
        assert out.dtypes == [pl.Categorical, pl.Int64]
        assert out.to_dict(False) == {"X": ["a", "b"], "Y": [2, 1]}


def test_streaming_block_on_literals_6054() -> None:
    df = pl.DataFrame({"col_1": [0] * 5 + [1] * 5})
    s = pl.Series("col_2", list(range(10)))

    assert df.lazy().with_columns(s).groupby("col_1").agg(pl.all().first()).collect(
        streaming=True
    ).sort("col_1").to_dict(False) == {"col_1": [0, 1], "col_2": [0, 5]}


def test_streaming_streamable_functions(monkeypatch: Any, capfd: Any) -> None:
    monkeypatch.setenv("POLARS_VERBOSE", "1")
    assert (
        pl.DataFrame({"a": [1, 2, 3]})
        .lazy()
        .map(
            function=lambda df: df.with_columns(pl.col("a").alias("b")),
            schema={"a": pl.Int64, "b": pl.Int64},
            streamable=True,
        )
    ).collect(streaming=True).to_dict(False) == {"a": [1, 2, 3], "b": [1, 2, 3]}

    (_, err) = capfd.readouterr()
    assert "df -> function -> ordered_sink" in err


@pytest.mark.slow()
def test_cross_join_stack() -> None:
    a = pl.Series(np.arange(100_000)).to_frame().lazy()
    t0 = time.time()
    # this should be instant if directly pushed into sink
    # if not the cross join will first fill the stack with all matches of a single chunk
    assert a.join(a, how="cross").head().collect(streaming=True).shape == (5, 2)
    t1 = time.time()
    assert (t1 - t0) < 0.5


@pytest.mark.slow()
def test_ooc_sort(monkeypatch: Any) -> None:
    monkeypatch.setenv("POLARS_FORCE_OOC", "1")

    s = pl.arange(0, 100_000, eager=True).rename("idx")

    df = s.shuffle().to_frame()

    for descending in [True, False]:
        out = (
            df.lazy().sort("idx", descending=descending).collect(streaming=True)
        ).to_series()

        assert_series_equal(out, s.sort(descending=descending))


def test_streaming_literal_expansion() -> None:
    df = pl.DataFrame(
        {
            "y": ["a", "b"],
            "z": [1, 2],
        }
    )

    q = df.lazy().select(
        x=pl.lit("constant"),
        y=pl.col("y"),
        z=pl.col("z"),
    )

    assert q.collect(streaming=True).to_dict(False) == {
        "x": ["constant", "constant"],
        "y": ["a", "b"],
        "z": [1, 2],
    }
    assert q.groupby(["x", "y"]).agg(pl.mean("z")).sort("y").collect(
        streaming=True
    ).to_dict(False) == {
        "x": ["constant", "constant"],
        "y": ["a", "b"],
        "z": [1.0, 2.0],
    }
    assert q.groupby(["x"]).agg(pl.mean("z")).collect().to_dict(False) == {
        "x": ["constant"],
        "z": [1.5],
    }


def test_tree_validation_streaming() -> None:
    # this query leads to a tree collection with an invalid branch
    # this test triggers the tree validation function.
    df_1 = pl.DataFrame(
        {
            "a": [22, 1, 1],
            "b": [500, 37, 20],
        },
    ).lazy()

    df_2 = pl.DataFrame(
        {"a": [23, 4, 20, 28, 3]},
    ).lazy()

    dfs = [df_2]
    cat = pl.concat(dfs, how="vertical")

    df_3 = df_1.select(
        [
            "a",
            # this expression is not allowed streaming, so it invalidates a branch
            pl.col("b")
            .filter(pl.col("a").min() > pl.col("a").rank())
            .alias("b_not_streaming"),
        ]
    ).join(
        cat,
        on=[
            "a",
        ],
    )

    out = df_1.join(df_3, on="a", how="left")
    assert out.collect(streaming=True).shape == (3, 3)


def test_streaming_apply(monkeypatch: Any, capfd: Any) -> None:
    monkeypatch.setenv("POLARS_VERBOSE", "1")
    q = pl.DataFrame({"a": [1, 2]}).lazy()

    (
        q.select(pl.col("a").apply(lambda x: x * 2, return_dtype=pl.Int64)).collect(
            streaming=True
        )
    )
    (_, err) = capfd.readouterr()
    assert "df -> projection -> ordered_sink" in err


def test_streaming_ternary() -> None:
    q = pl.LazyFrame({"a": [1, 2, 3]})

    assert (
        q.with_columns(
            pl.when(pl.col("a") >= 2).then(pl.col("a")).otherwise(None).alias("b"),
        )
        .explain(streaming=True)
        .startswith("--- PIPELINE")
    )


def test_streaming_unique(monkeypatch: Any, capfd: Any) -> None:
    monkeypatch.setenv("POLARS_VERBOSE", "1")
    df = pl.DataFrame({"a": [1, 2, 2, 2], "b": [3, 4, 4, 4], "c": [5, 6, 7, 7]})
    q = df.lazy().unique(subset=["a", "c"], maintain_order=False).sort(["a", "b", "c"])
    assert_frame_equal(q.collect(streaming=True), q.collect(streaming=False))

    q = df.lazy().unique(subset=["b", "c"], maintain_order=False).sort(["a", "b", "c"])
    assert_frame_equal(q.collect(streaming=True), q.collect(streaming=False))

    q = df.lazy().unique(subset=None, maintain_order=False).sort(["a", "b", "c"])
    assert_frame_equal(q.collect(streaming=True), q.collect(streaming=False))
    (_, err) = capfd.readouterr()
    assert "df -> re-project-sink -> sort_multiple" in err


@pytest.mark.write_disk()
def test_streaming_sort(monkeypatch: Any, capfd: Any) -> None:
    monkeypatch.setenv("POLARS_VERBOSE", "1")
    monkeypatch.setenv("POLARS_FORCE_OOC", "1")
    # this creates a lot of duplicate partitions and triggers: #7568
    assert (
        pl.Series(np.random.randint(0, 100, 100))
        .to_frame("s")
        .lazy()
        .sort("s")
        .collect(streaming=True)["s"]
        .is_sorted()
    )
    (_, err) = capfd.readouterr()
    assert "df -> sort" in err


@pytest.mark.write_disk()
def test_streaming_groupby_ooc(monkeypatch: Any) -> None:
    np.random.seed(1)
    s = pl.Series("a", np.random.randint(0, 10, 100))

    for env in ["POLARS_FORCE_OOC", "_NO_OP"]:
        monkeypatch.setenv(env, "1")
        q = (
            s.to_frame()
            .lazy()
            .groupby("a")
            .agg(pl.first("a").alias("a_first"), pl.last("a").alias("a_last"))
            .sort("a")
        )

        assert q.collect(streaming=True).to_dict(False) == {
            "a": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "a_first": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "a_last": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        }

        q = (
            s.cast(str)
            .to_frame()
            .lazy()
            .groupby("a")
            .agg(pl.first("a").alias("a_first"), pl.last("a").alias("a_last"))
            .sort("a")
        )

        assert q.collect(streaming=True).to_dict(False) == {
            "a": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
            "a_first": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
            "a_last": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
        }

        q = (
            pl.DataFrame(
                {
                    "a": s,
                    "b": s.rename("b"),
                }
            )
            .lazy()
            .groupby(["a", "b"])
            .agg(pl.first("a").alias("a_first"), pl.last("a").alias("a_last"))
            .sort("a")
        )

        assert q.collect(streaming=True).to_dict(False) == {
            "a": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "b": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "a_first": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "a_last": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        }


def test_streaming_groupby_struct_key() -> None:
    df = pl.DataFrame(
        {"A": [1, 2, 3, 2], "B": ["google", "ms", "apple", "ms"], "C": [2, 3, 4, 3]}
    )
    df1 = df.lazy().with_columns(pl.struct(["A", "C"]).alias("tuples"))
    assert df1.groupby("tuples").agg(pl.count(), pl.col("B").first()).sort("B").collect(
        streaming=True
    ).to_dict(False) == {
        "tuples": [{"A": 3, "C": 4}, {"A": 1, "C": 2}, {"A": 2, "C": 3}],
        "count": [1, 1, 2],
        "B": ["apple", "google", "ms"],
    }


@pytest.mark.slow()
def test_streaming_groupby_all_numeric_types_stability_8570() -> None:
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
                .groupby(keys)
                .agg(pl.col("z").sum().alias("z_sum"))
                .collect(streaming=True)
            )
            assert dfd["z_sum"].sum() == dfc["z"].sum()


@typing.no_type_check
def test_streaming_groupby_categorical_aggregate() -> None:
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
            .groupby(["a", "b"])
            .agg([pl.col("a").first().alias("sum")])
            .collect(streaming=True)
        )

    assert out.sort("b").to_dict(False) == {
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


def test_streaming_restart_non_streamable_groupby() -> None:
    df = pl.DataFrame({"id": [1], "id2": [1], "id3": [1], "value": [1]})
    res = (
        df.lazy()
        .join(df.lazy(), on=["id", "id2"], how="left")
        .filter(
            (pl.col("id3") > pl.col("id3_right"))
            & (pl.col("id3") - pl.col("id3_right") < 30)
        )
        .groupby(["id2", "id3", "id3_right"])
        .agg(
            pl.col("value").apply(lambda x: x).sum() * pl.col("value").sum()
        )  # non-streamable UDF + nested_agg
    )

    assert """--- PIPELINE""" in res.explain(streaming=True)


def test_streaming_sortedness_propagation_9494() -> None:
    assert (
        pl.DataFrame(
            {
                "when": [date(2023, 5, 10), date(2023, 5, 20), date(2023, 6, 10)],
                "what": [1, 2, 3],
            }
        )
        .lazy()
        .sort("when")
        .groupby_dynamic("when", every="1mo")
        .agg(pl.col("what").sum())
        .collect(streaming=True)
    ).to_dict(False) == {"when": [date(2023, 5, 1), date(2023, 6, 1)], "what": [3, 3]}


@pytest.mark.write_disk()
def test_out_of_core_sort_9503(monkeypatch: Any) -> None:
    monkeypatch.setenv("POLARS_FORCE_OOC", "1")
    np.random.seed(0)

    num_rows = 1_00_000
    num_columns = 2
    num_tables = 10

    # ensure we create many chunks
    # this will ensure we create more files
    # and that creates contention while dumping
    q = pl.concat(
        [
            pl.DataFrame(
                [
                    pl.Series(np.random.randint(0, 10000, size=num_rows))
                    for _ in range(num_columns)
                ]
            )
            for _ in range(num_tables)
        ],
        rechunk=False,
    ).lazy()
    q = q.sort(q.columns)
    df = q.collect(streaming=True)
    assert df.shape == (1_000_000, 2)
    assert df["column_0"].flags["SORTED_ASC"]
    assert df.head(20).to_dict(False) == {
        "column_0": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "column_1": [
            242,
            245,
            588,
            618,
            732,
            902,
            925,
            945,
            1009,
            1161,
            1352,
            1365,
            1451,
            1581,
            1778,
            1836,
            1976,
            2091,
            2120,
            2124,
        ],
    }
