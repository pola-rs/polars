from pathlib import Path
from typing import Any

import numpy as np
import pytest

import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal

pytestmark = [
    # OOC tests always write to disk
    pytest.mark.write_disk(),
    # OOC tests must run on the same worker due to garbage collection
    pytest.mark.xdist_group("ooc"),
]


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


@pytest.fixture(scope="module")
def random_integers() -> pl.Series:
    np.random.seed(1)
    return pl.Series("a", np.random.randint(0, 10, 100), dtype=pl.Int64)


def test_streaming_groupby_ooc_q1(monkeypatch: Any, random_integers: pl.Series) -> None:
    s = random_integers
    monkeypatch.setenv("POLARS_FORCE_OOC", "1")

    result = (
        s.to_frame()
        .lazy()
        .groupby("a")
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


def test_streaming_groupby_ooc_q2(monkeypatch: Any, random_integers: pl.Series) -> None:
    s = random_integers
    monkeypatch.setenv("POLARS_FORCE_OOC", "1")

    result = (
        s.cast(str)
        .to_frame()
        .lazy()
        .groupby("a")
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


def test_streaming_groupby_ooc_q3(monkeypatch: Any, random_integers: pl.Series) -> None:
    s = random_integers
    monkeypatch.setenv("POLARS_FORCE_OOC", "1")

    result = (
        pl.DataFrame({"a": s, "b": s})
        .lazy()
        .groupby(["a", "b"])
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


@pytest.mark.slow()
def test_streaming_sort_multiple_columns(
    monkeypatch: Any, capfd: Any, str_ints_df: pl.DataFrame
) -> None:
    monkeypatch.setenv("POLARS_FORCE_OOC", "1")
    monkeypatch.setenv("POLARS_VERBOSE", "1")

    df = str_ints_df

    out = df.lazy().sort(["strs", "vals"]).collect(streaming=True)
    assert_frame_equal(out, out.sort(["strs", "vals"]))
    err = capfd.readouterr().err
    assert "OOC sort forced" in err
    assert "RUN STREAMING PIPELINE" in err
    assert "df -> sort_multiple" in err
    assert out.columns == ["vals", "strs"]


@pytest.mark.slow()
def test_streaming_out_of_core_unique(
    io_files_path: Path, monkeypatch: Any, capfd: Any
) -> None:
    monkeypatch.setenv("POLARS_FORCE_OOC", "1")
    monkeypatch.setenv("POLARS_VERBOSE", "1")
    monkeypatch.setenv("POLARS_STREAMING_GROUPBY_SPILL_SIZE", "256")

    df = pl.read_csv(io_files_path / "foods*.csv")
    # this creates 10M rows
    q = df.lazy()
    q = q.join(q, how="cross").select(df.columns).head(10_000)

    # uses out-of-core unique
    df1 = q.join(q.head(1000), how="cross").unique().collect(streaming=True)
    # this ensures the cross join gives equal result but uses the in-memory unique
    df2 = q.join(q.head(1000), how="cross").collect(streaming=True).unique()
    assert df1.shape == df2.shape
    err = capfd.readouterr().err
    assert "OOC groupby started" in err


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
