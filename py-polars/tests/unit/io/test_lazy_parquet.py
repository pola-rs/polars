from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import pytest

import polars as pl
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from polars.internals.type_aliases import ParallelStrategy


@pytest.fixture()
def parquet_file_path(io_files_path: Path) -> Path:
    return io_files_path / "small.parquet"


@pytest.fixture()
def foods_parquet_path(io_files_path: Path) -> Path:
    return io_files_path / "foods1.parquet"


def test_scan_parquet(parquet_file_path: Path) -> None:
    df = pl.scan_parquet(parquet_file_path)
    assert df.collect().shape == (4, 3)


def test_row_count(foods_parquet_path: Path) -> None:
    df = pl.read_parquet(foods_parquet_path, row_count_name="row_count")
    assert df["row_count"].to_list() == list(range(27))

    df = (
        pl.scan_parquet(foods_parquet_path, row_count_name="row_count")
        .filter(pl.col("category") == pl.lit("vegetables"))
        .collect()
    )

    assert df["row_count"].to_list() == [0, 6, 11, 13, 14, 20, 25]

    df = (
        pl.scan_parquet(foods_parquet_path, row_count_name="row_count")
        .with_row_count("foo", 10)
        .filter(pl.col("category") == pl.lit("vegetables"))
        .collect()
    )

    assert df["foo"].to_list() == [10, 16, 21, 23, 24, 30, 35]


def test_categorical_parquet_statistics() -> None:
    df = pl.DataFrame(
        {
            "book": [
                "bookA",
                "bookA",
                "bookB",
                "bookA",
                "bookA",
                "bookC",
                "bookC",
                "bookC",
            ],
            "transaction_id": [1, 2, 3, 4, 5, 6, 7, 8],
            "user": ["bob", "bob", "bob", "tim", "lucy", "lucy", "lucy", "lucy"],
        }
    ).with_columns(pl.col("book").cast(pl.Categorical))

    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "books.parquet"
        df.write_parquet(file_path, statistics=True)

        parallel_options: list[ParallelStrategy] = [
            "auto",
            "columns",
            "row_groups",
            "none",
        ]
        for par in parallel_options:
            df = (
                pl.scan_parquet(file_path, parallel=par)
                .filter(pl.col("book") == "bookA")
                .collect()
            )
        assert df.shape == (4, 3)


def test_null_parquet() -> None:
    df = pl.DataFrame([pl.Series("foo", [], dtype=pl.Int8)])
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "null.parquet"
        df.write_parquet(file_path)
        out = pl.read_parquet(file_path)
    assert_frame_equal(out, df)


def test_parquet_stats() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "binary_stats.parquet"

        df1 = pd.DataFrame({"a": [None, 1, None, 2, 3, 3, 4, 4, 5, 5]})
        df1.to_parquet(file_path, engine="pyarrow")
        df = (
            pl.scan_parquet(file_path)
            .filter(pl.col("a").is_not_null() & (pl.col("a") > 4))
            .collect()
        )
        assert df["a"].to_list() == [5.0, 5.0]

        assert (
            pl.scan_parquet(file_path).filter(pl.col("a") > 4).select(pl.col("a").sum())
        ).collect()[0, "a"] == 10.0

        assert (
            pl.scan_parquet(file_path).filter(pl.col("a") < 4).select(pl.col("a").sum())
        ).collect()[0, "a"] == 9.0

        assert (
            pl.scan_parquet(file_path).filter(pl.col("a") < 4).select(pl.col("a").sum())
        ).collect()[0, "a"] == 9.0

        assert (
            pl.scan_parquet(file_path).filter(pl.col("a") > 4).select(pl.col("a").sum())
        ).collect()[0, "a"] == 10.0
        assert pl.scan_parquet(file_path).filter(
            (pl.col("a") * 10) > 5.0
        ).collect().shape == (8, 1)


def test_row_count_schema(parquet_file_path: Path) -> None:
    assert (
        pl.scan_parquet(str(parquet_file_path), row_count_name="id")
        .select(["id", "b"])
        .collect()
    ).dtypes == [pl.UInt32, pl.Utf8]


def test_parquet_statistics(monkeypatch: Any, capfd: Any) -> None:
    monkeypatch.setenv("POLARS_VERBOSE", "1")

    df = pl.DataFrame({"idx": pl.arange(0, 100, eager=True)}).with_columns(
        (pl.col("idx") // 25).alias("part")
    )
    df = pl.concat(df.partition_by("part", as_dict=False), rechunk=False)
    assert df.n_chunks("all") == [4, 4]

    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "stats.parquet"
        df.write_parquet(file_path, statistics=True, use_pyarrow=False)

        for pred in [
            pl.col("idx") < 50,
            pl.col("idx") > 50,
            pl.col("idx").null_count() != 0,
            pl.col("idx").null_count() == 0,
            pl.col("idx").min() == pl.col("part").null_count(),
        ]:
            result = pl.scan_parquet(file_path).filter(pred).collect()
            assert_frame_equal(result, df.filter(pred))

    captured = capfd.readouterr().err
    assert (
        "parquet file must be read, statistics not sufficient for predicate."
        in captured
    )
    assert (
        "parquet file can be skipped, the statistics were sufficient"
        " to apply the predicate." in captured
    )


@pytest.mark.xfail(sys.platform == "win32", reason="Does not work on Windows")
def test_streaming_categorical() -> None:
    df = pl.DataFrame(
        [
            pl.Series("name", ["Bob", "Alice", "Bob"], pl.Categorical),
            pl.Series("amount", [100, 200, 300]),
        ]
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "categorical.parquet"
        df.write_parquet(file_path)

        with pl.StringCache():
            result = (
                pl.scan_parquet(file_path)
                .groupby("name")
                .agg(pl.col("amount").sum())
                .collect()
            )
            expected = pl.DataFrame(
                {"name": ["Bob", "Alice"], "amount": [400, 200]},
                schema_overrides={"name": pl.Categorical},
            )
            assert_frame_equal(result, expected)


@pytest.mark.xfail(sys.platform == "win32", reason="Does not work on Windows")
def test_parquet_struct_categorical() -> None:
    df = pl.DataFrame(
        [
            pl.Series("a", ["bob"], pl.Categorical),
            pl.Series("b", ["foo"], pl.Categorical),
        ]
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "categorical.parquet"
        df.write_parquet(file_path)

        with pl.StringCache():
            out = pl.read_parquet(file_path).select(pl.col("b").value_counts())
        assert out.to_dict(False) == {"b": [{"b": "foo", "counts": 1}]}
