# flake8: noqa: W191,E101
from os import path
from pathlib import Path

import pandas as pd

import polars as pl


def test_scan_parquet() -> None:
    df = pl.scan_parquet(Path(__file__).parent.parent / "files" / "small.parquet")
    assert df.collect().shape == (4, 3)


def test_row_count(foods_parquet: str) -> None:
    df = pl.read_parquet(foods_parquet, row_count_name="row_count")
    assert df["row_count"].to_list() == list(range(27))

    df = (
        pl.scan_parquet(foods_parquet, row_count_name="row_count")
        .filter(pl.col("category") == pl.lit("vegetables"))
        .collect()
    )

    assert df["row_count"].to_list() == [0, 6, 11, 13, 14, 20, 25]

    df = (
        pl.scan_parquet(foods_parquet, row_count_name="row_count")
        .with_row_count("foo", 10)
        .filter(pl.col("category") == pl.lit("vegetables"))
        .collect()
    )

    assert df["foo"].to_list() == [10, 16, 21, 23, 24, 30, 35]


def test_categorical_parquet_statistics(io_test_dir: str) -> None:
    file = path.join(io_test_dir, "books.parquet")
    (
        pl.DataFrame(
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
        )
        .with_column(pl.col("book").cast(pl.Categorical))
        .to_parquet(file, statistics=True)
    )

    for par in [True, False]:
        df = (
            pl.scan_parquet(file, parallel=par)
            .filter(pl.col("book") == "bookA")
            .collect()
        )
    assert df.shape == (4, 3)


def test_null_parquet(io_test_dir: str) -> None:
    file = path.join(io_test_dir, "null.parquet")
    df = pl.DataFrame([pl.Series("foo", [], dtype=pl.Int8)])
    df.to_parquet(file)
    out = pl.read_parquet(file)
    assert out.frame_equal(df)


def test_parquet_stats(io_test_dir: str) -> None:
    file = path.join(io_test_dir, "binary_stats.parquet")
    df1 = pd.DataFrame({"a": [None, 1, None, 2, 3, 3, 4, 4, 5, 5]})
    df1.to_parquet(file, engine="pyarrow")
    df = (
        pl.scan_parquet(file)
        .filter(pl.col("a").is_not_null() & (pl.col("a") > 4))
        .collect()
    )
    assert df["a"].to_list() == [5.0, 5.0]

    assert (
        pl.scan_parquet(file).filter(pl.col("a") > 4).select(pl.col("a").sum())
    ).collect()[0, "a"] == 10.0

    assert (
        pl.scan_parquet(file).filter(pl.col("a") < 4).select(pl.col("a").sum())
    ).collect()[0, "a"] == 9.0

    assert (
        pl.scan_parquet(file).filter(4 > pl.col("a")).select(pl.col("a").sum())
    ).collect()[0, "a"] == 9.0

    assert (
        pl.scan_parquet(file).filter(4 < pl.col("a")).select(pl.col("a").sum())
    ).collect()[0, "a"] == 10.0
