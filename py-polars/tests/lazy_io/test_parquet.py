from os import path

import pandas as pd
import pytest

import polars as pl


@pytest.fixture
def cwd() -> str:
    return path.dirname(__file__)


def test_categorical_parquet_statistics(cwd: str) -> None:
    file = path.join(cwd, "books.parquet")
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


def test_null_parquet(cwd: str) -> None:
    file = path.join(cwd, "null.parquet")
    df = pl.DataFrame([pl.Series("foo", [], dtype=pl.Int8)])
    df.to_parquet(file)
    out = pl.read_parquet(file)
    assert out.frame_equal(df)


def test_binary_parquet_stats(cwd: str) -> None:
    file = path.join(cwd, "binary_stats.parquet")
    df1 = pd.DataFrame({"a": [None, 1, None, 2, 3, 3, 4, 4, 5, 5]})
    df1.to_parquet(file, engine="pyarrow")
    df = (
        pl.scan_parquet(file)
        .filter(pl.col("a").is_not_null() & (pl.col("a") > 4))
        .collect()
    )
    assert df["a"].to_list() == [5.0, 5.0]
