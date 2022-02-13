from os import path

import polars as pl


def test_categorical_parquet_statistics() -> None:
    file = path.join(path.dirname(__file__), "books.parquet")
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


def test_null_parquet() -> None:
    file = path.join(path.dirname(__file__), "null.parquet")
    df = pl.DataFrame([pl.Series("foo", [], dtype=pl.Int8)])
    df.to_parquet(file)
    out = pl.read_parquet(file)
    assert out.frame_equal(df)
