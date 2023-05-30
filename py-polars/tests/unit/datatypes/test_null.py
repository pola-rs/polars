import polars as pl


def test_null_index() -> None:
    df = pl.DataFrame({"a": [[1, 2], [3, 4], [5, 6]], "b": [[1, 2], [1, 2], [4, 5]]})

    df = df.with_columns(pl.lit(None).alias("null_col"))
    assert df[-1].to_dict(False) == {"a": [[5, 6]], "b": [[4, 5]], "null_col": [None]}
