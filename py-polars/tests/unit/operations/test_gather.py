import polars as pl


def test_negative_index_select() -> None:
    df = pl.DataFrame({"a": [[1, 2, 3], [4, 5, 6]]})
    assert df.select(pl.col("a").gather([0, -1])).to_dict(as_series=False) == {
        "a": [[1, 2, 3], [4, 5, 6]]
    }
