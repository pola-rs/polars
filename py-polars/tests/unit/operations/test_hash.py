import polars as pl


def test_hash_struct() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    df = df.select(pl.struct(pl.all()))
    assert df.select(pl.col("a").hash())["a"].to_list() == [
        2509182728763614268,
        6116564025432436932,
        49592145888590321,
    ]
