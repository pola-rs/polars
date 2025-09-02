import polars as pl


def test_hash_struct() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    df = df.select(pl.struct(pl.all()))
    assert df.select(pl.col("a").hash())["a"].to_list() == [
        13234032676922527816,
        14038929224386803749,
        13596073670754281839,
    ]
