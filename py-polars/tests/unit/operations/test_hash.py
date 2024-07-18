import polars as pl


def test_hash_struct() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    df = df.select(pl.struct(pl.all()))

    assert df.select(pl.col("a").hash())["a"].to_list() == [
        8045264196180950307,
        14608543421872010777,
        12464129093563214397,
    ]
