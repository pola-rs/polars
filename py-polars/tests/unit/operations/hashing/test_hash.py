import polars as pl


def test_hash_struct() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    df = df.select(pl.struct(pl.all()))
    assert df.select(pl.col("a").hash())["a"].to_list() == [
        5535262844797696299,
        15139341575481673729,
        12593759486533989774,
    ]


def test_0_width_df() -> None:
    df = pl.DataFrame(height=5)

    assert df.clear().height == 0
    assert df.clone().height == 5
    assert df.cast({}).height == 5
    assert df.equals(df)
    assert not df.equals(pl.DataFrame())
    assert df.estimated_size() == 0
    assert df.join(df, how="cross").height == 25

    out = df.hash_rows()
    assert out.value_counts()["count"].item() == 5

    assert pl.concat(2 * [pl.DataFrame(height=5)]).height == 10
