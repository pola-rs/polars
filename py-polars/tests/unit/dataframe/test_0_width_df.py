import polars as pl


def test_0_width_df() -> None:
    df = pl.DataFrame(height=5)

    assert df.clear().height == 0
    assert df.clone().height == 5
    assert df.cast({}).height == 5
    assert df.drop_nans().height == 5
    assert df.drop_nulls().height == 5
    assert df.equals(df)
    assert not df.equals(pl.DataFrame())
    assert df.estimated_size() == 0
    assert df.join(df, how="cross").height == 25

    out = df.hash_rows()
    assert out.value_counts()["count"].item() == 5

    assert pl.concat([df, df]).height == 10


def test_0_width_lf() -> None:
    lf = pl.LazyFrame(height=5)

    assert lf.clear().collect().height == 0
    assert lf.clone().collect().height == 5
    assert lf.cast({}).collect().height == 5
    assert lf.drop_nans().collect().height == 5
    assert lf.drop_nulls().collect().height == 5
    assert lf.join(lf, how="cross").collect().height == 25

    assert pl.concat([lf, lf]).collect().height == 10
