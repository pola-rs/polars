import polars as pl


def test_0_width_df() -> None:
    df = pl.DataFrame(height=5)

    assert df.bottom_k(3, by="*").height == 3
    assert df.clear().height == 0
    assert df.clone().height == 5
    assert df.cast({}).height == 5
    assert df.drop_nans().height == 5
    assert df.drop_nulls().height == 5
    assert df.equals(df)
    assert not df.equals(pl.DataFrame())
    assert df.estimated_size() == 0
    assert df.fill_null(0).height == 5
    assert df.fill_null(strategy="forward").height == 5
    assert df.gather_every(1).height == 5
    assert df.gather_every(5).height == 1
    assert df.interpolate().height == 5
    assert df.join(df, how="cross").height == 25
    assert df.reverse().height == 5
    assert df.shift(1).height == 5
    assert df.to_dummies().height == 5
    assert df.unique().height == 1

    # Aggregations reduce to a single row.
    assert df.count().height == 1
    assert df.max().height == 1
    assert df.mean().height == 1
    assert df.median().height == 1
    assert df.min().height == 1
    assert df.null_count().height == 1
    assert df.product().height == 1
    assert df.quantile(0.5).height == 1
    assert df.std().height == 1
    assert df.sum().height == 1
    assert df.var().height == 1

    # Comparison and arithmetic operators are element-wise.
    assert (df == 1).height == 5
    assert (df != 1).height == 5
    assert (df > 1).height == 5
    assert (df < 1).height == 5
    assert (df >= 1).height == 5
    assert (df <= 1).height == 5
    assert (df + 1).height == 5
    assert (df - 1).height == 5
    assert (df * 2).height == 5
    assert (df / 2).height == 5
    assert (df // 2).height == 5
    assert (df % 2).height == 5

    # With no columns every row is identical.
    assert df.is_duplicated().to_list() == [True] * 5
    assert df.is_unique().to_list() == [False] * 5
    assert df.n_unique() == 1

    out = df.hash_rows()
    assert out.value_counts()["count"].item() == 5

    assert pl.concat([df, df]).height == 10


def test_0_width_lf() -> None:
    lf = pl.LazyFrame(height=5)

    assert lf.bottom_k(3, by="*").collect().height == 3
    assert lf.clear().collect().height == 0
    assert lf.clone().collect().height == 5
    assert lf.cast({}).collect().height == 5
    assert lf.drop_nans().collect().height == 5
    assert lf.drop_nulls().collect().height == 5
    assert lf.fill_null(0).collect().height == 5
    assert lf.fill_null(strategy="forward").collect().height == 5
    assert lf.gather_every(1).collect().height == 5
    assert lf.gather_every(5).collect().height == 1
    assert lf.interpolate().collect().height == 5
    assert lf.join(lf, how="cross").collect().height == 25
    assert lf.reverse().collect().height == 5
    assert lf.shift(1).collect().height == 5
    assert lf.unique().collect().height == 1

    # Aggregations reduce to a single row.
    assert lf.count().collect().height == 1
    assert lf.max().collect().height == 1
    assert lf.mean().collect().height == 1
    assert lf.median().collect().height == 1
    assert lf.min().collect().height == 1
    assert lf.null_count().collect().height == 1
    assert lf.quantile(0.5).collect().height == 1
    assert lf.std().collect().height == 1
    assert lf.sum().collect().height == 1
    assert lf.var().collect().height == 1

    assert pl.concat([lf, lf]).collect().height == 10


def test_0_width_0_height() -> None:
    df = pl.DataFrame(height=0)

    assert df.reverse().height == 0
    assert df.shift(1).height == 0
    assert df.interpolate().height == 0
    assert df.gather_every(2).height == 0

    # Aggregating an empty frame still yields a single row.
    assert df.count().height == 1
    assert df.max().height == 1
    assert df.sum().height == 1

    assert df.unique().height == 0
    assert pl.LazyFrame(height=0).unique().collect().height == 0
    assert df.is_duplicated().to_list() == []
    assert df.is_unique().to_list() == []
    assert df.n_unique() == 0
