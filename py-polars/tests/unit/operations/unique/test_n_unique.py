import polars as pl


def test_n_unique() -> None:
    s = pl.Series("s", [11, 11, 11, 22, 22, 33, None, None, None])
    assert s.n_unique() == 4


def test_n_unique_subsets() -> None:
    df = pl.DataFrame(
        {
            "a": [1, 1, 2, 3, 4, 5],
            "b": [0.5, 0.5, 1.0, 2.0, 3.0, 3.0],
            "c": [True, True, True, False, True, True],
        }
    )
    # omitting 'subset' counts unique rows
    assert df.n_unique() == 5

    # providing it counts unique col/expr subsets
    assert df.n_unique(subset=["b", "c"]) == 4
    assert df.n_unique(subset=pl.col("c")) == 2
    assert (
        df.n_unique(subset=[(pl.col("a") // 2), (pl.col("c") | (pl.col("b") >= 2))])
        == 3
    )


def test_n_unique_null() -> None:
    assert pl.Series([]).n_unique() == 0
    assert pl.Series([None]).n_unique() == 1
    assert pl.Series([None, None]).n_unique() == 1
