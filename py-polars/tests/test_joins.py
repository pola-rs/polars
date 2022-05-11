import polars as pl


def test_semi_anti_join() -> None:
    df_a = pl.DataFrame({"key": [1, 2, 3], "payload": ["f", "i", None]})

    df_b = pl.DataFrame({"key": [3, 4, 5, None]})

    assert df_a.join(df_b, on="key", how="anti").to_dict(False) == {
        "key": [1, 2],
        "payload": ["f", "i"],
    }
    assert df_a.join(df_b, on="key", how="semi").to_dict(False) == {
        "key": [3],
        "payload": [None],
    }

    # lazy
    assert df_a.lazy().join(df_b.lazy(), on="key", how="anti").collect().to_dict(
        False
    ) == {
        "key": [1, 2],
        "payload": ["f", "i"],
    }
    assert df_a.lazy().join(df_b.lazy(), on="key", how="semi").collect().to_dict(
        False
    ) == {
        "key": [3],
        "payload": [None],
    }

    df_a = pl.DataFrame(
        {"a": [1, 2, 3, 1], "b": ["a", "b", "c", "a"], "payload": [10, 20, 30, 40]}
    )

    df_b = pl.DataFrame({"a": [3, 3, 4, 5], "b": ["c", "c", "d", "e"]})

    assert df_a.join(df_b, on=["a", "b"], how="anti").to_dict(False) == {
        "a": [1, 2, 1],
        "b": ["a", "b", "a"],
        "payload": [10, 20, 40],
    }
    assert df_a.join(df_b, on=["a", "b"], how="semi").to_dict(False) == {
        "a": [3],
        "b": ["c"],
        "payload": [30],
    }


def test_join_same_cat_src() -> None:
    df = pl.DataFrame(
        data={"column": ["a", "a", "b"], "more": [1, 2, 3]},
        columns=[("column", pl.Categorical), ("more", pl.Int32)],
    )
    df_agg = df.groupby("column").agg(pl.col("more").mean())
    assert df.join(df_agg, on="column").to_dict(False) == {
        "column": ["a", "a", "b"],
        "more": [1, 2, 3],
        "more_right": [1.5, 1.5, 3.0],
    }
