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
