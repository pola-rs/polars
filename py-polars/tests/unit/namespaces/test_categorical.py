import polars as pl
from polars.testing import assert_frame_equal


def test_categorical_lexical_sort() -> None:
    df = pl.DataFrame(
        {"cats": ["z", "z", "k", "a", "b"], "vals": [3, 1, 2, 2, 3]}
    ).with_columns(
        [
            pl.col("cats").cast(pl.Categorical).cat.set_ordering("lexical"),
        ]
    )

    out = df.sort(["cats"])
    assert out["cats"].dtype == pl.Categorical
    expected = pl.DataFrame(
        {"cats": ["a", "b", "k", "z", "z"], "vals": [2, 3, 2, 3, 1]}
    )
    assert_frame_equal(out.with_columns(pl.col("cats").cast(pl.Utf8)), expected)
    out = df.sort(["cats", "vals"])
    expected = pl.DataFrame(
        {"cats": ["a", "b", "k", "z", "z"], "vals": [2, 3, 2, 1, 3]}
    )
    assert_frame_equal(out.with_columns(pl.col("cats").cast(pl.Utf8)), expected)
    out = df.sort(["vals", "cats"])

    expected = pl.DataFrame(
        {"cats": ["z", "a", "k", "b", "z"], "vals": [1, 2, 2, 3, 3]}
    )
    assert_frame_equal(out.with_columns(pl.col("cats").cast(pl.Utf8)), expected)


def test_categorical_lexical_ordering_after_concat() -> None:
    with pl.StringCache():
        ldf1 = (
            pl.DataFrame([pl.Series("key1", [8, 5]), pl.Series("key2", ["fox", "baz"])])
            .lazy()
            .with_columns(
                pl.col("key2").cast(pl.Categorical).cat.set_ordering("lexical")
            )
        )
        ldf2 = (
            pl.DataFrame(
                [pl.Series("key1", [6, 8, 6]), pl.Series("key2", ["fox", "foo", "bar"])]
            )
            .lazy()
            .with_columns(
                pl.col("key2").cast(pl.Categorical).cat.set_ordering("lexical")
            )
        )
        df = (
            pl.concat([ldf1, ldf2])
            .with_columns(pl.col("key2").cat.set_ordering("lexical"))
            .collect()
        )

        df.sort(["key1", "key2"])


def test_sort_categoricals_6014() -> None:
    with pl.StringCache():
        # create basic categorical
        df1 = pl.DataFrame({"key": ["bbb", "aaa", "ccc"]}).with_columns(
            pl.col("key").cast(pl.Categorical)
        )
        # create lexically-ordered categorical
        df2 = pl.DataFrame({"key": ["bbb", "aaa", "ccc"]}).with_columns(
            pl.col("key").cast(pl.Categorical).cat.set_ordering("lexical")
        )

    out = df1.sort("key")
    assert out.to_dict(False) == {"key": ["bbb", "aaa", "ccc"]}
    out = df2.sort("key")
    assert out.to_dict(False) == {"key": ["aaa", "bbb", "ccc"]}
