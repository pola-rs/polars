import polars as pl
from polars.testing import assert_frame_equal


def test_categorical_lexical_sort() -> None:
    df = pl.DataFrame(
        {"cats": ["z", "z", "k", "a", "b"], "vals": [3, 1, 2, 2, 3]}
    ).with_columns(
        pl.col("cats").cast(pl.Categorical("lexical")),
    )

    out = df.sort(["cats"])
    assert out["cats"].dtype == pl.Categorical
    expected = pl.DataFrame(
        {"cats": ["a", "b", "k", "z", "z"], "vals": [2, 3, 2, 3, 1]}
    )
    assert_frame_equal(out.with_columns(pl.col("cats").cast(pl.String)), expected)
    out = df.sort(["cats", "vals"])
    expected = pl.DataFrame(
        {"cats": ["a", "b", "k", "z", "z"], "vals": [2, 3, 2, 1, 3]}
    )
    assert_frame_equal(out.with_columns(pl.col("cats").cast(pl.String)), expected)
    out = df.sort(["vals", "cats"])

    expected = pl.DataFrame(
        {"cats": ["z", "a", "k", "b", "z"], "vals": [1, 2, 2, 3, 3]}
    )
    assert_frame_equal(out.with_columns(pl.col("cats").cast(pl.String)), expected)

    s = pl.Series(["a", "c", "a", "b", "a"], dtype=pl.Categorical("lexical"))
    assert s.sort().cast(pl.String).to_list() == [
        "a",
        "a",
        "a",
        "b",
        "c",
    ]


def test_categorical_lexical_ordering_after_concat() -> None:
    with pl.StringCache():
        ldf1 = (
            pl.DataFrame([pl.Series("key1", [8, 5]), pl.Series("key2", ["fox", "baz"])])
            .lazy()
            .with_columns(pl.col("key2").cast(pl.Categorical("lexical")))
        )
        ldf2 = (
            pl.DataFrame(
                [pl.Series("key1", [6, 8, 6]), pl.Series("key2", ["fox", "foo", "bar"])]
            )
            .lazy()
            .with_columns(pl.col("key2").cast(pl.Categorical("lexical")))
        )
        df = pl.concat([ldf1, ldf2]).select(pl.col("key2")).collect()

        assert df.sort("key2").to_dict(as_series=False) == {
            "key2": ["bar", "baz", "foo", "fox", "fox"]
        }


def test_sort_categoricals_6014() -> None:
    with pl.StringCache():
        # create basic categorical
        df1 = pl.DataFrame({"key": ["bbb", "aaa", "ccc"]}).with_columns(
            pl.col("key").cast(pl.Categorical)
        )
        # create lexically-ordered categorical
        df2 = pl.DataFrame({"key": ["bbb", "aaa", "ccc"]}).with_columns(
            pl.col("key").cast(pl.Categorical("lexical"))
        )

    out = df1.sort("key")
    assert out.to_dict(as_series=False) == {"key": ["bbb", "aaa", "ccc"]}
    out = df2.sort("key")
    assert out.to_dict(as_series=False) == {"key": ["aaa", "bbb", "ccc"]}


def test_categorical_get_categories() -> None:
    assert pl.Series(
        "cats", ["foo", "bar", "foo", "foo", "ham"], dtype=pl.Categorical
    ).cat.get_categories().to_list() == ["foo", "bar", "ham"]


def test_cat_to_local() -> None:
    with pl.StringCache():
        s1 = pl.Series(["a", "b", "a"], dtype=pl.Categorical)
        s2 = pl.Series(["c", "b", "d"], dtype=pl.Categorical)

    # s2 physical starts after s1
    assert s1.to_physical().to_list() == [0, 1, 0]
    assert s2.to_physical().to_list() == [2, 1, 3]

    out = s2.cat.to_local()

    # Physical has changed and now starts at 0, string values are the same
    assert out.cat.is_local()
    assert out.to_physical().to_list() == [0, 1, 2]
    assert out.to_list() == s2.to_list()

    # s2 should be unchanged after the operation
    assert not s2.cat.is_local()
    assert s2.to_physical().to_list() == [2, 1, 3]
    assert s2.to_list() == ["c", "b", "d"]


def test_cat_to_local_missing_values() -> None:
    with pl.StringCache():
        _ = pl.Series(["a", "b"], dtype=pl.Categorical)
        s = pl.Series(["c", "b", None, "d"], dtype=pl.Categorical)

    out = s.cat.to_local()
    assert out.to_physical().to_list() == [0, 1, None, 2]


def test_cat_to_local_already_local() -> None:
    s = pl.Series(["a", "c", "a", "b"], dtype=pl.Categorical)

    assert s.cat.is_local()
    out = s.cat.to_local()

    assert out.to_physical().to_list() == [0, 1, 0, 2]
    assert out.to_list() == ["a", "c", "a", "b"]


def test_cat_is_local() -> None:
    s = pl.Series(["a", "c", "a", "b"], dtype=pl.Categorical)
    assert s.cat.is_local()

    with pl.StringCache():
        s2 = pl.Series(["a", "b", "a"], dtype=pl.Categorical)
    assert not s2.cat.is_local()


def test_cat_uses_lexical_ordering() -> None:
    s = pl.Series(["a", "b", None, "b"]).cast(pl.Categorical)
    assert s.cat.uses_lexical_ordering() is False

    s = s.cast(pl.Categorical("lexical"))
    assert s.cat.uses_lexical_ordering() is True

    s = s.cast(pl.Categorical("physical"))
    assert s.cat.uses_lexical_ordering() is False
