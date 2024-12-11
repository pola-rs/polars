from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal

if TYPE_CHECKING:
    from collections.abc import Generator
    from typing import Any

    FixtureRequest = Any


@pytest.fixture(params=[True, False])
def test_global_and_local(
    request: FixtureRequest,
) -> Generator[Any, Any, Any]:
    """Setup fixture which runs each test with and without global string cache."""
    use_global = request.param
    if use_global:
        with pl.StringCache():
            # Pre-fill some global items to ensure physical repr isn't 0..n.
            pl.Series(["a", "b", "c"], dtype=pl.Categorical)
            yield
    else:
        yield


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


@pytest.mark.may_fail_auto_streaming
def test_sort_categoricals_6014_internal() -> None:
    with pl.StringCache():
        # create basic categorical
        df = pl.DataFrame({"key": ["bbb", "aaa", "ccc"]}).with_columns(
            pl.col("key").cast(pl.Categorical)
        )

    out = df.sort("key")
    assert out.to_dict(as_series=False) == {"key": ["bbb", "aaa", "ccc"]}


def test_sort_categoricals_6014_lexical() -> None:
    with pl.StringCache():
        # create lexically-ordered categorical
        df = pl.DataFrame({"key": ["bbb", "aaa", "ccc"]}).with_columns(
            pl.col("key").cast(pl.Categorical("lexical"))
        )

    out = df.sort("key")
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


@pytest.mark.usefixtures("test_global_and_local")
def test_cat_len_bytes() -> None:
    # test Series
    s = pl.Series("a", ["Café", None, "Café", "345", "東京"], dtype=pl.Categorical)
    result = s.cat.len_bytes()
    expected = pl.Series("a", [5, None, 5, 3, 6], dtype=pl.UInt32)
    assert_series_equal(result, expected)

    # test DataFrame expr
    df = pl.DataFrame(s)
    result_df = df.select(pl.col("a").cat.len_bytes())
    expected_df = pl.DataFrame(expected)
    assert_frame_equal(result_df, expected_df)

    # test LazyFrame expr
    result_lf = df.lazy().select(pl.col("a").cat.len_bytes()).collect()
    assert_frame_equal(result_lf, expected_df)

    # test GroupBy
    result_df = (
        pl.LazyFrame({"key": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2], "value": s.extend(s)})
        .group_by("key", maintain_order=True)
        .agg(pl.col("value").cat.len_bytes().alias("len_bytes"))
        .explode("len_bytes")
        .collect()
    )
    expected_df = pl.DataFrame(
        {
            "key": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
            "len_bytes": pl.Series(
                [5, None, 5, 3, 6, 5, None, 5, 3, 6], dtype=pl.get_index_type()
            ),
        }
    )
    assert_frame_equal(result_df, expected_df)


@pytest.mark.usefixtures("test_global_and_local")
def test_cat_len_chars() -> None:
    # test Series
    s = pl.Series("a", ["Café", None, "Café", "345", "東京"], dtype=pl.Categorical)
    result = s.cat.len_chars()
    expected = pl.Series("a", [4, None, 4, 3, 2], dtype=pl.UInt32)
    assert_series_equal(result, expected)

    # test DataFrame expr
    df = pl.DataFrame(s)
    result_df = df.select(pl.col("a").cat.len_chars())
    expected_df = pl.DataFrame(expected)
    assert_frame_equal(result_df, expected_df)

    # test LazyFrame expr
    result_lf = df.lazy().select(pl.col("a").cat.len_chars()).collect()
    assert_frame_equal(result_lf, expected_df)

    # test GroupBy
    result_df = (
        pl.LazyFrame({"key": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2], "value": s.extend(s)})
        .group_by("key", maintain_order=True)
        .agg(pl.col("value").cat.len_chars().alias("len_bytes"))
        .explode("len_bytes")
        .collect()
    )
    expected_df = pl.DataFrame(
        {
            "key": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
            "len_bytes": pl.Series(
                [4, None, 4, 3, 2, 4, None, 4, 3, 2], dtype=pl.get_index_type()
            ),
        }
    )
    assert_frame_equal(result_df, expected_df)


@pytest.mark.usefixtures("test_global_and_local")
def test_starts_ends_with() -> None:
    s = pl.Series(
        "a",
        ["hamburger_with_tomatoes", "nuts", "nuts", "lollypop", None],
        dtype=pl.Categorical,
    )
    assert_series_equal(
        s.cat.ends_with("pop"), pl.Series("a", [False, False, False, True, None])
    )
    assert_series_equal(
        s.cat.starts_with("nu"), pl.Series("a", [False, True, True, False, None])
    )
    assert_series_equal(
        s.cat.starts_with(None),
        pl.Series("a", [None, None, None, None, None], dtype=pl.Boolean),
    )

    df = pl.DataFrame(
        {
            "a": pl.Series(
                ["hamburger_with_tomatoes", "nuts", "nuts", "lollypop", None],
                dtype=pl.Categorical,
            ),
            "sub": ["ham", "ts", "nu", None, "anything"],
        }
    )

    assert df.select(
        pl.col("a").cat.ends_with("pop").alias("ends_pop"),
        pl.col("a").cat.ends_with(pl.lit(None)).alias("ends_None"),
        pl.col("a").cat.ends_with(pl.col("sub")).alias("ends_sub"),
        pl.col("a").cat.starts_with("ham").alias("starts_ham"),
        pl.col("a").cat.starts_with(pl.lit(None)).alias("starts_None"),
        pl.col("a").cat.starts_with(pl.col("sub")).alias("starts_sub"),
    ).to_dict(as_series=False) == {
        "ends_pop": [False, False, False, True, None],
        "ends_None": [None, None, None, None, None],
        "ends_sub": [False, True, False, None, None],
        "starts_ham": [True, False, False, False, None],
        "starts_None": [None, None, None, None, None],
        "starts_sub": [True, False, True, None, None],
    }
