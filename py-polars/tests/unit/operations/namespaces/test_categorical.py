from __future__ import annotations

from io import BytesIO
from typing import TYPE_CHECKING

import pytest

import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal

if TYPE_CHECKING:
    from polars._typing import PolarsDataType


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


def test_sort_categoricals_6014_lexical() -> None:
    # create lexically-ordered categorical
    df = pl.DataFrame({"key": ["bbb", "aaa", "ccc"]}).with_columns(
        pl.col("key").cast(pl.Categorical("lexical"))
    )

    out = df.sort("key")
    assert out.to_dict(as_series=False) == {"key": ["aaa", "bbb", "ccc"]}


def test_categorical_get_categories() -> None:
    s = pl.Series("cats", ["foo", "bar", "foo", "foo", "ham"], dtype=pl.Categorical)
    assert set(s.cat.get_categories().to_list()) >= {"foo", "bar", "ham"}


def test_cat_to_local() -> None:
    s = pl.Series(["a", "b", "a"], dtype=pl.Categorical)
    assert_series_equal(s, s.cat.to_local())


def test_cat_uses_lexical_ordering() -> None:
    s = pl.Series(["a", "b", None, "b"]).cast(pl.Categorical)
    assert s.cat.uses_lexical_ordering()

    s = s.cast(pl.Categorical("lexical"))
    assert s.cat.uses_lexical_ordering()

    with pytest.warns(DeprecationWarning):
        s = s.cast(pl.Categorical("physical"))  # Deprecated.
        assert s.cat.uses_lexical_ordering()


@pytest.mark.parametrize("dtype", [pl.Categorical, pl.Enum])
def test_cat_len_bytes(dtype: PolarsDataType) -> None:
    # test Series
    values = ["Café", None, "Café", "345", "東京"]
    if dtype == pl.Enum:
        dtype = pl.Enum(list({x for x in values if x is not None}))
    s = pl.Series("a", values, dtype=dtype)
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


@pytest.mark.parametrize("dtype", [pl.Categorical, pl.Enum])
def test_cat_len_chars(dtype: PolarsDataType) -> None:
    values = ["Café", None, "Café", "345", "東京"]
    if dtype == pl.Enum:
        dtype = pl.Enum(list({x for x in values if x is not None}))
    # test Series
    s = pl.Series("a", values, dtype=dtype)
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


@pytest.mark.parametrize("dtype", [pl.Categorical, pl.Enum])
def test_starts_ends_with(dtype: PolarsDataType) -> None:
    values = ["hamburger_with_tomatoes", "nuts", "nuts", "lollypop", None]
    if dtype == pl.Enum:
        dtype = pl.Enum(list({x for x in values if x is not None}))
    s = pl.Series("a", values, dtype=dtype)
    assert_series_equal(
        s.cat.ends_with("pop"), pl.Series("a", [False, False, False, True, None])
    )
    assert_series_equal(
        s.cat.starts_with("nu"), pl.Series("a", [False, True, True, False, None])
    )

    with pytest.raises(TypeError, match="'prefix' must be a string; found"):
        s.cat.starts_with(None)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="'suffix' must be a string; found"):
        s.cat.ends_with(None)  # type: ignore[arg-type]

    df = pl.DataFrame({"a": pl.Series(values, dtype=dtype)})

    expected = {
        "ends_pop": [False, False, False, True, None],
        "starts_ham": [True, False, False, False, None],
    }

    assert (
        df.select(
            pl.col("a").cat.ends_with("pop").alias("ends_pop"),
            pl.col("a").cat.starts_with("ham").alias("starts_ham"),
        ).to_dict(as_series=False)
        == expected
    )

    with pytest.raises(TypeError, match="'prefix' must be a string; found"):
        df.select(pl.col("a").cat.starts_with(None))  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="'suffix' must be a string; found"):
        df.select(pl.col("a").cat.ends_with(None))  # type: ignore[arg-type]


@pytest.mark.parametrize("dtype", [pl.Categorical, pl.Enum])
def test_cat_slice(dtype: PolarsDataType) -> None:
    values = ["foobar", "barfoo", "foobar", "x", None]
    if dtype == pl.Enum:
        dtype = pl.Enum(list({x for x in values if x is not None}))
    df = pl.DataFrame({"a": pl.Series(values, dtype=dtype)})
    assert df["a"].cat.slice(-3).to_list() == ["bar", "foo", "bar", "x", None]
    assert df.select([pl.col("a").cat.slice(2, 4)])["a"].to_list() == [
        "obar",
        "rfoo",
        "obar",
        "",
        None,
    ]


def test_cat_order_flag_csv_read_23823() -> None:
    data = BytesIO(b"colx,coly\nabc,123\n#not_a_row\nxyz,456")
    lf = pl.scan_csv(
        source=data,
        comment_prefix="#",
        schema_overrides={"colx": pl.Categorical},
    )
    expected = pl.DataFrame(
        {"colx": ["abc", "xyz"], "coly": [123, 456]},
        schema_overrides={"colx": pl.Categorical},
    )
    assert_frame_equal(expected, lf.sort("colx", descending=False).collect())
