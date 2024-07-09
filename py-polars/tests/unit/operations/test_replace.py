from __future__ import annotations

from typing import Any

import pytest

import polars as pl
from polars.exceptions import InvalidOperationError
from polars.testing import assert_frame_equal, assert_series_equal


@pytest.fixture(scope="module")
def str_mapping() -> dict[str | None, str]:
    return {
        "CA": "Canada",
        "DE": "Germany",
        "FR": "France",
        None: "Not specified",
    }


def test_replace_str_to_str(str_mapping: dict[str | None, str]) -> None:
    df = pl.DataFrame({"country_code": ["FR", None, "ES", "DE"]})
    result = df.select(replaced=pl.col("country_code").replace(str_mapping))
    expected = pl.DataFrame({"replaced": ["France", "Not specified", "ES", "Germany"]})
    assert_frame_equal(result, expected)


def test_replace_enum() -> None:
    dtype = pl.Enum(["a", "b", "c", "d"])
    s = pl.Series(["a", "b", "c"], dtype=dtype)
    old = ["a", "b"]
    new = pl.Series(["c", "d"], dtype=dtype)

    result = s.replace(old, new)

    expected = pl.Series(["c", "d", "c"], dtype=dtype)
    assert_series_equal(result, expected)


def test_replace_enum_to_str() -> None:
    dtype = pl.Enum(["a", "b", "c", "d"])
    s = pl.Series(["a", "b", "c"], dtype=dtype)

    result = s.replace({"a": "c", "b": "d"})

    expected = pl.Series(["c", "d", "c"], dtype=dtype)
    assert_series_equal(result, expected)


@pl.StringCache()
def test_replace_cat_to_cat(str_mapping: dict[str | None, str]) -> None:
    lf = pl.LazyFrame(
        {"country_code": ["FR", None, "ES", "DE"]},
        schema={"country_code": pl.Categorical},
    )
    old = pl.Series(["CA", "DE", "FR", None], dtype=pl.Categorical)
    new = pl.Series(
        ["Canada", "Germany", "France", "Not specified"], dtype=pl.Categorical
    )

    result = lf.select(replaced=pl.col("country_code").replace(old, new))

    expected = pl.LazyFrame(
        {"replaced": ["France", "Not specified", "ES", "Germany"]},
        schema_overrides={"replaced": pl.Categorical},
    )
    assert_frame_equal(result, expected)


def test_replace_invalid_old_dtype() -> None:
    lf = pl.LazyFrame({"a": [1, 2, 3]})
    mapping = {"a": 10, "b": 20}
    with pytest.raises(
        InvalidOperationError, match="conversion from `str` to `i64` failed"
    ):
        lf.select(pl.col("a").replace(mapping)).collect()


def test_replace_int_to_int() -> None:
    df = pl.DataFrame({"int": [None, 1, None, 3]}, schema={"int": pl.Int16})
    mapping = {1: 5, 3: 7}
    result = df.select(replaced=pl.col("int").replace(mapping))
    expected = pl.DataFrame(
        {"replaced": [None, 5, None, 7]}, schema={"replaced": pl.Int16}
    )
    assert_frame_equal(result, expected)


def test_replace_int_to_int_keep_dtype() -> None:
    df = pl.DataFrame({"int": [None, 1, None, 3]}, schema={"int": pl.Int16})
    old = [1, 3]
    new = pl.Series([5, 7], dtype=pl.Int16)

    result = df.select(replaced=pl.col("int").replace(old, new))
    expected = pl.DataFrame(
        {"replaced": [None, 5, None, 7]}, schema={"replaced": pl.Int16}
    )
    assert_frame_equal(result, expected)


def test_replace_int_to_str() -> None:
    df = pl.DataFrame({"int": [None, 1, None, 3]}, schema={"int": pl.Int16})
    mapping = {1: "b", 3: "d"}
    with pytest.raises(
        InvalidOperationError, match="conversion from `str` to `i16` failed"
    ):
        df.select(replaced=pl.col("int").replace(mapping))


def test_replace_int_to_str_with_null() -> None:
    df = pl.DataFrame({"int": [None, 1, None, 3]}, schema={"int": pl.Int16})
    mapping = {1: "b", 3: "d", None: "e"}
    with pytest.raises(
        InvalidOperationError, match="conversion from `str` to `i16` failed"
    ):
        df.select(replaced=pl.col("int").replace(mapping))


def test_replace_empty_mapping() -> None:
    df = pl.DataFrame({"int": [None, 1, None, 3]}, schema={"int": pl.Int16})
    mapping: dict[Any, Any] = {}
    result = df.select(pl.col("int").replace(mapping))
    assert_frame_equal(result, df)


def test_replace_mapping_different_dtype_str_int() -> None:
    df = pl.DataFrame({"int": [None, "1", None, "3"]})
    mapping = {1: "b", 3: "d"}

    result = df.select(pl.col("int").replace(mapping))
    expected = pl.DataFrame({"int": [None, "b", None, "d"]})
    assert_frame_equal(result, expected)


def test_replace_mapping_different_dtype_map_none() -> None:
    df = pl.DataFrame({"int": [None, "1", None, "3"]})
    mapping = {1: "b", 3: "d", None: "e"}
    result = df.select(pl.col("int").replace(mapping))
    expected = pl.DataFrame({"int": ["e", "b", "e", "d"]})
    assert_frame_equal(result, expected)


def test_replace_mapping_different_dtype_str_float() -> None:
    df = pl.DataFrame({"int": [None, "1", None, "3"]})
    mapping = {1.0: "b", 3.0: "d"}

    result = df.select(pl.col("int").replace(mapping))
    assert_frame_equal(result, df)


# https://github.com/pola-rs/polars/issues/7132
def test_replace_str_to_str_replace_all() -> None:
    df = pl.DataFrame({"text": ["abc"]})
    mapping = {"abc": "123"}
    result = df.select(pl.col("text").replace(mapping).str.replace_all("1", "-"))
    expected = pl.DataFrame({"text": ["-23"]})
    assert_frame_equal(result, expected)


@pytest.fixture(scope="module")
def int_mapping() -> dict[int, int]:
    return {1: 11, 2: 22, 3: 33, 4: 44, 5: 55}


def test_replace_int_to_int1(int_mapping: dict[int, int]) -> None:
    s = pl.Series([-1, 22, None, 44, -5])
    result = s.replace(int_mapping)
    expected = pl.Series([-1, 22, None, 44, -5])
    assert_series_equal(result, expected)


def test_replace_int_to_int4(int_mapping: dict[int, int]) -> None:
    s = pl.Series([-1, 22, None, 44, -5])
    result = s.replace(int_mapping)
    expected = pl.Series([-1, 22, None, 44, -5])
    assert_series_equal(result, expected)


# https://github.com/pola-rs/polars/issues/12728
def test_replace_str_to_int2() -> None:
    s = pl.Series(["a", "b"])
    mapping = {"a": 1, "b": 2}
    result = s.replace(mapping)
    expected = pl.Series(["1", "2"])
    assert_series_equal(result, expected)


def test_replace_str_to_bool_without_default() -> None:
    s = pl.Series(["True", "False", "False", None])
    mapping = {"True": True, "False": False}
    result = s.replace(mapping)
    expected = pl.Series(["true", "false", "false", None])
    assert_series_equal(result, expected)


def test_replace_old_new() -> None:
    s = pl.Series([1, 2, 2, 3])
    result = s.replace(2, 9)
    expected = s = pl.Series([1, 9, 9, 3])
    assert_series_equal(result, expected)


def test_replace_old_new_many_to_one() -> None:
    s = pl.Series([1, 2, 2, 3])
    result = s.replace([2, 3], 9)
    expected = s = pl.Series([1, 9, 9, 9])
    assert_series_equal(result, expected)


def test_replace_old_new_mismatched_lengths() -> None:
    s = pl.Series([1, 2, 2, 3, 4])
    with pytest.raises(InvalidOperationError):
        s.replace([2, 3, 4], [8, 9])


def test_replace_fast_path_one_to_one() -> None:
    lf = pl.LazyFrame({"a": [1, 2, 2, 3]})
    result = lf.select(pl.col("a").replace(2, 100))
    expected = pl.LazyFrame({"a": [1, 100, 100, 3]})
    assert_frame_equal(result, expected)


def test_replace_fast_path_one_null_to_one() -> None:
    # https://github.com/pola-rs/polars/issues/13391
    lf = pl.LazyFrame({"a": [1, None]})
    result = lf.select(pl.col("a").replace(None, 100))
    expected = pl.LazyFrame({"a": [1, 100]})
    assert_frame_equal(result, expected)


def test_replace_fast_path_many_with_null_to_one() -> None:
    lf = pl.LazyFrame({"a": [1, 2, None]})
    result = lf.select(pl.col("a").replace([1, None], 100))
    expected = pl.LazyFrame({"a": [100, 2, 100]})
    assert_frame_equal(result, expected)


def test_replace_fast_path_many_to_one() -> None:
    lf = pl.LazyFrame({"a": [1, 2, 2, 3]})
    result = lf.select(pl.col("a").replace([2, 3], 100))
    expected = pl.LazyFrame({"a": [1, 100, 100, 100]})
    assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ([2, 2], 100),
        ([2, 2], [100, 200]),
        ([2, 2], [100, 100]),
    ],
)
def test_replace_duplicates_old(old: list[int], new: int | list[int]) -> None:
    s = pl.Series([1, 2, 3, 2, 3])
    with pytest.raises(
        InvalidOperationError,
        match="`old` input for `replace` must not contain duplicates",
    ):
        s.replace(old, new)


def test_replace_duplicates_new() -> None:
    s = pl.Series([1, 2, 3, 2, 3])
    result = s.replace([1, 2], [100, 100])
    expected = s = pl.Series([100, 100, 3, 100, 3])
    assert_series_equal(result, expected)


def test_replace_return_dtype_deprecated() -> None:
    s = pl.Series([1, 2, 3])
    with pytest.deprecated_call():
        result = s.replace(1, 10, return_dtype=pl.Int8)
    expected = pl.Series([10, 2, 3], dtype=pl.Int8)
    assert_series_equal(result, expected)


def test_replace_default_deprecated() -> None:
    s = pl.Series([1, 2, 3])
    with pytest.deprecated_call():
        result = s.replace(1, 10, default=None)
    expected = pl.Series([10, None, None], dtype=pl.Int32)
    assert_series_equal(result, expected)
