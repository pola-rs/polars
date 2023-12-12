from __future__ import annotations

from typing import Any

import pytest

import polars as pl
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


def test_replace_str_to_str_default_self(str_mapping: dict[str | None, str]) -> None:
    df = pl.DataFrame({"country_code": ["FR", None, "ES", "DE"]})
    result = df.select(
        replaced=pl.col("country_code").replace(
            str_mapping, default=pl.col("country_code")
        )
    )
    expected = pl.DataFrame({"replaced": ["France", "Not specified", "ES", "Germany"]})
    assert_frame_equal(result, expected)


def test_replace_str_to_str_default_null(str_mapping: dict[str | None, str]) -> None:
    df = pl.DataFrame({"country_code": ["FR", None, "ES", "DE"]})
    result = df.select(
        replaced=pl.col("country_code").replace(str_mapping, default=None)
    )
    expected = pl.DataFrame({"replaced": ["France", "Not specified", None, "Germany"]})
    assert_frame_equal(result, expected)


def test_replace_str_to_str_default_other(str_mapping: dict[str | None, str]) -> None:
    df = pl.DataFrame({"country_code": ["FR", None, "ES", "DE"]})

    result = df.with_row_count().select(
        replaced=pl.col("country_code").replace(str_mapping, default=pl.col("row_nr"))
    )
    expected = pl.DataFrame({"replaced": ["France", "Not specified", "2", "Germany"]})
    assert_frame_equal(result, expected)


# TODO: Check return dtype
@pl.StringCache()
def test_replace_cat_to_cat(str_mapping: dict[str | None, str]) -> None:
    lf = pl.LazyFrame(
        {"country_code": ["FR", None, "ES", "DE"]},
        schema={"country_code": pl.Categorical},
    )

    result = lf.select(
        replaced=pl.col("country_code").cast(pl.Categorical).replace(str_mapping)
    )
    expected = pl.LazyFrame(
        {"replaced": ["France", "Not specified", "ES", "Germany"]},
        schema={"replaced": pl.Utf8},
    )
    assert_frame_equal(result, expected)


# TODO: Check return dtype
def test_replace_int_to_int23() -> None:
    df = pl.DataFrame({"int": [None, 1, None, 3]}, schema={"int": pl.Int16})
    mapping = {1: 5, 3: 7}
    result = df.select(replaced=pl.col("int").replace(mapping))
    expected = pl.DataFrame(
        {"replaced": [None, 5, None, 7]}, schema={"replaced": pl.Int64}
    )
    assert_frame_equal(result, expected)


def test_replace_int_to_str2() -> None:
    df = pl.DataFrame({"int": [None, 1, None, 3]}, schema={"int": pl.Int16})
    mapping = {1: "b", 3: "d"}
    result = df.select(replaced=pl.col("int").replace(mapping))
    expected = pl.DataFrame({"replaced": [None, "b", None, "d"]})
    assert_frame_equal(result, expected)


def test_replace_int_to_str_with_null() -> None:
    df = pl.DataFrame({"int": [None, 1, None, 3]}, schema={"int": pl.Int16})
    mapping = {1: "b", 3: "d", None: "e"}
    result = df.select(replaced=pl.col("int").replace(mapping))
    expected = pl.DataFrame({"replaced": ["e", "b", "e", "d"]})
    assert_frame_equal(result, expected)


def test_replace_int_to_int_null() -> None:
    df = pl.DataFrame({"int": [None, 1, None, 3]}, schema={"int": pl.Int16})
    mapping = {3: None}
    result = df.select(
        replaced=pl.col("int").replace(mapping, default=pl.lit(6).cast(pl.Int16))
    )
    expected = pl.DataFrame(
        {"replaced": [6, 6, 6, None]}, schema={"replaced": pl.Int16}
    )
    assert_frame_equal(result, expected)


def test_replace_int_to_int_null_default_null() -> None:
    df = pl.DataFrame({"int": [None, 1, None, 3]}, schema={"int": pl.Int16})
    mapping = {3: None}
    result = df.select(replaced=pl.col("int").replace(mapping, default=None))
    expected = pl.DataFrame(
        {"replaced": [None, None, None, None]}, schema={"replaced": pl.Null}
    )
    assert_frame_equal(result, expected)


def test_replace_int_to_int_null_return_dtype() -> None:
    df = pl.DataFrame({"int": [None, 1, None, 3]}, schema={"int": pl.Int16})
    mapping = {3: None}

    with pytest.deprecated_call():
        result = df.select(
            replaced=pl.col("int").replace(mapping, default=6, return_dtype=pl.Int32)
        )

    expected = pl.DataFrame(
        {"replaced": [6, 6, 6, None]}, schema={"replaced": pl.Int32}
    )
    assert_frame_equal(result, expected)


def test_replace_empty_mapping() -> None:
    df = pl.DataFrame({"int": [None, 1, None, 3]}, schema={"int": pl.Int16})
    mapping: dict[Any, Any] = {}
    result = df.select(pl.col("int").replace(mapping))
    assert_frame_equal(result, df)


def test_replace_empty_mapping_default() -> None:
    df = pl.DataFrame({"int": [None, 1, None, 3]}, schema={"int": pl.Int16})
    mapping: dict[Any, Any] = {}
    result = df.select(pl.col("int").replace(mapping, default=pl.lit("A")))
    expected = pl.DataFrame({"int": ["A", "A", "A", "A"]})
    assert_frame_equal(result, expected)


# TODO: Probably does not need to raise an error anymore
@pytest.mark.skip()  # Skip for now
@pytest.mark.parametrize("mapping", [{1: "b", 3: "d"}, {1: "b", 3: "d", None: "e"}])
def test_replace_errors(mapping: dict[int | None, str]) -> None:
    df = pl.DataFrame({"int": [None, "1", None, "3"]})
    mapping = {1: "b", 3: "d", None: "e"}

    with pytest.raises(
        pl.ComputeError,
        match="mapping keys for `replace` could not be converted to Utf8 without losing values in the conversion",
    ):
        df.select(pl.col("int").replace(mapping))


# TODO: Probably does not need to raise an error anymore
@pytest.mark.skip()  # Skip for now
def test_replace_errors2() -> None:
    df = pl.DataFrame({"int": [None, "1", None, "3"]})
    mapping = {1.0: "b", 3.0: "d"}

    with pytest.raises(
        pl.ComputeError,
        match=".*'float' object cannot be interpreted as an integer",
    ):
        df.select(pl.col("int").replace(mapping))


# https://github.com/pola-rs/polars/issues/7132
def test_replace_str_to_str_replace_all() -> None:
    df = pl.DataFrame({"text": ["abc"]})
    mapping = {"abc": "123"}
    result = df.select(pl.col("text").replace(mapping).str.replace_all("1", "-"))
    expected = pl.DataFrame({"text": ["-23"]})
    assert_frame_equal(result, expected)


# TODO: Check return dtype
def test_replace_int_to_int_df() -> None:
    lf = pl.LazyFrame({"a": [1, 2, 3]})
    mapping = {1: 11, 2: 22}
    result = lf.select(pl.col("a").cast(pl.UInt8).replace(mapping, default=99))
    expected = pl.LazyFrame({"a": [11, 22, 99]}, schema_overrides={"a": pl.Int64})
    assert_frame_equal(result, expected)


def test_replace_str_to_int_fill_null() -> None:
    lf = pl.LazyFrame({"a": ["one", "two"]})
    mapping = {"one": 1}

    with pytest.deprecated_call():
        result = lf.select(
            pl.col("a")
            .replace(mapping, default=None, return_dtype=pl.UInt32)
            .fill_null(999)
        )

    expected = pl.LazyFrame({"a": [1, 999]})
    assert_frame_equal(result, expected)


def test_replace_mix() -> None:
    df = pl.DataFrame(
        [
            pl.Series("float_to_boolean", [1.0, None]),
            pl.Series("boolean_to_int", [True, False]),
            pl.Series("boolean_to_str", [True, False]),
        ]
    )

    result = df.with_columns(
        pl.col("float_to_boolean").replace({1.0: True}, default=None),
        pl.col("boolean_to_int").replace({True: 1, False: 0}),
        pl.col("boolean_to_str").replace({True: "1", False: "0"}),
    )

    expected = pl.DataFrame(
        [
            pl.Series("float_to_boolean", [True, None], dtype=pl.Boolean),
            pl.Series("boolean_to_int", [1, 0], dtype=pl.Int64),
            pl.Series("boolean_to_str", ["1", "0"], dtype=pl.Utf8),
        ]
    )
    assert_frame_equal(result, expected)


@pytest.fixture(scope="module")
def int_mapping() -> dict[int, int]:
    return {1: 11, 2: 22, 3: 33, 4: 44, 5: 55}


def test_replace_int_to_int1(int_mapping: dict[int, int]) -> None:
    s = pl.Series([-1, 22, None, 44, -5])
    result = s.replace(int_mapping)
    expected = pl.Series([-1, 22, None, 44, -5])
    assert_series_equal(result, expected)


# TODO: Check return dtype
def test_replace_int_to_int2(int_mapping: dict[int, int]) -> None:
    s = pl.Series([1, 22, None, 44, -5], dtype=pl.Int16)
    result = s.replace(int_mapping, default=None)
    expected = pl.Series([11, None, None, None, None], dtype=pl.Int64)
    assert_series_equal(result, expected)


# TODO: Check return dtype
def test_replace_int_to_int21(int_mapping: dict[int, int]) -> None:
    s = pl.Series([1, 22, None, 44, -5], dtype=pl.Int16)
    result = s.replace(int_mapping, default=9)
    expected = pl.Series([11, 9, 9, 9, 9], dtype=pl.Int64)
    assert_series_equal(result, expected)


# TODO: Check return dtype
def test_replace_int_to_int3(int_mapping: dict[int, int]) -> None:
    s = pl.Series([-1, 22, None, 44, -5], dtype=pl.Int16)
    result = s.replace(int_mapping)
    expected = pl.Series([-1, 22, None, 44, -5], dtype=pl.Int64)
    assert_series_equal(result, expected)


def test_replace_int_to_int4_return_dtype(int_mapping: dict[int, int]) -> None:
    s = pl.Series([-1, 22, None, 44, -5], dtype=pl.Int16)
    with pytest.deprecated_call():
        result = s.replace(int_mapping, return_dtype=pl.Float32)
    expected = pl.Series([-1.0, 22.0, None, 44.0, -5.0], dtype=pl.Float32)
    assert_series_equal(result, expected)


def test_replace_int_to_int5_return_dtype(int_mapping: dict[int, int]) -> None:
    s = pl.Series([1, 22, None, 44, -5], dtype=pl.Int16)
    with pytest.deprecated_call():
        result = s.replace(int_mapping, default=9, return_dtype=pl.Float32)
    expected = pl.Series([11.0, 9.0, 9.0, 9.0, 9.0], dtype=pl.Float32)
    assert_series_equal(result, expected)


def test_replace_bool_to_int() -> None:
    s = pl.Series([True, False, False, None])
    mapping = {True: 1, False: 0}
    result = s.replace(mapping)
    expected = pl.Series([1, 0, 0, None])
    assert_series_equal(result, expected)


def test_replace_bool_to_str() -> None:
    s = pl.Series([True, False, False, None])
    mapping = {True: "1", False: "0"}
    result = s.replace(mapping)
    expected = pl.Series(["1", "0", "0", None])
    assert_series_equal(result, expected)


def test_replace_int_to_str() -> None:
    s = pl.Series("a", [-1, 2, None, 4, -5])
    mapping = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}

    result = s.replace(mapping)

    expected = pl.Series("a", ["-1", "two", None, "four", "-5"])
    assert_series_equal(result, expected)


def test_replace_int_to_str_with_default() -> None:
    s = pl.Series("a", [1, 2, None, 4, 5])
    mapping = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}

    result = s.replace(mapping, default="?")

    expected = pl.Series("a", ["one", "two", "?", "four", "five"])
    assert_series_equal(result, expected)


# https://github.com/pola-rs/polars/issues/12728
def test_replace_str_to_int2() -> None:
    s = pl.Series(["a", "b"])
    mapping = {"a": 1, "b": 2}

    result = s.replace(mapping)

    expected = pl.Series(["1", "2"])
    assert_series_equal(result, expected)


def test_replace_str_to_int_with_default() -> None:
    s = pl.Series(["a", "b"])
    mapping = {"a": 1, "b": 2}

    result = s.replace(mapping, default=None)

    expected = pl.Series([1, 2])
    assert_series_equal(result, expected)


def test_map_dict_deprecated() -> None:
    s = pl.Series("a", [1, 2, 3])
    with pytest.deprecated_call():
        result = s.map_dict({2: 100})
    expected = pl.Series("a", [None, 100, None])
    assert_series_equal(result, expected)

    with pytest.deprecated_call():
        result = s.to_frame().select(pl.col("a").map_dict({2: 100})).to_series()
    assert_series_equal(result, expected)
