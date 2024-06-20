from __future__ import annotations

import contextlib
from typing import Any

import pytest

import polars as pl
from polars.exceptions import CategoricalRemappingWarning, InvalidOperationError
from polars.testing import assert_frame_equal, assert_series_equal


def test_replace_strict_incomplete_mapping() -> None:
    lf = pl.LazyFrame({"a": [1, 2, 2, 3]})

    with pytest.raises(InvalidOperationError, match="incomplete mapping"):
        lf.select(pl.col("a").replace_strict({2: 200, 3: 300})).collect()


def test_replace_strict_incomplete_mapping_null_raises() -> None:
    s = pl.Series("a", [1, 2, 2, None, None])
    with pytest.raises(InvalidOperationError):
        s.replace_strict({1: 10})


def test_replace_strict_mapping_null_not_specified() -> None:
    s = pl.Series("a", [1, 2, 2, None, None])

    result = s.replace_strict({1: 10, 2: 20})

    expected = pl.Series("a", [10, 20, 20, None, None])
    assert_series_equal(result, expected)


def test_replace_strict_mapping_null_specified() -> None:
    s = pl.Series("a", [1, 2, 2, None, None])

    result = s.replace_strict({1: 10, 2: 20, None: 0})

    expected = pl.Series("a", [10, 20, 20, 0, 0])
    assert_series_equal(result, expected)


def test_replace_strict_mapping_null_replace_by_null() -> None:
    s = pl.Series("a", [1, 2, 2, None])

    result = s.replace_strict({1: 10, 2: None, None: 0})

    expected = pl.Series("a", [10, None, None, 0])
    assert_series_equal(result, expected)


def test_replace_strict_mapping_null_with_default() -> None:
    s = pl.Series("a", [1, 2, 2, None, None])

    result = s.replace_strict({1: 10}, default=0)

    expected = pl.Series("a", [10, 0, 0, 0, 0])
    assert_series_equal(result, expected)


def test_replace_strict_empty() -> None:
    lf = pl.LazyFrame({"a": [None, None]})
    result = lf.select(pl.col("a").replace_strict({}))
    assert_frame_equal(lf, result)


def test_replace_strict_fast_path_many_to_one_default() -> None:
    lf = pl.LazyFrame({"a": [1, 2, 2, 3]})
    result = lf.select(pl.col("a").replace_strict([2, 3], 100, default=-1))
    expected = pl.LazyFrame({"a": [-1, 100, 100, 100]}, schema={"a": pl.Int32})
    assert_frame_equal(result, expected)


def test_replace_strict_fast_path_many_to_one_null() -> None:
    lf = pl.LazyFrame({"a": [1, 2, 2, 3]})
    result = lf.select(pl.col("a").replace_strict([2, 3], None, default=-1))
    expected = pl.LazyFrame({"a": [-1, None, None, None]}, schema={"a": pl.Int32})
    assert_frame_equal(result, expected)


@pytest.fixture(scope="module")
def str_mapping() -> dict[str | None, str]:
    return {
        "CA": "Canada",
        "DE": "Germany",
        "FR": "France",
        None: "Not specified",
    }


def test_replace_strict_str_to_str_default_self(
    str_mapping: dict[str | None, str],
) -> None:
    df = pl.DataFrame({"country_code": ["FR", None, "ES", "DE"]})
    result = df.select(
        replaced=pl.col("country_code").replace_strict(
            str_mapping, default=pl.col("country_code")
        )
    )
    expected = pl.DataFrame({"replaced": ["France", "Not specified", "ES", "Germany"]})
    assert_frame_equal(result, expected)


def test_replace_strict_str_to_str_default_null(
    str_mapping: dict[str | None, str],
) -> None:
    df = pl.DataFrame({"country_code": ["FR", None, "ES", "DE"]})
    result = df.select(
        replaced=pl.col("country_code").replace_strict(str_mapping, default=None)
    )
    expected = pl.DataFrame({"replaced": ["France", "Not specified", None, "Germany"]})
    assert_frame_equal(result, expected)


def test_replace_strict_str_to_str_default_other(
    str_mapping: dict[str | None, str],
) -> None:
    df = pl.DataFrame({"country_code": ["FR", None, "ES", "DE"]})

    result = df.with_row_index().select(
        replaced=pl.col("country_code").replace_strict(
            str_mapping, default=pl.col("index")
        )
    )
    expected = pl.DataFrame({"replaced": ["France", "Not specified", "2", "Germany"]})
    assert_frame_equal(result, expected)


def test_replace_strict_str_to_cat() -> None:
    s = pl.Series(["a", "b", "c"])
    mapping = {"a": "c", "b": "d"}
    result = s.replace_strict(mapping, default=None, return_dtype=pl.Categorical)
    expected = pl.Series(["c", "d", None], dtype=pl.Categorical)
    assert_series_equal(result, expected, categorical_as_str=True)


def test_replace_strict_enum_to_new_enum() -> None:
    s = pl.Series(["a", "b", "c"], dtype=pl.Enum(["a", "b", "c", "d"]))
    old = ["a", "b"]

    new_dtype = pl.Enum(["a", "b", "c", "d", "e"])
    new = pl.Series(["c", "e"], dtype=new_dtype)

    result = s.replace_strict(old, new, default=None, return_dtype=new_dtype)

    expected = pl.Series(["c", "e", None], dtype=new_dtype)
    assert_series_equal(result, expected)


def test_replace_strict_int_to_int_null() -> None:
    df = pl.DataFrame({"int": [None, 1, None, 3]}, schema={"int": pl.Int16})
    mapping = {3: None}
    result = df.select(
        replaced=pl.col("int").replace_strict(mapping, default=pl.lit(6).cast(pl.Int16))
    )
    expected = pl.DataFrame(
        {"replaced": [6, 6, 6, None]}, schema={"replaced": pl.Int16}
    )
    assert_frame_equal(result, expected)


def test_replace_strict_int_to_int_null_default_null() -> None:
    df = pl.DataFrame({"int": [None, 1, None, 3]}, schema={"int": pl.Int16})
    mapping = {3: None}
    result = df.select(replaced=pl.col("int").replace_strict(mapping, default=None))
    expected = pl.DataFrame(
        {"replaced": [None, None, None, None]}, schema={"replaced": pl.Null}
    )
    assert_frame_equal(result, expected)


def test_replace_strict_int_to_int_null_return_dtype() -> None:
    df = pl.DataFrame({"int": [None, 1, None, 3]}, schema={"int": pl.Int16})
    mapping = {3: None}

    result = df.select(
        replaced=pl.col("int").replace_strict(mapping, default=6, return_dtype=pl.Int32)
    )

    expected = pl.DataFrame(
        {"replaced": [6, 6, 6, None]}, schema={"replaced": pl.Int32}
    )
    assert_frame_equal(result, expected)


def test_replace_strict_empty_mapping_default() -> None:
    df = pl.DataFrame({"int": [None, 1, None, 3]}, schema={"int": pl.Int16})
    mapping: dict[Any, Any] = {}
    result = df.select(pl.col("int").replace_strict(mapping, default=pl.lit("A")))
    expected = pl.DataFrame({"int": ["A", "A", "A", "A"]})
    assert_frame_equal(result, expected)


def test_replace_strict_int_to_int_df() -> None:
    lf = pl.LazyFrame({"a": [1, 2, 3]}, schema={"a": pl.UInt8})
    mapping = {1: 11, 2: 22}

    result = lf.select(
        pl.col("a").replace_strict(
            old=pl.Series(mapping.keys()),
            new=pl.Series(mapping.values(), dtype=pl.UInt8),
            default=pl.lit(99).cast(pl.UInt8),
        )
    )
    expected = pl.LazyFrame({"a": [11, 22, 99]}, schema_overrides={"a": pl.UInt8})
    assert_frame_equal(result, expected)


def test_replace_strict_str_to_int_fill_null() -> None:
    lf = pl.LazyFrame({"a": ["one", "two"]})
    mapping = {"one": 1}

    result = lf.select(
        pl.col("a")
        .replace_strict(mapping, default=None, return_dtype=pl.UInt32)
        .fill_null(999)
    )

    expected = pl.LazyFrame({"a": pl.Series([1, 999], dtype=pl.UInt32)})
    assert_frame_equal(result, expected)


def test_replace_strict_mix() -> None:
    df = pl.DataFrame(
        [
            pl.Series("float_to_boolean", [1.0, None]),
            pl.Series("boolean_to_int", [True, False]),
            pl.Series("boolean_to_str", [True, False]),
        ]
    )

    result = df.with_columns(
        pl.col("float_to_boolean").replace_strict({1.0: True}),
        pl.col("boolean_to_int").replace_strict({True: 1, False: 0}),
        pl.col("boolean_to_str").replace_strict({True: "1", False: "0"}),
    )

    expected = pl.DataFrame(
        [
            pl.Series("float_to_boolean", [True, None], dtype=pl.Boolean),
            pl.Series("boolean_to_int", [1, 0], dtype=pl.Int64),
            pl.Series("boolean_to_str", ["1", "0"], dtype=pl.String),
        ]
    )
    assert_frame_equal(result, expected)


@pytest.fixture(scope="module")
def int_mapping() -> dict[int, int]:
    return {1: 11, 2: 22, 3: 33, 4: 44, 5: 55}


def test_replace_strict_int_to_int2(int_mapping: dict[int, int]) -> None:
    s = pl.Series([1, 22, None, 44, -5])
    result = s.replace_strict(int_mapping, default=None)
    expected = pl.Series([11, None, None, None, None], dtype=pl.Int64)
    assert_series_equal(result, expected)


def test_replace_strict_int_to_int3(int_mapping: dict[int, int]) -> None:
    s = pl.Series([1, 22, None, 44, -5], dtype=pl.Int16)
    result = s.replace_strict(int_mapping, default=9)
    expected = pl.Series([11, 9, 9, 9, 9], dtype=pl.Int64)
    assert_series_equal(result, expected)


def test_replace_strict_int_to_int4_return_dtype(int_mapping: dict[int, int]) -> None:
    s = pl.Series([-1, 22, None, 44, -5], dtype=pl.Int16)
    result = s.replace_strict(int_mapping, default=s, return_dtype=pl.Float32)
    expected = pl.Series([-1.0, 22.0, None, 44.0, -5.0], dtype=pl.Float32)
    assert_series_equal(result, expected)


def test_replace_strict_int_to_int5_return_dtype(int_mapping: dict[int, int]) -> None:
    s = pl.Series([1, 22, None, 44, -5], dtype=pl.Int16)
    result = s.replace_strict(int_mapping, default=9, return_dtype=pl.Float32)
    expected = pl.Series([11.0, 9.0, 9.0, 9.0, 9.0], dtype=pl.Float32)
    assert_series_equal(result, expected)


def test_replace_strict_bool_to_int() -> None:
    s = pl.Series([True, False, False, None])
    mapping = {True: 1, False: 0}
    result = s.replace_strict(mapping)
    expected = pl.Series([1, 0, 0, None])
    assert_series_equal(result, expected)


def test_replace_strict_bool_to_str() -> None:
    s = pl.Series([True, False, False, None])
    mapping = {True: "1", False: "0"}
    result = s.replace_strict(mapping)
    expected = pl.Series(["1", "0", "0", None])
    assert_series_equal(result, expected)


def test_replace_strict_str_to_bool() -> None:
    s = pl.Series(["True", "False", "False", None])
    mapping = {"True": True, "False": False}
    result = s.replace_strict(mapping)
    expected = pl.Series([True, False, False, None])
    assert_series_equal(result, expected)


def test_replace_strict_int_to_str() -> None:
    s = pl.Series("a", [-1, 2, None, 4, -5])
    mapping = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}

    with pytest.raises(InvalidOperationError, match="incomplete mapping"):
        s.replace_strict(mapping)
    result = s.replace_strict(mapping, default=None)

    expected = pl.Series("a", [None, "two", None, "four", None])
    assert_series_equal(result, expected)


def test_replace_strict_int_to_str2() -> None:
    s = pl.Series("a", [1, 2, None, 4, 5])
    mapping = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}

    result = s.replace_strict(mapping)

    expected = pl.Series("a", ["one", "two", None, "four", "five"])
    assert_series_equal(result, expected)


def test_replace_strict_int_to_str_with_default() -> None:
    s = pl.Series("a", [1, 2, None, 4, 5])
    mapping = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}

    result = s.replace_strict(mapping, default="?")

    expected = pl.Series("a", ["one", "two", "?", "four", "five"])
    assert_series_equal(result, expected)


def test_replace_strict_str_to_int() -> None:
    s = pl.Series(["a", "b"])
    mapping = {"a": 1, "b": 2}
    result = s.replace_strict(mapping)
    expected = pl.Series([1, 2])
    assert_series_equal(result, expected)


@pytest.mark.parametrize(
    ("context", "dtype"),
    [
        (pl.StringCache(), pl.Categorical),
        (pytest.warns(CategoricalRemappingWarning), pl.Categorical),
        (contextlib.nullcontext(), pl.Enum(["a", "b", "OTHER"])),
    ],
)
def test_replace_strict_cat_str(
    context: contextlib.AbstractContextManager,  # type: ignore[type-arg]
    dtype: pl.DataType,
) -> None:
    with context:
        for old, new, expected in [
            ("a", "c", pl.Series("s", ["c", None], dtype=pl.Utf8)),
            (["a", "b"], ["c", "d"], pl.Series("s", ["c", "d"], dtype=pl.Utf8)),
            (pl.lit("a", dtype=dtype), "c", pl.Series("s", ["c", None], dtype=pl.Utf8)),
            (
                pl.Series(["a", "b"], dtype=dtype),
                ["c", "d"],
                pl.Series("s", ["c", "d"], dtype=pl.Utf8),
            ),
        ]:
            s = pl.Series("s", ["a", "b"], dtype=dtype)
            s_replaced = s.replace_strict(old, new, default=None)  # type: ignore[arg-type]
            assert_series_equal(s_replaced, expected)

            s = pl.Series("s", ["a", "b"], dtype=dtype)
            s_replaced = s.replace_strict(old, new, default="OTHER")  # type: ignore[arg-type]
            assert_series_equal(s_replaced, expected.fill_null("OTHER"))


@pytest.mark.parametrize(
    "context", [pl.StringCache(), pytest.warns(CategoricalRemappingWarning)]
)
def test_replace_strict_cat_cat(
    context: contextlib.AbstractContextManager,  # type: ignore[type-arg]
) -> None:
    with context:
        dt = pl.Categorical
        for old, new, expected in [
            ("a", pl.lit("c", dtype=dt), pl.Series("s", ["c", None], dtype=dt)),
            (
                ["a", "b"],
                pl.Series(["c", "d"], dtype=dt),
                pl.Series("s", ["c", "d"], dtype=dt),
            ),
        ]:
            s = pl.Series("s", ["a", "b"], dtype=dt)
            s_replaced = s.replace_strict(old, new, default=None)  # type: ignore[arg-type]
            assert_series_equal(s_replaced, expected)

            s = pl.Series("s", ["a", "b"], dtype=dt)
            s_replaced = s.replace_strict(old, new, default=pl.lit("OTHER", dtype=dt))  # type: ignore[arg-type]
            assert_series_equal(s_replaced, expected.fill_null("OTHER"))
