# mypy: disable-error-code="redundant-expr"
from __future__ import annotations

import enum
import io
import operator
import re
import sys
from datetime import date
from textwrap import dedent
from typing import Any, Callable

import pytest

import polars as pl
from polars import StringCache
from polars.exceptions import (
    ComputeError,
    InvalidOperationError,
    OutOfBoundsError,
    SchemaError,
)
from polars.testing import assert_frame_equal, assert_series_equal
from tests.unit.conftest import INTEGER_DTYPES

if sys.version_info >= (3, 11):
    from enum import StrEnum

    PyStrEnum: type[enum.Enum] | None = StrEnum
else:
    PyStrEnum = None


def test_enum_creation() -> None:
    dtype = pl.Enum(["a", "b"])
    s = pl.Series([None, "a", "b"], dtype=dtype)
    assert s.null_count() == 1
    assert s.len() == 3
    assert s.dtype == dtype

    # from iterables
    e = pl.Enum(f"x{i}" for i in range(5))
    assert e.categories.to_list() == ["x0", "x1", "x2", "x3", "x4"]

    e = pl.Enum("abcde")
    assert e.categories.to_list() == ["a", "b", "c", "d", "e"]


@pytest.mark.parametrize("categories", [[], pl.Series("foo", dtype=pl.Int16), None])
def test_enum_init_empty(categories: pl.Series | list[str] | None) -> None:
    dtype = pl.Enum(categories)  # type: ignore[arg-type]
    expected = pl.Series("category", dtype=pl.String)
    assert_series_equal(dtype.categories, expected)


def test_enum_init_from_python() -> None:
    # standard string enum
    class Color1(str, enum.Enum):
        RED = "red"
        GREEN = "green"
        BLUE = "blue"

    dtype = pl.Enum(Color1)
    assert dtype == pl.Enum(["red", "green", "blue"])

    # standard generic enum
    class Color2(enum.Enum):
        RED = "red"
        GREEN = "green"
        BLUE = "blue"

    dtype = pl.Enum(Color2)
    assert dtype == pl.Enum(["red", "green", "blue"])

    # specialised string enum
    if sys.version_info >= (3, 11):

        class Color3(enum.Enum):
            RED = "red"
            GREEN = "green"
            BLUE = "blue"

        dtype = pl.Enum(Color3)
        assert dtype == pl.Enum(["red", "green", "blue"])


def test_enum_init_from_python_invalid() -> None:
    class Color(int, enum.Enum):
        RED = 1
        GREEN = 2
        BLUE = 3

    with pytest.raises(
        TypeError,
        match="Enum categories must be strings",
    ):
        pl.Enum(Color)

    # flag/int enums
    for EnumBase in (enum.Flag, enum.IntFlag, enum.IntEnum):

        class Color(EnumBase):  # type: ignore[no-redef,misc,valid-type]
            RED = enum.auto()
            GREEN = enum.auto()
            BLUE = enum.auto()

        with pytest.raises(
            TypeError,
            match="Enum categories must be strings; `Color` values are integers",
        ):
            pl.Enum(Color)


def test_enum_non_existent() -> None:
    with pytest.raises(
        InvalidOperationError,
        match="conversion from `str` to `enum` failed in column '' for 1 out of 4 values: \\[\"c\"\\]",
    ):
        pl.Series([None, "a", "b", "c"], dtype=pl.Enum(categories=["a", "b"]))


def test_enum_non_existent_non_strict() -> None:
    s = pl.Series(
        [None, "a", "b", "c"], dtype=pl.Enum(categories=["a", "b"]), strict=False
    )
    expected = pl.Series([None, "a", "b", None], dtype=pl.Enum(categories=["a", "b"]))
    assert_series_equal(s, expected)


def test_enum_from_schema_argument() -> None:
    df = pl.DataFrame(
        {"col1": ["a", "b", "c"]}, schema={"col1": pl.Enum(["a", "b", "c"])}
    )
    assert df.get_column("col1").dtype == pl.Enum
    assert dedent(
        """
        │ col1 │
        │ ---  │
        │ enum │
        ╞══════╡
        """
    ) in str(df)


def test_equality_of_two_separately_constructed_enums() -> None:
    s = pl.Series([None, "a", "b"], dtype=pl.Enum(categories=["a", "b"]))
    s2 = pl.Series([None, "a", "b"], dtype=pl.Enum(categories=["a", "b"]))
    assert_series_equal(s, s2)


def test_nested_enum_creation() -> None:
    dtype = pl.List(pl.Enum(["a", "b", "c"]))
    s = pl.Series([[None, "a"], ["b", "c"]], dtype=dtype)
    assert s.len() == 2
    assert s.dtype == dtype


def test_enum_union() -> None:
    e1 = pl.Enum(["a", "b"])
    e2 = pl.Enum(["b", "c"])
    assert e1 | e2 == pl.Enum(["a", "b", "c"])
    assert e1.union(e2) == pl.Enum(["a", "b", "c"])


def test_nested_enum_concat() -> None:
    dtype = pl.List(pl.Enum(["a", "b", "c", "d"]))
    s1 = pl.Series([[None, "a"], ["b", "c"]], dtype=dtype)
    s2 = pl.Series([["c", "d"], ["a", None]], dtype=dtype)
    expected = pl.Series(
        [
            [None, "a"],
            ["b", "c"],
            ["c", "d"],
            ["a", None],
        ],
        dtype=dtype,
    )

    assert_series_equal(pl.concat((s1, s2)), expected)
    assert_series_equal(s1.extend(s2), expected)


def test_nested_enum_agg_sort_18026() -> None:
    df = (
        pl.DataFrame({"a": [1, 1, 2, 2], "b": ["Y", "Z", "Z", "Y"]})
        .cast({"b": pl.Enum(["Z", "Y"])})
        .with_columns(pl.struct("b", "a").alias("c"))
    )
    result = df.group_by("a").agg("c").sort("a")
    expected = pl.DataFrame(
        {
            "a": [1, 2],
            "c": [
                [{"b": "Y", "a": 1}, {"b": "Z", "a": 1}],
                [{"b": "Z", "a": 2}, {"b": "Y", "a": 2}],
            ],
        },
        schema={
            "a": pl.Int64,
            "c": pl.List(
                pl.Struct([pl.Field("b", pl.Enum(["Z", "Y"])), pl.Field("a", pl.Int64)])
            ),
        },
    )
    assert_frame_equal(result, expected)


def test_casting_to_an_enum_from_utf() -> None:
    dtype = pl.Enum(["a", "b", "c"])
    s = pl.Series([None, "a", "b", "c"])
    s2 = s.cast(dtype)
    assert s2.dtype == dtype
    assert s2.null_count() == 1


def test_casting_to_an_enum_from_categorical() -> None:
    dtype = pl.Enum(["a", "b", "c"])
    s = pl.Series([None, "a", "b", "c"], dtype=pl.Categorical)
    s2 = s.cast(dtype)
    assert s2.dtype == dtype
    assert s2.null_count() == 1
    expected = pl.Series([None, "a", "b", "c"], dtype=dtype)
    assert_series_equal(s2, expected)


def test_casting_to_an_enum_from_categorical_nonstrict() -> None:
    dtype = pl.Enum(["a", "b"])
    s = pl.Series([None, "a", "b", "c"], dtype=pl.Categorical)
    s2 = s.cast(dtype, strict=False)
    assert s2.dtype == dtype
    assert s2.null_count() == 2  # "c" mapped to null
    expected = pl.Series([None, "a", "b", None], dtype=dtype)
    assert_series_equal(s2, expected)


def test_casting_to_an_enum_from_enum_nonstrict() -> None:
    dtype = pl.Enum(["a", "b"])
    s = pl.Series([None, "a", "b", "c"], dtype=pl.Enum(["a", "b", "c"]))
    s2 = s.cast(dtype, strict=False)
    assert s2.dtype == dtype
    assert s2.null_count() == 2  # "c" mapped to null
    expected = pl.Series([None, "a", "b", None], dtype=dtype)
    assert_series_equal(s2, expected)


def test_casting_to_an_enum_from_integer() -> None:
    dtype = pl.Enum(["a", "b", "c"])
    expected = pl.Series([None, "b", "a", "c"], dtype=dtype)
    s = pl.Series([None, 1, 0, 2], dtype=pl.UInt32)
    s_enum = s.cast(dtype)
    assert s_enum.dtype == dtype
    assert s_enum.null_count() == 1
    assert_series_equal(s_enum, expected)


def test_casting_to_an_enum_oob_from_integer() -> None:
    dtype = pl.Enum(["a", "b", "c"])
    s = pl.Series([None, 1, 0, 5], dtype=pl.UInt32)
    with pytest.raises(
        OutOfBoundsError, match=("index 5 is bigger than the number of categories 3")
    ):
        s.cast(dtype)


def test_casting_to_an_enum_from_categorical_nonexistent() -> None:
    with pytest.raises(
        InvalidOperationError,
        match=(
            r"conversion from `cat` to `enum` failed in column '' for 1 out of 4 values: \[\"c\"\]"
        ),
    ):
        pl.Series([None, "a", "b", "c"], dtype=pl.Categorical).cast(pl.Enum(["a", "b"]))


@StringCache()
def test_casting_to_an_enum_from_global_categorical() -> None:
    dtype = pl.Enum(["a", "b", "c"])
    s = pl.Series([None, "a", "b", "c"], dtype=pl.Categorical)
    s2 = s.cast(dtype)
    assert s2.dtype == dtype
    assert s2.null_count() == 1
    expected = pl.Series([None, "a", "b", "c"], dtype=dtype)
    assert_series_equal(s2, expected)


@StringCache()
def test_casting_to_an_enum_from_global_categorical_nonexistent() -> None:
    with pytest.raises(
        InvalidOperationError,
        match=(
            r"conversion from `cat` to `enum` failed in column '' for 1 out of 4 values: \[\"c\"\]"
        ),
    ):
        pl.Series([None, "a", "b", "c"], dtype=pl.Categorical).cast(pl.Enum(["a", "b"]))


def test_casting_from_an_enum_to_local() -> None:
    dtype = pl.Enum(["a", "b", "c"])
    s = pl.Series([None, "a", "b", "c"], dtype=dtype)
    s2 = s.cast(pl.Categorical)
    expected = pl.Series([None, "a", "b", "c"], dtype=pl.Categorical)
    assert_series_equal(s2, expected)


@StringCache()
def test_casting_from_an_enum_to_global() -> None:
    dtype = pl.Enum(["a", "b", "c"])
    s = pl.Series([None, "a", "b", "c"], dtype=dtype)
    s2 = s.cast(pl.Categorical)
    expected = pl.Series([None, "a", "b", "c"], dtype=pl.Categorical)
    assert_series_equal(s2, expected)


def test_append_to_an_enum() -> None:
    s = pl.Series([None, "a", "b", "c"], dtype=pl.Enum(["a", "b", "c"]))
    s2 = pl.Series(["c", "a", "b", "c"], dtype=pl.Enum(["a", "b", "c"]))
    s.append(s2)
    assert s.len() == 8


def test_append_to_an_enum_with_new_category() -> None:
    with pytest.raises(
        SchemaError,
        match=("type Enum.*is incompatible with expected type Enum.*"),
    ):
        pl.Series([None, "a", "b", "c"], dtype=pl.Enum(["a", "b", "c"])).append(
            pl.Series(["d", "a", "b", "c"], dtype=pl.Enum(["a", "b", "c", "d"]))
        )


def test_extend_to_an_enum() -> None:
    s = pl.Series([None, "a", "b", "c"], dtype=pl.Enum(["a", "b", "c"]))
    s2 = pl.Series(["c", "a", "b", "c"], dtype=pl.Enum(["a", "b", "c"]))
    s.extend(s2)
    assert s.len() == 8
    assert s.null_count() == 1


def test_series_init_uninstantiated_enum() -> None:
    with pytest.raises(
        ComputeError,
        match="can not cast / initialize Enum without categories present",
    ):
        pl.Series(["a", "b", "a"], dtype=pl.Enum)


@pytest.mark.parametrize(
    ("op", "expected"),
    [
        (operator.le, pl.Series([None, True, True, True])),
        (operator.lt, pl.Series([None, True, False, False])),
        (operator.ge, pl.Series([None, False, True, True])),
        (operator.gt, pl.Series([None, False, False, False])),
        (operator.eq, pl.Series([None, False, True, True])),
        (operator.ne, pl.Series([None, True, False, False])),
        (pl.Series.ne_missing, pl.Series([False, True, False, False])),
        (pl.Series.eq_missing, pl.Series([True, False, True, True])),
    ],
)
def test_equality_enum(
    op: Callable[[pl.Series, pl.Series], pl.Series], expected: pl.Series
) -> None:
    dtype = pl.Enum(["a", "b", "c"])
    s = pl.Series([None, "a", "b", "c"], dtype=dtype)
    s2 = pl.Series([None, "c", "b", "c"], dtype=dtype)

    assert_series_equal(op(s, s2), expected)
    assert_series_equal(op(s, s2.cast(pl.String)), expected)


@pytest.mark.parametrize(
    ("op", "expected"),
    [
        (operator.le, pl.Series([None, False, True, True])),
        (operator.lt, pl.Series([None, False, False, True])),
        (operator.ge, pl.Series([None, True, True, False])),
        (operator.gt, pl.Series([None, True, False, False])),
        (operator.eq, pl.Series([None, False, True, False])),
        (operator.ne, pl.Series([None, True, False, True])),
        (pl.Series.ne_missing, pl.Series([True, True, False, True])),
        (pl.Series.eq_missing, pl.Series([False, False, True, False])),
    ],
)
def test_compare_enum_str_single(
    op: Callable[[pl.Series, pl.Series], pl.Series], expected: pl.Series
) -> None:
    s = pl.Series(
        [None, "HIGH", "MEDIUM", "LOW"], dtype=pl.Enum(["LOW", "MEDIUM", "HIGH"])
    )
    s2 = "MEDIUM"

    assert_series_equal(op(s, s2), expected)  # type: ignore[arg-type]


def test_equality_missing_enum_scalar() -> None:
    dtype = pl.Enum(["a", "b", "c"])
    df = pl.DataFrame({"a": pl.Series([None, "a", "b", "c"], dtype=dtype)})

    out = df.select(
        pl.col("a").eq_missing(pl.lit("c", dtype=dtype)).alias("cmp")
    ).get_column("cmp")
    expected = pl.Series("cmp", [False, False, False, True], dtype=pl.Boolean)
    assert_series_equal(out, expected)

    out_str = df.select(pl.col("a").eq_missing(pl.lit("c")).alias("cmp")).get_column(
        "cmp"
    )
    assert_series_equal(out_str, expected)

    out = df.select(
        pl.col("a").ne_missing(pl.lit("c", dtype=dtype)).alias("cmp")
    ).get_column("cmp")
    expected = pl.Series("cmp", [True, True, True, False], dtype=pl.Boolean)
    assert_series_equal(out, expected)

    out_str = df.select(pl.col("a").ne_missing(pl.lit("c")).alias("cmp")).get_column(
        "cmp"
    )
    assert_series_equal(out_str, expected)


def test_equality_missing_enum_none_scalar() -> None:
    dtype = pl.Enum(["a", "b", "c"])
    df = pl.DataFrame({"a": pl.Series([None, "a", "b", "c"], dtype=dtype)})

    out = df.select(
        pl.col("a").eq_missing(pl.lit(None, dtype=dtype)).alias("cmp")
    ).get_column("cmp")
    expected = pl.Series("cmp", [True, False, False, False], dtype=pl.Boolean)
    assert_series_equal(out, expected)

    out = df.select(
        pl.col("a").ne_missing(pl.lit(None, dtype=dtype)).alias("cmp")
    ).get_column("cmp")
    expected = pl.Series("cmp", [False, True, True, True], dtype=pl.Boolean)
    assert_series_equal(out, expected)


@pytest.mark.parametrize(("op"), [operator.le, operator.lt, operator.ge, operator.gt])
def test_compare_enum_str_single_raise(
    op: Callable[[pl.Series, pl.Series], pl.Series],
) -> None:
    s = pl.Series(
        [None, "HIGH", "MEDIUM", "LOW"], dtype=pl.Enum(["LOW", "MEDIUM", "HIGH"])
    )
    s2 = "NOTEXIST"

    with pytest.raises(
        InvalidOperationError,
        match=re.escape(
            "conversion from `str` to `enum` failed in column '' for 1 out of 1 values: [\"NOTEXIST\"]"
        ),
    ):
        op(s, s2)  # type: ignore[arg-type]


def test_compare_enum_str_raise() -> None:
    s = pl.Series([None, "a", "b", "c"], dtype=pl.Enum(["a", "b", "c"]))
    s2 = pl.Series([None, "d", "d", "d"])
    s_broadcast = pl.Series(["d"])

    for s_compare in [s2, s_broadcast]:
        for op in [operator.le, operator.gt, operator.ge, operator.lt]:
            with pytest.raises(
                InvalidOperationError,
                match="conversion from `str` to `enum` failed in column",
            ):
                op(s, s_compare)


def test_different_enum_comparison_order() -> None:
    df_enum = pl.DataFrame(
        [
            pl.Series(
                "a_cat", ["c", "a", "b", "c", "b"], dtype=pl.Enum(["a", "b", "c"])
            ),
            pl.Series(
                "b_cat", ["F", "G", "E", "G", "G"], dtype=pl.Enum(["F", "G", "E"])
            ),
        ]
    )
    for op in [operator.gt, operator.ge, operator.lt, operator.le]:
        with pytest.raises(
            ComputeError,
            match="can only compare categoricals of the same type",
        ):
            df_enum.filter(op(pl.col("a_cat"), pl.col("b_cat")))


@pytest.mark.parametrize("categories", [[None], ["x", "y", None]])
def test_enum_categories_null(categories: list[str | None]) -> None:
    with pytest.raises(TypeError, match="Enum categories must not contain null values"):
        pl.Enum(categories)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("categories", "type"), [([date.today()], "Date"), ([-10, 10], "Int64")]
)
def test_valid_enum_category_types(categories: Any, type: str) -> None:
    with pytest.raises(
        TypeError, match=f"Enum categories must be strings; found data of type {type}"
    ):
        pl.Enum(categories)


def test_enum_categories_unique() -> None:
    with pytest.raises(ValueError, match="must be unique; found duplicate 'a'"):
        pl.Enum(["a", "a", "b", "b", "b", "c"])


def test_enum_categories_series_input() -> None:
    categories = pl.Series("a", ["a", "b", "c"])
    dtype = pl.Enum(categories)
    assert_series_equal(dtype.categories, categories.alias("category"))


def test_enum_categories_series_zero_copy() -> None:
    categories = pl.Series(["a", "b"])
    dtype = pl.Enum(categories)

    s = pl.Series([None, "a", "b"], dtype=dtype)
    result_dtype = s.dtype

    assert result_dtype == dtype


@pytest.mark.parametrize("dtype", INTEGER_DTYPES)
def test_enum_cast_from_other_integer_dtype(dtype: pl.DataType) -> None:
    enum_dtype = pl.Enum(["a", "b", "c", "d"])
    series = pl.Series([1, 2, 3, 3, 2, 1], dtype=dtype)
    series.cast(enum_dtype)


def test_enum_cast_from_other_integer_dtype_oob() -> None:
    enum_dtype = pl.Enum(["a", "b", "c", "d"])
    series = pl.Series([-1, 2, 3, 3, 2, 1], dtype=pl.Int8)
    with pytest.raises(
        InvalidOperationError, match="conversion from `i8` to `u32` failed in column"
    ):
        series.cast(enum_dtype)

    series = pl.Series([2**34, 2, 3, 3, 2, 1], dtype=pl.UInt64)
    with pytest.raises(
        InvalidOperationError,
        match="conversion from `u64` to `u32` failed in column",
    ):
        series.cast(enum_dtype)


def test_enum_creating_col_expr() -> None:
    df = pl.DataFrame(
        {
            "col1": ["a", "b", "c"],
            "col2": ["d", "e", "f"],
            "col3": ["g", "h", "i"],
        },
        schema={
            "col1": pl.Enum(["a", "b", "c"]),
            "col2": pl.Categorical(),
            "col3": pl.Enum(["g", "h", "i"]),
        },
    )

    out = df.select(pl.col(pl.Enum))
    expected = df.select("col1", "col3")
    assert_frame_equal(out, expected)


def test_enum_cse_eq() -> None:
    df = pl.DataFrame({"a": [1]})

    # these both share the value "a", which is used in both expressions
    dt1 = pl.Enum(["a", "b"])
    dt2 = pl.Enum(["a", "c"])

    out = (
        df.lazy()
        .select(
            pl.when(True).then(pl.lit("a", dtype=dt1)).alias("dt1"),
            pl.when(True).then(pl.lit("a", dtype=dt2)).alias("dt2"),
        )
        .collect()
    )

    assert out["dt1"].item() == "a"
    assert out["dt2"].item() == "a"
    assert out["dt1"].dtype == pl.Enum(["a", "b"])
    assert out["dt2"].dtype == pl.Enum(["a", "c"])
    assert out["dt1"].dtype != out["dt2"].dtype


def test_category_comparison_subset() -> None:
    dt1 = pl.Enum(["a"])
    dt2 = pl.Enum(["a", "b"])
    out = (
        pl.LazyFrame()
        .select(
            pl.lit("a", dtype=dt1).alias("dt1"),
            pl.lit("a", dtype=dt2).alias("dt2"),
        )
        .collect()
    )

    assert out["dt1"].item() == "a"
    assert out["dt2"].item() == "a"
    assert out["dt1"].dtype == pl.Enum(["a"])
    assert out["dt2"].dtype == pl.Enum(["a", "b"])
    assert out["dt1"].dtype != out["dt2"].dtype


@pytest.mark.parametrize("dt", INTEGER_DTYPES)
def test_integer_cast_to_enum_15738(dt: pl.DataType) -> None:
    s = pl.Series([0, 1, 2], dtype=dt).cast(pl.Enum(["a", "b", "c"]))
    assert s.to_list() == ["a", "b", "c"]
    expected_s = pl.Series(["a", "b", "c"], dtype=pl.Enum(["a", "b", "c"]))
    assert_series_equal(s, expected_s)


def test_enum_19269() -> None:
    en = pl.Enum(["X", "Z", "Y"])
    df = pl.DataFrame(
        {"test": pl.Series(["X", "Y", "Z"], dtype=en), "group": [1, 2, 2]}
    )
    out = (
        df.group_by("group", maintain_order=True)
        .agg(pl.col("test").mode())
        .select(
            a=pl.col("test").list.max(),
            b=pl.col("test").list.min(),
        )
    )

    assert out.to_dict(as_series=False) == {"a": ["X", "Y"], "b": ["X", "Z"]}
    assert out.dtypes == [en, en]


def test_roundtrip_enum_parquet() -> None:
    dtype = pl.Enum(["foo", "bar", "ham"])
    df = pl.DataFrame(pl.Series("d", ["foo", "bar"]), schema=pl.Schema({"d": dtype}))
    f = io.BytesIO()
    df.write_parquet(f)
    f.seek(0)
    assert pl.scan_parquet(f).collect_schema()["d"] == dtype


@pytest.mark.parametrize(
    "EnumBase",
    [
        (enum.Enum,),
        (str, enum.Enum),
        *([(PyStrEnum,)] if PyStrEnum is not None else []),
    ],
)
def test_init_frame_from_enums(EnumBase: tuple[type, ...]) -> None:
    class Portfolio(*EnumBase):  # type: ignore[misc]
        TECH = "Technology"
        RETAIL = "Retail"
        OTHER = "Other"

    # confirm that we can infer the enum dtype from various enum bases
    df = pl.DataFrame(
        {"trade_id": [123, 456], "portfolio": [Portfolio.OTHER, Portfolio.TECH]}
    )
    expected = pl.DataFrame(
        {"trade_id": [123, 456], "portfolio": ["Other", "Technology"]},
        schema={
            "trade_id": pl.Int64,
            "portfolio": pl.Enum(["Technology", "Retail", "Other"]),
        },
    )
    assert_frame_equal(expected, df)

    # if schema indicates string, ensure we do *not* convert to enum
    df = pl.DataFrame(
        {
            "trade_id": [123, 456, 789],
            "portfolio": [Portfolio.OTHER, Portfolio.TECH, Portfolio.RETAIL],
        },
        schema_overrides={"portfolio": pl.String},
    )
    assert df.schema == {"trade_id": pl.Int64, "portfolio": pl.String}


@pytest.mark.parametrize(
    "EnumBase",
    [
        (enum.Enum,),
        (enum.Flag,),
        (enum.IntEnum,),
        (enum.IntFlag,),
        (int, enum.Enum),
    ],
)
def test_init_series_from_int_enum(EnumBase: tuple[type, ...]) -> None:
    # note: we do not support integer enums as polars enums,
    # but we should be able to load the values

    class Number(*EnumBase):  # type: ignore[misc]
        ONE = 1
        TWO = 2
        FOUR = 4
        EIGHT = 8

    s = pl.Series(values=[Number.EIGHT, Number.TWO, Number.FOUR])

    expected = pl.Series(values=[8, 2, 4], dtype=pl.Int64)
    assert_series_equal(expected, s)


def test_read_enum_from_csv() -> None:
    df = pl.DataFrame(
        {
            "foo": ["ham", "spam", None, "and", "such"],
            "bar": ["ham", "spam", None, "and", "such"],
        }
    )
    f = io.BytesIO()
    df.write_csv(f)
    f.seek(0)

    schema = {"foo": pl.Enum(["ham", "and", "such", "spam"]), "bar": pl.String()}
    read = pl.read_csv(f, schema=schema)
    assert read.schema == schema
    assert_frame_equal(df.cast(schema), read)  # type: ignore[arg-type]
