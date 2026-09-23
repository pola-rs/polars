from __future__ import annotations

import re
from collections.abc import Collection
from datetime import date, datetime, time, timedelta
from decimal import Decimal as D
from typing import TYPE_CHECKING, Any

import pytest

import polars as pl
from polars.exceptions import InvalidOperationError
from polars.testing import assert_frame_equal, assert_series_equal

if TYPE_CHECKING:
    from collections.abc import Iterator

    from polars._typing import EngineType, PolarsDataType
    from tests.conftest import PlMonkeyPatch


def test_struct_logical_is_in() -> None:
    df1 = pl.DataFrame(
        {
            "x": pl.date_range(date(2022, 1, 1), date(2022, 1, 7), eager=True),
            "y": [0, 4, 6, 2, 3, 4, 5],
        }
    )
    df2 = pl.DataFrame(
        {
            "x": pl.date_range(date(2022, 1, 3), date(2022, 1, 9), eager=True),
            "y": [6, 2, 3, 4, 5, 0, 1],
        }
    )

    s1 = df1.select(pl.struct(["x", "y"])).to_series()
    s2 = df2.select(pl.struct(["x", "y"])).to_series()
    assert s1.is_in(s2).to_list() == [False, False, True, True, True, True, True]


def test_struct_logical_is_in_nonullpropagate() -> None:
    s = pl.Series([date(2022, 1, 1), date(2022, 1, 2), date(2022, 1, 3), None])
    df1 = pl.DataFrame(
        {
            "x": s,
            "y": [0, 4, 6, None],
        }
    )
    s = pl.Series([date(2022, 2, 1), date(2022, 1, 2), date(2022, 2, 3), None])
    df2 = pl.DataFrame(
        {
            "x": s,
            "y": [6, 4, 3, None],
        }
    )

    # Left has no nulls, right has nulls
    s1 = df1.select(pl.struct(["x", "y"])).to_series()
    s1 = s1.extend_constant(s1[0], 1)
    s2 = df2.select(pl.struct(["x", "y"])).to_series().extend_constant(None, 1)
    assert s1.is_in(s2, nulls_equal=False).to_list() == [
        False,
        True,
        False,
        True,
        False,
    ]
    assert s1.is_in(s2, nulls_equal=True).to_list() == [
        False,
        True,
        False,
        True,
        False,
    ]

    # Left has nulls, right has no nulls
    s1 = df1.select(pl.struct(["x", "y"])).to_series().extend_constant(None, 1)
    s2 = df2.select(pl.struct(["x", "y"])).to_series()
    s2 = s2.extend_constant(s2[0], 1)
    assert s1.is_in(s2, nulls_equal=False).to_list() == [
        False,
        True,
        False,
        True,
        None,
    ]
    assert s1.is_in(s2, nulls_equal=True).to_list() == [
        False,
        True,
        False,
        True,
        False,
    ]

    # Both have nulls
    # {None, None} is a valid element unaffected by the missing parameter.
    s1 = df1.select(pl.struct(["x", "y"])).to_series().extend_constant(None, 1)
    s2 = df2.select(pl.struct(["x", "y"])).to_series().extend_constant(None, 1)
    assert s1.is_in(s2, nulls_equal=False).to_list() == [
        False,
        True,
        False,
        True,
        None,
    ]
    assert s1.is_in(s2, nulls_equal=True).to_list() == [
        False,
        True,
        False,
        True,
        True,
    ]


@pytest.mark.parametrize("nulls_equal", [False, True])
def test_is_in_bool(nulls_equal: bool) -> None:
    vals = [True, None]
    df = pl.DataFrame({"A": [True, False, None]})
    missing_value = True if nulls_equal else None
    assert df.select(pl.col("A").is_in(vals, nulls_equal=nulls_equal)).to_dict(
        as_series=False
    ) == {"A": [True, False, missing_value]}


def test_is_in_bool_11216() -> None:
    s = pl.Series([False]).is_in([False, None])
    expected = pl.Series([True])
    assert_series_equal(s, expected)


@pytest.mark.parametrize("nulls_equal", [False, True])
def test_is_in_empty_list_4559(nulls_equal: bool) -> None:
    assert pl.Series(["a"]).is_in([], nulls_equal=nulls_equal).to_list() == [False]


def test_is_in_empty_list_4639() -> None:
    df = pl.DataFrame({"a": [1, None]})
    empty_list: list[int] = []

    result = df.with_columns([pl.col("a").is_in(empty_list).alias("a_in_list")])
    expected = pl.DataFrame({"a": [1, None], "a_in_list": [False, None]})
    assert_frame_equal(result, expected)


def test_is_in_struct() -> None:
    df = pl.DataFrame(
        {
            "struct_elem": [{"a": 1, "b": 11}, {"a": 1, "b": 90}],
            "struct_list": [
                [{"a": 1, "b": 11}, {"a": 2, "b": 12}, {"a": 3, "b": 13}],
                [{"a": 3, "b": 3}],
            ],
        }
    )

    assert df.filter(pl.col("struct_elem").is_in("struct_list")).to_dict(
        as_series=False
    ) == {
        "struct_elem": [{"a": 1, "b": 11}],
        "struct_list": [[{"a": 1, "b": 11}, {"a": 2, "b": 12}, {"a": 3, "b": 13}]],
    }


def test_is_in_null_prop() -> None:
    assert (
        pl.Series([None], dtype=pl.Float32)
        .is_in(pl.Series([42], dtype=pl.Float32))
        .item()
        is None
    )
    assert pl.Series([{"a": None}, None], dtype=pl.Struct({"a": pl.Float32})).is_in(
        pl.Series([{"a": 42}], dtype=pl.Struct({"a": pl.Float32}))
    ).to_list() == [False, None]

    assert pl.Series([{"a": None}, None], dtype=pl.Struct({"a": pl.Boolean})).is_in(
        pl.Series([{"a": 42}], dtype=pl.Struct({"a": pl.Boolean}))
    ).to_list() == [False, None]


def test_is_in_9070() -> None:
    with pytest.raises(
        InvalidOperationError,
        match=r"'is_in' cannot check for Int64 values in List\(Float64\) data",
    ):
        pl.Series([1]).is_in(pl.Series([1.99])).item()


def test_is_in_large_uint64_21966() -> None:
    # https://github.com/pola-rs/polars/issues/21966
    # Large integers beyond Float64 precision (2^53) should compare exactly,
    # not lose precision by casting to Float64.

    # Original issue: values differing only beyond float64 precision
    s = pl.Series([58830407606777880], dtype=pl.UInt64)
    assert not s.is_in([58830407606777883]).item()
    assert s.is_in([58830407606777880]).item()

    # Values at and beyond the float64 precision boundary (2^53)
    boundary = 2**53
    s = pl.Series([boundary, boundary + 1, boundary + 2], dtype=pl.UInt64)
    assert s.is_in([boundary]).to_list() == [True, False, False]
    assert s.is_in([boundary + 1]).to_list() == [False, True, False]

    # UInt64 vs Int64: should use Int128 supertype to preserve precision
    val = 2**53 + 1000
    s = pl.Series([val], dtype=pl.UInt64)
    assert s.is_in(pl.Series([val], dtype=pl.Int64)).item()
    assert not s.is_in(pl.Series([val + 1], dtype=pl.Int64)).item()

    # Int64 vs UInt64 (reverse direction)
    s = pl.Series([val], dtype=pl.Int64)
    assert s.is_in(pl.Series([val], dtype=pl.UInt64)).item()
    assert not s.is_in(pl.Series([val + 1], dtype=pl.UInt64)).item()

    # Negative values in signed list vs unsigned series (uses Int128 supertype)
    s = pl.Series([100], dtype=pl.UInt64)
    assert s.is_in(pl.Series([-1, 100, 200], dtype=pl.Int64)).item()
    assert not s.is_in(pl.Series([-1, 99, 200], dtype=pl.Int64)).item()

    # Smaller integer type combinations that have lossless supertypes
    s = pl.Series([100, 200], dtype=pl.UInt32)
    assert s.is_in(pl.Series([100, 300], dtype=pl.Int32)).to_list() == [True, False]

    s = pl.Series([100, 200], dtype=pl.Int16)
    assert s.is_in(pl.Series([100, 300], dtype=pl.UInt16)).to_list() == [True, False]

    # UInt64 max value (should use Int128)
    s = pl.Series([2**64 - 1], dtype=pl.UInt64)
    assert s.is_in(pl.Series([2**64 - 1], dtype=pl.UInt64)).item()
    assert not s.is_in(pl.Series([2**64 - 2], dtype=pl.UInt64)).item()

    # UInt128 and Int64 have no common supertype. The needle is cast, and a needle out
    # of range is a miss.
    s = pl.Series([100, 2**127], dtype=pl.UInt128)
    assert s.is_in(pl.Series([100, -1], dtype=pl.Int64)).to_list() == [True, False]


def test_is_in_float_list_10764() -> None:
    df = pl.DataFrame(
        {
            "lst": [[1.0, 2.0, 3.0, 4.0, 5.0], [3.14, 5.28]],
            "n": [3.0, 2.0],
        }
    )
    assert df.select(pl.col("n").is_in("lst").alias("is_in")).to_dict(
        as_series=False
    ) == {"is_in": [True, False]}


def test_is_in_df() -> None:
    df = pl.DataFrame({"a": [1, 2, 3]})
    assert df.select(pl.col("a").is_in([1, 2]))["a"].to_list() == [True, True, False]


def test_is_in_series() -> None:
    s = pl.Series(["a", "b", "c"])

    out = s.is_in(["a", "b"])
    assert out.to_list() == [True, True, False]

    # Check if empty list is converted to pl.String
    out = s.is_in([])
    assert out.to_list() == [False] * out.len()

    for x_y_z in (["x", "y", "z"], {"x", "y", "z"}):
        out = s.is_in(x_y_z)
        assert out.to_list() == [False, False, False]

    df = pl.DataFrame({"a": [1.0, 2.0], "b": [1, 4], "c": ["e", "d"]})
    assert df.select(
        pl.col("a").is_in(pl.col("b").cast(pl.Float64))
    ).to_series().to_list() == [
        True,
        False,
    ]
    assert df.select(pl.col("b").is_in([])).to_series().to_list() == [False] * df.height

    with pytest.raises(
        InvalidOperationError,
        match=r"'is_in' cannot check for Int64 values in List\(String\) data",
    ):
        df.select(pl.col("b").is_in(["x", "x"]))

    # check we don't shallow-copy and accidentally modify 'a' (see: #10072)
    a = pl.Series("a", [1, 2])
    b = pl.Series("b", [1, 3]).is_in(a)

    assert a.name == "a"
    assert_series_equal(b, pl.Series("b", [True, False]))


@pytest.mark.parametrize("nulls_equal", [False, True])
def test_is_in_null(nulls_equal: bool) -> None:
    # No nulls in right
    s = pl.Series([None, None], dtype=pl.Null)
    result = s.is_in([1, 2], nulls_equal=nulls_equal)
    missing_value = False if nulls_equal else None
    expected = pl.Series([missing_value, missing_value], dtype=pl.Boolean)
    assert_series_equal(result, expected)

    # Nulls in right
    s = pl.Series([None, None], dtype=pl.Null)
    result = s.is_in([None, None], nulls_equal=nulls_equal)
    missing_value = True if nulls_equal else None
    expected = pl.Series([missing_value, missing_value], dtype=pl.Boolean)
    assert_series_equal(result, expected)


@pytest.mark.parametrize("nulls_equal", [False, True])
def test_is_in_boolean(nulls_equal: bool) -> None:
    # Nulls in neither left nor right
    s = pl.Series([True, False])
    result = s.is_in([True, False], nulls_equal=nulls_equal)
    expected = pl.Series([True, True])
    assert_series_equal(result, expected)

    # Nulls in left only
    s = pl.Series([True, None])
    result = s.is_in([False, False], nulls_equal=nulls_equal)
    missing_value = False if nulls_equal else None
    expected = pl.Series([False, missing_value])
    assert_series_equal(result, expected)

    # Nulls in right only
    s = pl.Series([True, False])
    result = s.is_in([True, None], nulls_equal=nulls_equal)
    expected = pl.Series([True, False])
    assert_series_equal(result, expected)

    # Nulls in both
    s = pl.Series([True, False, None])
    result = s.is_in([True, None], nulls_equal=nulls_equal)
    missing_value = True if nulls_equal else None
    expected = pl.Series([True, False, missing_value])
    assert_series_equal(result, expected)


@pytest.mark.parametrize("dtype", [pl.List(pl.Boolean), pl.Array(pl.Boolean, 2)])
@pytest.mark.parametrize("nulls_equal", [False, True])
def test_is_in_boolean_list(dtype: PolarsDataType, nulls_equal: bool) -> None:
    # Note list is_in does not propagate nulls.
    df = pl.DataFrame(
        {
            "a": [True, False, None, None, None],
            "b": pl.Series(
                [
                    [True, False],
                    [True, True],
                    [None, True],
                    [False, True],
                    [True, True],
                ],
                dtype=dtype,
            ),
        }
    )
    missing_true = True if nulls_equal else None
    missing_false = False if nulls_equal else None
    result = df.select(pl.col("a").is_in("b", nulls_equal=nulls_equal))["a"]
    expected = pl.Series("a", [True, False, missing_true, missing_false, missing_false])
    assert_series_equal(result, expected)


def test_is_in_invalid_shape() -> None:
    with pytest.raises(InvalidOperationError):
        pl.Series("a", [1, 2, 3]).is_in([[], []])


def test_is_in_list_rhs() -> None:
    assert_series_equal(
        pl.Series([1, 2, 3, 4, 5]).is_in(pl.Series([[1], [2, 9], [None], None, None])),
        pl.Series([True, True, False, None, None]),
    )


@pytest.mark.parametrize("dtype", [pl.Float32, pl.Float64])
def test_is_in_float(dtype: PolarsDataType) -> None:
    s = pl.Series([float("nan"), 0.0], dtype=dtype)
    result = s.is_in([-0.0, -float("nan")])
    expected = pl.Series([True, True], dtype=pl.Boolean)
    assert_series_equal(result, expected)


@pytest.mark.parametrize(
    ("df", "matches", "expected_error"),
    [
        (
            pl.DataFrame({"a": [1, 2], "b": [[1, 2], [3, 4]]}),
            [True, False],
            None,
        ),
        (
            pl.DataFrame({"a": [2.5, 3.0], "b": [[1.5, 2.0], [2.5, 3.0]]}),
            [False, True],
            None,
        ),
        (
            pl.DataFrame(
                {"a": [None, None], "b": [[1, 2], [3, 4]]},
                schema_overrides={"a": pl.Null},
            ),
            [None, None],
            None,
        ),
        (
            pl.DataFrame({"a": ["1", "2"], "b": [[1, 2], [3, 4]]}),
            None,
            r"'is_in' cannot check for String values in List\(Int64\) data",
        ),
        (
            pl.DataFrame({"a": [date.today(), None], "b": [[1, 2], [3, 4]]}),
            None,
            r"'is_in' cannot check for Date values in List\(Int64\) data",
        ),
    ],
)
def test_is_in_expr_list_series(
    df: pl.DataFrame, matches: list[bool] | None, expected_error: str | None
) -> None:
    expr_is_in = pl.col("a").is_in(pl.col("b"))
    if matches:
        assert df.select(expr_is_in).to_series().to_list() == matches
    else:
        with pytest.raises(InvalidOperationError, match=expected_error):
            df.select(expr_is_in)


@pytest.mark.parametrize(
    ("df", "matches"),
    [
        (
            pl.DataFrame({"a": [1.0, None], "b": [[1.0, 2.5, 4.0], [3.0, 4.0, 5.0]]}),
            [True, False],
        ),
        (
            pl.DataFrame({"a": [1.0, None], "b": [[0.0, 2.5, None], [3.0, 4.0, None]]}),
            [False, True],
        ),
        (
            pl.DataFrame(
                {"a": [None, None], "b": [[1, 2], [3, 4]]},
                schema_overrides={"a": pl.Null},
            ),
            [False, False],
        ),
        (
            pl.DataFrame(
                {"a": [None, None], "b": [[1, 2], [3, None]]},
                schema_overrides={"a": pl.Null},
            ),
            [False, True],
        ),
    ],
)
def test_is_in_expr_list_series_nonullpropagate(
    df: pl.DataFrame, matches: list[bool]
) -> None:
    expr_is_in = pl.col("a").is_in(pl.col("b"), nulls_equal=True)
    assert df.select(expr_is_in).to_series().to_list() == matches


@pytest.mark.parametrize("nulls_equal", [False, True])
def test_is_in_null_series(nulls_equal: bool) -> None:
    df = pl.DataFrame({"a": ["a", "b", None]})
    result = df.select(pl.col("a").is_in([None], nulls_equal=nulls_equal))
    missing_value = True if nulls_equal else None
    expected = pl.DataFrame({"a": [False, False, missing_value]})
    assert_frame_equal(result, expected)


def test_is_in_int_range() -> None:
    r = pl.int_range(0, 3, eager=False)
    out = pl.select(r.is_in([1, 2])).to_series()
    assert out.to_list() == [False, True, True]

    r = pl.int_range(0, 3, eager=True)  # type: ignore[assignment]
    out = r.is_in([1, 2])  # type: ignore[assignment]
    assert out.to_list() == [False, True, True]


def test_is_in_date_range() -> None:
    r = pl.date_range(date(2023, 1, 1), date(2023, 1, 3), eager=False)
    out = pl.select(r.is_in([date(2023, 1, 2), date(2023, 1, 3)])).to_series()
    assert out.to_list() == [False, True, True]

    r = pl.date_range(date(2023, 1, 1), date(2023, 1, 3), eager=True)  # type: ignore[assignment]
    out = r.is_in([date(2023, 1, 2), date(2023, 1, 3)])  # type: ignore[assignment]
    assert out.to_list() == [False, True, True]


@pytest.mark.parametrize("dtype", [pl.Categorical, pl.Enum(["a", "b", "c"])])
@pytest.mark.parametrize("nulls_equal", [False, True])
def test_cat_is_in_series(dtype: pl.DataType, nulls_equal: bool) -> None:
    s = pl.Series(["a", "b", "c", None], dtype=dtype)
    s2 = pl.Series(["b", "c"], dtype=dtype)
    missing_value = False if nulls_equal else None
    expected = pl.Series([False, True, True, missing_value])
    assert_series_equal(s.is_in(s2, nulls_equal=nulls_equal), expected)

    s2_str = s2.cast(pl.String)
    assert_series_equal(s.is_in(s2_str, nulls_equal=nulls_equal), expected)


@pytest.mark.parametrize("nulls_equal", [False, True])
def test_cat_is_in_series_non_existent(nulls_equal: bool) -> None:
    dtype = pl.Categorical
    s = pl.Series(["a", "b", "c", None], dtype=dtype)
    s2 = pl.Series(["a", "d", "e"], dtype=dtype)
    missing_value = False if nulls_equal else None
    expected = pl.Series([True, False, False, missing_value])
    assert_series_equal(s.is_in(s2, nulls_equal=nulls_equal), expected)

    s2_str = s2.cast(pl.String)
    assert_series_equal(s.is_in(s2_str, nulls_equal=nulls_equal), expected)


@pytest.mark.parametrize(
    "nulls_equal",
    [False, True],
)
def test_enum_is_in_series_non_existent(nulls_equal: bool) -> None:
    dtype = pl.Enum(["a", "b", "c"])
    missing_value = False if nulls_equal else None
    s = pl.Series(["a", "b", "c", None], dtype=dtype)
    s2_str = pl.Series(["a", "d", "e"])
    expected = pl.Series([True, False, False, missing_value])

    # A label outside the categories cannot match; it is not an error.
    assert_series_equal(s.is_in(s2_str.implode(), nulls_equal=nulls_equal), expected)
    assert_series_equal(s.is_in(["a", "d", "e"], nulls_equal=nulls_equal), expected)

    out = s.is_in(["a"], nulls_equal=nulls_equal)
    assert_series_equal(out, expected)


@pytest.mark.parametrize("dtype", [pl.Categorical, pl.Enum(["a", "b", "c"])])
@pytest.mark.parametrize("nulls_equal", [False, True])
def test_cat_is_in_with_lit_str(dtype: pl.DataType, nulls_equal: bool) -> None:
    missing_value = False if nulls_equal else None
    s = pl.Series(["a", "b", "c", None], dtype=dtype)
    lit = ["b"]
    expected = pl.Series([False, True, False, missing_value])

    assert_series_equal(s.is_in(lit, nulls_equal=nulls_equal), expected)


@pytest.mark.parametrize("nulls_equal", [False, True])
def test_cat_is_in_with_lit_str_non_existent(nulls_equal: bool) -> None:
    dtype = pl.Categorical()
    missing_value = False if nulls_equal else None
    s = pl.Series(["a", "b", "c", None], dtype=dtype)
    lit = ["d"]
    expected = pl.Series([False, False, False, missing_value])

    assert_series_equal(s.is_in(lit, nulls_equal=nulls_equal), expected)


@pytest.mark.parametrize("dtype", [pl.Categorical, pl.Enum(["a", "b", "c"])])
def test_cat_is_in_with_lit_str_cache_setup(dtype: pl.DataType) -> None:
    # init the global cache
    _ = pl.Series(["c", "b", "a"], dtype=dtype)

    assert_series_equal(pl.Series(["a"], dtype=dtype).is_in(["a"]), pl.Series([True]))
    assert_series_equal(pl.Series(["b"], dtype=dtype).is_in(["b"]), pl.Series([True]))
    assert_series_equal(pl.Series(["c"], dtype=dtype).is_in(["c"]), pl.Series([True]))


def test_is_in_with_wildcard_13809() -> None:
    out = pl.DataFrame({"A": ["B"]}).select(pl.all().is_in(["C"]))
    expected = pl.DataFrame({"A": [False]})
    assert_frame_equal(out, expected)


@pytest.mark.parametrize(
    "dtype",
    [
        pl.Categorical,
        pl.Enum(["a", "b", "c", "d"]),
    ],
)
def test_cat_is_in_from_str(dtype: pl.DataType) -> None:
    s = pl.Series(["c", "c", "b"], dtype=dtype)

    # test local
    assert_series_equal(
        pl.Series(["a", "d", "b"]).is_in(s),
        pl.Series([False, False, True]),
    )


@pytest.mark.parametrize("dtype", [pl.Categorical, pl.Enum(["a", "b", "c", "d"])])
def test_cat_list_is_in_from_cat(dtype: pl.DataType) -> None:
    df = pl.DataFrame(
        [
            (["a", "b"], "c"),
            (["a", "b"], "a"),
            (["a", None], None),
            (["a", "c"], None),
            (["a"], "d"),
        ],
        schema={"li": pl.List(dtype), "x": dtype},
        orient="row",
    )
    res = df.select(pl.col("li").list.contains(pl.col("x")))
    expected_df = pl.DataFrame({"li": [False, True, True, False, False]})
    assert_frame_equal(res, expected_df)


@pytest.mark.parametrize(
    ("val", "expected"),
    [
        ("b", [True, False, False, None, True]),
        (None, [False, False, True, None, False]),
        ("e", [False, False, False, None, False]),
    ],
)
def test_cat_list_is_in_from_cat_single(val: str | None, expected: list[bool]) -> None:
    df = pl.Series(
        "li",
        [["a", "b"], ["a", "c"], ["a", None], None, ["b"]],
        dtype=pl.List(pl.Categorical),
    ).to_frame()
    res = df.select(pl.col("li").list.contains(pl.lit(val, dtype=pl.Categorical)))
    expected_df = pl.DataFrame({"li": expected})
    assert_frame_equal(res, expected_df)


def test_cat_list_is_in_from_str() -> None:
    df = pl.DataFrame(
        [
            (["a", "b"], "c"),
            (["a", "b"], "a"),
            (["a", None], None),
            (["a", "c"], None),
            (["a"], "d"),
        ],
        schema={"li": pl.List(pl.Categorical), "x": pl.String},
        orient="row",
    )
    res = df.select(pl.col("li").list.contains(pl.col("x")))
    expected_df = pl.DataFrame({"li": [False, True, True, False, False]})
    assert_frame_equal(res, expected_df)


@pytest.mark.parametrize(
    ("val", "expected"),
    [
        ("b", [True, False, False, None, True]),
        (None, [False, False, True, None, False]),
        ("e", [False, False, False, None, False]),
    ],
)
def test_cat_list_is_in_from_single_str(val: str | None, expected: list[bool]) -> None:
    df = pl.Series(
        "li",
        [["a", "b"], ["a", "c"], ["a", None], None, ["b"]],
        dtype=pl.List(pl.Categorical),
    ).to_frame()
    res = df.select(pl.col("li").list.contains(pl.lit(val, dtype=pl.String)))
    expected_df = pl.DataFrame({"li": expected})
    assert_frame_equal(res, expected_df)


@pytest.mark.parametrize("nulls_equal", [False, True])
def test_is_in_struct_enum_17618(nulls_equal: bool) -> None:
    df = pl.DataFrame()
    dtype = pl.Enum(categories=["HBS"])
    df = df.insert_column(0, pl.Series("category", [], dtype=dtype))
    assert df.filter(
        pl.struct("category").is_in(
            pl.Series(
                [{"category": "HBS"}],
                dtype=pl.Struct({"category": df["category"].dtype}),
            ),
            nulls_equal=nulls_equal,
        )
    ).shape == (0, 1)


@pytest.mark.parametrize("nulls_equal", [False, True])
def test_is_in_decimal(nulls_equal: bool) -> None:
    assert pl.DataFrame({"a": [D("0.0"), D("0.2"), D("0.1")]}).select(
        pl.col("a").is_in([D("0.0"), D("0.1")], nulls_equal=nulls_equal)
    )["a"].to_list() == [True, False, True]
    missing_value = True if nulls_equal else None
    assert pl.DataFrame({"a": [D("0.0"), D("0.2"), None]}).select(
        pl.col("a").is_in([D("0.0"), D("0.1"), None], nulls_equal=nulls_equal)
    )["a"].to_list() == [True, False, missing_value]

    for haystack in ([0.0, 0.1], [1, 0, 2]):
        with pytest.raises(InvalidOperationError, match="cannot check for Decimal"):
            pl.DataFrame({"a": [D("0.0")]}).select(
                pl.col("a").is_in(haystack, nulls_equal=nulls_equal)
            )


def test_is_in_collection() -> None:
    df = pl.DataFrame(
        {
            "lbl": ["aa", "bb", "cc", "dd", "ee"],
            "val": [0, 1, 2, 3, 4],
        }
    )

    class CustomCollection(Collection[int]):
        def __init__(self, vals: Collection[int]) -> None:
            super().__init__()
            self.vals = vals

        def __contains__(self, x: object) -> bool:
            return x in self.vals

        def __iter__(self) -> Iterator[int]:
            yield from self.vals

        def __len__(self) -> int:
            return len(self.vals)

    for constraint_values in (
        {3, 2, 1},
        frozenset({3, 2, 1}),
        CustomCollection([3, 2, 1]),
    ):
        res = df.filter(pl.col("val").is_in(constraint_values))
        assert set(res["lbl"]) == {"bb", "cc", "dd"}


@pytest.mark.parametrize("nulls_equal", [False, True])
def test_null_propagate_all_paths(nulls_equal: bool) -> None:
    # No nulls in either
    s = pl.Series([1, 2, 3])
    result = s.is_in([1, 3, 8], nulls_equal=nulls_equal)
    expected = pl.Series([True, False, True])
    assert_series_equal(result, expected)

    # Nulls in left only
    s = pl.Series([1, 2, None])
    result = s.is_in([1, 3, 8], nulls_equal=nulls_equal)
    missing_value = False if nulls_equal else None
    expected = pl.Series([True, False, missing_value])
    assert_series_equal(result, expected)

    # Nulls in right only
    s = pl.Series([1, 2, 3])
    result = s.is_in([1, 3, None], nulls_equal=nulls_equal)
    expected = pl.Series([True, False, True])
    assert_series_equal(result, expected)

    # Nulls in both
    s = pl.Series([1, 2, None])
    result = s.is_in([1, 3, None], nulls_equal=nulls_equal)
    missing_value = True if nulls_equal else None
    expected = pl.Series([True, False, missing_value])
    assert_series_equal(result, expected)


@pytest.mark.parametrize("nulls_equal", [False, True])
def test_null_propagate_all_paths_cat(nulls_equal: bool) -> None:
    # No nulls in either
    s = pl.Series(["1", "2", "3"])
    result = s.is_in(["1", "3", "8"], nulls_equal=nulls_equal)
    expected = pl.Series([True, False, True])
    assert_series_equal(result, expected)

    # Nulls in left only
    s = pl.Series(["1", "2", None])
    result = s.is_in(["1", "3", "8"], nulls_equal=nulls_equal)
    missing_value = False if nulls_equal else None
    expected = pl.Series([True, False, missing_value])
    assert_series_equal(result, expected)

    # Nulls in right only
    s = pl.Series(["1", "2", "3"])
    result = s.is_in(["1", "3", None], nulls_equal=nulls_equal)
    expected = pl.Series([True, False, True])
    assert_series_equal(result, expected)

    # Nulls in both
    s = pl.Series(["1", "2", None])
    result = s.is_in(["1", "3", None], nulls_equal=nulls_equal)
    missing_value = True if nulls_equal else None
    expected = pl.Series([True, False, missing_value])
    assert_series_equal(result, expected)


def test_is_in_non_nested_container() -> None:
    # The right-hand side (the container) must be a nested dtype.
    df = pl.DataFrame(
        {
            "a": pl.Series([[1, 2], [3, 4]], dtype=pl.List(pl.Int64)),
            "b": pl.Series([1, 3], dtype=pl.Int64),
        }
    )
    with pytest.raises(
        InvalidOperationError,
        match=r"(?s)'is_in' cannot check for List\(Int64\) values in Int64 data.*container dtype \(Int64\) must be nested",
    ):
        df.select(pl.col("a").is_in(pl.col("b")))


MEMBERSHIP_OPS = ["is_in-list", "is_in-array", "list.contains", "arr.contains"]


def _container(
    op: str, rows: list[list[Any] | None], inner: PolarsDataType
) -> pl.Series:
    if op.endswith("list") or op == "list.contains":
        return pl.Series("h", rows, dtype=pl.List(inner))
    width = len(next(r for r in rows if r is not None))
    return pl.Series("h", rows, dtype=pl.Array(inner, width))


def _membership(
    op: str, needle: pl.Expr, container: pl.Expr, *, nulls_equal: bool | None = None
) -> pl.Expr:
    kwargs = {} if nulls_equal is None else {"nulls_equal": nulls_equal}
    if op.startswith("is_in"):
        return needle.is_in(container, **kwargs)
    if op == "list.contains":
        return container.list.contains(needle, **kwargs)
    return container.arr.contains(needle, **kwargs)


_RUST_DTYPE: dict[PolarsDataType, str] = {
    pl.Int8: "i8",
    pl.Int64: "i64",
    pl.UInt64: "u64",
    pl.Float16: "f16",
    pl.Float32: "f32",
    pl.Datetime("ms"): "datetime[ms]",
    pl.Datetime("ns"): "datetime[ns]",
    pl.Duration("ms"): "duration[ms]",
}


def _assert_needle_cast(lf: pl.LazyFrame, dtype: str) -> None:
    # The needle is cast as the function runs; the container is never rewritten.
    plan = lf.explain()
    assert f"[needle: {dtype}]" in plan
    assert ".cast(" not in plan


@pytest.mark.parametrize("op", MEMBERSHIP_OPS)
@pytest.mark.parametrize(
    ("needle_dtype", "inner", "hit", "miss"),
    [
        pytest.param(pl.Int64, pl.Int8, 7, 300, id="int-narrowing"),
        pytest.param(pl.Int8, pl.UInt64, 7, -1, id="int-sign"),
        pytest.param(pl.UInt128, pl.Int64, 7, 2**64, id="u128-signed"),
        pytest.param(
            pl.Datetime("us"),
            pl.Datetime("ms"),
            datetime(2020, 1, 1, 0, 0, 0, 1000),
            datetime(2020, 1, 1, 0, 0, 0, 1001),
            id="datetime-finer",
        ),
        pytest.param(
            pl.Datetime("us"),
            pl.Datetime("ns"),
            datetime(2020, 1, 1),
            datetime(2500, 1, 1),
            id="datetime-coarser",
        ),
        pytest.param(
            pl.Duration("us"),
            pl.Duration("ms"),
            timedelta(milliseconds=1),
            timedelta(microseconds=1001),
            id="duration-finer",
        ),
    ],
)
def test_is_in_casts_the_needle_to_the_element_dtype(
    op: str, needle_dtype: PolarsDataType, inner: PolarsDataType, hit: Any, miss: Any
) -> None:
    # A needle the cast cannot represent exactly (out of range, rounded, overflowing)
    # equals no element, so it is a miss rather than an error or a rounded hit.
    hay = [hit, hit] if op.endswith("array") or op == "arr.contains" else [hit]
    lf = pl.LazyFrame(
        {
            "n": pl.Series([hit, miss, None], dtype=needle_dtype),
            "h": _container(op, [hay, hay, hay], inner),
        }
    )

    column = lf.select(_membership(op, pl.col("n"), pl.col("h")).alias("o"))
    _assert_needle_cast(column, _RUST_DTYPE[inner])
    # `is_in` defaults to `nulls_equal=False`, the `contains` methods to `True`.
    null = None if op.startswith("is_in") else False
    assert column.collect()["o"].to_list() == [True, False, null]

    for value, expected in ((hit, True), (miss, False)):
        literal = lf.select(
            _membership(op, pl.lit(value, needle_dtype), pl.col("h")).alias("o")
        )
        plan = literal.explain()
        assert "[needle:" not in plan
        assert ".cast(" not in plan
        # An inexact literal is resolved at plan time and never reaches the kernel.
        assert (op.split("-")[0] in plan) == expected
        assert literal.collect()["o"].to_list() == [expected] * 3


@pytest.mark.parametrize("op", MEMBERSHIP_OPS)
@pytest.mark.parametrize("nulls_equal", [False, True])
@pytest.mark.parametrize(
    ("needle", "needle_dtype", "element", "inner"),
    [
        # Floors to a `ms` value whose cast back to `us` overflows to null.
        pytest.param(
            -(2**63) + 1,
            pl.Datetime("us"),
            -(2**63) // 1000,
            pl.Datetime("ms"),
            id="min-datetime",
        ),
        # Overflows to null, which must not read as a null element.
        pytest.param(
            16725225600000000, pl.Datetime("us"), None, pl.Datetime("ns"), id="overflow"
        ),
    ],
)
def test_is_in_needle_with_an_inexact_cast_never_matches(
    op: str,
    nulls_equal: bool,
    needle: int,
    needle_dtype: PolarsDataType,
    element: int | None,
    inner: PolarsDataType,
) -> None:
    hay = (
        [element, element]
        if op.endswith("array") or op == "arr.contains"
        else [element]
    )
    df = pl.DataFrame(
        {
            "n": pl.Series([needle, needle]).cast(needle_dtype),
            "h": _container(op, [hay, None], pl.Int64).cast(
                _container(op, [hay], inner).dtype
            ),
        }
    )

    for n in (pl.col("n"), pl.lit(needle).cast(needle_dtype)):
        out = df.select(
            _membership(op, n, pl.col("h"), nulls_equal=nulls_equal).alias("o")
        )
        # A null container stays null.
        assert out["o"].to_list() == [False, None]


@pytest.mark.parametrize("op", MEMBERSHIP_OPS)
@pytest.mark.parametrize("dtype", [pl.Enum(["a", "b"]), pl.Categorical])
@pytest.mark.parametrize("needle_is_string", [False, True])
def test_is_in_compares_strings_with_categories_natively(
    op: str, dtype: PolarsDataType, needle_is_string: bool
) -> None:
    # Neither side is cast: an unknown label matches nothing, on whichever side it is.
    needle_dtype, inner = (pl.String, dtype) if needle_is_string else (dtype, pl.String)
    hay = ["a", "b"] if needle_is_string else ["a", "z"]
    lf = pl.LazyFrame(
        {
            "n": pl.Series(["a", "b", None], dtype=needle_dtype),
            "h": _container(op, [hay, hay, hay], inner),
        }
    )
    if needle_is_string:
        lf = lf.with_columns(pl.lit(pl.Series(["a", "z", None])).alias("n"))

    q = lf.select(
        _membership(op, pl.col("n"), pl.col("h"), nulls_equal=True).alias("o")
    )
    assert ".cast(" not in q.explain()
    expected = [True, False, False]
    assert q.collect()["o"].to_list() == expected

    for value, found in (("a", True), ("z" if needle_is_string else "b", False)):
        needle = pl.lit(value, needle_dtype)
        out = lf.select(_membership(op, needle, pl.col("h")).alias("o")).collect()
        assert out["o"].to_list() == [found] * 3


@pytest.mark.parametrize(
    "op", ["is_in-list", "is_in-array", "list.contains", "arr.contains"]
)
def test_is_in_null_category_does_not_match_an_unknown_label(op: str) -> None:
    # An element without a category must not read as a null element.
    dtype = pl.Enum(["a"])
    df = pl.DataFrame(
        {
            "n": pl.Series([None, None], dtype=dtype),
            "h": _container(op, [["z", "z"], ["z", None]], pl.String),
        }
    )

    out = df.select(_membership(op, pl.col("n"), pl.col("h"), nulls_equal=True))
    assert out.to_series().to_list() == [False, True]


@pytest.mark.parametrize("op", MEMBERSHIP_OPS)
@pytest.mark.parametrize(
    ("needle", "found"),
    [
        pytest.param(D("1.50"), True, id="hit"),
        pytest.param(D("1.55"), False, id="rounded"),
        # Rounds to `10.0`, whose cast back to `Decimal(3, 2)` overflows.
        pytest.param(D("9.99"), False, id="reverse-overflow"),
    ],
)
def test_is_in_decimal_of_another_scale(op: str, needle: D, found: bool) -> None:
    hay = [D("1.5"), D("10.0")]
    lf = pl.LazyFrame(
        {
            "n": pl.Series([needle] * 2, dtype=pl.Decimal(3, 2)),
            "h": _container(op, [hay, None], pl.Decimal(3, 1)),
        }
    )

    for n in (pl.col("n"), pl.lit(needle, pl.Decimal(3, 2))):
        q = lf.select(_membership(op, n, pl.col("h")).alias("o"))
        # The kernel compares any precision and scale; neither side is cast.
        assert ".cast(" not in q.explain()
        assert q.collect()["o"].to_list() == [found, None]


@pytest.mark.parametrize("op", MEMBERSHIP_OPS)
@pytest.mark.parametrize(
    ("needle_dtype", "inner"),
    [
        pytest.param(pl.Float64, pl.Float32, id="f64-f32"),
        pytest.param(pl.Float64, pl.Float16, id="f64-f16"),
        pytest.param(pl.Float32, pl.Float16, id="f32-f16"),
    ],
)
def test_is_in_narrows_a_float_needle(
    op: str, needle_dtype: PolarsDataType, inner: PolarsDataType
) -> None:
    # `1.1` has no exact value in the narrower type, so it must not round onto one.
    hay = [1.5, 1.1]
    lf = pl.LazyFrame(
        {
            "n": pl.Series([1.5, 1.1, float("nan")], dtype=needle_dtype),
            "h": _container(op, [hay, hay, hay], inner),
        }
    )

    q = lf.select(_membership(op, pl.col("n"), pl.col("h")).alias("o"))
    _assert_needle_cast(q, _RUST_DTYPE[inner])
    assert q.collect()["o"].to_list() == [True, False, False]
    for value, found in ((1.5, True), (1.1, False)):
        needle = pl.lit(value, needle_dtype)
        out = lf.select(_membership(op, needle, pl.col("h")).alias("o")).collect()
        assert out["o"].to_list() == [found] * 3


@pytest.mark.parametrize("op", MEMBERSHIP_OPS)
@pytest.mark.parametrize(
    ("needle_dtype", "inner"),
    [
        pytest.param(pl.Float16, pl.Float64, id="f16-f64"),
        pytest.param(pl.Float32, pl.Float64, id="f32-f64"),
        pytest.param(pl.Int8, pl.Int64, id="i8-i64"),
    ],
)
def test_is_in_widens_a_numeric_needle_exactly(
    op: str, needle_dtype: PolarsDataType, inner: PolarsDataType
) -> None:
    hay = [1, 2]
    lf = pl.LazyFrame(
        {
            "n": pl.Series([1, 3], dtype=needle_dtype),
            "h": _container(op, [hay, hay], inner),
        }
    )

    q = lf.select(_membership(op, pl.col("n"), pl.col("h")).alias("o"))
    plan = q.explain()
    assert re.search(r'col\("n"\)\.(strict_)?cast\(', plan)
    assert "[needle:" not in plan
    assert 'col("h").cast' not in plan
    assert q.collect()["o"].to_list() == [True, False]


@pytest.mark.parametrize("op", MEMBERSHIP_OPS)
@pytest.mark.parametrize("unit", ["us", "ms"])
def test_is_in_compares_instants_across_time_zones(op: str, unit: str) -> None:
    midnight = pl.lit(datetime(2020, 1, 1)).dt.cast_time_unit(unit)  # type: ignore[arg-type]
    hay = [datetime(2020, 1, 1), datetime(2020, 1, 2)]
    df = pl.DataFrame({"h": _container(op, [hay], pl.Datetime("us"))}).with_columns(
        pl.col("h").cast(_container(op, [hay], pl.Datetime("us", "UTC")).dtype)
    )

    # The same instant, shown in another zone.
    same = midnight.dt.replace_time_zone("UTC").dt.convert_time_zone("America/New_York")
    # Midnight in New York is a different instant from midnight UTC.
    other = midnight.dt.replace_time_zone("America/New_York")
    for needle, found in ((same, True), (other, False)):
        for n in (needle, pl.col("n")):
            out = df.with_columns(n=needle).select(
                _membership(op, n, pl.col("h")).alias("o")
            )
            assert out["o"].to_list() == [found]


@pytest.mark.parametrize("op", MEMBERSHIP_OPS)
@pytest.mark.parametrize("needle_is_aware", [False, True])
def test_is_in_rejects_naive_and_aware_datetimes(
    op: str, needle_is_aware: bool
) -> None:
    aware = pl.Datetime("us", "UTC")
    needle_dtype, inner = (
        (aware, pl.Datetime("us")) if needle_is_aware else (pl.Datetime("us"), aware)
    )
    hay = [datetime(2020, 1, 1), datetime(2020, 1, 2)]
    df = pl.DataFrame(
        {
            "n": pl.Series([datetime(2020, 1, 1)]).cast(needle_dtype),
            "h": _container(op, [hay], pl.Datetime("us")),
        }
    ).with_columns(pl.col("h").cast(_container(op, [hay], inner).dtype))

    with pytest.raises(InvalidOperationError, match="time-zone-aware"):
        df.select(_membership(op, pl.col("n"), pl.col("h")))


@pytest.mark.parametrize("op", MEMBERSHIP_OPS)
@pytest.mark.parametrize(
    ("needle", "needle_dtype", "inner"),
    [
        pytest.param([1], pl.List(pl.Int64), pl.List(pl.Null), id="list"),
        pytest.param(
            {"a": 1},
            pl.Struct({"a": pl.Int64}),
            pl.Struct({"a": pl.Null}),
            id="struct",
        ),
        pytest.param("a", pl.Enum(["a"]), pl.Null, id="null"),
    ],
)
def test_is_in_null_elements_take_the_needle_dtype(
    op: str, needle: Any, needle_dtype: PolarsDataType, inner: PolarsDataType
) -> None:
    # Elements that can only be null are cast to the needle's dtype, which is always
    # valid.
    null = (
        {"a": None}
        if isinstance(needle, dict)
        else ([None] if isinstance(needle, list) else None)
    )
    df = pl.DataFrame({"h": _container(op, [[null, null]], inner)})

    out = df.select(
        _membership(op, pl.lit(needle, needle_dtype), pl.col("h")).alias("o")
    )
    assert out["o"].to_list() == [False]


@pytest.mark.parametrize(
    ("values", "haystack", "same_dtype_haystack"),
    [
        pytest.param(
            pl.Series(["a", "b"]),
            pl.Series([["a"]], dtype=pl.List(pl.Enum(["a", "b"]))),
            pl.Series([["a"]]),
            id="string-enum",
        ),
        pytest.param(
            pl.Series([D("1.50"), D("2.50")], dtype=pl.Decimal(3, 2)),
            pl.Series([[D("1.5")]], dtype=pl.List(pl.Decimal(3, 1))),
            pl.Series([[D("1.50")]], dtype=pl.List(pl.Decimal(3, 2))),
            id="decimal-scale",
        ),
    ],
)
def test_is_in_haystack_of_another_dtype_is_not_a_filter_constraint(
    values: pl.Series, haystack: pl.Series, same_dtype_haystack: pl.Series
) -> None:
    # The kernel compares these natively; their values differ from the column's as
    # scalars, so intersecting them as allowed sets would wrongly empty the filter.
    lf = pl.LazyFrame({"c": values})
    q = lf.filter(
        pl.col("c").is_in(pl.lit(haystack))
        & pl.col("c").is_in(pl.lit(same_dtype_haystack))
    )

    assert "FILTER" in q.explain()
    assert q.collect()["c"].to_list() == values.head(1).to_list()


def test_is_in_does_not_match_null_on_temporal_overflow() -> None:
    # A needle outside the stored unit's range must not be confused with a null element.
    df = pl.DataFrame(
        {
            "k": [datetime(2500, 1, 1)],
            "h": pl.Series([[None]], dtype=pl.List(pl.Datetime("ns"))),
        }
    )

    for expr in (
        pl.col("h").list.contains(pl.col("k")),
        pl.col("k").is_in(pl.col("h"), nulls_equal=True),
    ):
        assert df.select(expr.alias("o"))["o"].to_list() == [False]


@pytest.mark.parametrize("op", MEMBERSHIP_OPS)
def test_is_in_inexact_literal_needle_keeps_the_shape(op: str) -> None:
    hay = [1, 2] if op.endswith("array") or op == "arr.contains" else [1]
    df = pl.DataFrame({"g": [1, 1, 2], "h": _container(op, [hay, None, hay], pl.Int8)})
    expr = _membership(op, pl.lit(300, pl.Int64), pl.col("h")).alias("o")

    assert df.select(expr)["o"].to_list() == [False, None, False]
    assert df.head(0).select(expr).height == 0
    out = df.group_by("g", maintain_order=True).agg(expr)
    assert out["o"].to_list() == [[False, None], [False]]


@pytest.mark.parametrize("op", MEMBERSHIP_OPS)
@pytest.mark.parametrize(
    ("needle_dtype", "inner", "values"),
    [
        pytest.param(pl.Int64, pl.Int8, list(range(30)), id="int"),
        pytest.param(
            pl.Float64, pl.Float32, [i / 4 for i in range(30)], id="float-narrowing"
        ),
        pytest.param(
            pl.Datetime("us"),
            pl.Datetime("ms"),
            [datetime(2020, 1, 1) + timedelta(milliseconds=i) for i in range(30)],
            id="temporal",
        ),
    ],
)
def test_is_in_needle_cast_evaluates_the_needle_once(
    op: str, needle_dtype: PolarsDataType, inner: PolarsDataType, values: list[Any]
) -> None:
    # Casting and guarding a needle must not evaluate it twice: a shuffled needle would
    # then be checked against a different permutation than the one searched for.
    df = pl.DataFrame({"h": _container(op, [values] * 30, inner)})
    needle = pl.lit(pl.Series(values, dtype=needle_dtype)).shuffle()

    out = df.select(_membership(op, needle, pl.col("h")).alias("o"))
    assert out["o"].to_list() == [True] * 30


@pytest.mark.parametrize("op", MEMBERSHIP_OPS)
@pytest.mark.parametrize("nulls_equal", [False, True])
def test_is_in_needle_cast_masks_with_the_evaluated_container(
    op: str, nulls_equal: bool
) -> None:
    # The null-container mask must come from the same evaluation the kernel searched.
    n = 30
    hay = [0, 1] if op.endswith("array") or op == "arr.contains" else [0]
    rows = [hay if i % 3 else None for i in range(n)]
    df = pl.DataFrame({"h": _container(op, rows, pl.Int8)})
    needle = pl.lit(pl.Series([300] * n, dtype=pl.Int64))

    out = df.select(
        _membership(op, needle, pl.col("h").shuffle(), nulls_equal=nulls_equal).alias(
            "o"
        )
    )
    assert out["o"].null_count() == sum(r is None for r in rows)
    assert out["o"].drop_nulls().to_list() == [False] * (n - out["o"].null_count())


def test_is_in_needle_cast_with_a_scalar_haystack() -> None:
    # A scalar haystack can lower to a semi join, which needs matching key dtypes.
    lf = pl.LazyFrame(
        {
            "n": pl.Series([1, 2, 300, None], dtype=pl.Int64),
            "h": pl.Series([[1, 2], [3], [4], [5]], dtype=pl.List(pl.Int8)),
        }
    )

    expr = pl.col("n").is_in(pl.col("h").first())
    _assert_needle_cast(lf.select(expr), "i8")
    assert lf.select(expr).collect()["n"].to_list() == [True, True, False, None]


@pytest.mark.parametrize("op", MEMBERSHIP_OPS)
@pytest.mark.parametrize(
    ("inner", "value"),
    [
        # Casting the elements to the needle's scale would round `1.005` onto `1.00`.
        pytest.param(pl.Float64, 1.005, id="float"),
        # Casting the elements to the needle's precision would overflow on this value.
        pytest.param(pl.Int64, 2**62, id="int"),
    ],
)
def test_is_in_rejects_a_decimal_needle_in_primitive_numeric_data(
    op: str, inner: PolarsDataType, value: Any
) -> None:
    df = pl.DataFrame({"h": _container(op, [[value]], inner)})

    with pytest.raises(InvalidOperationError, match="cannot check for Decimal"):
        df.select(_membership(op, pl.lit(D("1.00"), pl.Decimal(10, 2)), pl.col("h")))


@pytest.mark.parametrize("op", MEMBERSHIP_OPS)
@pytest.mark.parametrize("needle_dtype", [pl.Categorical, pl.Enum(["a"])])
@pytest.mark.parametrize("nulls_equal", [False, True])
def test_is_in_categorical_needle_in_null_data(
    op: str, needle_dtype: PolarsDataType, nulls_equal: bool
) -> None:
    df = pl.DataFrame({"h": _container(op, [[None]], pl.Null)})

    def search(value: str | None) -> list[bool | None]:
        needle = pl.lit(value, needle_dtype)
        expr = _membership(op, needle, pl.col("h"), nulls_equal=nulls_equal)
        return df.select(expr.alias("o"))["o"].to_list()

    assert search("a") == [False]
    assert search(None) == [True if nulls_equal else None]


@pytest.mark.parametrize("op", MEMBERSHIP_OPS)
def test_is_in_error_names_the_materialized_needle_dtype(op: str) -> None:
    df = pl.DataFrame({"h": _container(op, [[1]], pl.Int64)})

    with pytest.raises(InvalidOperationError) as exc:
        df.select(_membership(op, pl.lit(2.5), pl.col("h")))
    assert f"'{op.split('-')[0]}' cannot check for Float64 values" in str(exc.value)
    assert "Unknown" not in str(exc.value)


def _reference_is_in(
    needles: list[object], haystack: list[object], nulls_equal: bool
) -> list[bool | None]:
    def same(a: object, b: object) -> bool:
        if isinstance(a, float) and isinstance(b, float):
            return (a != a and b != b) or a == b
        return a == b

    out: list[bool | None] = []
    for n in needles:
        if n is None:
            out.append(any(h is None for h in haystack) if nulls_equal else None)
        else:
            out.append(any(h is not None and same(n, h) for h in haystack))
    return out


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
@pytest.mark.parametrize("nulls_equal", [False, True])
@pytest.mark.parametrize("null_in_haystack", [False, True])
@pytest.mark.parametrize(
    ("dtype", "needles", "haystack"),
    [
        # linear scan
        (pl.Int64, [1, 2, 3, None, -5, 2**62], [1, -5, 2**62, 9, 10]),
        # bitset with negative values
        (pl.Int32, [-100, 0, 100, 5000, None, 7], list(range(-100, 100, 3)) + [5000]),
        # bitset with values spanning most of the type
        (pl.Int8, [-128, 127, 0, None], list(range(-128, 128, 2))),
        # hash set: range too wide for a bitset
        (
            pl.Int64,
            [0, 10**12, -(10**12), 5, None, 6],
            [i * 10**9 for i in range(-1000, 1000, 7)] + [5],
        ),
        # extremes make the range overflow
        (
            pl.Int64,
            [-(2**63), 2**63 - 1, 0, None],
            [-(2**63), 2**63 - 1] + list(range(20)),
        ),
        (pl.UInt64, [0, 2**64 - 1, 3, None], [2**64 - 1, 3] + list(range(10, 30))),
        (
            pl.Int128,
            [-(2**100), 2**100, 1, None],
            [-(2**100), 2**100, 2, 3, 4, 5, 6, 7, 8],
        ),
        (pl.Float64, [1.0, -0.0, float("nan"), None, 2.5], [0.0, float("nan"), 2.5]),
        (
            pl.Float32,
            [1.0, -0.0, float("nan"), None, 2.5],
            [0.0, float("nan")] + [float(i) for i in range(10)],
        ),
        (pl.String, ["a", "", "bb", None, "zzz"], ["a", "", "ccc", "dddd"]),
        (
            pl.String,
            ["a", "", "bb", None, "x50", "x51"],
            [f"x{i}" for i in range(100)] + [""],
        ),
        (pl.Binary, [b"a", b"", b"bb", None], [b"a", b"", b"ccc"]),
        (pl.Boolean, [True, False, None], [True]),
        (pl.Boolean, [True, False, None], [False]),
        (pl.Boolean, [True, False, None], []),
        (pl.Date, [date(2020, 1, 1), date(2021, 1, 1), None], [date(2020, 1, 1)]),
        (pl.Decimal(10, 2), [D("1.50"), D("2.00"), None], [D("1.5"), D("3")]),
        (pl.List(pl.Int64), [[1, 2], [3], [], None, [None]], [[1, 2], [], [None]]),
        (
            pl.Struct({"a": pl.Int64, "b": pl.String}),
            [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}, None, {"a": None, "b": None}],
            [{"a": 1, "b": "x"}, {"a": None, "b": None}],
        ),
    ],
)
def test_is_in_literal_haystack_paths(
    engine: EngineType,
    nulls_equal: bool,
    null_in_haystack: bool,
    dtype: pl.DataType,
    needles: list[object],
    haystack: list[object],
) -> None:
    if null_in_haystack:
        haystack = [*haystack, None]
    needle_s = pl.Series("n", needles, dtype=dtype)
    haystack_s = pl.Series(haystack, dtype=dtype)
    expected = _reference_is_in(needles, haystack, nulls_equal)

    result = (
        needle_s.to_frame()
        .lazy()
        .select(pl.col("n").is_in(haystack_s.implode(), nulls_equal=nulls_equal))
        .collect(engine=engine)
        .to_series()
    )
    assert result.to_list() == expected


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
def test_is_in_literal_haystack_many_chunks(
    engine: EngineType, plmonkeypatch: PlMonkeyPatch
) -> None:
    plmonkeypatch.setenv("POLARS_IDEAL_MORSEL_SIZE", "100")
    n = 2_000
    df = pl.DataFrame({"i": range(n), "s": [str(i) for i in range(n)]})
    df = pl.concat([df.slice(i * 100, 100) for i in range(n // 100)])
    haystack = list(range(0, n, 10))
    result = (
        df.lazy()
        .select(
            i=pl.col("i").is_in(haystack),
            s=pl.col("s").is_in([str(i) for i in haystack]),
        )
        .collect(engine=engine)
    )
    expected = [i % 10 == 0 for i in range(n)]
    assert result["i"].to_list() == expected
    assert result["s"].to_list() == expected


def test_is_in_literal_haystack_in_group_by() -> None:
    df = pl.DataFrame({"g": [1, 1, 2, 2, 3], "v": [1, 5, 2, 9, None]})
    result = (
        df.lazy()
        .group_by("g", maintain_order=True)
        .agg(pl.col("v").is_in([1, 2, 9]).sum())
        .collect()
    )
    assert result["v"].to_list() == [1, 2, 0]


def test_is_in_all_null_literal_haystack() -> None:
    s = pl.Series("n", [1, None])
    haystack = pl.Series([None], dtype=pl.Int64)
    assert s.is_in(haystack).to_list() == [False, None]
    assert s.is_in(haystack, nulls_equal=True).to_list() == [False, True]


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
def test_is_in_multi_row_literal_haystack(
    engine: EngineType, plmonkeypatch: PlMonkeyPatch
) -> None:
    plmonkeypatch.setenv("POLARS_IDEAL_MORSEL_SIZE", "3")
    n = 10
    result = (
        pl.DataFrame({"n": range(n)})
        .lazy()
        .select(pl.col("n").is_in(pl.Series([[i] for i in range(n)])).sum())
        .collect(engine=engine)
    )
    assert result.item() == n


LONG = "x" * 20


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
@pytest.mark.parametrize("nulls_equal", [False, True])
@pytest.mark.parametrize(
    ("dtype", "needles", "haystack"),
    [
        # empty haystacks
        (pl.Int64, [1, None], []),
        (pl.Float64, [1.0, None], []),
        (pl.String, ["a", None], []),
        (pl.List(pl.Int64), [[1], None], []),
        # all-null needles
        (pl.Int64, [None, None], [1, 2]),
        (pl.String, [None, None], ["a"]),
        (pl.Boolean, [None, None], [True]),
        (pl.Struct({"a": pl.Int64}), [None, None], [{"a": 1}]),
        # many duplicates of a few values
        (pl.Int64, [1, 2, 3], [1, 2] * 20),
        (pl.String, ["a", "b", "c"], ["a", "b"] * 20),
        # bitset near the ends of the type
        (
            pl.UInt64,
            [2**64 - 1, 2**64 - 3, 0, 5, None],
            list(range(2**64 - 20, 2**64, 2)),
        ),
        (
            pl.Int64,
            [-(2**63), -(2**63) + 4, 0, None],
            list(range(-(2**63), -(2**63) + 20)),
        ),
        (pl.Int16, [-32768, 32767, 0, 1, None], list(range(-32768, 32767, 1000))),
        # needles below and far above the bitset range
        (pl.Int32, [-1, 0, 99, 100, 10**6, -(10**6)], list(range(100))),
        # long strings: linear scan and hash table, duplicates, shared prefixes
        (
            pl.String,
            [LONG, LONG + "y", "x" * 19, "short", None],
            [LONG, LONG, "x" * 21, "short"],
        ),
        (
            pl.String,
            [LONG, LONG + "y", "x" * 19, f"{LONG}5", f"{LONG}50", "short", None],
            [f"{LONG}{i}" for i in range(40)] + [LONG, LONG, "short"],
        ),
        (
            pl.String,
            ["abcdefghijkl", "abcdefghijklm", "abcdefghijk", "", None],
            ["abcdefghijkl", "abcdefghijklm", ""] + [str(i) for i in range(10)],
        ),
        # zero bytes must not be confused with inline padding
        (
            pl.Binary,
            [bytes(20), bytes(12), bytes(11), bytes(1), b"", None],
            [bytes(20), bytes(12), bytes(1)] + [bytes([i]) for i in range(1, 10)],
        ),
        (pl.Binary, [bytes(12), bytes(11), bytes(13), b"", None], [bytes(12), b""]),
        # nested values through the hash table
        (
            pl.List(pl.Int64),
            [[1, 2], [3], [], [None], None, [1, 2, 3]],
            [[i] for i in range(20)] + [[1, 2], [None], []],
        ),
        (
            pl.Struct({"a": pl.Int64, "b": pl.String}),
            [{"a": 1, "b": LONG}, {"a": 1, "b": None}, {"a": None, "b": None}, None],
            [{"a": i, "b": LONG} for i in range(20)] + [{"a": 1, "b": None}],
        ),
        (
            pl.List(pl.List(pl.Int64)),
            [[[1], [2]], [[1]], [], None],
            [[[1], [2]], []],
        ),
        # temporal types
        (
            pl.Datetime("ms", "UTC"),
            [datetime(2020, 1, 1), datetime(2021, 1, 1), None],
            [datetime(2020, 1, 1)],
        ),
        (
            pl.Duration("us"),
            [timedelta(days=1), timedelta(days=2), None],
            [timedelta(days=1)],
        ),
        (pl.Time, [time(1, 2, 3), time(4, 5, 6), None], [time(1, 2, 3)]),
        # decimals with different scales, and values that do not fit the common scale
        (
            pl.Decimal(5, 2),
            [D("1.50"), D("999.99"), D("0.01"), None],
            [D("1.5"), D("0.01")],
        ),
    ],
)
def test_is_in_literal_haystack_edge_cases(
    engine: EngineType,
    nulls_equal: bool,
    dtype: pl.DataType,
    needles: list[object],
    haystack: list[object],
) -> None:
    needle_s = pl.Series("n", needles, dtype=dtype)
    haystack_s = pl.Series(haystack, dtype=dtype)
    expected = _reference_is_in(needles, haystack, nulls_equal)

    result = (
        needle_s.to_frame()
        .lazy()
        .select(pl.col("n").is_in(haystack_s.implode(), nulls_equal=nulls_equal))
        .collect(engine=engine)
        .to_series()
    )
    assert result.to_list() == expected


@pytest.mark.parametrize("nulls_equal", [False, True])
@pytest.mark.parametrize(
    ("dtype", "needles", "haystack"),
    [
        (pl.Int64, [1, 2, None, 5, 7], list(range(0, 10, 2)) + [None]),
        (pl.Int64, [1, 2, None, 5, 7], list(range(0, 10**7, 10**5)) + [None]),
        (pl.Float64, [1.0, float("nan"), -0.0, None], [0.0, float("nan"), None]),
        (pl.String, ["a", LONG, None, "b"], ["a", LONG, None]),
        (
            pl.String,
            ["a", LONG, None, "b"],
            [str(i) for i in range(30)] + ["a", LONG, None],
        ),
        (pl.Boolean, [True, False, None], [False, None]),
        (pl.List(pl.Int64), [[1], [], None], [[1], None]),
        (
            pl.Struct({"a": pl.Int64}),
            [{"a": 1}, {"a": None}, None],
            [{"a": 1}, {"a": None}, None],
        ),
    ],
)
def test_is_in_literal_matches_per_row_haystack(
    nulls_equal: bool,
    dtype: pl.DataType,
    needles: list[object],
    haystack: list[object],
) -> None:
    # The same haystack in every row goes through the per-row kernel.
    haystack_s = pl.Series(haystack, dtype=dtype)
    df = pl.DataFrame(
        {
            "n": pl.Series(needles, dtype=dtype),
            "h": pl.Series([haystack] * len(needles), dtype=pl.List(dtype)),
        }
    )
    result = df.select(
        literal=pl.col("n").is_in(haystack_s.implode(), nulls_equal=nulls_equal),
        per_row=pl.col("n").is_in(pl.col("h"), nulls_equal=nulls_equal),
    )
    assert result["literal"].to_list() == result["per_row"].to_list()


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
def test_is_in_literal_haystack_array_dtype(engine: EngineType) -> None:
    haystack = pl.Series([[1, 2, 3]], dtype=pl.Array(pl.Int64, 3))
    result = (
        pl.LazyFrame({"n": [1, 4, None]})
        .select(pl.col("n").is_in(pl.lit(haystack)))
        .collect(engine=engine)
        .to_series()
    )
    assert result.to_list() == [True, False, None]


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
def test_is_in_literal_haystack_scalar_needle(engine: EngineType) -> None:
    lf = pl.LazyFrame({"n": [1, 2, 3]})
    result = lf.select(
        a=pl.lit(2).is_in([1, 2]),
        b=pl.lit(5).is_in([1, 2]),
        c=pl.lit(None, dtype=pl.Int64).is_in([1, 2]),
        d=pl.lit(None, dtype=pl.Int64).is_in([1, None], nulls_equal=True),
    ).collect(engine=engine)
    assert result.row(0) == (True, False, None, True)
    assert result.height == 1


def test_is_in_literal_haystack_streaming_filter_and_group_by(
    plmonkeypatch: PlMonkeyPatch,
) -> None:
    plmonkeypatch.setenv("POLARS_IDEAL_MORSEL_SIZE", "7")
    n = 100
    lf = pl.LazyFrame({"g": [i % 3 for i in range(n)], "v": range(n)})
    keep = [1, 5, 9, 50, 99]

    result = lf.filter(pl.col("v").is_in(keep)).collect(engine="streaming")
    assert result["v"].to_list() == keep

    result = (
        lf.group_by("g", maintain_order=True)
        .agg(pl.col("v").is_in(keep).sum())
        .collect(engine="streaming")
    )
    assert result["v"].to_list() == [2, 1, 2]


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
def test_is_in_literal_haystack_chunked_needle_with_nulls(engine: EngineType) -> None:
    a = pl.Series("s", ["a", None, "b", LONG])
    b = pl.Series("s", [None, "c", LONG, "a"])
    s = pl.concat([a, b], rechunk=False)
    assert s.n_chunks() == 2
    result = (
        s.to_frame()
        .lazy()
        .select(
            a=pl.col("s").is_in(["a", LONG]),
            b=pl.col("s").is_in(["a", LONG, None], nulls_equal=True),
        )
        .collect(engine=engine)
    )
    assert result["a"].to_list() == [True, None, False, True, None, False, True, True]
    assert result["b"].to_list() == [True, True, False, True, True, False, True, True]


def test_is_in_literal_haystack_categorical_mismatch() -> None:
    s = pl.Series("n", ["a", "b"], dtype=pl.Enum(["a", "b"]))
    haystack = pl.Series(["a"], dtype=pl.Enum(["a", "c"]))
    with pytest.raises(InvalidOperationError):
        s.is_in(haystack.implode())


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
@pytest.mark.parametrize(
    ("dtype", "needles"),
    [
        (pl.List(pl.Int64), [None, [1], [2], None]),
        (pl.Array(pl.Int64, 1), [None, [1], [2], None]),
        (pl.Struct({"a": pl.Int64}), [None, {"a": 1}, {"a": 2}, None]),
    ],
)
def test_is_in_nested_null_needles_in_aggregation(
    engine: EngineType, dtype: pl.DataType, needles: list[object]
) -> None:
    haystack = pl.Series([needles[0], needles[1]], dtype=dtype).implode()
    df = pl.DataFrame({"g": [0, 1, 1, 0], "n": pl.Series(needles, dtype=dtype)})
    is_in = pl.col("n").is_in(haystack)
    result = (
        df.lazy()
        .group_by("g", maintain_order=True)
        .agg(
            total=is_in.sum(),
            nulls=is_in.null_count(),
            first=is_in.first(),
            last=is_in.last(),
        )
        .collect(engine=engine)
    )
    assert result["total"].to_list() == [0, 1]
    assert result["nulls"].to_list() == [2, 0]
    assert result["first"].to_list() == [None, True]
    assert result["last"].to_list() == [None, False]
    assert df.select(is_in.null_count()).item() == 2
