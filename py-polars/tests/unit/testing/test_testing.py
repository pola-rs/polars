from __future__ import annotations

import math
from datetime import datetime, time, timedelta
from typing import Any

import pytest

import polars as pl
from polars.exceptions import InvalidAssert
from polars.testing import (
    assert_frame_equal,
    assert_frame_not_equal,
    assert_series_equal,
    assert_series_not_equal,
)


def test_compare_series_value_mismatch() -> None:
    srs1 = pl.Series([1, 2, 3])
    srs2 = pl.Series([2, 3, 4])

    assert_series_not_equal(srs1, srs2)
    with pytest.raises(
        AssertionError, match=r"Series are different \(value mismatch\)"
    ):
        assert_series_equal(srs1, srs2)


def test_compare_series_empty_equal() -> None:
    srs1 = pl.Series([])
    srs2 = pl.Series(())

    assert_series_equal(srs1, srs2)
    with pytest.raises(AssertionError):
        assert_series_not_equal(srs1, srs2)


def test_compare_series_nans_assert_equal() -> None:
    # note: NaN values do not _compare_ equal, but should _assert_ equal (by default)
    nan = float("NaN")

    srs1 = pl.Series([1.0, 2.0, nan, 4.0, None, 6.0])
    srs2 = pl.Series([1.0, nan, 3.0, 4.0, None, 6.0])
    srs3 = pl.Series([1.0, 2.0, 3.0, 4.0, None, 6.0])

    for srs in (srs1, srs2, srs3):
        assert_series_equal(srs, srs)
        assert_series_equal(srs, srs, check_exact=True)

    with pytest.raises(AssertionError):
        assert_series_equal(srs1, srs1, nans_compare_equal=False)
    assert_series_not_equal(srs1, srs1, nans_compare_equal=False)

    with pytest.raises(AssertionError):
        assert_series_equal(srs1, srs1, nans_compare_equal=False, check_exact=True)
    assert_series_not_equal(srs1, srs1, nans_compare_equal=False, check_exact=True)

    for check_exact, nans_equal in (
        (False, False),
        (False, True),
        (True, False),
        (True, True),
    ):
        if check_exact:
            check_msg = "exact value mismatch"
        else:
            check_msg = "Series are different.*value mismatch.*"

        with pytest.raises(AssertionError, match=check_msg):
            assert_series_equal(
                srs1, srs2, check_exact=check_exact, nans_compare_equal=nans_equal
            )
        with pytest.raises(AssertionError, match=check_msg):
            assert_series_equal(
                srs1, srs3, check_exact=check_exact, nans_compare_equal=nans_equal
            )

    srs4 = pl.Series([1.0, 2.0, 3.0, 4.0, None, 6.0])
    srs5 = pl.Series([1.0, 2.0, 3.0, 4.0, nan, 6.0])
    srs6 = pl.Series([1, 2, 3, 4, None, 6])

    assert_series_equal(srs4, srs6, check_dtype=False)
    with pytest.raises(AssertionError):
        assert_series_equal(srs5, srs6, check_dtype=False)
    assert_series_not_equal(srs5, srs6, check_dtype=True)

    # nested
    for float_type in (pl.Float32, pl.Float64):
        srs = pl.Series([[0.0, nan]], dtype=pl.List(float_type))
        assert srs.dtype == pl.List(float_type)
        assert_series_equal(srs, srs)


def test_compare_series_nulls() -> None:
    srs1 = pl.Series([1, 2, None])
    srs2 = pl.Series([1, 2, None])
    assert_series_equal(srs1, srs2)

    srs1 = pl.Series([1, 2, 3])
    srs2 = pl.Series([1, None, None])
    assert_series_not_equal(srs1, srs2)

    with pytest.raises(AssertionError, match="value mismatch"):
        assert_series_equal(srs1, srs2)


def test_series_cmp_fast_paths() -> None:
    assert (
        pl.Series([None], dtype=pl.Int32) != pl.Series([1, 2], dtype=pl.Int32)
    ).to_list() == [None, None]
    assert (
        pl.Series([None], dtype=pl.Int32) == pl.Series([1, 2], dtype=pl.Int32)
    ).to_list() == [None, None]

    assert (
        pl.Series([None], dtype=pl.Utf8) != pl.Series(["a", "b"], dtype=pl.Utf8)
    ).to_list() == [None, None]
    assert (
        pl.Series([None], dtype=pl.Utf8) == pl.Series(["a", "b"], dtype=pl.Utf8)
    ).to_list() == [None, None]

    assert (
        pl.Series([None], dtype=pl.Boolean)
        != pl.Series([True, False], dtype=pl.Boolean)
    ).to_list() == [None, None]
    assert (
        pl.Series([None], dtype=pl.Boolean)
        == pl.Series([False, False], dtype=pl.Boolean)
    ).to_list() == [None, None]


def test_compare_series_value_mismatch_string() -> None:
    srs1 = pl.Series(["hello", "no"])
    srs2 = pl.Series(["hello", "yes"])

    assert_series_not_equal(srs1, srs2)
    with pytest.raises(
        AssertionError, match=r"Series are different \(exact value mismatch\)"
    ):
        assert_series_equal(srs1, srs2)


def test_compare_series_type_mismatch() -> None:
    srs1 = pl.Series([1, 2, 3])
    srs2 = pl.DataFrame({"col1": [2, 3, 4]})

    with pytest.raises(
        AssertionError, match=r"Inputs are different \(unexpected input types\)"
    ):
        assert_series_equal(srs1, srs2)  # type: ignore[arg-type]

    srs3 = pl.Series([1.0, 2.0, 3.0])
    assert_series_not_equal(srs1, srs3)
    with pytest.raises(
        AssertionError, match=r"Series are different \(dtype mismatch\)"
    ):
        assert_series_equal(srs1, srs3)


def test_compare_series_name_mismatch() -> None:
    srs1 = pl.Series(values=[1, 2, 3], name="srs1")
    srs2 = pl.Series(values=[1, 2, 3], name="srs2")
    with pytest.raises(AssertionError, match=r"Series are different \(name mismatch\)"):
        assert_series_equal(srs1, srs2)


def test_compare_series_shape_mismatch() -> None:
    srs1 = pl.Series(values=[1, 2, 3, 4], name="srs1")
    srs2 = pl.Series(values=[1, 2, 3], name="srs2")

    assert_series_not_equal(srs1, srs2)
    with pytest.raises(
        AssertionError, match=r"Series are different \(length mismatch\)"
    ):
        assert_series_equal(srs1, srs2)


def test_compare_series_value_exact_mismatch() -> None:
    srs1 = pl.Series([1.0, 2.0, 3.0])
    srs2 = pl.Series([1.0, 2.0 + 1e-7, 3.0])
    with pytest.raises(
        AssertionError, match=r"Series are different \(exact value mismatch\)"
    ):
        assert_series_equal(srs1, srs2, check_exact=True)


def test_compare_frame_equal_nans() -> None:
    nan = float("NaN")
    df1 = pl.DataFrame(
        data={"x": [1.0, nan], "y": [nan, 2.0]},
        schema=[("x", pl.Float32), ("y", pl.Float64)],
    )
    assert_frame_equal(df1, df1, check_exact=True)

    df2 = pl.DataFrame(
        data={"x": [1.0, nan], "y": [None, 2.0]},
        schema=[("x", pl.Float32), ("y", pl.Float64)],
    )
    assert_frame_not_equal(df1, df2)
    with pytest.raises(AssertionError, match="values for column 'y' are different"):
        assert_frame_equal(df1, df2, check_exact=True)


def test_compare_frame_equal_nested_nans() -> None:
    nan = float("NaN")

    # list dtype
    df1 = pl.DataFrame(
        data={"x": [[1.0, nan]], "y": [[nan, 2.0]]},
        schema=[("x", pl.List(pl.Float32)), ("y", pl.List(pl.Float64))],
    )
    assert_frame_equal(df1, df1, check_exact=True)

    df2 = pl.DataFrame(
        data={"x": [[1.0, nan]], "y": [[None, 2.0]]},
        schema=[("x", pl.List(pl.Float32)), ("y", pl.List(pl.Float64))],
    )
    assert_frame_not_equal(df1, df2)
    with pytest.raises(AssertionError, match="values for column 'y' are different"):
        assert_frame_equal(df1, df2, check_exact=True)

    # struct dtype
    df3 = pl.from_dicts(
        [
            {
                "id": 1,
                "struct": [
                    {"x": "text", "y": [0.0, nan]},
                    {"x": "text", "y": [0.0, nan]},
                ],
            },
            {
                "id": 2,
                "struct": [
                    {"x": "text", "y": [1]},
                    {"x": "text", "y": [1]},
                ],
            },
        ]
    )
    df4 = pl.from_dicts(
        [
            {
                "id": 1,
                "struct": [
                    {"x": "text", "y": [0.0, nan], "z": ["$"]},
                    {"x": "text", "y": [0.0, nan], "z": ["$"]},
                ],
            },
            {
                "id": 2,
                "struct": [
                    {"x": "text", "y": [nan, 1], "z": ["!"]},
                    {"x": "text", "y": [nan, 1], "z": ["?"]},
                ],
            },
        ]
    )

    assert_frame_equal(df3, df3)
    assert_frame_not_equal(df3, df3, nans_compare_equal=False)

    assert_frame_equal(df4, df4)
    assert_frame_not_equal(df4, df4, nans_compare_equal=False)

    assert_frame_not_equal(df3, df4)
    for check_dtype in (True, False):
        with pytest.raises(AssertionError, match="mismatch|different"):
            assert_frame_equal(df3, df4, check_dtype=check_dtype)


def test_assert_frame_equal_pass() -> None:
    df1 = pl.DataFrame({"a": [1, 2]})
    df2 = pl.DataFrame({"a": [1, 2]})
    assert_frame_equal(df1, df2)


def test_assert_frame_equal_types() -> None:
    df1 = pl.DataFrame({"a": [1, 2]})
    srs1 = pl.Series(values=[1, 2], name="a")
    with pytest.raises(
        AssertionError, match=r"inputs are different \(unexpected input types\)"
    ):
        assert_frame_equal(df1, srs1)  # type: ignore[arg-type]


def test_assert_frame_equal_length_mismatch() -> None:
    df1 = pl.DataFrame({"a": [1, 2]})
    df2 = pl.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(
        AssertionError, match=r"DataFrames are different \(length mismatch\)"
    ):
        assert_frame_equal(df1, df2)


def test_assert_frame_equal_column_mismatch() -> None:
    df1 = pl.DataFrame({"a": [1, 2]})
    df2 = pl.DataFrame({"b": [1, 2]})
    with pytest.raises(
        AssertionError, match="columns \\['a'\\] in left frame, but not in right"
    ):
        assert_frame_equal(df1, df2)


def test_assert_frame_equal_column_mismatch2() -> None:
    df1 = pl.DataFrame({"a": [1, 2]})
    df2 = pl.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
    with pytest.raises(
        AssertionError, match="columns \\['b', 'c'\\] in right frame, but not in left"
    ):
        assert_frame_equal(df1, df2)


def test_assert_frame_equal_column_mismatch_order() -> None:
    df1 = pl.DataFrame({"b": [3, 4], "a": [1, 2]})
    df2 = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    with pytest.raises(AssertionError, match="columns are not in the same order"):
        assert_frame_equal(df1, df2)

    assert_frame_equal(df1, df2, check_column_order=False)


def test_assert_frame_equal_ignore_row_order() -> None:
    df1 = pl.DataFrame({"a": [1, 2], "b": [4, 3]})
    df2 = pl.DataFrame({"a": [2, 1], "b": [3, 4]})
    df3 = pl.DataFrame({"b": [3, 4], "a": [2, 1]})
    with pytest.raises(AssertionError, match="values for column 'a' are different"):
        assert_frame_equal(df1, df2)

    assert_frame_equal(df1, df2, check_row_order=False)
    # eg:
    # ┌─────┬─────┐      ┌─────┬─────┐
    # │ a   ┆ b   │      │ a   ┆ b   │
    # │ --- ┆ --- │      │ --- ┆ --- │
    # │ i64 ┆ i64 │ (eq) │ i64 ┆ i64 │
    # ╞═════╪═════╡  ==  ╞═════╪═════╡
    # │ 1   ┆ 4   │      │ 2   ┆ 3   │
    # │ 2   ┆ 3   │      │ 1   ┆ 4   │
    # └─────┴─────┘      └─────┴─────┘

    with pytest.raises(AssertionError, match="columns are not in the same order"):
        assert_frame_equal(df1, df3, check_row_order=False)

    assert_frame_equal(df1, df3, check_row_order=False, check_column_order=False)

    # note: not all column types support sorting
    with pytest.raises(
        InvalidAssert,
        match="cannot set `check_row_order=False`.*unsortable columns",
    ):
        assert_frame_equal(
            left=pl.DataFrame({"a": [[1, 2], [3, 4]], "b": [3, 4]}),
            right=pl.DataFrame({"a": [[3, 4], [1, 2]], "b": [4, 3]}),
            check_row_order=False,
        )


@pytest.mark.parametrize(
    ("df1", "df2", "kwargs"),
    [
        pytest.param(
            pl.DataFrame({"a": [0.2, 0.3]}),
            pl.DataFrame({"a": [0.2, 0.3]}),
            {"atol": 1e-15},
            id="equal_floats_low_atol",
        ),
        pytest.param(
            pl.DataFrame({"a": [0.2, 0.3]}),
            pl.DataFrame({"a": [0.2, 0.3000000000000001]}),
            {"atol": 1e-15},
            id="approx_equal_float_low_atol",
        ),
        pytest.param(
            pl.DataFrame({"a": [0.2, 0.3]}),
            pl.DataFrame({"a": [0.2, 0.31]}),
            {"atol": 0.1},
            id="approx_equal_float_high_atol",
        ),
        pytest.param(
            pl.DataFrame({"a": [0.2, 1.3]}),
            pl.DataFrame({"a": [0.2, 0.9]}),
            {"atol": 1},
            id="approx_equal_float_integer_atol",
        ),
        pytest.param(
            pl.DataFrame({"a": [0.0, 1.0, 2.0]}, schema={"a": pl.Float64}),
            pl.DataFrame({"a": [0, 1, 2]}, schema={"a": pl.Int64}),
            {"check_dtype": False},
            id="equal_int_float_integer_no_check_dtype",
        ),
        pytest.param(
            pl.DataFrame({"a": [0, 1, 2]}, schema={"a": pl.Float64}),
            pl.DataFrame({"a": [0, 1, 2]}, schema={"a": pl.Float32}),
            {"check_dtype": False},
            id="equal_int_float_integer_no_check_dtype",
        ),
        pytest.param(
            pl.DataFrame({"a": [0, 1, 2]}, schema={"a": pl.Int64}),
            pl.DataFrame({"a": [0, 1, 2]}, schema={"a": pl.Int64}),
            {},
            id="equal_int",
        ),
        pytest.param(
            pl.DataFrame({"a": ["a", "b", "c"]}, schema={"a": pl.Utf8}),
            pl.DataFrame({"a": ["a", "b", "c"]}, schema={"a": pl.Utf8}),
            {},
            id="equal_str",
        ),
        pytest.param(
            pl.DataFrame({"a": [[0.2, 0.3]]}),
            pl.DataFrame({"a": [[0.2, 0.300001]]}),
            {"atol": 1e-5},
            id="list_of_float_low_atol",
        ),
        pytest.param(
            pl.DataFrame({"a": [[0.2, 0.3]]}),
            pl.DataFrame({"a": [[0.2, 0.31]]}),
            {"atol": 0.1},
            id="list_of_float_high_atol",
        ),
        pytest.param(
            pl.DataFrame({"a": [[0.2, 1.3]]}),
            pl.DataFrame({"a": [[0.2, 0.9]]}),
            {"atol": 1},
            id="list_of_float_integer_atol",
        ),
        pytest.param(
            pl.DataFrame({"a": [[0.2, 0.3]]}),
            pl.DataFrame({"a": [[0.2, 0.300000001]]}),
            {"rtol": 1e-5},
            id="list_of_float_low_rtol",
        ),
        pytest.param(
            pl.DataFrame({"a": [[0.2, 0.3]]}),
            pl.DataFrame({"a": [[0.2, 0.301]]}),
            {"rtol": 0.1},
            id="list_of_float_high_rtol",
        ),
        pytest.param(
            pl.DataFrame({"a": [[0.2, 1.3]]}),
            pl.DataFrame({"a": [[0.2, 0.9]]}),
            {"rtol": 1},
            id="list_of_float_integer_rtol",
        ),
        pytest.param(
            pl.DataFrame({"a": [[None, 1.3]]}),
            pl.DataFrame({"a": [[None, 0.9]]}),
            {"rtol": 1},
            id="list_of_none_and_float_integer_rtol",
        ),
        pytest.param(
            pl.DataFrame({"a": [[None, 1.3]]}),
            pl.DataFrame({"a": [[None, 0.9]]}),
            {"rtol": 1, "nans_compare_equal": False},
            id="list_of_none_and_float_integer_rtol",
        ),
        pytest.param(
            pl.DataFrame({"a": [[[0.2, 3.0]]]}),
            pl.DataFrame({"a": [[[0.2, 3.00000001]]]}),
            {"atol": 0.1, "nans_compare_equal": True},
            id="nested_list_of_float_atol_high_nans_compare_equal_true",
        ),
        pytest.param(
            pl.DataFrame({"a": [[[0.2, 3.0]]]}),
            pl.DataFrame({"a": [[[0.2, 3.00000001]]]}),
            {"atol": 0.1, "nans_compare_equal": False},
            id="nested_list_of_float_atol_high_nans_compare_equal_false",
        ),
    ],
)
def test_assert_frame_equal_passes_assertion(
    df1: pl.DataFrame,
    df2: pl.DataFrame,
    kwargs: Any,
) -> None:
    assert_frame_equal(df1, df2, **kwargs)
    with pytest.raises(AssertionError):
        assert_frame_not_equal(df1, df2, **kwargs)


@pytest.mark.parametrize(
    ("df1", "df2", "kwargs"),
    [
        pytest.param(
            pl.DataFrame({"a": [[0.2, 0.3]]}),
            pl.DataFrame({"a": [[0.2, 0.3, 0.4]]}),
            {},
            id="list_of_float_different_lengths",
        ),
        pytest.param(
            pl.DataFrame({"a": [[0.2, 0.3]]}),
            pl.DataFrame({"a": [[0.2, 0.3000000000000001]]}),
            {"check_exact": True},
            id="list_of_float_check_exact",
        ),
        pytest.param(
            pl.DataFrame({"a": [[0.2, 0.3]]}),
            pl.DataFrame({"a": [[0.2, 0.300001]]}),
            {"atol": 1e-15, "rtol": 0},
            id="list_of_float_too_low_atol",
        ),
        pytest.param(
            pl.DataFrame({"a": [[0.2, 0.3]]}),
            pl.DataFrame({"a": [[0.2, 0.30000001]]}),
            {"atol": -1, "rtol": 0},
            id="list_of_float_negative_atol",
        ),
        pytest.param(
            pl.DataFrame({"a": [[math.nan, 1.3]]}),
            pl.DataFrame({"a": [[math.nan, 0.9]]}),
            {"rtol": 1, "nans_compare_equal": False},
            id="list_of_nan_and_float_integer_rtol",
        ),
        pytest.param(
            pl.DataFrame({"a": [[2.0, 3.0]]}),
            pl.DataFrame({"a": [[2, 3]]}),
            {"check_exact": False, "check_dtype": True},
            id="list_of_float_list_of_int_check_dtype_true",
        ),
        pytest.param(
            pl.DataFrame({"a": [[[0.2, math.nan, 3.0]]]}),
            pl.DataFrame({"a": [[[0.2, math.nan, 3.11]]]}),
            {"atol": 0.1, "rtol": 0, "nans_compare_equal": True},
            id="nested_list_of_float_and_nan_atol_high_nans_compare_equal_true",
        ),
        pytest.param(
            pl.DataFrame({"a": [[[0.2, math.nan, 3.0]]]}),
            pl.DataFrame({"a": [[[0.2, math.nan, 3.0]]]}),
            {"nans_compare_equal": False},
            id="nested_list_of_float_and_nan_atol_high_nans_compare_equal_false",
        ),
        pytest.param(
            pl.DataFrame({"a": [[[0.2, 3.0]]]}),
            pl.DataFrame({"a": [[[0.2, 3.11]]]}),
            {"atol": 0.1, "nans_compare_equal": False},
            id="nested_list_of_float_atol_high_nans_compare_equal_false",
        ),
        pytest.param(
            pl.DataFrame({"a": [[[[0.2, 3.0]]]]}),
            pl.DataFrame({"a": [[[[0.2, 3.11]]]]}),
            {"atol": 0.1, "rtol": 0, "nans_compare_equal": True},
            id="double_nested_list_of_float_atol_high_nans_compare_equal_true",
        ),
        pytest.param(
            pl.DataFrame({"a": [[[[0.2, math.nan, 3.0]]]]}),
            pl.DataFrame({"a": [[[[0.2, math.nan, 3.0]]]]}),
            {"atol": 0.1, "nans_compare_equal": False},
            id="double_nested_list_of_float_and_nan_atol_high_nans_compare_equal_false",
        ),
        pytest.param(
            pl.DataFrame({"a": [[[[0.2, 3.0]]]]}),
            pl.DataFrame({"a": [[[[0.2, 3.11]]]]}),
            {"atol": 0.1, "rtol": 0, "nans_compare_equal": False},
            id="double_nested_list_of_float_atol_high_nans_compare_equal_false",
        ),
        pytest.param(
            pl.DataFrame({"a": [[[[[0.2, 3.0]]]]]}),
            pl.DataFrame({"a": [[[[[0.2, 3.11]]]]]}),
            {"atol": 0.1, "rtol": 0, "nans_compare_equal": True},
            id="triple_nested_list_of_float_atol_high_nans_compare_equal_true",
        ),
        pytest.param(
            pl.DataFrame({"a": [[[[[0.2, math.nan, 3.0]]]]]}),
            pl.DataFrame({"a": [[[[[0.2, math.nan, 3.0]]]]]}),
            {"atol": 0.1, "nans_compare_equal": False},
            id="triple_nested_list_of_float_and_nan_atol_high_nans_compare_equal_true",
        ),
    ],
)
def test_assert_frame_equal_raises_assertion_error(
    df1: pl.DataFrame,
    df2: pl.DataFrame,
    kwargs: Any,
) -> None:
    with pytest.raises(AssertionError):
        assert_frame_equal(df1, df2, **kwargs)
    assert_frame_not_equal(df1, df2, **kwargs)


def test_assert_series_equal_int_overflow() -> None:
    # internally may call 'abs' if not check_exact, which can overflow on signed int
    s0 = pl.Series([-128], dtype=pl.Int8)
    s1 = pl.Series([0, -128], dtype=pl.Int8)
    s2 = pl.Series([1, -128], dtype=pl.Int8)

    for check_exact in (True, False):
        assert_series_equal(s0, s0, check_exact=check_exact)
        with pytest.raises(AssertionError):
            assert_series_equal(s1, s2, check_exact=check_exact)


def test_assert_series_equal_uint_overflow() -> None:
    # 'atol' is checked following "(left-right).abs()", which can overflow on uint
    s1 = pl.Series([1, 2, 3], dtype=pl.UInt8)
    s2 = pl.Series([2, 3, 4], dtype=pl.UInt8)

    with pytest.raises(AssertionError):
        assert_series_equal(s1, s2, atol=0)
    assert_series_equal(s1, s2, atol=1)

    # confirm no OverflowError in the below test case:
    # as "(left-right).abs()" > max(Int64)
    left = pl.Series(
        values=[2810428175213635359],
        dtype=pl.UInt64,
    )
    right = pl.Series(
        values=[15807433754238349345],
        dtype=pl.UInt64,
    )
    with pytest.raises(AssertionError):
        assert_series_equal(left, right)


@pytest.mark.parametrize(
    ("data1", "data2"),
    [
        ([datetime(2022, 10, 2, 12)], [datetime(2022, 10, 2, 13)]),
        ([time(10, 0, 0)], [time(10, 0, 10)]),
        ([timedelta(10, 0, 0)], [timedelta(10, 0, 10)]),
    ],
)
def test_assert_series_equal_temporal(data1: Any, data2: Any) -> None:
    s1 = pl.Series(data1)
    s2 = pl.Series(data2)
    assert_series_not_equal(s1, s2)


@pytest.mark.parametrize(
    ("s1", "s2", "kwargs"),
    [
        pytest.param(
            pl.Series([0.2, 0.3]),
            pl.Series([0.2, 0.3]),
            {"atol": 1e-15},
            id="equal_floats_low_atol",
        ),
        pytest.param(
            pl.Series([0.2, 0.3]),
            pl.Series([0.2, 0.3000000000000001]),
            {"atol": 1e-15},
            id="approx_equal_float_low_atol",
        ),
        pytest.param(
            pl.Series([0.2, 0.3]),
            pl.Series([0.2, 0.31]),
            {"atol": 0.1},
            id="approx_equal_float_high_atol",
        ),
        pytest.param(
            pl.Series([0.2, 1.3]),
            pl.Series([0.2, 0.9]),
            {"atol": 1},
            id="approx_equal_float_integer_atol",
        ),
        pytest.param(
            pl.Series([1.0, 2.0, float("nan")]),
            pl.Series([1.005, 2.005, float("nan")]),
            {"atol": 1e-2, "rtol": 0.0},
            id="approx_equal_float_nan_atol",
        ),
        pytest.param(
            pl.Series([1.0, 2.0, None]),
            pl.Series([1.005, 2.005, None]),
            {"atol": 1e-2},
            id="approx_equal_float_none_atol",
        ),
        pytest.param(
            pl.Series([1.0, 2.0, None]),
            pl.Series([1.005, 2.005, None]),
            {"atol": 1e-2, "nans_compare_equal": False},
            id="approx_equal_float_none_atol_with_nans_compare_equal_false",
        ),
        pytest.param(
            pl.Series([1.0, 2.0, float("nan")]),
            pl.Series([1.005, 2.015, float("nan")]),
            {"atol": 0.0, "rtol": 1e-2},
            id="approx_equal_float_nan_rtol",
        ),
        pytest.param(
            pl.Series([1.0, 2.0, None]),
            pl.Series([1.005, 2.015, None]),
            {"rtol": 1e-2},
            id="approx_equal_float_none_rtol",
        ),
        pytest.param(
            pl.Series([1.0, 2.0, None]),
            pl.Series([1.005, 2.015, None]),
            {"rtol": 1e-2, "nans_compare_equal": False},
            id="approx_equal_float_none_rtol_with_nans_compare_equal_false",
        ),
        pytest.param(
            pl.Series([0.0, 1.0, 2.0], dtype=pl.Float64),
            pl.Series([0, 1, 2], dtype=pl.Int64),
            {"check_dtype": False},
            id="equal_int_float_integer_no_check_dtype",
        ),
        pytest.param(
            pl.Series([0, 1, 2], dtype=pl.Float64),
            pl.Series([0, 1, 2], dtype=pl.Float32),
            {"check_dtype": False},
            id="equal_int_float_integer_no_check_dtype",
        ),
        pytest.param(
            pl.Series([0, 1, 2], dtype=pl.Int64),
            pl.Series([0, 1, 2], dtype=pl.Int64),
            {},
            id="equal_int",
        ),
        pytest.param(
            pl.Series(["a", "b", "c"], dtype=pl.Utf8),
            pl.Series(["a", "b", "c"], dtype=pl.Utf8),
            {},
            id="equal_str",
        ),
        pytest.param(
            pl.Series([[0.2, 0.3]]),
            pl.Series([[0.2, 0.31]]),
            {"atol": 0.1},
            id="list_of_float_high_atol",
        ),
        pytest.param(
            pl.Series([[0.2, 1.3]]),
            pl.Series([[0.2, 0.9]]),
            {"atol": 1},
            id="list_of_float_integer_atol",
        ),
        pytest.param(
            pl.Series([[0.2, 0.3]]),
            pl.Series([[0.2, 0.300000001]]),
            {"rtol": 1e-15},
            id="list_of_float_low_rtol",
        ),
        pytest.param(
            pl.Series([[0.2, 0.3]]),
            pl.Series([[0.2, 0.301]]),
            {"rtol": 0.1},
            id="list_of_float_high_rtol",
        ),
        pytest.param(
            pl.Series([[0.2, 1.3]]),
            pl.Series([[0.2, 0.9]]),
            {"rtol": 1},
            id="list_of_float_integer_rtol",
        ),
        pytest.param(
            pl.Series([[None, 1.3]]),
            pl.Series([[None, 0.9]]),
            {"rtol": 1},
            id="list_of_none_and_float_integer_rtol",
        ),
        pytest.param(
            pl.Series([[None, 1.3]]),
            pl.Series([[None, 0.9]]),
            {"rtol": 1, "nans_compare_equal": False},
            id="list_of_none_and_float_integer_rtol",
        ),
        pytest.param(
            pl.Series([[None, 1]], dtype=pl.List(pl.Int64)),
            pl.Series([[None, 1]], dtype=pl.List(pl.Int64)),
            {"rtol": 1},
            id="list_of_none_and_int_integer_rtol",
        ),
        pytest.param(
            pl.Series([[math.nan, 1.3]]),
            pl.Series([[math.nan, 0.9]]),
            {"rtol": 1, "nans_compare_equal": True},
            id="list_of_none_and_float_integer_rtol",
        ),
        pytest.param(
            pl.Series([[2.0, 3.0]]),
            pl.Series([[2, 3]]),
            {"check_exact": False, "check_dtype": False},
            id="list_of_float_list_of_int_check_dtype_false",
        ),
        pytest.param(
            pl.Series([[[0.2, 3.0]]]),
            pl.Series([[[0.2, 3.00000001]]]),
            {"atol": 0.1, "nans_compare_equal": True},
            id="nested_list_of_float_atol_high_nans_compare_equal_true",
        ),
        pytest.param(
            pl.Series([[[0.2, math.nan, 3.0]]]),
            pl.Series([[[0.2, math.nan, 3.00000001]]]),
            {"atol": 0.1, "nans_compare_equal": True},
            id="nested_list_of_float_and_nan_atol_high_nans_compare_equal_true",
        ),
        pytest.param(
            pl.Series([[[0.2, 3.0]]]),
            pl.Series([[[0.2, 3.00000001]]]),
            {"atol": 0.1, "nans_compare_equal": False},
            id="nested_list_of_float_atol_high_nans_compare_equal_false",
        ),
        pytest.param(
            pl.Series([[[[0.2, 3.0]]]]),
            pl.Series([[[[0.2, 3.00000001]]]]),
            {"atol": 0.1, "nans_compare_equal": True},
            id="double_nested_list_of_float_atol_high_nans_compare_equal_true",
        ),
        pytest.param(
            pl.Series([[[[0.2, math.nan, 3.0]]]]),
            pl.Series([[[[0.2, math.nan, 3.00000001]]]]),
            {"atol": 0.1, "nans_compare_equal": True},
            id="double_nested_list_of_float__and_nan_atol_high_nans_compare_equal_true",
        ),
        pytest.param(
            pl.Series([[[[0.2, 3.0]]]]),
            pl.Series([[[[0.2, 3.00000001]]]]),
            {"atol": 0.1, "nans_compare_equal": False},
            id="double_nested_list_of_float_atol_high_nans_compare_equal_false",
        ),
        pytest.param(
            pl.Series([[[[[0.2, 3.0]]]]]),
            pl.Series([[[[[0.2, 3.00000001]]]]]),
            {"atol": 0.1, "nans_compare_equal": True},
            id="triple_nested_list_of_float_atol_high_nans_compare_equal_true",
        ),
        pytest.param(
            pl.Series([[[[[0.2, math.nan, 3.0]]]]]),
            pl.Series([[[[[0.2, math.nan, 3.00000001]]]]]),
            {"atol": 0.1, "nans_compare_equal": True},
            id="triple_nested_list_of_float_and_nan_atol_high_nans_compare_equal_true",
        ),
        pytest.param(
            pl.Series([[[[[0.2, 3.0]]]]]),
            pl.Series([[[[[0.2, 3.00000001]]]]]),
            {"atol": 0.1, "nans_compare_equal": False},
            id="triple_nested_list_of_float_atol_high_nans_compare_equal_false",
        ),
        pytest.param(
            pl.struct(a=0, b=1, eager=True),
            pl.struct(a=0, b=1, eager=True),
            {},
            id="struct_equal",
        ),
        pytest.param(
            pl.struct(a=0, b=1.1, eager=True),
            pl.struct(a=0, b=1.01, eager=True),
            {"atol": 0.1, "rtol": 0},
            id="struct_approx_equal",
        ),
        pytest.param(
            pl.struct(a=0, b=1.09, eager=True),
            pl.struct(a=0, b=1, eager=True),
            {"atol": 0.1, "rtol": 0, "check_dtype": False},
            id="struct_approx_equal_different_type",
        ),
        pytest.param(
            pl.struct(a=0, b=[0.0, 1.1], eager=True),
            pl.struct(a=0, b=[0.0, 1.11], eager=True),
            {"atol": 0.1},
            id="struct_with_list_approx_equal",
        ),
        pytest.param(
            pl.struct(a=0, b=[0.0, math.nan], eager=True),
            pl.struct(a=0, b=[0.0, math.nan], eager=True),
            {"atol": 0.1, "nans_compare_equal": True},
            id="struct_with_list_with_nan_compare_equal_true",
        ),
    ],
)
def test_assert_series_equal_passes_assertion(
    s1: pl.Series,
    s2: pl.Series,
    kwargs: Any,
) -> None:
    assert_series_equal(s1, s2, **kwargs)
    with pytest.raises(AssertionError):
        assert_series_not_equal(s1, s2, **kwargs)


@pytest.mark.parametrize(
    ("s1", "s2", "kwargs"),
    [
        pytest.param(
            pl.Series([0.2, 0.3]),
            pl.Series([0.2, 0.39]),
            {"atol": 0.09, "rtol": 0},
            id="approx_equal_float_high_atol_zero_rtol",
        ),
        pytest.param(
            pl.Series([0.2, 1.3]),
            pl.Series([0.2, 2.31]),
            {"atol": 1, "rtol": 0},
            id="approx_equal_float_integer_atol_zero_rtol",
        ),
        pytest.param(
            pl.Series([0, 1, 2], dtype=pl.Float64),
            pl.Series([0, 1, 2], dtype=pl.Int64),
            {"check_dtype": True},
            id="equal_int_float_integer_check_dtype",
        ),
        pytest.param(
            pl.Series([0, 1, 2], dtype=pl.Float64),
            pl.Series([0, 1, 2], dtype=pl.Float32),
            {"check_dtype": True},
            id="equal_int_float_integer_check_dtype",
        ),
        pytest.param(
            pl.Series([1.0, 2.0, float("nan")]),
            pl.Series([1.005, 2.005, float("nan")]),
            {"atol": 1e-2, "rtol": 0.0, "nans_compare_equal": False},
            id="approx_equal_float_nan_atol_with_nan_compare_equal_false",
        ),
        pytest.param(
            pl.Series([1.0, 2.0, float("nan")]),
            pl.Series([1.005, 2.005, 3.005]),
            {"atol": 1e-2, "rtol": 0.0},
            id="approx_equal_float_left_nan_atol",
        ),
        pytest.param(
            pl.Series([1.0, 2.0, float("nan")]),
            pl.Series([1.005, 2.005, 3.005]),
            {"atol": 1e-2, "rtol": 0.0, "nans_compare_equal": False},
            id="approx_equal_float_left_nan_atol_with_nan_compare_equal_false",
        ),
        pytest.param(
            pl.Series([1.0, 2.0, 3.0]),
            pl.Series([1.005, 2.005, float("nan")]),
            {"atol": 1e-2, "rtol": 0.0},
            id="approx_equal_float_right_nan_atol",
        ),
        pytest.param(
            pl.Series([1.0, 2.0, 3.0]),
            pl.Series([1.005, 2.005, float("nan")]),
            {"atol": 1e-2, "rtol": 0.0, "nans_compare_equal": False},
            id="approx_equal_float_right_nan_atol_with_nan_compare_equal_false",
        ),
        pytest.param(
            pl.Series([1.0, 2.0, float("nan")]),
            pl.Series([1.005, 2.015, float("nan")]),
            {"atol": 0.0, "rtol": 1e-2, "nans_compare_equal": False},
            id="approx_equal_float_nan_rtol_with_nan_compare_equal_false",
        ),
        pytest.param(
            pl.Series([1.0, 2.0, float("nan")]),
            pl.Series([1.005, 2.015, 3.025]),
            {"atol": 0.0, "rtol": 1e-2},
            id="approx_equal_float_left_nan_rtol",
        ),
        pytest.param(
            pl.Series([1.0, 2.0, float("nan")]),
            pl.Series([1.005, 2.015, 3.025]),
            {"atol": 0.0, "rtol": 1e-2, "nans_compare_equal": False},
            id="approx_equal_float_left_nan_rtol_with_nan_compare_equal_false",
        ),
        pytest.param(
            pl.Series([1.0, 2.0, 3.0]),
            pl.Series([1.005, 2.015, float("nan")]),
            {"atol": 0.0, "rtol": 1e-2},
            id="approx_equal_float_right_nan_rtol",
        ),
        pytest.param(
            pl.Series([1.0, 2.0, 3.0]),
            pl.Series([1.005, 2.015, float("nan")]),
            {"atol": 0.0, "rtol": 1e-2, "nans_compare_equal": False},
            id="approx_equal_float_right_nan_rtol_with_nan_compare_equal_false",
        ),
        pytest.param(
            pl.Series([[0.2, 0.3]]),
            pl.Series([[0.2, 0.3, 0.4]]),
            {},
            id="list_of_float_different_lengths",
        ),
        pytest.param(
            pl.Series([[0.2, 0.3]]),
            pl.Series([[0.2, 0.3000000000000001]]),
            {"check_exact": True},
            id="list_of_float_check_exact",
        ),
        pytest.param(
            pl.Series([[0.2, 0.3]]),
            pl.Series([[0.2, 0.300001]]),
            {"atol": 1e-15, "rtol": 0},
            id="list_of_float_too_low_atol",
        ),
        pytest.param(
            pl.Series([[0.2, 0.3]]),
            pl.Series([[0.2, 0.30000001]]),
            {"atol": -1, "rtol": 0},
            id="list_of_float_negative_atol",
        ),
        pytest.param(
            pl.Series([[math.nan, 1.3]]),
            pl.Series([[math.nan, 0.9]]),
            {"rtol": 1, "nans_compare_equal": False},
            id="list_of_nan_and_float_integer_rtol",
        ),
        pytest.param(
            pl.Series([[2.0, 3.0]]),
            pl.Series([[2, 3]]),
            {"check_exact": False, "check_dtype": True},
            id="list_of_float_list_of_int_check_dtype_true",
        ),
        pytest.param(
            pl.struct(a=0, b=1.1, eager=True),
            pl.struct(a=0, b=1, eager=True),
            {"atol": 0.1, "rtol": 0, "check_dtype": True},
            id="struct_approx_equal_different_type",
        ),
        pytest.param(
            pl.struct(a=0, b=[0.0, math.nan], eager=True),
            pl.struct(a=0, b=[0.0, math.nan], eager=True),
            {"atol": 0.1, "nans_compare_equal": False},
            id="struct_with_list_with_nan_compare_equal_false",
        ),
    ],
)
def test_assert_series_equal_raises_assertion_error(
    s1: pl.Series,
    s2: pl.Series,
    kwargs: Any,
) -> None:
    with pytest.raises(AssertionError):
        assert_series_equal(s1, s2, **kwargs)
    assert_series_not_equal(s1, s2, **kwargs)


def test_assert_series_equal_categorical() -> None:
    s1 = pl.Series(["a", "b", "a"], dtype=pl.Categorical)
    s2 = pl.Series(["a", "b", "a"], dtype=pl.Categorical)
    with pytest.raises(pl.ComputeError, match="cannot compare categoricals"):
        assert_series_equal(s1, s2)

    assert_series_equal(s1, s2, categorical_as_str=True)


def test_assert_series_equal_categorical_vs_str() -> None:
    s1 = pl.Series(["a", "b", "a"], dtype=pl.Categorical)
    s2 = pl.Series(["a", "b", "a"], dtype=pl.Utf8)

    with pytest.raises(AssertionError, match="dtype mismatch"):
        assert_series_equal(s1, s2, categorical_as_str=True)


def test_assert_series_equal_full_series() -> None:
    s1 = pl.Series([1, 2, 3])
    s2 = pl.Series([1, 2, 4])
    msg = (
        r"Series are different \(value mismatch\)\n"
        r"\[left\]:  \[1, 2, 3\]\n"
        r"\[right\]: \[1, 2, 4\]"
    )
    with pytest.raises(AssertionError, match=msg):
        assert_series_equal(s1, s2)


def test_assert_frame_equal_dtypes_mismatch() -> None:
    data = {"a": [1, 2], "b": [3, 4]}
    df1 = pl.DataFrame(data, schema={"a": pl.Int8, "b": pl.Int16})
    df2 = pl.DataFrame(data, schema={"b": pl.Int16, "a": pl.Int16})

    with pytest.raises(AssertionError, match="dtypes do not match"):
        assert_frame_equal(df1, df2, check_column_order=False)


def test_assert_frame_not_equal() -> None:
    df = pl.DataFrame({"a": [1, 2]})
    with pytest.raises(AssertionError, match="frames are equal"):
        assert_frame_not_equal(df, df)


def test_assert_series_not_equal() -> None:
    s = pl.Series("a", [1, 2])
    with pytest.raises(AssertionError, match="Series are equal"):
        assert_series_not_equal(s, s)
