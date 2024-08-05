from __future__ import annotations

import math
from typing import Any

import pytest
from hypothesis import given

import polars as pl
from polars.testing import assert_frame_equal, assert_frame_not_equal
from polars.testing.parametric import dataframes

nan = float("nan")
pytest_plugins = ["pytester"]


@given(df=dataframes())
def test_equal(df: pl.DataFrame) -> None:
    assert_frame_equal(df, df.clone(), check_exact=True)


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
            {"check_dtypes": False},
            id="equal_int_float_integer_no_check_dtype",
        ),
        pytest.param(
            pl.DataFrame({"a": [0, 1, 2]}, schema={"a": pl.Float64}),
            pl.DataFrame({"a": [0, 1, 2]}, schema={"a": pl.Float32}),
            {"check_dtypes": False},
            id="equal_int_float_integer_no_check_dtype",
        ),
        pytest.param(
            pl.DataFrame({"a": [0, 1, 2]}, schema={"a": pl.Int64}),
            pl.DataFrame({"a": [0, 1, 2]}, schema={"a": pl.Int64}),
            {},
            id="equal_int",
        ),
        pytest.param(
            pl.DataFrame({"a": ["a", "b", "c"]}, schema={"a": pl.String}),
            pl.DataFrame({"a": ["a", "b", "c"]}, schema={"a": pl.String}),
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
            pl.DataFrame({"a": [[[0.2, 3.0]]]}),
            pl.DataFrame({"a": [[[0.2, 3.00000001]]]}),
            {"atol": 0.1},
            id="nested_list_of_float_atol_high",
        ),
    ],
)
def test_assert_frame_equal_passes_assertion(
    df1: pl.DataFrame,
    df2: pl.DataFrame,
    kwargs: dict[str, Any],
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
            pl.DataFrame({"a": [[2.0, 3.0]]}),
            pl.DataFrame({"a": [[2, 3]]}),
            {"check_exact": False, "check_dtypes": True},
            id="list_of_float_list_of_int_check_dtype_true",
        ),
        pytest.param(
            pl.DataFrame({"a": [[[0.2, math.nan, 3.0]]]}),
            pl.DataFrame({"a": [[[0.2, math.nan, 3.11]]]}),
            {"atol": 0.1, "rtol": 0},
            id="nested_list_of_float_and_nan_atol_high",
        ),
        pytest.param(
            pl.DataFrame({"a": [[[[0.2, 3.0]]]]}),
            pl.DataFrame({"a": [[[[0.2, 3.11]]]]}),
            {"atol": 0.1, "rtol": 0},
            id="double_nested_list_of_float_atol_high",
        ),
        pytest.param(
            pl.DataFrame({"a": [[[[[0.2, 3.0]]]]]}),
            pl.DataFrame({"a": [[[[[0.2, 3.11]]]]]}),
            {"atol": 0.1, "rtol": 0},
            id="triple_nested_list_of_float_atol_high",
        ),
    ],
)
def test_assert_frame_equal_raises_assertion_error(
    df1: pl.DataFrame,
    df2: pl.DataFrame,
    kwargs: dict[str, Any],
) -> None:
    with pytest.raises(AssertionError):
        assert_frame_equal(df1, df2, **kwargs)
    assert_frame_not_equal(df1, df2, **kwargs)


def test_compare_frame_equal_nans() -> None:
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
    with pytest.raises(AssertionError, match="value mismatch for column 'y'"):
        assert_frame_equal(df1, df2, check_exact=True)


def test_compare_frame_equal_nested_nans() -> None:
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
    with pytest.raises(AssertionError, match="value mismatch for column 'y'"):
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
                    {"x": "text", "y": [nan, 1.0], "z": ["!"]},
                    {"x": "text", "y": [nan, 1.0], "z": ["?"]},
                ],
            },
        ]
    )

    assert_frame_equal(df3, df3)
    assert_frame_equal(df4, df4)

    assert_frame_not_equal(df3, df4)
    for check_dtype in (True, False):
        with pytest.raises(AssertionError, match="mismatch|different"):
            assert_frame_equal(df3, df4, check_dtypes=check_dtype)


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
        AssertionError,
        match=r"DataFrames are different \(number of rows does not match\)",
    ):
        assert_frame_equal(df1, df2)


def test_assert_frame_equal_column_mismatch() -> None:
    df1 = pl.DataFrame({"a": [1, 2]})
    df2 = pl.DataFrame({"b": [1, 2]})
    with pytest.raises(
        AssertionError, match="columns \\['a'\\] in left DataFrame, but not in right"
    ):
        assert_frame_equal(df1, df2)


def test_assert_frame_equal_column_mismatch2() -> None:
    df1 = pl.LazyFrame({"a": [1, 2]})
    df2 = pl.LazyFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
    with pytest.raises(
        AssertionError,
        match="columns \\['b', 'c'\\] in right LazyFrame, but not in left",
    ):
        assert_frame_equal(df1, df2)


def test_assert_frame_equal_column_mismatch_order() -> None:
    df1 = pl.DataFrame({"b": [3, 4], "a": [1, 2]})
    df2 = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    with pytest.raises(AssertionError, match="columns are not in the same order"):
        assert_frame_equal(df1, df2)

    assert_frame_equal(df1, df2, check_column_order=False)


def test_assert_frame_equal_check_row_order() -> None:
    df1 = pl.DataFrame({"a": [1, 2], "b": [4, 3]})
    df2 = pl.DataFrame({"a": [2, 1], "b": [3, 4]})

    with pytest.raises(AssertionError, match="value mismatch for column 'a'"):
        assert_frame_equal(df1, df2)
    assert_frame_equal(df1, df2, check_row_order=False)


def test_assert_frame_equal_check_row_col_order() -> None:
    df1 = pl.DataFrame({"a": [1, 2], "b": [4, 3]})
    df3 = pl.DataFrame({"b": [3, 4], "a": [2, 1]})

    with pytest.raises(AssertionError, match="columns are not in the same order"):
        assert_frame_equal(df1, df3, check_row_order=False)
    assert_frame_equal(df1, df3, check_row_order=False, check_column_order=False)


def test_assert_frame_equal_check_row_order_unsortable() -> None:
    df1 = pl.DataFrame({"a": [object(), object()], "b": [3, 4]})
    df2 = pl.DataFrame({"a": [object(), object()], "b": [4, 3]})
    with pytest.raises(
        TypeError, match="cannot set `check_row_order=False`.*unsortable columns"
    ):
        assert_frame_equal(df1, df2, check_row_order=False)


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


def test_assert_frame_equal_check_dtype_deprecated() -> None:
    df1 = pl.DataFrame({"a": [1, 2]})
    df2 = pl.DataFrame({"a": [1.0, 2.0]})
    df3 = pl.DataFrame({"a": [2, 1]})

    with pytest.deprecated_call():
        assert_frame_equal(df1, df2, check_dtype=False)  # type: ignore[call-arg]

    with pytest.deprecated_call():
        assert_frame_not_equal(df1, df3, check_dtype=False)  # type: ignore[call-arg]


def test_tracebackhide(testdir: pytest.Testdir) -> None:
    testdir.makefile(
        ".py",
        test_path="""\
import polars as pl
from polars.testing import assert_frame_equal, assert_frame_not_equal

def test_frame_equal_fail():
    df1 = pl.DataFrame({"a": [1, 2]})
    df2 = pl.DataFrame({"a": [1, 3]})
    assert_frame_equal(df1, df2)

def test_frame_not_equal_fail():
    df1 = pl.DataFrame({"a": [1, 2]})
    df2 = pl.DataFrame({"a": [1, 2]})
    assert_frame_not_equal(df1, df2)

def test_frame_data_type_fail():
    df1 = pl.DataFrame({"a": [1, 2]})
    df2 = {"a": [1, 2]}
    assert_frame_equal(df1, df2)

def test_frame_schema_fail():
    df1 = pl.DataFrame({"a": [1, 2]}, {"a": pl.Int64})
    df2 = pl.DataFrame({"a": [1, 2]}, {"a": pl.Int32})
    assert_frame_equal(df1, df2)
""",
    )
    result = testdir.runpytest()
    result.assert_outcomes(passed=0, failed=4)
    stdout = "\n".join(result.outlines)

    assert "polars/py-polars/polars/testing" not in stdout

    # The above should catch any polars testing functions that appear in the
    # stack trace. But we keep the following checks (for specific function
    # names) just to double-check.

    assert "def assert_frame_equal" not in stdout
    assert "def assert_frame_not_equal" not in stdout
    assert "def _assert_correct_input_type" not in stdout
    assert "def _assert_frame_schema_equal" not in stdout

    assert "def assert_series_equal" not in stdout
    assert "def assert_series_not_equal" not in stdout
    assert "def _assert_series_values_equal" not in stdout
    assert "def _assert_series_nested_values_equal" not in stdout
    assert "def _assert_series_null_values_match" not in stdout
    assert "def _assert_series_nan_values_match" not in stdout
    assert "def _assert_series_values_within_tolerance" not in stdout

    # Make sure the tests are failing for the expected reason (e.g. not because
    # an import is missing or something like that):

    assert (
        "AssertionError: DataFrames are different (value mismatch for column 'a')"
        in stdout
    )
    assert "AssertionError: frames are equal" in stdout
    assert "AssertionError: inputs are different (unexpected input types)" in stdout
    assert "AssertionError: DataFrames are different (dtypes do not match)" in stdout
