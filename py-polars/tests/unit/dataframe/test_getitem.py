from __future__ import annotations

from typing import Any

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given

import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal
from polars.testing.parametric import column, dataframes


@given(
    df=dataframes(
        max_size=10,
        cols=[
            column(
                "start",
                dtype=pl.Int8,
                allow_null=True,
                strategy=st.integers(min_value=-8, max_value=8),
            ),
            column(
                "stop",
                dtype=pl.Int8,
                allow_null=True,
                strategy=st.integers(min_value=-6, max_value=6),
            ),
            column(
                "step",
                dtype=pl.Int8,
                allow_null=True,
                strategy=st.integers(min_value=-4, max_value=4).filter(
                    lambda x: x != 0
                ),
            ),
            column("misc", dtype=pl.Int32),
        ],
    )
    # generated dataframe example -
    # ┌───────┬──────┬──────┬───────┐
    # │ start ┆ stop ┆ step ┆ misc  │
    # │ ---   ┆ ---  ┆ ---  ┆ ---   │
    # │ i8    ┆ i8   ┆ i8   ┆ i32   │
    # ╞═══════╪══════╪══════╪═══════╡
    # │ 2     ┆ -1   ┆ null ┆ -55   │
    # │ -3    ┆ 0    ┆ -2   ┆ 61582 │
    # │ null  ┆ 1    ┆ 2    ┆ 5865  │
    # └───────┴──────┴──────┴───────┘
)
def test_df_getitem_row_slice(df: pl.DataFrame) -> None:
    # take strategy-generated integer values from the frame as slice bounds.
    # use these bounds to slice the same frame, and then validate the result
    # against a py-native slice of the same data using the same bounds.
    #
    # given the average number of rows in the frames, and the value of
    # max_examples, this will result in close to 5000 test permutations,
    # running in around ~1.5 secs (depending on hardware/etc).
    py_data = df.rows()

    for start, stop, step, _ in py_data:
        s = slice(start, stop, step)
        sliced_py_data = py_data[s]
        sliced_df_data = df[s].rows()

        assert (
            sliced_py_data == sliced_df_data
        ), f"slice [{start}:{stop}:{step}] failed on df w/len={len(df)}"


def test_df_getitem_col_single_name() -> None:
    df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    result = df[:, "a"]
    expected = df.select("a").to_series()
    assert_series_equal(result, expected)


@pytest.mark.parametrize(
    ("input", "expected_cols"),
    [
        (["a"], ["a"]),
        (["a", "d"], ["a", "d"]),
        (slice("b", "d"), ["b", "c", "d"]),
        (pl.Series(["a", "b"]), ["a", "b"]),
        (np.array(["c", "d"]), ["c", "d"]),
    ],
)
def test_df_getitem_col_multiple_names(input: Any, expected_cols: list[str]) -> None:
    df = pl.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6], "d": [7, 8]})
    result = df[:, input]
    expected = df.select(expected_cols)
    assert_frame_equal(result, expected)


def test_df_getitem_col_single_index() -> None:
    df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    result = df[:, 1]
    expected = df.select("b").to_series()
    assert_series_equal(result, expected)


def test_df_getitem_col_two_entries() -> None:
    df = pl.DataFrame({"x": [1.0], "y": [1.0]})

    assert_frame_equal(df["x", "y"], df)
    assert_frame_equal(df[True, True], df)


@pytest.mark.parametrize(
    ("input", "expected_cols"),
    [
        ([0], ["a"]),
        ([0, 3], ["a", "d"]),
        (slice(1, 4), ["b", "c", "d"]),
        (pl.Series([0, 1]), ["a", "b"]),
        (np.array([2, 3]), ["c", "d"]),
    ],
)
def test_df_getitem_col_multiple_indices(input: Any, expected_cols: list[str]) -> None:
    df = pl.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6], "d": [7, 8]})
    result = df[:, input]
    expected = df.select(expected_cols)
    assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "mask",
    [
        [True, False, True],
        pl.Series([True, False, True]),
        np.array([True, False, True]),
    ],
)
def test_df_getitem_col_boolean_mask(mask: Any) -> None:
    df = pl.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
    result = df[:, mask]
    expected = df.select("a", "c")
    assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ("rng", "expected_cols"),
    [
        (range(2), ["a", "b"]),
        (range(1, 4), ["b", "c", "d"]),
        (range(3, 0, -2), ["d", "b"]),
    ],
)
def test_df_getitem_col_range(rng: range, expected_cols: list[str]) -> None:
    df = pl.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6], "d": [7, 8]})
    result = df[:, rng]
    expected = df.select(expected_cols)
    assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "input", [[], (), pl.Series(dtype=pl.Int64), np.array([], dtype=np.uint32)]
)
def test_df_getitem_col_empty_inputs(input: Any) -> None:
    df = pl.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
    result = df[:, input]
    expected = pl.DataFrame()
    assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ("input", "match"),
    [
        (
            [0.0, 1.0],
            "cannot select columns using Sequence with elements of type 'float'",
        ),
        (
            pl.Series([[1, 2], [3, 4]]),
            "cannot select columns using Series of type List\\(Int64\\)",
        ),
        (
            np.array([0.0, 1.0]),
            "cannot select columns using NumPy array of type float64",
        ),
        (object(), "cannot select columns using key of type 'object'"),
    ],
)
def test_df_getitem_col_invalid_inputs(input: Any, match: str) -> None:
    df = pl.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
    with pytest.raises(TypeError, match=match):
        df[:, input]


@pytest.mark.parametrize(
    ("input", "match"),
    [
        (["a", 2], "'int' object cannot be converted to 'PyString'"),
        ([1, "c"], "'str' object cannot be interpreted as an integer"),
    ],
)
def test_df_getitem_col_mixed_inputs(input: list[Any], match: str) -> None:
    df = pl.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
    with pytest.raises(TypeError, match=match):
        df[:, input]


@pytest.mark.parametrize(
    ("input", "match"),
    [
        ([0.0, 1.0], "unexpected value while building Series of type Int64"),
        (
            pl.Series([[1, 2], [3, 4]]),
            "cannot treat Series of type List\\(Int64\\) as indices",
        ),
        (np.array([0.0, 1.0]), "cannot treat NumPy array of type float64 as indices"),
        (object(), "cannot select rows using key of type 'object'"),
    ],
)
def test_df_getitem_row_invalid_inputs(input: Any, match: str) -> None:
    df = pl.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
    with pytest.raises(TypeError, match=match):
        df[input, :]


def test_df_getitem_row_range() -> None:
    df = pl.DataFrame({"a": [1, 2, 3, 4], "b": [5.0, 6.0, 7.0, 8.0]})
    result = df[range(3, 0, -2), :]
    expected = pl.DataFrame({"a": [4, 2], "b": [8.0, 6.0]})
    assert_frame_equal(result, expected)


def test_df_getitem_row_range_single_input() -> None:
    df = pl.DataFrame({"a": [1, 2, 3, 4], "b": [5.0, 6.0, 7.0, 8.0]})
    result = df[range(1, 3)]
    expected = pl.DataFrame({"a": [2, 3], "b": [6.0, 7.0]})
    assert_frame_equal(result, expected)


def test_df_getitem_row_empty_list_single_input() -> None:
    df = pl.DataFrame({"a": [1, 2], "b": [5.0, 6.0]})
    result = df[[]]
    expected = df.clear()
    assert_frame_equal(result, expected)


def test_df_getitem() -> None:
    """Test all the methods to use [] on a dataframe."""
    df = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [3, 4, 5, 6]})

    # multiple slices.
    # The first element refers to the rows, the second element to columns
    assert_frame_equal(df[:, :], df)

    # str, always refers to a column name
    assert_series_equal(df["a"], pl.Series("a", [1.0, 2.0, 3.0, 4.0]))

    # int, always refers to a row index (zero-based): index=1 => second row
    assert_frame_equal(df[1], pl.DataFrame({"a": [2.0], "b": [4]}))

    # int, int.
    # The first element refers to the rows, the second element to columns
    assert df[2, 1] == 5
    assert df[2, -2] == 3.0

    with pytest.raises(IndexError):
        # Column index out of bounds
        df[2, 2]

    with pytest.raises(IndexError):
        # Column index out of bounds
        df[2, -3]

    # int, list[int].
    # The first element refers to the rows, the second element to columns
    assert_frame_equal(df[2, [1, 0]], pl.DataFrame({"b": [5], "a": [3.0]}))
    assert_frame_equal(df[2, [-1, -2]], pl.DataFrame({"b": [5], "a": [3.0]}))

    with pytest.raises(IndexError):
        # Column index out of bounds
        df[2, [2, 0]]

    with pytest.raises(IndexError):
        # Column index out of bounds
        df[2, [2, -3]]

    # slice. Below an example of taking every second row
    assert_frame_equal(df[1::2], pl.DataFrame({"a": [2.0, 4.0], "b": [4, 6]}))

    # slice, empty slice
    assert df[:0].columns == ["a", "b"]
    assert len(df[:0]) == 0

    # make mypy happy
    empty: list[int] = []

    # empty list with column selector drops rows but keeps columns
    assert_frame_equal(df[empty, :], df[:0])

    # numpy array: assumed to be row indices if integers, or columns if strings

    # numpy array: positive idxs and empty idx
    for np_dtype in (
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
    ):
        assert_frame_equal(
            df[np.array([1, 0, 3, 2, 3, 0], dtype=np_dtype)],
            pl.DataFrame(
                {"a": [2.0, 1.0, 4.0, 3.0, 4.0, 1.0], "b": [4, 3, 6, 5, 6, 3]}
            ),
        )
        assert df[np.array([], dtype=np_dtype)].columns == ["a", "b"]

    # numpy array: positive and negative idxs.
    for np_dtype in (np.int8, np.int16, np.int32, np.int64):
        assert_frame_equal(
            df[np.array([-1, 0, -3, -2, 3, -4], dtype=np_dtype)],
            pl.DataFrame(
                {"a": [4.0, 1.0, 2.0, 3.0, 4.0, 1.0], "b": [6, 3, 4, 5, 6, 3]}
            ),
        )

    # note that we cannot use floats (even if they could be casted to integer without
    # loss)
    with pytest.raises(TypeError):
        _ = df[np.array([1.0])]

    with pytest.raises(
        TypeError, match="multi-dimensional NumPy arrays not supported as index"
    ):
        df[np.array([[0], [1]])]

    # sequences (lists or tuples; tuple only if length != 2)
    # if strings or list of expressions, assumed to be column names
    # if bools, assumed to be a row mask
    # if integers, assumed to be row indices
    assert_frame_equal(df[["a", "b"]], df)
    assert_frame_equal(df.select([pl.col("a"), pl.col("b")]), df)
    assert_frame_equal(
        df[[1, -4, -1, 2, 1]],
        pl.DataFrame({"a": [2.0, 1.0, 4.0, 3.0, 2.0], "b": [4, 3, 6, 5, 4]}),
    )

    # pl.Series: strings for column selections.
    assert_frame_equal(df[pl.Series("", ["a", "b"])], df)

    # pl.Series: positive idxs or empty idxs for row selection.
    for pl_dtype in (
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
    ):
        assert_frame_equal(
            df[pl.Series("", [1, 0, 3, 2, 3, 0], dtype=pl_dtype)],
            pl.DataFrame(
                {"a": [2.0, 1.0, 4.0, 3.0, 4.0, 1.0], "b": [4, 3, 6, 5, 6, 3]}
            ),
        )
        assert df[pl.Series("", [], dtype=pl_dtype)].columns == ["a", "b"]

    # pl.Series: positive and negative idxs for row selection.
    for pl_dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64):
        assert_frame_equal(
            df[pl.Series("", [-1, 0, -3, -2, 3, -4], dtype=pl_dtype)],
            pl.DataFrame(
                {"a": [4.0, 1.0, 2.0, 3.0, 4.0, 1.0], "b": [6, 3, 4, 5, 6, 3]}
            ),
        )

    # Boolean masks for rows not supported
    with pytest.raises(TypeError):
        df[[True, False, True], [False, True]]
    with pytest.raises(TypeError):
        df[pl.Series([True, False, True]), "b"]

    assert_frame_equal(df[np.array([True, False])], df[:, :1])

    # wrong length boolean mask for column selection
    with pytest.raises(
        ValueError,
        match=f"expected {df.width} values when selecting columns by boolean mask",
    ):
        df[:, [True, False, True]]


def test_df_getitem2() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0], "c": ["a", "b", "c"]})

    # select columns by mask
    assert df[:2, :1].rows() == [(1,), (2,)]
    assert df[:2, ["a"]].rows() == [(1,), (2,)]

    # column selection by string(s) in first dimension
    assert df["a"].to_list() == [1, 2, 3]
    assert df["b"].to_list() == [1.0, 2.0, 3.0]
    assert df["c"].to_list() == ["a", "b", "c"]

    # row selection by integers(s) in first dimension
    assert_frame_equal(df[0], pl.DataFrame({"a": [1], "b": [1.0], "c": ["a"]}))
    assert_frame_equal(df[-1], pl.DataFrame({"a": [3], "b": [3.0], "c": ["c"]}))

    # row, column selection when using two dimensions
    assert df[:, "a"].to_list() == [1, 2, 3]
    assert df[:, 1].to_list() == [1.0, 2.0, 3.0]
    assert df[:2, 2].to_list() == ["a", "b"]

    assert_frame_equal(
        df[[1, 2]], pl.DataFrame({"a": [2, 3], "b": [2.0, 3.0], "c": ["b", "c"]})
    )
    assert_frame_equal(
        df[[-1, -2]], pl.DataFrame({"a": [3, 2], "b": [3.0, 2.0], "c": ["c", "b"]})
    )

    assert df[["a", "b"]].columns == ["a", "b"]
    assert_frame_equal(
        df[[1, 2], [1, 2]], pl.DataFrame({"b": [2.0, 3.0], "c": ["b", "c"]})
    )
    assert df[1, 2] == "b"
    assert df[1, 1] == 2.0
    assert df[2, 0] == 3

    assert df[[2], ["a", "b"]].rows() == [(3, 3.0)]
    assert df.to_series(0).name == "a"
    assert (df["a"] == df["a"]).sum() == 3
    assert (df["c"] == df["a"].cast(str)).sum() == 0
    assert df[:, "a":"b"].rows() == [(1, 1.0), (2, 2.0), (3, 3.0)]  # type: ignore[index, misc]
    assert df[:, "a":"c"].columns == ["a", "b", "c"]  # type: ignore[index, misc]
    assert df[:, []].shape == (0, 0)
    expect = pl.DataFrame({"c": ["b"]})
    assert_frame_equal(df[1, [2]], expect)
    expect = pl.DataFrame({"b": [1.0, 3.0]})
    assert_frame_equal(df[[0, 2], [1]], expect)
    assert df[0, "c"] == "a"
    assert df[1, "c"] == "b"
    assert df[2, "c"] == "c"
    assert df[0, "a"] == 1

    # more slicing
    expect = pl.DataFrame({"a": [3, 2, 1], "b": [3.0, 2.0, 1.0], "c": ["c", "b", "a"]})
    assert_frame_equal(df[::-1], expect)
    expect = pl.DataFrame({"a": [1, 2], "b": [1.0, 2.0], "c": ["a", "b"]})
    assert_frame_equal(df[:-1], expect)

    expect = pl.DataFrame({"a": [1, 3], "b": [1.0, 3.0], "c": ["a", "c"]})
    assert_frame_equal(df[::2], expect)

    # only allow boolean values in column position
    df = pl.DataFrame(
        {
            "a": [1, 2],
            "b": [2, 3],
            "c": [3, 4],
        }
    )

    assert df[:, [False, True, True]].columns == ["b", "c"]
    assert df[:, pl.Series([False, True, True])].columns == ["b", "c"]
    assert df[:, pl.Series([False, False, False])].columns == []


def test_df_getitem_5343() -> None:
    # https://github.com/pola-rs/polars/issues/5343
    df = pl.DataFrame(
        {
            f"foo{col}": [n**col for n in range(5)]  # 5 rows
            for col in range(12)  # 12 columns
        }
    )
    assert df[4, 4] == 256
    assert df[4, 5] == 1024
    assert_frame_equal(df[4, [2]], pl.DataFrame({"foo2": [16]}))
    assert_frame_equal(df[4, [5]], pl.DataFrame({"foo5": [1024]}))
