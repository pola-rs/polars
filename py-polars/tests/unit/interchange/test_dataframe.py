from __future__ import annotations

import pytest

import polars as pl
from polars.interchange.dataframe import PolarsDataFrame
from polars.interchange.protocol import CopyNotAllowedError
from polars.testing import assert_frame_equal, assert_series_equal


def test_dataframe_dunder() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    dfi = PolarsDataFrame(df)

    assert_frame_equal(dfi._df, df)
    assert dfi._allow_copy is True

    dfi_new = dfi.__dataframe__(allow_copy=False)

    assert_frame_equal(dfi_new._df, df)
    assert dfi_new._allow_copy is False


def test_dataframe_dunder_nan_as_null_not_implemented() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    dfi = PolarsDataFrame(df)

    with pytest.raises(NotImplementedError, match="has not been implemented"):
        df.__dataframe__(nan_as_null=True)

    with pytest.raises(NotImplementedError, match="has not been implemented"):
        dfi.__dataframe__(nan_as_null=True)


def test_metadata() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    dfi = PolarsDataFrame(df)
    assert dfi.metadata == {}


def test_num_columns() -> None:
    df = pl.DataFrame({"a": [1], "b": [2]})
    dfi = PolarsDataFrame(df)
    assert dfi.num_columns() == 2


def test_num_rows() -> None:
    df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    dfi = PolarsDataFrame(df)
    assert dfi.num_rows() == 2


def test_num_chunks() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    dfi = PolarsDataFrame(df)
    assert dfi.num_chunks() == 1

    df2 = pl.concat([df, df], rechunk=False)
    dfi2 = df2.__dataframe__()
    assert dfi2.num_chunks() == 2


def test_column_names() -> None:
    df = pl.DataFrame({"a": [1], "b": [2]})
    dfi = PolarsDataFrame(df)
    assert dfi.column_names() == ["a", "b"]


def test_get_column() -> None:
    df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    dfi = PolarsDataFrame(df)

    out = dfi.get_column(1)

    expected = pl.Series("b", [3, 4])
    assert_series_equal(out._col, expected)


def test_get_column_by_name() -> None:
    df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    dfi = PolarsDataFrame(df)

    out = dfi.get_column_by_name("b")

    expected = pl.Series("b", [3, 4])
    assert_series_equal(out._col, expected)


def test_get_columns() -> None:
    s1 = pl.Series("a", [1, 2])
    s2 = pl.Series("b", [3, 4])
    df = pl.DataFrame([s1, s2])
    dfi = PolarsDataFrame(df)

    out = dfi.get_columns()

    expected = [s1, s2]
    for o, e in zip(out, expected):
        assert_series_equal(o._col, e)


def test_select_columns() -> None:
    df = pl.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
    dfi = PolarsDataFrame(df)

    out = dfi.select_columns([0, 2])

    expected = pl.DataFrame({"a": [1, 2], "c": [5, 6]})
    assert_frame_equal(out._df, expected)


def test_select_columns_nonlist_input() -> None:
    df = pl.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
    dfi = PolarsDataFrame(df)

    out = dfi.select_columns((2,))

    expected = pl.DataFrame({"c": [5, 6]})
    assert_frame_equal(out._df, expected)


def test_select_columns_invalid_input() -> None:
    df = pl.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
    dfi = PolarsDataFrame(df)

    with pytest.raises(TypeError):
        dfi.select_columns(1)  # type: ignore[arg-type]


def test_select_columns_by_name() -> None:
    df = pl.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
    dfi = PolarsDataFrame(df)

    out = dfi.select_columns_by_name(["a", "c"])

    expected = pl.DataFrame({"a": [1, 2], "c": [5, 6]})
    assert_frame_equal(out._df, expected)


def test_select_columns_by_name_invalid_input() -> None:
    df = pl.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
    dfi = PolarsDataFrame(df)

    with pytest.raises(TypeError):
        dfi.select_columns_by_name(1)  # type: ignore[arg-type]


@pytest.mark.parametrize("n_chunks", [None, 2])
def test_get_chunks(n_chunks: int | None) -> None:
    df1 = pl.DataFrame({"a": [1, 2], "b": [4, 5]})
    df2 = pl.DataFrame({"a": [3], "b": [6]})
    df = pl.concat([df1, df2], rechunk=False)
    dfi = PolarsDataFrame(df)

    out = dfi.get_chunks(n_chunks)

    expected = dfi._get_chunks_from_col_chunks()
    for o, e in zip(out, expected):
        assert_frame_equal(o._df, e)


def test_get_chunks_invalid_input() -> None:
    df1 = pl.DataFrame({"a": [1, 2], "b": [4, 5]})
    df2 = pl.DataFrame({"a": [3], "b": [6]})
    df = pl.concat([df1, df2], rechunk=False)

    dfi = PolarsDataFrame(df)

    with pytest.raises(ValueError):
        next(dfi.get_chunks(0))

    with pytest.raises(ValueError):
        next(dfi.get_chunks(3))


def test_get_chunks_subdivided_chunks() -> None:
    df1 = pl.DataFrame({"a": [1, 2, 3], "b": [6, 7, 8]})
    df2 = pl.DataFrame({"a": [4, 5], "b": [9, 0]})
    df = pl.concat([df1, df2], rechunk=False)

    dfi = PolarsDataFrame(df)
    out = dfi.get_chunks(4)

    chunk1 = next(out)
    expected1 = pl.DataFrame({"a": [1, 2], "b": [6, 7]})
    assert_frame_equal(chunk1._df, expected1)

    chunk2 = next(out)
    expected2 = pl.DataFrame({"a": [3], "b": [8]})
    assert_frame_equal(chunk2._df, expected2)

    chunk3 = next(out)
    expected3 = pl.DataFrame({"a": [4], "b": [9]})
    assert_frame_equal(chunk3._df, expected3)

    chunk4 = next(out)
    expected4 = pl.DataFrame({"a": [5], "b": [0]})
    assert_frame_equal(chunk4._df, expected4)

    with pytest.raises(StopIteration):
        next(out)


def test_get_chunks_zero_copy_fail() -> None:
    col1 = pl.Series([1, 2])
    col2 = pl.concat([pl.Series([3]), pl.Series([4])], rechunk=False)
    df = pl.DataFrame({"a": col1, "b": col2})

    dfi = PolarsDataFrame(df, allow_copy=False)

    with pytest.raises(
        CopyNotAllowedError, match="unevenly chunked columns must be rechunked"
    ):
        next(dfi.get_chunks())


@pytest.mark.parametrize("allow_copy", [True, False])
def test_get_chunks_from_col_chunks_single_chunk(allow_copy: bool) -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    dfi = PolarsDataFrame(df, allow_copy=allow_copy)
    out = dfi._get_chunks_from_col_chunks()

    chunk1 = next(out)
    assert_frame_equal(chunk1, df)

    with pytest.raises(StopIteration):
        next(out)


@pytest.mark.parametrize("allow_copy", [True, False])
def test_get_chunks_from_col_chunks_even_chunks(allow_copy: bool) -> None:
    df1 = pl.DataFrame({"a": [1, 2], "b": [4, 5]})
    df2 = pl.DataFrame({"a": [3], "b": [6]})
    df = pl.concat([df1, df2], rechunk=False)

    dfi = PolarsDataFrame(df, allow_copy=allow_copy)
    out = dfi._get_chunks_from_col_chunks()

    chunk1 = next(out)
    assert_frame_equal(chunk1, df1)

    chunk2 = next(out)
    assert_frame_equal(chunk2, df2)

    with pytest.raises(StopIteration):
        next(out)


def test_get_chunks_from_col_chunks_uneven_chunks_allow_copy() -> None:
    col1 = pl.concat([pl.Series([1, 2]), pl.Series([3, 4, 5])], rechunk=False)
    col2 = pl.concat(
        [pl.Series([6, 7]), pl.Series([8]), pl.Series([9, 0])], rechunk=False
    )
    df = pl.DataFrame({"a": col1, "b": col2})

    dfi = PolarsDataFrame(df, allow_copy=True)
    out = dfi._get_chunks_from_col_chunks()

    expected1 = pl.DataFrame({"a": [1, 2], "b": [6, 7]})
    chunk1 = next(out)
    assert_frame_equal(chunk1, expected1)

    expected2 = pl.DataFrame({"a": [3, 4, 5], "b": [8, 9, 0]})
    chunk2 = next(out)
    assert_frame_equal(chunk2, expected2)

    with pytest.raises(StopIteration):
        next(out)


def test_get_chunks_from_col_chunks_uneven_chunks_zero_copy_fails() -> None:
    col1 = pl.concat([pl.Series([1, 2]), pl.Series([3, 4, 5])], rechunk=False)
    col2 = pl.concat(
        [pl.Series([6, 7]), pl.Series([8]), pl.Series([9, 0])], rechunk=False
    )
    df = pl.DataFrame({"a": col1, "b": col2})

    dfi = PolarsDataFrame(df, allow_copy=False)
    out = dfi._get_chunks_from_col_chunks()

    # First chunk can be yielded zero copy
    expected1 = pl.DataFrame({"a": [1, 2], "b": [6, 7]})
    chunk1 = next(out)
    assert_frame_equal(chunk1, expected1)

    # Second chunk requires a rechunk of the second column
    with pytest.raises(CopyNotAllowedError, match="columns must be rechunked"):
        next(out)


def test_dataframe_unsupported_types() -> None:
    df = pl.DataFrame({"a": [[4], [5, 6]]})
    dfi = PolarsDataFrame(df)

    # Generic dataframe operations work fine
    assert dfi.num_rows() == 2

    # Certain column operations also work
    col = dfi.get_column_by_name("a")
    assert col.num_chunks() == 1

    # Error is raised when unsupported operations are requested
    with pytest.raises(ValueError, match="not supported"):
        col.dtype
