from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import pytest

import polars as pl
from polars.interchange.column import PolarsColumn
from polars.interchange.protocol import ColumnNullType, CopyNotAllowedError, DtypeKind
from polars.testing import assert_series_equal

if TYPE_CHECKING:
    from polars.interchange.protocol import Dtype


def test_size() -> None:
    s = pl.Series([1, 2, 3])
    col = PolarsColumn(s)
    assert col.size() == 3


def test_offset() -> None:
    s = pl.Series([1, 2, 3])
    col = PolarsColumn(s)
    assert col.offset == 0


def test_dtype_int() -> None:
    s = pl.Series([1, 2, 3], dtype=pl.Int32)
    col = PolarsColumn(s)
    assert col.dtype == (DtypeKind.INT, 32, "i", "=")


def test_dtype_categorical() -> None:
    s = pl.Series(["a", "b", "a"], dtype=pl.Categorical)
    col = PolarsColumn(s)
    assert col.dtype == (DtypeKind.CATEGORICAL, 32, "I", "=")


def test_describe_categorical() -> None:
    s = pl.Series(["b", "a", "a", "c", None, "b"], dtype=pl.Categorical)
    col = PolarsColumn(s)

    out = col.describe_categorical

    assert out["is_ordered"] is True
    assert out["is_dictionary"] is True

    expected_categories = pl.Series(["b", "a", "c"])
    assert_series_equal(out["categories"]._col, expected_categories)


def test_describe_categorical_lexical_ordering() -> None:
    s = pl.Series(["b", "a", "a", "c", None, "b"], dtype=pl.Categorical("lexical"))
    col = PolarsColumn(s)

    out = col.describe_categorical

    assert out["is_ordered"] is False


def test_describe_categorical_enum() -> None:
    s = pl.Series(["b", "a", "a", "c", None, "b"], dtype=pl.Enum(["a", "b", "c"]))
    col = PolarsColumn(s)

    out = col.describe_categorical

    assert out["is_ordered"] is True
    assert out["is_dictionary"] is True

    expected_categories = pl.Series("category", ["a", "b", "c"])
    assert_series_equal(out["categories"]._col, expected_categories)


def test_describe_categorical_other_dtype() -> None:
    s = pl.Series(["a", "b", "a"], dtype=pl.String)
    col = PolarsColumn(s)
    with pytest.raises(TypeError):
        col.describe_categorical


def test_describe_null() -> None:
    s = pl.Series([1, 2, None])
    col = PolarsColumn(s)
    assert col.describe_null == (ColumnNullType.USE_BITMASK, 0)


def test_describe_null_no_null_values() -> None:
    s = pl.Series([1, 2, 3])
    col = PolarsColumn(s)
    assert col.describe_null == (ColumnNullType.NON_NULLABLE, None)


def test_null_count() -> None:
    s = pl.Series([None, 2, None])
    col = PolarsColumn(s)
    assert col.null_count == 2


def test_metadata() -> None:
    s = pl.Series([1, 2])
    col = PolarsColumn(s)
    assert col.metadata == {}


def test_num_chunks() -> None:
    s = pl.Series([1, 2])
    col = PolarsColumn(s)
    assert col.num_chunks() == 1

    s2 = pl.concat([s, s], rechunk=False)
    col2 = s2.to_frame().__dataframe__().get_column(0)
    assert col2.num_chunks() == 2


@pytest.mark.parametrize("n_chunks", [None, 2])
def test_get_chunks(n_chunks: int | None) -> None:
    s1 = pl.Series([1, 2, 3])
    s2 = pl.Series([4, 5])
    s = pl.concat([s1, s2], rechunk=False)
    col = PolarsColumn(s)

    out = col.get_chunks(n_chunks)

    expected = [s1, s2]
    for o, e in zip(out, expected):
        assert_series_equal(o._col, e)


def test_get_chunks_invalid_input() -> None:
    s1 = pl.Series([1, 2, 3])
    s2 = pl.Series([4, 5])
    s = pl.concat([s1, s2], rechunk=False)
    col = PolarsColumn(s)

    with pytest.raises(ValueError):
        next(col.get_chunks(0))

    with pytest.raises(ValueError):
        next(col.get_chunks(3))


def test_get_chunks_subdivided_chunks() -> None:
    s1 = pl.Series([1, 2, 3])
    s2 = pl.Series([4, 5])
    s = pl.concat([s1, s2], rechunk=False)
    col = PolarsColumn(s)

    out = col.get_chunks(4)

    chunk1 = next(out)
    expected1 = pl.Series([1, 2])
    assert_series_equal(chunk1._col, expected1)

    chunk2 = next(out)
    expected2 = pl.Series([3])
    assert_series_equal(chunk2._col, expected2)

    chunk3 = next(out)
    expected3 = pl.Series([4])
    assert_series_equal(chunk3._col, expected3)

    chunk4 = next(out)
    expected4 = pl.Series([5])
    assert_series_equal(chunk4._col, expected4)

    with pytest.raises(StopIteration):
        next(out)


@pytest.mark.parametrize(
    ("series", "expected_data", "expected_dtype"),
    [
        (
            pl.Series([1, None, 3], dtype=pl.Int16),
            pl.Series([1, 0, 3], dtype=pl.Int16),
            (DtypeKind.INT, 16, "s", "="),
        ),
        (
            pl.Series([-1.5, 3.0, None], dtype=pl.Float64),
            pl.Series([-1.5, 3.0, 0.0], dtype=pl.Float64),
            (DtypeKind.FLOAT, 64, "g", "="),
        ),
        (
            pl.Series(["a", "bc", None, "éâç"], dtype=pl.String),
            pl.Series([97, 98, 99, 195, 169, 195, 162, 195, 167], dtype=pl.UInt8),
            (DtypeKind.UINT, 8, "C", "="),
        ),
        (
            pl.Series(
                [datetime(1988, 1, 2), None, datetime(2022, 12, 3)], dtype=pl.Datetime
            ),
            pl.Series([568080000000000, 0, 1670025600000000], dtype=pl.Int64),
            (DtypeKind.INT, 64, "l", "="),
        ),
        (
            pl.Series(["a", "b", None, "a"], dtype=pl.Categorical),
            pl.Series([0, 1, 0, 0], dtype=pl.UInt32),
            (DtypeKind.UINT, 32, "I", "="),
        ),
    ],
)
def test_get_buffers_data(
    series: pl.Series,
    expected_data: pl.Series,
    expected_dtype: Dtype,
) -> None:
    col = PolarsColumn(series)

    out = col.get_buffers()

    data_buffer, data_dtype = out["data"]
    assert_series_equal(data_buffer._data, expected_data)
    assert data_dtype == expected_dtype


def test_get_buffers_int() -> None:
    s = pl.Series([1, 2, 3], dtype=pl.Int8)
    col = PolarsColumn(s)

    out = col.get_buffers()

    data_buffer, data_dtype = out["data"]
    assert_series_equal(data_buffer._data, s)
    assert data_dtype == (DtypeKind.INT, 8, "c", "=")

    assert out["validity"] is None
    assert out["offsets"] is None


def test_get_buffers_with_validity_and_offsets() -> None:
    s = pl.Series(["a", "bc", None, "éâç"])
    col = PolarsColumn(s)

    out = col.get_buffers()

    data_buffer, data_dtype = out["data"]
    expected = pl.Series([97, 98, 99, 195, 169, 195, 162, 195, 167], dtype=pl.UInt8)
    assert_series_equal(data_buffer._data, expected)
    assert data_dtype == (DtypeKind.UINT, 8, "C", "=")

    validity = out["validity"]
    assert validity is not None
    val_buffer, val_dtype = validity
    expected = pl.Series([True, True, False, True])
    assert_series_equal(val_buffer._data, expected)
    assert val_dtype == (DtypeKind.BOOL, 1, "b", "=")

    offsets = out["offsets"]
    assert offsets is not None
    offsets_buffer, offsets_dtype = offsets
    expected = pl.Series([0, 1, 3, 3, 9], dtype=pl.Int64)
    assert_series_equal(offsets_buffer._data, expected)
    assert offsets_dtype == (DtypeKind.INT, 64, "l", "=")


def test_get_buffers_chunked_bitmask() -> None:
    s = pl.Series([True, False], dtype=pl.Boolean)
    s_chunked = pl.concat([s[:1], s[1:]], rechunk=False)
    col = PolarsColumn(s_chunked)

    chunks = list(col.get_chunks())
    assert chunks[0].get_buffers()["data"][0]._data.item() is True
    assert chunks[1].get_buffers()["data"][0]._data.item() is False


def test_get_buffers_string_zero_copy_fails() -> None:
    s = pl.Series("a", ["a", "bc"], dtype=pl.String)

    col = PolarsColumn(s, allow_copy=False)

    msg = "string buffers must be converted"
    with pytest.raises(CopyNotAllowedError, match=msg):
        col.get_buffers()


def test_get_buffers_global_categorical() -> None:
    with pl.StringCache():
        _ = pl.Series("a", ["a", "b"], dtype=pl.Categorical)
        s = pl.Series("a", ["c", "b"], dtype=pl.Categorical)

    # Converted to local categorical
    col = PolarsColumn(s, allow_copy=True)
    result = col.get_buffers()

    data_buffer, _ = result["data"]
    expected = pl.Series("a", [0, 1], dtype=pl.UInt32)
    assert_series_equal(data_buffer._data, expected)

    # Zero copy fails
    col = PolarsColumn(s, allow_copy=False)

    msg = "column 'a' must be converted to a local categorical"
    with pytest.raises(CopyNotAllowedError, match=msg):
        col.get_buffers()


def test_get_buffers_chunked_zero_copy_fails() -> None:
    s1 = pl.Series([1, 2, 3])
    s = pl.concat([s1, s1], rechunk=False)
    col = PolarsColumn(s, allow_copy=False)

    with pytest.raises(
        CopyNotAllowedError, match="non-contiguous buffer must be made contiguous"
    ):
        col.get_buffers()


def test_wrap_data_buffer() -> None:
    values = pl.Series([1, 2, 3])
    col = PolarsColumn(pl.Series())

    result_buffer, result_dtype = col._wrap_data_buffer(values)

    assert_series_equal(result_buffer._data, values)
    assert result_dtype == (DtypeKind.INT, 64, "l", "=")


def test_wrap_validity_buffer() -> None:
    validity = pl.Series([True, False, True])
    col = PolarsColumn(pl.Series())

    result = col._wrap_validity_buffer(validity)

    assert result is not None

    result_buffer, result_dtype = result
    assert_series_equal(result_buffer._data, validity)
    assert result_dtype == (DtypeKind.BOOL, 1, "b", "=")


def test_wrap_validity_buffer_no_nulls() -> None:
    col = PolarsColumn(pl.Series())
    assert col._wrap_validity_buffer(None) is None


def test_wrap_offsets_buffer() -> None:
    offsets = pl.Series([0, 1, 3, 3, 9], dtype=pl.Int64)
    col = PolarsColumn(pl.Series())

    result = col._wrap_offsets_buffer(offsets)

    assert result is not None

    result_buffer, result_dtype = result
    assert_series_equal(result_buffer._data, offsets)
    assert result_dtype == (DtypeKind.INT, 64, "l", "=")


def test_wrap_offsets_buffer_none() -> None:
    col = PolarsColumn(pl.Series())
    assert col._wrap_validity_buffer(None) is None


def test_column_unsupported_type() -> None:
    s = pl.Series("a", [[4], [5, 6]])
    col = PolarsColumn(s)

    # Certain column operations work
    assert col.num_chunks() == 1
    assert col.null_count == 0

    # Error is raised when unsupported operations are requested
    with pytest.raises(ValueError, match="not supported"):
        col.dtype
