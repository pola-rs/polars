from __future__ import annotations

from datetime import datetime, time, timedelta
from decimal import Decimal as D
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from hypothesis import given, settings
from numpy.testing import assert_array_equal

import polars as pl
from polars.testing.parametric import series

if TYPE_CHECKING:
    import numpy.typing as npt


def assert_zero_copy(s: pl.Series, arr: np.ndarray[Any, Any]) -> None:
    if s.len() == 0:
        return
    s_ptr = s._get_buffers()["values"]._get_buffer_info()[0]
    arr_ptr = arr.__array_interface__["data"][0]
    assert s_ptr == arr_ptr


def assert_zero_copy_only_raises(s: pl.Series) -> None:
    with pytest.raises(ValueError, match="cannot return a zero-copy array"):
        s.to_numpy(use_pyarrow=False, zero_copy_only=True)


@pytest.mark.parametrize(
    ("dtype", "expected_dtype"),
    [
        (pl.Int8, np.int8),
        (pl.Int16, np.int16),
        (pl.Int32, np.int32),
        (pl.Int64, np.int64),
        (pl.UInt8, np.uint8),
        (pl.UInt16, np.uint16),
        (pl.UInt32, np.uint32),
        (pl.UInt64, np.uint64),
        (pl.Float32, np.float32),
        (pl.Float64, np.float64),
    ],
)
def test_series_to_numpy_numeric_zero_copy(
    dtype: pl.PolarsDataType, expected_dtype: npt.DTypeLike
) -> None:
    s = pl.Series([1, 2, 3], dtype=dtype, strict=False)
    result = s.to_numpy(use_pyarrow=False, zero_copy_only=True)

    assert_zero_copy(s, result)
    assert result.tolist() == s.to_list()
    assert result.dtype == expected_dtype


@pytest.mark.parametrize(
    ("dtype", "expected_dtype"),
    [
        (pl.Int8, np.float32),
        (pl.Int16, np.float32),
        (pl.Int32, np.float64),
        (pl.Int64, np.float64),
        (pl.UInt8, np.float32),
        (pl.UInt16, np.float32),
        (pl.UInt32, np.float64),
        (pl.UInt64, np.float64),
        (pl.Float32, np.float32),
        (pl.Float64, np.float64),
    ],
)
def test_series_to_numpy_numeric_with_nulls(
    dtype: pl.PolarsDataType, expected_dtype: npt.DTypeLike
) -> None:
    s = pl.Series([1, 2, None], dtype=dtype, strict=False)
    result = s.to_numpy(use_pyarrow=False)

    assert result.tolist()[:-1] == s.to_list()[:-1]
    assert np.isnan(result[-1])
    assert result.dtype == expected_dtype
    assert_zero_copy_only_raises(s)


@pytest.mark.parametrize(
    ("dtype", "values"),
    [
        (pl.Categorical, ["a", "b", "a"]),
        (pl.Enum(["a", "b", "c"]), ["a", "b", "a"]),
        (pl.String, ["a", "bc", "def"]),
        (pl.Binary, [b"a", b"bc", b"def"]),
        (pl.Object, [Path(), Path("abc")]),
        # TODO: Implement for List types
        # (pl.List, [[1], [2, 3]]),
        # (pl.List, [["a"], ["b", "c"], []]),
    ],
)
def test_to_numpy_various_dtypes(dtype: pl.PolarsDataType, values: list[Any]) -> None:
    values.append(None)
    s = pl.Series(values, dtype=dtype)
    result = s.to_numpy(use_pyarrow=False)

    assert result.tolist() == values
    assert result.dtype == np.object_
    assert_zero_copy_only_raises(s)


def test_series_to_numpy_bool() -> None:
    s = pl.Series([True, False])
    result = s.to_numpy(use_pyarrow=False)

    assert s.to_list() == result.tolist()
    assert result.dtype == np.bool_
    assert_zero_copy_only_raises(s)


def test_series_to_numpy_bool_with_nulls() -> None:
    s = pl.Series([True, False, None])
    result = s.to_numpy(use_pyarrow=False)

    assert s.to_list() == result.tolist()
    assert result.dtype == np.object_
    assert_zero_copy_only_raises(s)


def test_series_to_numpy_array_of_int() -> None:
    values = [[1, 2], [3, 4], [5, 6]]
    s = pl.Series(values, dtype=pl.Array(pl.Int64, 2))
    result = s.to_numpy(use_pyarrow=False)

    expected = np.array(values)
    assert_array_equal(result, expected)
    assert result.dtype == np.int64


def test_series_to_numpy_array_of_str() -> None:
    values = [["1", "2", "3"], ["4", "5", "10000"]]
    s = pl.Series(values, dtype=pl.Array(pl.String, 3))
    result = s.to_numpy(use_pyarrow=False)
    assert result.tolist() == values
    assert result.dtype == np.object_


@pytest.mark.skip(
    reason="Currently bugged, see: https://github.com/pola-rs/polars/issues/14268"
)
def test_series_to_numpy_array_with_nulls() -> None:
    values = [[1, 2], [3, 4], None]
    s = pl.Series(values, dtype=pl.Array(pl.Int64, 2))
    result = s.to_numpy(use_pyarrow=False)

    expected = np.array([[1.0, 2.0], [3.0, 4.0], [np.nan, np.nan]])
    assert_array_equal(result, expected)
    assert result.dtype == np.float64
    assert_zero_copy_only_raises(s)


def test_to_numpy_null() -> None:
    s = pl.Series([None, None], dtype=pl.Null)
    result = s.to_numpy(use_pyarrow=False)
    expected = np.array([np.nan, np.nan], dtype=np.float32)
    assert_array_equal(result, expected)
    assert result.dtype == np.float32
    assert_zero_copy_only_raises(s)


@pytest.mark.skip(
    reason="Currently bugged, see: https://github.com/pola-rs/polars/issues/14274"
)
def test_to_numpy_empty() -> None:
    series = pl.Series(dtype=pl.String)
    result = series.to_numpy(use_pyarrow=False, zero_copy_only=True)
    assert result.dtype == np.int64
    assert result.shape == (0,)
    assert result.size == 0


@given(
    s=series(
        min_size=1, max_size=10, excluded_dtypes=[pl.Categorical, pl.List, pl.Struct]
    ).filter(
        lambda s: (
            getattr(s.dtype, "time_unit", None) != "ms"
            and not (s.dtype == pl.String and s.str.contains("\x00").any())
            and not (s.dtype == pl.Binary and s.bin.contains(b"\x00").any())
        )
    ),
)
@settings(max_examples=250)
def test_series_to_numpy(s: pl.Series) -> None:
    result = s.to_numpy(use_pyarrow=False)

    values = s.to_list()
    dtype_map = {
        pl.Datetime("ns"): "datetime64[ns]",
        pl.Datetime("us"): "datetime64[us]",
        pl.Duration("ns"): "timedelta64[ns]",
        pl.Duration("us"): "timedelta64[us]",
    }
    np_dtype = dtype_map.get(s.dtype)  # type: ignore[call-overload]
    expected = np.array(values, dtype=np_dtype)

    assert_array_equal(result, expected)


@pytest.mark.parametrize("use_pyarrow", [True, False])
@pytest.mark.parametrize("has_null", [True, False])
@pytest.mark.parametrize("dtype", [pl.Time, pl.Boolean, pl.String])
def test_to_numpy_no_zero_copy(
    use_pyarrow: bool, has_null: bool, dtype: pl.PolarsDataType
) -> None:
    data: list[Any] = ["a", None] if dtype == pl.String else [0, None]
    series = pl.Series(data if has_null else data[:1], dtype=dtype)
    with pytest.raises(ValueError):
        series.to_numpy(zero_copy_only=True, use_pyarrow=use_pyarrow)


@pytest.mark.parametrize("writable", [False, True])
@pytest.mark.parametrize("pyarrow_available", [False, True])
def test_to_numpy2(
    writable: bool, pyarrow_available: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pl.series.series, "_PYARROW_AVAILABLE", pyarrow_available)

    np_array = pl.Series("a", [1, 2, 3], pl.UInt8).to_numpy(writable=writable)

    np.testing.assert_array_equal(np_array, np.array([1, 2, 3], dtype=np.uint8))
    # Test if numpy array is readonly or writable.
    assert np_array.flags.writeable == writable

    if writable:
        np_array[1] += 10
        np.testing.assert_array_equal(np_array, np.array([1, 12, 3], dtype=np.uint8))

    np_array_with_missing_values = pl.Series("a", [None, 2, 3], pl.UInt8).to_numpy(
        writable=writable
    )

    np.testing.assert_array_equal(
        np_array_with_missing_values,
        np.array(
            [np.nan, 2.0, 3.0],
            dtype=(np.float64 if pyarrow_available else np.float32),
        ),
    )

    if writable:
        # As Null values can't be encoded natively in a numpy array,
        # this array will never be a view.
        assert np_array_with_missing_values.flags.writeable == writable


def test_view() -> None:
    s = pl.Series("a", [1.0, 2.5, 3.0])
    result = s._view()
    assert isinstance(result, np.ndarray)
    assert np.all(result == np.array([1.0, 2.5, 3.0]))


def test_view_nulls() -> None:
    s = pl.Series("b", [1, 2, None])
    assert s.has_validity()
    with pytest.raises(AssertionError):
        s._view()


def test_view_nulls_sliced() -> None:
    s = pl.Series("b", [1, 2, None])
    sliced = s[:2]
    assert np.all(sliced._view() == np.array([1, 2]))
    assert not sliced.has_validity()


def test_view_ub() -> None:
    # this would be UB if the series was dropped and not passed to the view
    s = pl.Series([3, 1, 5])
    result = s.sort()._view()
    assert np.sum(result) == 9


def test_view_deprecated() -> None:
    s = pl.Series("a", [1.0, 2.5, 3.0])
    with pytest.deprecated_call():
        result = s.view()
    assert isinstance(result, np.ndarray)
    assert np.all(result == np.array([1.0, 2.5, 3.0]))


def test_numpy_disambiguation() -> None:
    a = np.array([1, 2])
    df = pl.DataFrame({"a": a})
    result = df.with_columns(b=a).to_dict(as_series=False)  # type: ignore[arg-type]
    expected = {
        "a": [1, 2],
        "b": [1, 2],
    }
    assert result == expected


def test_to_numpy_datelike() -> None:
    s = pl.Series(
        "dt",
        [
            datetime(2022, 7, 5, 10, 30, 45, 123456),
            None,
            datetime(2023, 2, 5, 15, 22, 30, 987654),
        ],
    )
    assert str(s.to_numpy()) == str(
        np.array(
            ["2022-07-05T10:30:45.123456", "NaT", "2023-02-05T15:22:30.987654"],
            dtype="datetime64[us]",
        )
    )
    assert str(s.drop_nulls().to_numpy()) == str(
        np.array(
            ["2022-07-05T10:30:45.123456", "2023-02-05T15:22:30.987654"],
            dtype="datetime64[us]",
        )
    )


def test_series_to_numpy_temporal() -> None:
    s0 = pl.Series("date", [123543, 283478, 1243]).cast(pl.Date)
    s1 = pl.Series(
        "datetime", [datetime(2021, 1, 2, 3, 4, 5), datetime(2021, 2, 3, 4, 5, 6)]
    )
    s2 = pl.datetime_range(
        datetime(2021, 1, 1, 0),
        datetime(2021, 1, 1, 1),
        interval="1h",
        time_unit="ms",
        eager=True,
    )
    assert str(s0.to_numpy()) == "['2308-04-02' '2746-02-20' '1973-05-28']"
    assert (
        str(s1.to_numpy()[:2])
        == "['2021-01-02T03:04:05.000000' '2021-02-03T04:05:06.000000']"
    )
    assert (
        str(s2.to_numpy()[:2])
        == "['2021-01-01T00:00:00.000' '2021-01-01T01:00:00.000']"
    )
    s3 = pl.Series([timedelta(hours=1), timedelta(hours=-2)])
    out = np.array([3_600_000_000_000, -7_200_000_000_000], dtype="timedelta64[ns]")
    assert (s3.to_numpy() == out).all()

    s4 = pl.Series([time(10, 30, 45), time(23, 59, 59)])
    out = np.array([time(10, 30, 45), time(23, 59, 59)], dtype="object")
    assert (s4.to_numpy() == out).all()


@pytest.mark.parametrize("use_pyarrow", [True, False])
def test_decimal_numpy_export(use_pyarrow: bool) -> None:
    decimal_data = [D("1.234"), D("2.345"), D("-3.456")]

    s = pl.Series("n", decimal_data)
    df = s.to_frame()

    assert_array_equal(
        np.array(decimal_data),
        s.to_numpy(use_pyarrow=use_pyarrow),
    )
    assert_array_equal(
        np.array(decimal_data).reshape((-1, 1)),
        df.to_numpy(use_pyarrow=use_pyarrow),
    )
