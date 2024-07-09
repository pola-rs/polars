from __future__ import annotations

from datetime import date, datetime, time, timedelta
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

    from polars._typing import PolarsDataType


def assert_zero_copy(s: pl.Series, arr: np.ndarray[Any, Any]) -> None:
    if s.len() == 0:
        return
    s_ptr = s._get_buffers()["values"]._get_buffer_info()[0]
    arr_ptr = arr.__array_interface__["data"][0]
    assert s_ptr == arr_ptr


def assert_allow_copy_false_raises(s: pl.Series) -> None:
    with pytest.raises(RuntimeError, match="copy not allowed"):
        s.to_numpy(allow_copy=False)


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
    dtype: PolarsDataType, expected_dtype: npt.DTypeLike
) -> None:
    s = pl.Series([1, 2, 3]).cast(dtype)
    result = s.to_numpy(allow_copy=False)

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
    dtype: PolarsDataType, expected_dtype: npt.DTypeLike
) -> None:
    s = pl.Series([1, 2, None], dtype=dtype, strict=False)
    result = s.to_numpy()

    assert result.tolist()[:-1] == s.to_list()[:-1]
    assert np.isnan(result[-1])
    assert result.dtype == expected_dtype
    assert_allow_copy_false_raises(s)


@pytest.mark.parametrize(
    ("dtype", "expected_dtype"),
    [
        (pl.Duration, np.dtype("timedelta64[us]")),
        (pl.Duration("ms"), np.dtype("timedelta64[ms]")),
        (pl.Duration("us"), np.dtype("timedelta64[us]")),
        (pl.Duration("ns"), np.dtype("timedelta64[ns]")),
        (pl.Datetime, np.dtype("datetime64[us]")),
        (pl.Datetime("ms"), np.dtype("datetime64[ms]")),
        (pl.Datetime("us"), np.dtype("datetime64[us]")),
        (pl.Datetime("ns"), np.dtype("datetime64[ns]")),
    ],
)
def test_series_to_numpy_temporal_zero_copy(
    dtype: PolarsDataType, expected_dtype: npt.DTypeLike
) -> None:
    values = [0, 2_000, 1_000_000]
    s = pl.Series(values, dtype=dtype, strict=False)
    result = s.to_numpy(allow_copy=False)

    assert_zero_copy(s, result)
    # NumPy tolist returns integers for ns precision
    if s.dtype.time_unit == "ns":  # type: ignore[attr-defined]
        assert result.tolist() == values
    else:
        assert result.tolist() == s.to_list()
    assert result.dtype == expected_dtype


def test_series_to_numpy_datetime_with_tz_zero_copy() -> None:
    values = [datetime(1970, 1, 1), datetime(2024, 2, 28)]
    s = pl.Series(values).dt.convert_time_zone("Europe/Amsterdam")
    result = s.to_numpy(allow_copy=False)

    assert_zero_copy(s, result)
    assert result.tolist() == values
    assert result.dtype == np.dtype("datetime64[us]")


def test_series_to_numpy_date() -> None:
    values = [date(1970, 1, 1), date(2024, 2, 28)]
    s = pl.Series(values)

    result = s.to_numpy()

    assert s.to_list() == result.tolist()
    assert result.dtype == np.dtype("datetime64[D]")
    assert result.flags.writeable is True
    assert_allow_copy_false_raises(s)


@pytest.mark.parametrize(
    ("dtype", "expected_dtype"),
    [
        (pl.Date, np.dtype("datetime64[D]")),
        (pl.Duration("ms"), np.dtype("timedelta64[ms]")),
        (pl.Duration("us"), np.dtype("timedelta64[us]")),
        (pl.Duration("ns"), np.dtype("timedelta64[ns]")),
        (pl.Datetime, np.dtype("datetime64[us]")),
        (pl.Datetime("ms"), np.dtype("datetime64[ms]")),
        (pl.Datetime("us"), np.dtype("datetime64[us]")),
        (pl.Datetime("ns"), np.dtype("datetime64[ns]")),
    ],
)
def test_series_to_numpy_temporal_with_nulls(
    dtype: PolarsDataType, expected_dtype: npt.DTypeLike
) -> None:
    values = [0, 2_000, 1_000_000, None]
    s = pl.Series(values, dtype=dtype, strict=False)
    result = s.to_numpy()

    # NumPy tolist returns integers for ns precision
    if getattr(s.dtype, "time_unit", None) == "ns":
        assert result.tolist() == values
    else:
        assert result.tolist() == s.to_list()
    assert result.dtype == expected_dtype
    assert_allow_copy_false_raises(s)


def test_series_to_numpy_datetime_with_tz_with_nulls() -> None:
    values = [datetime(1970, 1, 1), datetime(2024, 2, 28), None]
    s = pl.Series(values).dt.convert_time_zone("Europe/Amsterdam")
    result = s.to_numpy()

    assert result.tolist() == values
    assert result.dtype == np.dtype("datetime64[us]")
    assert_allow_copy_false_raises(s)


@pytest.mark.parametrize(
    ("dtype", "values"),
    [
        (pl.Time, [time(10, 30, 45), time(23, 59, 59)]),
        (pl.Categorical, ["a", "b", "a"]),
        (pl.Enum(["a", "b", "c"]), ["a", "b", "a"]),
        (pl.String, ["a", "bc", "def"]),
        (pl.Binary, [b"a", b"bc", b"def"]),
        (pl.Decimal, [D("1.234"), D("2.345"), D("-3.456")]),
        (pl.Object, [Path(), Path("abc")]),
    ],
)
@pytest.mark.parametrize("with_nulls", [False, True])
def test_to_numpy_object_dtypes(
    dtype: PolarsDataType, values: list[Any], with_nulls: bool
) -> None:
    if with_nulls:
        values.append(None)

    s = pl.Series(values, dtype=dtype)
    result = s.to_numpy()

    assert result.tolist() == values
    assert result.dtype == np.object_
    assert_allow_copy_false_raises(s)


def test_series_to_numpy_bool() -> None:
    s = pl.Series([True, False])
    result = s.to_numpy()

    assert s.to_list() == result.tolist()
    assert result.dtype == np.bool_
    assert result.flags.writeable is True
    assert_allow_copy_false_raises(s)


def test_series_to_numpy_bool_with_nulls() -> None:
    s = pl.Series([True, False, None])
    result = s.to_numpy()

    assert s.to_list() == result.tolist()
    assert result.dtype == np.object_
    assert_allow_copy_false_raises(s)


def test_series_to_numpy_array_of_int() -> None:
    values = [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]
    s = pl.Series(values, dtype=pl.Array(pl.Array(pl.Int8, 3), 2))
    result = s.to_numpy(allow_copy=False)

    expected = np.array(values)
    assert_array_equal(result, expected)
    assert result.dtype == np.int8
    assert result.shape == (2, 2, 3)


def test_series_to_numpy_array_of_str() -> None:
    values = [["1", "2", "3"], ["4", "5", "10000"]]
    s = pl.Series(values, dtype=pl.Array(pl.String, 3))
    result = s.to_numpy()
    assert result.tolist() == values
    assert result.dtype == np.object_


def test_series_to_numpy_array_with_nulls() -> None:
    values = [[1, 2], [3, 4], None]
    s = pl.Series(values, dtype=pl.Array(pl.Int64, 2))
    result = s.to_numpy()

    expected = np.array([[1.0, 2.0], [3.0, 4.0], [np.nan, np.nan]])
    assert_array_equal(result, expected)
    assert result.dtype == np.float64
    assert_allow_copy_false_raises(s)


def test_series_to_numpy_array_with_nested_nulls() -> None:
    values = [[None, 2], [3, 4], [5, None]]
    s = pl.Series(values, dtype=pl.Array(pl.Int64, 2))
    result = s.to_numpy()

    expected = np.array([[np.nan, 2.0], [3.0, 4.0], [5.0, np.nan]])
    assert_array_equal(result, expected)
    assert result.dtype == np.float64
    assert_allow_copy_false_raises(s)


def test_series_to_numpy_array_of_arrays() -> None:
    values = [[[None, 2], [3, 4]], [None, [7, 8]]]
    s = pl.Series(values, dtype=pl.Array(pl.Array(pl.Int64, 2), 2))
    result = s.to_numpy()

    expected = np.array([[[np.nan, 2], [3, 4]], [[np.nan, np.nan], [7, 8]]])
    assert_array_equal(result, expected)
    assert result.dtype == np.float64
    assert result.shape == (2, 2, 2)
    assert_allow_copy_false_raises(s)


@pytest.mark.parametrize("chunked", [True, False])
def test_series_to_numpy_list(chunked: bool) -> None:
    values = [[1, 2], [3, 4, 5], [6], []]
    s = pl.Series(values)
    if chunked:
        s = pl.concat([s[:2], s[2:]])
    result = s.to_numpy()

    expected = np.array([np.array(v, dtype=np.int64) for v in values], dtype=np.object_)
    for res, exp in zip(result, expected):
        assert_array_equal(res, exp)
        assert res.flags.writeable == chunked
    assert result.dtype == expected.dtype
    assert_allow_copy_false_raises(s)


def test_series_to_numpy_struct_numeric_supertype() -> None:
    values = [{"a": 1, "b": 2.0}, {"a": 3, "b": 4.0}, {"a": 5, "b": None}]
    s = pl.Series(values)
    result = s.to_numpy()

    expected = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, np.nan]])
    assert_array_equal(result, expected)
    assert result.dtype == np.float64
    assert_allow_copy_false_raises(s)


def test_to_numpy_null() -> None:
    s = pl.Series([None, None], dtype=pl.Null)
    result = s.to_numpy()
    expected = np.array([np.nan, np.nan], dtype=np.float32)
    assert_array_equal(result, expected)
    assert result.dtype == np.float32
    assert_allow_copy_false_raises(s)


def test_to_numpy_empty() -> None:
    s = pl.Series(dtype=pl.String)
    result = s.to_numpy(allow_copy=False)
    assert result.dtype == np.object_
    assert result.shape == (0,)


def test_to_numpy_empty_writable() -> None:
    s = pl.Series(dtype=pl.Int64)
    result = s.to_numpy(allow_copy=False, writable=True)
    assert result.dtype == np.int64
    assert result.shape == (0,)
    assert result.flags.writeable is True


def test_to_numpy_chunked() -> None:
    s1 = pl.Series([1, 2])
    s2 = pl.Series([3, 4])
    s = pl.concat([s1, s2], rechunk=False)

    result = s.to_numpy()

    assert result.tolist() == s.to_list()
    assert result.dtype == np.int64
    assert result.flags.writeable is True
    assert_allow_copy_false_raises(s)

    # Check that writing to the array doesn't change the original data
    result[0] = 10
    assert result.tolist() == [10, 2, 3, 4]
    assert s.to_list() == [1, 2, 3, 4]


def test_to_numpy_chunked_temporal_nested() -> None:
    dtype = pl.Array(pl.Datetime("us"), 1)
    s1 = pl.Series([[datetime(2020, 1, 1)], [datetime(2021, 1, 1)]], dtype=dtype)
    s2 = pl.Series([[datetime(2022, 1, 1)], [datetime(2023, 1, 1)]], dtype=dtype)
    s = pl.concat([s1, s2], rechunk=False)

    result = s.to_numpy()

    assert result.tolist() == s.to_list()
    assert result.dtype == np.dtype("datetime64[us]")
    assert result.shape == (4, 1)
    assert result.flags.writeable is True
    assert_allow_copy_false_raises(s)


def test_zero_copy_only_deprecated() -> None:
    values = [1, 2]
    s = pl.Series([1, 2])
    with pytest.deprecated_call():
        result = s.to_numpy(zero_copy_only=True)
    assert result.tolist() == values


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


@given(
    s=series(
        min_size=1,
        max_size=10,
        excluded_dtypes=[
            pl.Categorical,
            pl.List,
            pl.Struct,
            pl.Datetime("ms"),
            pl.Duration("ms"),
        ],
        allow_null=False,
        allow_time_zones=False,  # NumPy does not support parsing time zone aware data
    ).filter(
        lambda s: (
            not (s.dtype == pl.String and s.str.contains("\x00").any())
            and not (s.dtype == pl.Binary and s.bin.contains(b"\x00").any())
        )
    ),
)
@settings(max_examples=250)
def test_series_to_numpy(s: pl.Series) -> None:
    result = s.to_numpy()

    values = s.to_list()
    dtype_map = {
        pl.Datetime("ns"): "datetime64[ns]",
        pl.Datetime("us"): "datetime64[us]",
        pl.Duration("ns"): "timedelta64[ns]",
        pl.Duration("us"): "timedelta64[us]",
        pl.Null(): "float32",
    }
    np_dtype = dtype_map.get(s.dtype)
    expected = np.array(values, dtype=np_dtype)

    assert_array_equal(result, expected)


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
