from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from typing import Any

import numpy as np
import pytest

import polars as pl
from polars._typing import PolarsDataType, TimeUnit
from polars.exceptions import (
    ComputeError,
    InvalidOperationError,
    OutOfBoundsError,
    PanicException,
    PolarsError,
)
from polars.testing import assert_frame_equal, assert_series_equal


@pytest.mark.parametrize(
    "input",
    [
        (),
        [],
        pl.Series(),
        pl.Series(dtype=pl.Int8),
        np.array([]),
    ],
)
def test_scatter_noop(input: Any) -> None:
    s = pl.Series("s", [1, 2, 3])
    s.scatter(input, 8)
    assert s.to_list() == [1, 2, 3]


def test_scatter() -> None:
    s = pl.Series("s", [1, 2, 3])

    # set new values, one index at a time
    s.scatter(0, 8)
    s.scatter([1], None)
    assert s.to_list() == [8, None, 3]

    # set new value at multiple indexes in one go
    s.scatter([0, 2], None)
    assert s.to_list() == [None, None, None]

    # try with different series dtype
    s = pl.Series("s", ["a", "b", "c"])
    s.scatter((1, 2), "x")
    assert s.to_list() == ["a", "x", "x"]
    assert s.scatter([0, 2], 0.12345).to_list() == ["0.12345", "x", "0.12345"]

    # set multiple values
    s = pl.Series(["z", "z", "z"])
    assert s.scatter([0, 1], ["a", "b"]).to_list() == ["a", "b", "z"]
    s = pl.Series([True, False, True])
    assert s.scatter([0, 1], [False, True]).to_list() == [False, True, True]

    # set negative indices
    a = pl.Series("r", range(5))
    a[-2] = None
    a[-5] = None
    assert a.to_list() == [None, 1, 2, None, 4]

    a = pl.Series("x", [1, 2])
    with pytest.raises(OutOfBoundsError):
        a[-100] = None
    assert_series_equal(a, pl.Series("x", [1, 2]))


def test_index_with_None_errors_16905() -> None:
    s = pl.Series("s", [1, 2, 3])
    with pytest.raises(ComputeError, match="index values should not be null"):
        s[[1, None]] = 5
    # The error doesn't trash the series, as it used to:
    assert_series_equal(s, pl.Series("s", [1, 2, 3]))


def test_object_dtype_16905() -> None:
    obj = object()
    s = pl.Series("s", [obj, 27], dtype=pl.Object)
    # This operation is not semantically wrong, it might be supported in the
    # future, but for now it isn't.
    with pytest.raises(InvalidOperationError):
        s[0] = 5
    # The error doesn't trash the series, as it used to:
    assert s.dtype.is_object()
    assert s.name == "s"
    assert s.to_list() == [obj, 27]


def test_scatter_datetime() -> None:
    s = pl.Series("dt", [None, datetime(2024, 1, 31)])
    result = s.scatter(0, datetime(2022, 2, 2))
    expected = pl.Series("dt", [datetime(2022, 2, 2), datetime(2024, 1, 31)])
    assert_series_equal(result, expected)


@pytest.mark.parametrize("time_unit", ["ms", "us", "ns"])
def test_scatter_datetime_time_unit_21474(time_unit: TimeUnit) -> None:
    dtype = pl.Datetime(time_unit)
    s = pl.Series("dt", [datetime(2024, 5, 26, 19), datetime(2024, 5, 27, 19)], dtype)
    result = s.scatter(1, datetime(2024, 5, 27, 19, 15))
    expected = pl.Series(
        "dt", [datetime(2024, 5, 26, 19), datetime(2024, 5, 27, 19, 15)], dtype
    )
    assert_series_equal(result, expected)

    df = pl.DataFrame({"dt": [datetime(2024, 5, 26, 19)]}, schema={"dt": dtype})
    df[0, "dt"] = datetime(2024, 5, 27, 19, 15)
    expected_df = pl.DataFrame(
        {"dt": [datetime(2024, 5, 27, 19, 15)]}, schema={"dt": dtype}
    )
    assert_frame_equal(df, expected_df)


def test_scatter_temporal_cast_21474() -> None:
    s = pl.Series([timedelta(1)], dtype=pl.Duration("ms"))
    expected = pl.Series([timedelta(seconds=1)], dtype=pl.Duration("ms"))
    assert_series_equal(s.scatter(0, timedelta(seconds=1)), expected)

    s = pl.Series([datetime(2024, 1, 1)], dtype=pl.Datetime("ms"))
    expected = pl.Series([datetime(2024, 5, 1)], dtype=pl.Datetime("ms"))
    assert_series_equal(s.scatter(0, date(2024, 5, 1)), expected)

    s = pl.Series([datetime(2024, 1, 1)], dtype=pl.Datetime("us"))
    expected = pl.Series([datetime(2024, 5, 1, 12)], dtype=pl.Datetime("us"))
    values = np.array(["2024-05-01T12:00"], dtype="datetime64[ms]")
    assert_series_equal(s.scatter(0, values), expected)

    with pytest.raises(InvalidOperationError):
        s.scatter(0, time(12))
    assert_series_equal(s, expected)


def test_scatter_logical_all_null() -> None:
    s = pl.Series("dt", [None, None], dtype=pl.Date)
    result = s.scatter(0, date(2022, 2, 2))
    expected = pl.Series("dt", [date(2022, 2, 2), None])
    assert_series_equal(result, expected)


def test_scatter_categorical_21175() -> None:
    s = pl.Series(["a", "b", "c"], dtype=pl.Categorical)
    assert_series_equal(
        s.scatter(0, "b"), pl.Series(["b", "b", "c"], dtype=pl.Categorical)
    )
    v = pl.Series(["v"], dtype=pl.Categorical)
    assert_series_equal(
        s.scatter([0, 2], v), pl.Series(["v", "b", "v"], dtype=pl.Categorical)
    )

    with pytest.raises(InvalidOperationError):
        s.scatter(1, 2)


def test_scatter_enum() -> None:
    e = pl.Enum(["a", "b", "c", "v"])
    s = pl.Series(["a", "b", "c"], dtype=e)
    assert_series_equal(s.scatter(0, "b"), pl.Series(["b", "b", "c"], dtype=e))
    v = pl.Series(["v"], dtype=pl.Categorical)
    assert_series_equal(s.scatter([0, 2], v), pl.Series(["v", "b", "v"], dtype=e))

    with pytest.raises(InvalidOperationError):
        s.scatter(1, "d")

    with pytest.raises(InvalidOperationError):
        s.scatter(1, 2)


@pytest.mark.parametrize("dtype", [pl.Categorical, pl.Enum(["a", "b"])])
def test_scatter_null(dtype: PolarsDataType) -> None:
    s = pl.Series("a", ["a", "b"], dtype=dtype)
    result = s.scatter(0, None)
    expected = pl.Series("a", [None, "b"], dtype=dtype)
    assert_series_equal(result, expected)


def test_scatter_decimal_25869() -> None:
    s = pl.Series([1, 2, 3], dtype=pl.Decimal(scale=10))
    assert_series_equal(
        s.scatter(0, 15), pl.Series([15, 2, 3], dtype=pl.Decimal(scale=10))
    )

    with pytest.raises(InvalidOperationError):
        s.scatter(1, "test")


@pytest.mark.parametrize(
    ("s", "value", "expected"),
    [
        (pl.Series([1.0, 2.0]), pl.Series([Decimal("9.9")]), 9.9),
        (pl.Series([1, 2]), pl.Series([Decimal("7.00")]), 7),
        (pl.Series(["a", "b"]), pl.Series([Decimal("9.9")]), "9.9"),
        (pl.Series(["a", "b"]), pl.Series(["zz"], dtype=pl.Categorical), "zz"),
        (pl.Series(["a", "b"]), pl.Series(["zz"], dtype=pl.Enum(["zz"])), "zz"),
        (pl.Series(["a", "b"]), pl.Series([date(2024, 5, 6)]), "2024-05-06"),
        (
            pl.Series([datetime(2024, 1, 1)], dtype=pl.Datetime("us", "UTC")),
            pl.Series([datetime(2024, 5, 6, 7)], dtype=pl.Datetime("ms", "UTC")),
            datetime(2024, 5, 6, 7, tzinfo=UTC),
        ),
    ],
)
def test_scatter_logical_value_30872(
    s: pl.Series, value: pl.Series, expected: Any
) -> None:
    # The value is cast to the column dtype, not stored as its physical representation.
    assert s.clone().scatter(0, value)[0] == expected

    t = s.clone()
    t[0] = value
    assert t[0] == expected

    t = s.clone()
    t[np.array([True, False])] = value
    assert t[0] == expected

    df = s.to_frame("c")
    df[0, "c"] = value
    assert df["c"][0] == expected


def test_scatter_logical_value_setitem_30872() -> None:
    s = pl.Series([1.0, 2.0, 3.0])
    s[0] = Decimal("9.9")
    assert s.to_list() == [9.9, 2.0, 3.0]

    s = pl.Series([1, 2, 3])
    s[[0, 1]] = [Decimal("7"), Decimal("8.00")]
    assert s.to_list() == [7, 8, 3]

    df = pl.DataFrame({"id": [1, 2], "amount": [10.0, 20.0]})
    df[1, "amount"] = Decimal("12.34")
    assert df.rows() == [(1, 10.0), (2, 12.34)]


@pytest.mark.parametrize(
    ("s", "idx", "value", "error"),
    [
        (pl.Series([1, 5, 3], dtype=pl.Int8), 1, 300, InvalidOperationError),
        (pl.Series([1, 5, 3], dtype=pl.UInt8), 1, -1, InvalidOperationError),
        (pl.Series([1, 5, 3]), 1, float("nan"), InvalidOperationError),
        (pl.Series([1, 5, 3]), 1, "x", InvalidOperationError),
        # One value that doesn't fit: nothing is written.
        (pl.Series([1, 5, 3], dtype=pl.Int8), [0, 1], [5, 300], InvalidOperationError),
        (pl.Series([1, 5, 3]), 1, pl.Series(["z"], dtype=pl.Categorical), PolarsError),
        (pl.Series([1, 5, 3]), 1, [{"a": 9}], InvalidOperationError),
    ],
)
def test_scatter_value_does_not_fit_raises_30872(
    s: pl.Series, idx: Any, value: Any, error: type[Exception]
) -> None:
    expected = s.clone()
    with pytest.raises(error):
        s.scatter(idx, value)
    assert_series_equal(s, expected)


def test_scatter_panic_in_cast_keeps_series_30872() -> None:
    # Casting this out-of-range date to String panics; the series must survive it.
    s = pl.Series("keep", ["a", "b"])
    with pytest.raises(PanicException):
        s.scatter(0, pl.Series([2**31 - 1]).cast(pl.Date))
    assert_series_equal(s, pl.Series("keep", ["a", "b"]))


def test_scatter_value_does_not_fit_frame_unchanged_30872() -> None:
    df = pl.DataFrame({"a": pl.Series([1, 2], dtype=pl.Int8)})
    with pytest.raises(InvalidOperationError):
        df[1, "a"] = 300
    assert_frame_equal(df, pl.DataFrame({"a": pl.Series([1, 2], dtype=pl.Int8)}))


def test_scatter_null_into_decimal_30872() -> None:
    dtype = pl.Decimal(5, 2)
    s = pl.Series([1, 2], dtype=dtype)
    s[0] = None
    assert_series_equal(s, pl.Series([None, 2], dtype=dtype))

    s[[0, 1]] = [None, None]
    assert_series_equal(s, pl.Series([None, None], dtype=dtype))

    df = pl.DataFrame({"d": pl.Series([1, 2], dtype=dtype)})
    df[0, "d"] = None
    assert df["d"].to_list() == [None, Decimal("2.00")]


def test_scatter_boolean_offset_slice_30105() -> None:
    # The slice is the only owner of its buffers.
    s = pl.Series("s", [True, True, False, False]).slice(2)
    s.scatter(0, False)
    assert s.to_list() == [False, False]


@pytest.mark.parametrize(
    ("dtype", "value"), [(pl.Int64, 1), (pl.String, "x"), (pl.Boolean, True)]
)
def test_scatter_offset_slice_validity_30105(dtype: PolarsDataType, value: Any) -> None:
    values = [value, value, value, None, value, None, value]
    s = pl.Series("s", values, dtype=dtype).slice(3)
    # Position 1 already holds `value`, so nothing should change.
    s.scatter(1, value)
    assert s.to_list() == [None, value, None, value]
