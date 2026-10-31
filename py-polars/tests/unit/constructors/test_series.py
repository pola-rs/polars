from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from typing import TYPE_CHECKING, Any, Literal

import dateutil.tz
import numpy as np
import pandas as pd
import pytest
import pytz

import polars as pl
from polars.exceptions import InvalidOperationError
from polars.testing.asserts.series import assert_series_equal

if TYPE_CHECKING:
    from polars._typing import PolarsDataType


requires_pandas_as_unit = pytest.mark.skipif(
    not hasattr(pd.Timestamp, "as_unit"),
    reason="pandas version does not support Timestamp.as_unit",
)


def test_series_mixed_dtypes_list() -> None:
    values = [[0.1, 1]]
    s = pl.Series(values, strict=True)
    assert s.dtype == pl.List(pl.Float64)
    assert s.to_list() == [[0.1, 1.0]]


def test_series_mixed_dtypes_string() -> None:
    values = [[12], "foo", 9]

    with pytest.raises(TypeError, match="unexpected value"):
        pl.Series(values)

    s = pl.Series(values, strict=False)
    assert s.dtype == pl.String
    assert s.to_list() == ["[12]", "foo", "9"]
    assert s[1] == "foo"


def test_series_mixed_dtypes_object() -> None:
    values = [[12], b"foo", 9]

    with pytest.raises(TypeError, match="unexpected value"):
        pl.Series(values)

    s = pl.Series(values, strict=False)
    assert s.dtype.is_object()
    assert s.to_list() == values
    assert s[1] == b"foo"


# https://github.com/pola-rs/polars/issues/15139
@pytest.mark.parametrize("dtype", [pl.List(pl.Int64), None])
def test_sequence_of_series_with_dtype(dtype: PolarsDataType | None) -> None:
    values = [1, 2, 3]
    int_series = pl.Series(values)
    list_series = pl.Series([int_series], dtype=dtype)

    assert list_series.to_list() == [values]
    assert list_series.dtype == pl.List(pl.Int64)


@pytest.mark.parametrize(
    ("values", "dtype", "expected_dtype"),
    [
        ([1, 1.0, 1], None, pl.Float64),
        ([1, 1, "1.0"], None, pl.String),
        ([1, 1.0, "1.0"], None, pl.String),
        ([True, 1], None, pl.Int64),
        ([True, 1.0], None, pl.Float64),
        ([True, 1], pl.Boolean, pl.Boolean),
        ([True, 1.0], pl.Boolean, pl.Boolean),
        ([False, "1.0"], None, pl.String),
    ],
)
def test_upcast_primitive_and_strings(
    values: list[Any], dtype: PolarsDataType, expected_dtype: PolarsDataType
) -> None:
    with pytest.raises(TypeError):
        pl.Series(values, dtype=dtype, strict=True)

    assert pl.Series(values, dtype=dtype, strict=False).dtype == expected_dtype


def test_preserve_decimal_precision() -> None:
    dtype = pl.Decimal(None, 1)
    s = pl.Series(dtype=dtype)
    assert s.dtype == dtype


@pytest.mark.parametrize("dtype", [None, pl.Duration("ms")])
def test_large_timedelta(dtype: pl.DataType | None) -> None:
    values = [timedelta.min, timedelta.max]
    s = pl.Series(values, dtype=dtype)
    assert s.dtype == pl.Duration("ms")

    # Microsecond precision is lost
    expected = [timedelta.min, timedelta.max - timedelta(microseconds=999)]
    assert s.to_list() == expected


def test_array_large_u64() -> None:
    u64_max = 2**64 - 1
    values = [[u64_max]]
    dtype = pl.Array(pl.UInt64, 1)
    s = pl.Series(values, dtype=dtype)
    assert s.dtype == dtype
    assert s.to_list() == values


def test_array_creation_idx_size() -> None:
    s = pl.Series([None])
    width = 2**31
    s = s.new_from_index(0, width)
    assert pl.Series("a", [s, s, s, s], dtype=pl.Array(pl.Null, width)).shape == (4,)


def test_series_init_ambiguous_datetime() -> None:
    value = datetime(2001, 10, 28, 2)
    dtype = pl.Datetime(time_zone="Europe/Belgrade")

    result = pl.Series([value], dtype=dtype, strict=True)
    expected = pl.Series([datetime(2001, 10, 28, 3)]).dt.replace_time_zone(
        "Europe/Belgrade"
    )
    assert_series_equal(result, expected)

    result = pl.Series([value], dtype=dtype, strict=False)
    assert_series_equal(result, expected)


def test_series_init_nonexistent_datetime() -> None:
    value = datetime(2024, 3, 31, 2, 30)
    dtype = pl.Datetime(time_zone="Europe/Amsterdam")

    result = pl.Series([value], dtype=dtype, strict=True)
    expected = pl.Series([datetime(2024, 3, 31, 4, 30)]).dt.replace_time_zone(
        "Europe/Amsterdam"
    )
    assert_series_equal(result, expected)

    result = pl.Series([value], dtype=dtype, strict=False)
    assert_series_equal(result, expected)


# https://github.com/pola-rs/polars/issues/15518
def test_series_init_np_temporal_with_nat_15518() -> None:
    arr = np.array(["2020-01-01", "2020-01-02", "2020-01-03"], "datetime64[D]")
    arr[1] = np.datetime64("NaT", "D")

    result = pl.Series(arr)

    expected = pl.Series([date(2020, 1, 1), None, date(2020, 1, 3)])
    assert_series_equal(result, expected)


def test_series_init_pandas_timestamp_18127() -> None:
    result = pl.Series([pd.Timestamp("2000-01-01T00:00:00.123456789", tz="UTC")])
    assert result.dtype == pl.Datetime("ns", "UTC")


@pytest.mark.parametrize(
    ("tz", "expected_tz"),
    [
        ("Asia/Tokyo", "Asia/Tokyo"),
        ("UTC", "UTC"),
        (pytz.timezone("Europe/London"), "Europe/London"),
        (pytz.UTC, "UTC"),
        (timezone(timedelta(hours=2)), "UTC"),  # (fixed offsets become UTC)
    ],
)
def test_series_init_pandas_timestamp_time_zones(tz: Any, expected_tz: str) -> None:
    value = pd.Timestamp("2021-06-15 12:30:00.000000001", tz=tz)
    result = pl.Series([value])
    assert result.dtype == pl.Datetime("ns", expected_tz)
    assert result.dt.timestamp("ns").item() == value.value


NEW_YORK = pytz.timezone("America/New_York")


@pytest.mark.parametrize(
    "value",
    [
        NEW_YORK.localize(datetime(2020, 6, 1, 12, 30, 15, 123456)),
        NEW_YORK.localize(datetime(2020, 11, 1, 1, 30), is_dst=True),
        NEW_YORK.localize(datetime(2020, 11, 1, 1, 30), is_dst=False),
        datetime(2020, 1, 1, tzinfo=NEW_YORK),  # (not localized: LMT offset)
        pytz.timezone("Europe/Amsterdam").localize(datetime(2150, 7, 1, 8)),
        pytz.timezone("Etc/GMT+5").localize(datetime(2020, 1, 1)),
        datetime(2020, 1, 1, tzinfo=pytz.UTC),
    ],
)
def test_series_init_pytz_datetime(value: datetime) -> None:
    result = pl.Series([value])
    assert result.dtype == pl.Datetime("us", str(value.tzinfo))

    epoch = datetime(1970, 1, 1, tzinfo=pytz.UTC)
    ts = result.dt.timestamp("us").item()
    assert ts == (value - epoch) // timedelta(microseconds=1)


def test_series_init_pandas_timestamp_unsupported_time_zone() -> None:
    # zones that are neither named nor fixed-offset are not silently made UTC
    value = pd.Timestamp("2021-06-15 12:30", tz=dateutil.tz.gettz("Europe/Paris"))
    with pytest.raises(TypeError, match="is not a fixed offset timezone"):
        pl.Series([value])


def test_series_init_pandas_nat() -> None:
    ts, td = pd.Timestamp("2020-01-01 00:00:00.000000001"), pd.Timedelta(1, unit="ns")
    assert pl.Series([ts, pd.NaT]).dt.timestamp("ns").to_list() == [ts.value, None]
    assert pl.Series([td, pd.NaT]).dt.total_nanoseconds().to_list() == [1, None]
    assert pl.DataFrame({"x": [pd.NaT]}).to_series().to_list() == [None]


@requires_pandas_as_unit
@pytest.mark.parametrize("unit", ["ns", "us", "ms", "s"])
def test_series_init_pandas_units_28869(unit: Literal["ns", "us", "ms", "s"]) -> None:
    # pandas values keep their own time unit (with seconds as milliseconds)
    time_unit = "ms" if unit == "s" else unit
    for value, dtype in (
        (
            pd.Timestamp("1969-12-31 23:59:59.123456789", tz="UTC"),
            pl.Datetime(time_unit, "UTC"),
        ),
        (pd.Timedelta(-123_456_789, unit="ns"), pl.Duration(time_unit)),
    ):
        value = value.as_unit(unit)
        result = pl.Series([None, value])
        assert result.dtype == dtype
        assert result.dt.cast_time_unit("ns").cast(pl.Int64).to_list() == [
            None,
            value.value,
        ]


@requires_pandas_as_unit
def test_series_init_pandas_outside_nanosecond_range() -> None:
    # values that cannot be expressed in nanoseconds still convert
    ts = pd.Timestamp("3000-01-01 00:00:00.5", tz="Asia/Tokyo").as_unit("us")
    td = pd.Timedelta(np.timedelta64(300 * 365, "D")).as_unit("ms")
    assert pl.Series([ts]).dt.timestamp("us").item() == ts.asm8.view("i8")
    assert pl.Series([td]).to_list() == [timedelta(days=300 * 365)]

    # ...unless they cannot be expressed as milliseconds either
    with pytest.raises(
        InvalidOperationError,
        match=r"conversion from `Timestamp\[s\]` to `datetime\[ms\]` failed",
    ):
        pl.Series([pd.Timestamp(np.datetime64(2**63 - 1, "s"))])


@requires_pandas_as_unit
@pytest.mark.parametrize("time_unit", [None, "ns", "us", "ms"])
@pytest.mark.parametrize(
    ("value", "other", "dtype"),
    [
        (
            pd.Timestamp("1969-12-31 23:59:59.123456789"),
            datetime(2020, 1, 1),
            pl.Datetime,
        ),
        (
            pd.Timedelta(-123_456_789, unit="ns"),
            timedelta(days=1, microseconds=1),
            pl.Duration,
        ),
    ],
)
def test_series_init_mixed_units_28869(
    value: Any, other: Any, dtype: Any, time_unit: Literal["ns", "us", "ms"] | None
) -> None:
    values = [value, value.as_unit("us"), value.as_unit("ms"), other]
    divisor = {None: 1, "ns": 1, "us": 1_000, "ms": 1_000_000}[time_unit]

    # (datetimes round down to coarser units, while durations round towards zero)
    def rescaled(v: Any) -> int:
        ns = (pd.Timestamp if dtype == pl.Datetime else pd.Timedelta)(v).value
        return ns // divisor if dtype == pl.Datetime or ns >= 0 else -(-ns // divisor)

    # the finest (or explicitly given) unit is used, independent of value order
    expected_dtype = dtype(time_unit or "ns")
    for strict in (True, False):
        for vals in (values, values[::-1]):
            dtype_arg = None if time_unit is None else expected_dtype
            result = pl.Series(vals, dtype=dtype_arg, strict=strict)
            assert result.dtype == expected_dtype
            assert result.cast(pl.Int64).to_list() == [rescaled(v) for v in vals]

    # nested and row inference also use the finest unit
    if time_unit is None:
        for result in (
            pl.Series([values]).explode(),
            pl.Series([[v] for v in values]).explode(),
            pl.Series([{"t": v} for v in values]).struct.field("t"),
            pl.from_dicts([{"t": v} for v in values])["t"],
            pl.DataFrame({"i": range(4)})
            .map_rows(lambda r: (values[r[0]],))
            .to_series(),
        ):
            assert result.dtype == expected_dtype
            assert result.cast(pl.Int64).to_list() == [rescaled(v) for v in values]


@pytest.mark.parametrize(
    ("values", "dtype"),
    [
        (
            [pd.Timestamp("2020-01-01 00:00:00.000000001"), datetime(3000, 1, 1)],
            pl.Datetime,
        ),
        ([pd.Timedelta(1, unit="ns"), timedelta(days=300 * 365)], pl.Duration),
    ],
)
def test_series_init_mixed_units_out_of_range_28869(
    values: list[Any], dtype: Any
) -> None:
    # the second value cannot be represented with "ns" precision
    name = dtype.__name__.lower()
    msg = rf"conversion from `{name}\[μs\]` to `{name}\[ns\]` failed"
    for vals in (values, [values], [[v] for v in values]):
        with pytest.raises(InvalidOperationError, match=msg):
            pl.Series(vals)
    with pytest.raises(InvalidOperationError, match=msg):
        pl.Series(values, dtype=dtype("ns"))

    # (non-strict construction makes it null)
    for result in (
        pl.Series(values, strict=False),
        pl.Series(values, dtype=dtype("ns"), strict=False),
    ):
        assert result.cast(pl.Int64).to_list() == [values[0].value, None]

    # an explicit (coarser) time unit is used directly
    for strict in (True, False):
        result = pl.Series(values, dtype=dtype("us"), strict=strict)
        assert result.dtype == dtype("us")
        assert result.to_list()[1] == values[1]


def test_series_init_np_2d_zero_zero_shape() -> None:
    arr = np.array([]).reshape(0, 0)
    assert_series_equal(
        pl.Series("a", arr),
        pl.Series("a", [], pl.Array(pl.Float64, 0)),
    )


def test_series_init_np_2d_empty() -> None:
    arr = np.array([]).reshape(0, 2)
    assert_series_equal(
        pl.Series("a", arr),
        pl.Series("a", [], pl.Array(pl.Float64, 2)),
    )


def test_list_null_constructor_schema() -> None:
    expected = pl.List(pl.Null)
    assert pl.Series([[]]).dtype == expected
    assert pl.Series([[]], dtype=pl.List).dtype == expected


@pytest.mark.parametrize(
    ("data", "dtype", "expected_values"),
    [
        (
            ["2024-01-01", "2025-10-07"],
            pl.Date,
            [date(2024, 1, 1), date(2025, 10, 7)],
        ),
        (
            ["12:00:00", "13:30:00"],
            pl.Time,
            [time(12, 0), time(13, 30)],
        ),
        (
            ["2024-01-01 23:59:59", "2024-01-02T13:30:00.123456"],
            pl.Datetime,
            [datetime(2024, 1, 1, 23, 59, 59), datetime(2024, 1, 2, 13, 30, 0, 123456)],
        ),
    ],
)
def test_temporal_dtype_string_values(
    data: list[str], dtype: PolarsDataType, expected_values: list[Any]
) -> None:
    for tp in (dtype, dtype()):  # type: ignore[operator]
        s = pl.Series(name="srs", values=data, dtype=tp)
        assert s.to_list() == expected_values
        assert s.dtype == dtype


def test_invalid() -> None:
    with pytest.raises(TypeError, match="constructor"):
        # Note: the `type: ignore` is correct here, as `0` is
        # an invalid type. If you make a PR which removes this
        # `type: ignore`, you may have inadvertently introduced
        # an `Any` type in the `Series.__init__` signature.
        pl.Series(1)  # type: ignore[arg-type]
