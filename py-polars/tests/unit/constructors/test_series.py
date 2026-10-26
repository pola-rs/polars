from __future__ import annotations

import io
import zoneinfo
from datetime import UTC, date, datetime, time, timedelta, timezone, tzinfo
from importlib import resources
from itertools import permutations
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import dateutil.tz
import numpy as np
import pandas as pd
import pytest

import polars as pl
from polars.exceptions import InvalidOperationError
from polars.testing.asserts.frame import assert_frame_equal
from polars.testing.asserts.series import assert_series_equal

if TYPE_CHECKING:
    from collections.abc import Callable

    from polars._typing import PolarsDataType, TimeUnit


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
        (timezone(timedelta(hours=2)), "UTC"),  # (fixed offsets become UTC)
    ],
)
def test_series_init_pandas_timestamp_time_zones(tz: Any, expected_tz: str) -> None:
    value = pd.Timestamp("2021-06-15 12:30:00.000000001", tz=tz)
    result = pl.Series([value])
    assert result.dtype == pl.Datetime("ns", expected_tz)
    assert result.dt.timestamp("ns").item() == value.value


@pytest.mark.parametrize("zone", ["Europe/London", "UTC"])
def test_series_init_pandas_timestamp_pytz_time_zones(zone: str) -> None:
    # (`pytz` is not installed in every test environment)
    pytz = pytest.importorskip("pytz")
    test_series_init_pandas_timestamp_time_zones(pytz.timezone(zone), zone)


@pytest.mark.parametrize(
    ("zone", "value", "is_dst"),
    [
        ("America/New_York", datetime(2020, 6, 1, 12, 30, 15, 123456), False),
        ("America/New_York", datetime(2020, 11, 1, 1, 30), True),
        ("America/New_York", datetime(2020, 11, 1, 1, 30), False),
        ("America/New_York", datetime(2020, 1, 1), None),  # (not localized: LMT)
        ("Europe/Amsterdam", datetime(2150, 7, 1, 8), False),
        ("Etc/GMT+5", datetime(2020, 1, 1), False),
        ("UTC", datetime(2020, 1, 1), None),
    ],
)
def test_series_init_pytz_datetime(
    zone: str, value: datetime, is_dst: bool | None
) -> None:
    # (`pytz` is not installed in every test environment)
    pytz = pytest.importorskip("pytz")
    tz = pytz.timezone(zone)
    # (with `is_dst=None` the zone is attached as-is, rather than localized)
    value = (
        value.replace(tzinfo=tz)
        if is_dst is None
        else tz.localize(value, is_dst=is_dst)
    )
    result = pl.Series([value])
    assert result.dtype == pl.Datetime("us", str(value.tzinfo))

    epoch = datetime(1970, 1, 1, tzinfo=pytz.UTC)
    ts = result.dt.timestamp("us").item()
    assert ts == (value - epoch) // timedelta(microseconds=1)


def _zone_file(key: str) -> io.BytesIO:
    # (as `zoneinfo` resolves keys: the system time zone database, else `tzdata`)
    for root in zoneinfo.TZPATH:
        if (path := Path(root, key)).is_file():
            return io.BytesIO(path.read_bytes())
    return io.BytesIO(resources.files("tzdata.zoneinfo").joinpath(key).read_bytes())


@pytest.mark.parametrize(
    "tz_from_file", [zoneinfo.ZoneInfo.from_file, dateutil.tz.tzfile]
)
@pytest.mark.parametrize("as_pandas", [False, True])
def test_series_init_unnamed_time_zone(
    tz_from_file: Callable[[io.BytesIO], tzinfo], as_pandas: bool
) -> None:
    # zones loaded from a file are neither named (their `key` is None, or absent in
    # older dateutil versions) nor fixed-offset, so must not be silently made UTC
    tz = tz_from_file(_zone_file("Europe/Paris"))
    value = datetime(2021, 6, 15, 12, 30, tzinfo=tz)
    with pytest.raises(TypeError, match="is not a fixed offset timezone"):
        pl.Series([pd.Timestamp(value) if as_pandas else value])


@pytest.mark.skipif(
    getattr(dateutil.tz.gettz("Europe/Paris"), "key", None) is None,
    reason="dateutil version does not name its time zones",
)
@pytest.mark.parametrize("as_pandas", [False, True])
def test_series_init_dateutil_named_time_zone(as_pandas: bool) -> None:
    value = datetime(2021, 6, 15, 12, 30, tzinfo=dateutil.tz.gettz("Europe/Paris"))
    s = pl.Series([pd.Timestamp(value) if as_pandas else value])
    assert s.dtype.time_zone == "Europe/Paris"  # type: ignore[attr-defined]
    assert s.dt.convert_time_zone("UTC").item() == datetime(
        2021, 6, 15, 10, 30, tzinfo=UTC
    )


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
    # (strict inference restarts on finding a finer unit: once, twice, midway or last)
    expected_dtype = dtype(time_unit or "ns")
    for strict in (True, False):
        for vals in map(list, permutations(values)):
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

    # (reversed, it only overflows once the finer unit restarts strict inference)
    for ordered in (values, values[::-1]):
        for vals in (ordered, [ordered], [[v] for v in ordered]):
            with pytest.raises(InvalidOperationError, match=msg):
                pl.Series(vals)

        with pytest.raises(InvalidOperationError, match=msg):
            pl.Series(ordered, dtype=dtype("ns"))

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


def test_series_init_pandas_timestamp_out_of_datetime_range() -> None:
    result = pl.Series([pd.Timestamp(np.datetime64("10000-01-01", "s"))])
    assert result.dtype == pl.Datetime("ms")
    assert result.dt.year().item() == 10000
    assert result.dt.epoch("s").item() == 253402300800


def test_series_init_pandas_timestamp_out_of_datetime_range_time_zone() -> None:
    ts = pd.Timestamp(np.datetime64("10000-01-01", "s"))
    try:
        ts = ts.tz_localize("America/New_York")
    except (NotImplementedError, ValueError):
        pytest.skip("this pandas version cannot localize out-of-range Timestamps")
    result = pl.Series([ts])
    assert result.dtype == pl.Datetime("ms", "America/New_York")
    assert result.dt.epoch("s").item() == 253402300800 + 5 * 3600


def test_series_init_pandas_timestamp_time_zone() -> None:
    tz = zoneinfo.ZoneInfo("Europe/Amsterdam")
    result = pl.Series([pd.Timestamp("2200-07-01 12:00", tz=tz)])
    assert result.dtype == pl.Datetime("us", "Europe/Amsterdam")
    assert result.item() == datetime(2200, 7, 1, 12, tzinfo=tz)


def test_series_init_datetime_subclass_from_pandas_named_module() -> None:
    class Datetime(datetime):
        @property
        def asm8(self) -> np.datetime64:
            return np.datetime64("2001-05-04", "s")

    Datetime.__module__ = "pandas_helpers"
    assert pl.Series([Datetime(2025, 1, 2)]).item() == datetime(2025, 1, 2)


def test_series_init_pandas_timedelta_non_nanosecond_unit() -> None:
    result = pl.Series([pd.Timedelta(np.timedelta64(1, "s"))])
    assert result.dtype == pl.Duration("ms")
    assert result.item() == timedelta(seconds=1)


def test_series_init_pandas_timedelta_out_of_microsecond_range() -> None:
    result = pl.Series([pd.Timedelta(np.timedelta64(10**9, "D"))])
    assert result.dtype == pl.Duration("ms")
    assert result.cast(pl.Int64).item() == 86_400_000_000_000_000


def test_series_init_pandas_timedelta_nanoseconds() -> None:
    result = pl.Series([pd.Timedelta(-1, "ns"), pd.Timedelta(1_999, "ns")])
    assert result.dtype == pl.Duration("ns")
    assert result.cast(pl.Int64).to_list() == [-1, 1_999]


def test_series_init_datetime_subclass_with_asm8() -> None:
    class Datetime(datetime):
        @property
        def asm8(self) -> np.timedelta64:
            return np.timedelta64(1, "s")

    assert pl.Series([Datetime(2025, 1, 2)]).item() == datetime(2025, 1, 2)


def test_series_init_pandas_timestamp_nanoseconds_before_epoch() -> None:
    result = pl.Series([pd.Timestamp("1969-12-31T23:59:59.999999999")])
    assert result.item() == datetime(1969, 12, 31, 23, 59, 59, 999999)


def test_series_init_pandas_nat() -> None:
    with pytest.raises(TypeError, match="cannot convert pandas NaT"):
        pl.Series([type(pd.NaT)()])
    with pytest.raises(TypeError, match="cannot convert pandas NaT"):
        pl.Series([pd.NaT])


def test_series_init_pandas_timestamp_subclass() -> None:
    class Timestamp(pd.Timestamp):
        pass

    ts = pd.Timestamp(np.datetime64("10000-01-01", "s"))
    ts.__class__ = Timestamp
    assert pl.Series([ts]).dt.year().item() == 10000


class _VariableOffset(tzinfo):
    def utcoffset(self, dt: datetime | None) -> timedelta | None:
        if dt is None:
            return None
        return timedelta(hours=2 if 4 <= dt.month <= 9 else 1)

    def dst(self, dt: datetime | None) -> timedelta | None:
        return None

    def tzname(self, dt: datetime | None) -> str | None:
        return None


def test_series_init_pandas_timestamp_variable_offset_time_zone() -> None:
    dt = datetime(2020, 7, 1, 12, tzinfo=_VariableOffset())
    with pytest.raises(TypeError):
        pl.Series([dt])
    with pytest.raises(TypeError):
        pl.Series([pd.Timestamp(dt)])


def test_series_init_pandas_timestamp_fixed_offset_time_zone() -> None:
    dt = datetime(2020, 7, 1, 12, tzinfo=timezone(timedelta(hours=2)))
    assert_series_equal(pl.Series([pd.Timestamp(dt)]), pl.Series([dt]))


def test_series_init_uint128_index_error_propagates() -> None:
    class Index:
        def __index__(self) -> int:
            msg = "bad index"
            raise ValueError(msg)

        def __float__(self) -> float:
            return 7.0

    with pytest.raises(ValueError, match="bad index"):
        pl.Series([1, Index()], dtype=pl.UInt128, strict=False)


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


@pytest.mark.parametrize("time_unit", ["ms", "us", "ns"])
def test_series_from_numpy_datetime64_scalars_18158(time_unit: TimeUnit) -> None:
    values = [
        np.datetime64("2022-01-01T12:34:56.789", time_unit),
        None,
        np.datetime64("NaT", time_unit),
    ]
    expected = pl.Series(
        [datetime(2022, 1, 1, 12, 34, 56, 789_000), None, None],
        dtype=pl.Datetime(time_unit),
    )
    assert_series_equal(pl.Series(values), expected)


@pytest.mark.parametrize("time_unit", ["ms", "us", "ns"])
def test_series_from_numpy_timedelta64_scalars_16029(time_unit: TimeUnit) -> None:
    s = pl.Series(
        [np.timedelta64(1500, time_unit), None, np.timedelta64("NaT", time_unit)]
    )
    expected = pl.Series([1500, None, None]).cast(pl.Duration(time_unit))
    assert_series_equal(s, expected)


def test_series_from_numpy_temporal_scalars_other_units() -> None:
    # 'D' becomes Date, 's' becomes milliseconds.
    s = pl.Series([np.datetime64("2022-01-01", "D"), np.datetime64("NaT", "D")])
    assert_series_equal(s, pl.Series([date(2022, 1, 1), None], dtype=pl.Date))

    s = pl.Series([np.datetime64("2022-01-01T00:00:01", "s")])
    expected = pl.Series([datetime(2022, 1, 1, 0, 0, 1)], dtype=pl.Datetime("ms"))
    assert_series_equal(s, expected)

    s = pl.Series([np.timedelta64(3, "s")])
    assert_series_equal(s, pl.Series([3_000]).cast(pl.Duration("ms")))

    # The unit count is applied: 5 * 10ms.
    s = pl.Series([np.datetime64(5, ("ms", 10))])
    assert_series_equal(s, pl.Series([50]).cast(pl.Datetime("ms")))

    with pytest.raises(ValueError, match="unit 'M'"):
        pl.Series([np.datetime64("2022-01", "M")])


@pytest.mark.parametrize(
    ("value", "msg"),
    [
        (np.datetime64(2**62, "s"), r"`datetime64\[s\]` to `datetime\[ms\]`"),
        (np.datetime64(2**40, "D"), r"`datetime64\[D\]` to `date`"),
        (np.datetime64(2**62, ("ms", 10)), r"`datetime64\[10ms\]` to `datetime\[ms\]`"),
        (np.timedelta64(2**62, "s"), r"`timedelta64\[s\]` to `duration\[ms\]`"),
    ],
)
def test_series_from_numpy_temporal_scalars_out_of_range(
    value: np.datetime64 | np.timedelta64, msg: str
) -> None:
    # (consistent with pandas `Timestamp`/`Timedelta` values that are out of range)
    with pytest.raises(InvalidOperationError, match=rf"conversion from {msg} failed"):
        pl.Series([value])


# Newer NumPy versions deprecate values without a unit.
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_series_from_numpy_temporal_scalars_generic_unit() -> None:
    s = pl.Series([None, np.datetime64("NaT")])
    assert_series_equal(s, pl.Series([None, None], dtype=pl.Null))

    with pytest.raises(ValueError, match="unit 'generic'"):
        pl.Series([np.timedelta64(1)])


def test_series_from_numpy_temporal_scalars_14380() -> None:
    arr = np.array(["2020-08-21"], dtype="datetime64")
    s = pl.Series("date", list(arr))
    assert_series_equal(s, pl.Series("date", [date(2020, 8, 21)], dtype=pl.Date))

    expected = pl.Series("date", [datetime(2020, 8, 21)], dtype=pl.Datetime("us"))
    assert_series_equal(pl.Series("date", list(arr), dtype=pl.Datetime), expected)
    assert_series_equal(pl.Series("date", arr, dtype=pl.Datetime), expected)


def test_series_from_numpy_temporal_scalars_mixed_units() -> None:
    values = [
        np.datetime64("2022-01-01T00:00:00.001", "ms"),
        np.datetime64("2022-01-01T00:00:00.000000002", "ns"),
    ]
    # the finest unit is used (without loss of precision), independent of value order
    ns = [1_640_995_200_001_000_000, 1_640_995_200_000_000_002]
    for strict in (True, False):
        for vals, expected in ((values, ns), (values[::-1], ns[::-1])):
            assert_series_equal(
                pl.Series(vals, strict=strict),
                pl.Series(expected).cast(pl.Datetime("ns")),
            )


@pytest.mark.parametrize(
    ("dtype", "expected_dtype", "expected"),
    [
        (pl.Datetime("ms"), pl.Datetime("ms"), 1_640_995_200_123),
        (pl.Datetime("us", "UTC"), pl.Datetime("us", "UTC"), 1_640_995_200_123_456),
        (pl.Datetime, pl.Datetime("ns"), 1_640_995_200_123_456_789),
        (pl.Date, pl.Date, 18_993),
        (pl.Int64, pl.Int64, 1_640_995_200_123_456_789),
    ],
)
def test_series_from_numpy_datetime64_scalars_with_dtype(
    dtype: PolarsDataType, expected_dtype: PolarsDataType, expected: int
) -> None:
    value = np.datetime64("2022-01-01T00:00:00.123456789", "ns")
    s = pl.Series([value, None], dtype=dtype)
    assert_series_equal(s, pl.Series([expected, None]).cast(expected_dtype))


def test_frame_from_numpy_temporal_scalars_16029() -> None:
    td = np.timedelta64(1, "ns")
    expected = pl.DataFrame({"column_0": pl.Series([1]).cast(pl.Duration("ns"))})
    assert_frame_equal(pl.DataFrame([[td]]), expected)
    assert_frame_equal(pl.DataFrame({"column_0": [td]}), expected)

    row = (np.timedelta64(5, "ms"), np.datetime64("2022-01-01T00:00:00.000000001"))
    expected = pl.DataFrame(
        {
            "a": pl.Series([5]).cast(pl.Duration("ms")),
            "b": pl.Series([1_640_995_200_000_000_001]).cast(pl.Datetime("ns")),
        }
    )
    assert_frame_equal(pl.DataFrame([{"a": row[0], "b": row[1]}]), expected)
    assert_frame_equal(pl.DataFrame([row], schema=["a", "b"], orient="row"), expected)

    s = pl.Series([{"a": np.datetime64("2022-01-01T00:00:00.000000001")}])
    expected_field = pl.Series("a", [1_640_995_200_000_000_001]).cast(pl.Datetime("ns"))
    assert_series_equal(s, expected_field.to_frame().to_struct(""))

    s = pl.Series([[np.datetime64("2022-01-01", "us")]])
    expected_list = pl.Series(
        [[datetime(2022, 1, 1)]], dtype=pl.List(pl.Datetime("us"))
    )
    assert_series_equal(s, expected_list)
