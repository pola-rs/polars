from __future__ import annotations

from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import pytest

import polars as pl
from polars._utils.constants import MS_PER_SECOND, NS_PER_SECOND, US_PER_SECOND
from polars.exceptions import ComputeError, InvalidOperationError
from polars.testing import assert_frame_equal
from polars.testing.asserts.series import assert_series_equal

if TYPE_CHECKING:
    from polars._typing import PolarsDataType


def test_string_date() -> None:
    df = pl.DataFrame({"x1": ["2021-01-01"]}).with_columns(
        **{"x1-date": pl.col("x1").cast(pl.Date)}
    )
    expected = pl.DataFrame({"x1-date": [date(2021, 1, 1)]})
    out = df.select(pl.col("x1-date"))
    assert_frame_equal(expected, out)


def test_invalid_string_date() -> None:
    df = pl.DataFrame({"x1": ["2021-01-aa"]})

    with pytest.raises(InvalidOperationError):
        df.with_columns(**{"x1-date": pl.col("x1").cast(pl.Date)})


def test_string_datetime() -> None:
    df = pl.DataFrame(
        {"x1": ["2021-12-19T00:39:57", "2022-12-19T16:39:57"]}
    ).with_columns(
        **{
            "x1-datetime-ns": pl.col("x1").cast(pl.Datetime(time_unit="ns")),
            "x1-datetime-ms": pl.col("x1").cast(pl.Datetime(time_unit="ms")),
            "x1-datetime-us": pl.col("x1").cast(pl.Datetime(time_unit="us")),
        }
    )
    first_row = datetime(year=2021, month=12, day=19, hour=00, minute=39, second=57)
    second_row = datetime(year=2022, month=12, day=19, hour=16, minute=39, second=57)
    expected = pl.DataFrame(
        {
            "x1-datetime-ns": [first_row, second_row],
            "x1-datetime-ms": [first_row, second_row],
            "x1-datetime-us": [first_row, second_row],
        }
    ).select(
        pl.col("x1-datetime-ns").dt.cast_time_unit("ns"),
        pl.col("x1-datetime-ms").dt.cast_time_unit("ms"),
        pl.col("x1-datetime-us").dt.cast_time_unit("us"),
    )

    out = df.select(
        pl.col("x1-datetime-ns"), pl.col("x1-datetime-ms"), pl.col("x1-datetime-us")
    )
    assert_frame_equal(expected, out)


def test_invalid_string_datetime() -> None:
    df = pl.DataFrame({"x1": ["2021-12-19 00:39:57", "2022-12-19 16:39:57"]})
    with pytest.raises(InvalidOperationError):
        df.with_columns(
            **{"x1-datetime-ns": pl.col("x1").cast(pl.Datetime(time_unit="ns"))}
        )


def test_string_datetime_timezone() -> None:
    ccs_tz = "America/Caracas"
    stg_tz = "America/Santiago"
    utc_tz = "UTC"
    df = pl.DataFrame(
        {"x1": ["1996-12-19T16:39:57 +00:00", "2022-12-19T00:39:57 +00:00"]}
    ).with_columns(
        **{
            "x1-datetime-ns": pl.col("x1").cast(
                pl.Datetime(time_unit="ns", time_zone=ccs_tz)
            ),
            "x1-datetime-ms": pl.col("x1").cast(
                pl.Datetime(time_unit="ms", time_zone=stg_tz)
            ),
            "x1-datetime-us": pl.col("x1").cast(
                pl.Datetime(time_unit="us", time_zone=utc_tz)
            ),
        }
    )

    expected = pl.DataFrame(
        {
            "x1-datetime-ns": [
                datetime(year=1996, month=12, day=19, hour=12, minute=39, second=57),
                datetime(year=2022, month=12, day=18, hour=20, minute=39, second=57),
            ],
            "x1-datetime-ms": [
                datetime(year=1996, month=12, day=19, hour=13, minute=39, second=57),
                datetime(year=2022, month=12, day=18, hour=21, minute=39, second=57),
            ],
            "x1-datetime-us": [
                datetime(year=1996, month=12, day=19, hour=16, minute=39, second=57),
                datetime(year=2022, month=12, day=19, hour=00, minute=39, second=57),
            ],
        }
    ).select(
        pl.col("x1-datetime-ns").dt.cast_time_unit("ns").dt.replace_time_zone(ccs_tz),
        pl.col("x1-datetime-ms").dt.cast_time_unit("ms").dt.replace_time_zone(stg_tz),
        pl.col("x1-datetime-us").dt.cast_time_unit("us").dt.replace_time_zone(utc_tz),
    )

    out = df.select(
        pl.col("x1-datetime-ns"), pl.col("x1-datetime-ms"), pl.col("x1-datetime-us")
    )

    assert_frame_equal(expected, out)


@pytest.mark.parametrize(("dtype"), [pl.Int8, pl.Int16, pl.Int32, pl.Int64])
def test_leading_plus_zero_int(dtype: pl.DataType) -> None:
    s_int = pl.Series(
        [
            "-000000000000002",
            "-1",
            "-0",
            "0",
            "+0",
            "1",
            "+1",
            "0000000000000000000002",
            "+000000000000000000003",
        ]
    )
    assert_series_equal(
        s_int.cast(dtype), pl.Series([-2, -1, 0, 0, 0, 1, 1, 2, 3], dtype=dtype)
    )


@pytest.mark.parametrize(("dtype"), [pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64])
def test_leading_plus_zero_uint(dtype: pl.DataType) -> None:
    s_int = pl.Series(
        ["0", "+0", "1", "+1", "0000000000000000000002", "+000000000000000000003"]
    )
    assert_series_equal(s_int.cast(dtype), pl.Series([0, 0, 1, 1, 2, 3], dtype=dtype))


@pytest.mark.parametrize(("dtype"), [pl.Float32, pl.Float64])
def test_leading_plus_zero_float(dtype: pl.DataType) -> None:
    s_float = pl.Series(
        [
            "-000000000000002.0",
            "-1.0",
            "-.5",
            "-0.0",
            "0.",
            "+0",
            "+.5",
            "1",
            "+1",
            "0000000000000000000002",
            "+000000000000000000003",
        ]
    )
    assert_series_equal(
        s_float.cast(dtype),
        pl.Series(
            [-2.0, -1.0, -0.5, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 2.0, 3.0], dtype=dtype
        ),
    )


def _cast_series(
    val: int | datetime | date | time | timedelta,
    dtype_in: PolarsDataType,
    dtype_out: PolarsDataType,
    strict: bool,
) -> int | datetime | date | time | timedelta | None:
    return pl.Series("a", [val], dtype=dtype_in).cast(dtype_out, strict=strict).item()  # type: ignore[no-any-return]


def _cast_expr(
    val: int | datetime | date | time | timedelta,
    dtype_in: PolarsDataType,
    dtype_out: PolarsDataType,
    strict: bool,
) -> int | datetime | date | time | timedelta | None:
    return (  # type: ignore[no-any-return]
        pl.Series("a", [val], dtype=dtype_in)
        .to_frame()
        .select(pl.col("a").cast(dtype_out, strict=strict))
        .item()
    )


def _cast_lit(
    val: int | datetime | date | time | timedelta,
    dtype_in: PolarsDataType,
    dtype_out: PolarsDataType,
    strict: bool,
) -> int | datetime | date | time | timedelta | None:
    return pl.select(pl.lit(val, dtype=dtype_in).cast(dtype_out, strict=strict)).item()  # type: ignore[no-any-return]


@pytest.mark.parametrize(
    ("value", "from_dtype", "to_dtype", "should_succeed", "expected_value"),
    [
        (-1, pl.Int8, pl.UInt8, False, None),
        (-1, pl.Int16, pl.UInt16, False, None),
        (-1, pl.Int32, pl.UInt32, False, None),
        (-1, pl.Int64, pl.UInt64, False, None),
        (2**7, pl.UInt8, pl.Int8, False, None),
        (2**15, pl.UInt16, pl.Int16, False, None),
        (2**31, pl.UInt32, pl.Int32, False, None),
        (2**63, pl.UInt64, pl.Int64, False, None),
        (2**7 - 1, pl.UInt8, pl.Int8, True, 2**7 - 1),
        (2**15 - 1, pl.UInt16, pl.Int16, True, 2**15 - 1),
        (2**31 - 1, pl.UInt32, pl.Int32, True, 2**31 - 1),
        (2**63 - 1, pl.UInt64, pl.Int64, True, 2**63 - 1),
    ],
)
def test_strict_cast_int(
    value: int,
    from_dtype: PolarsDataType,
    to_dtype: PolarsDataType,
    should_succeed: bool,
    expected_value: Any,
) -> None:
    args = [value, from_dtype, to_dtype, True]
    if should_succeed:
        assert _cast_series(*args) == expected_value  # type: ignore[arg-type]
        assert _cast_expr(*args) == expected_value  # type: ignore[arg-type]
        assert _cast_lit(*args) == expected_value  # type: ignore[arg-type]
    else:
        with pytest.raises(InvalidOperationError):
            _cast_series(*args)  # type: ignore[arg-type]
        with pytest.raises(InvalidOperationError):
            _cast_expr(*args)  # type: ignore[arg-type]
        with pytest.raises(InvalidOperationError):
            _cast_lit(*args)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("value", "from_dtype", "to_dtype", "expected_value"),
    [
        (-1, pl.Int8, pl.UInt8, None),
        (-1, pl.Int16, pl.UInt16, None),
        (-1, pl.Int32, pl.UInt32, None),
        (-1, pl.Int64, pl.UInt64, None),
        (2**7, pl.UInt8, pl.Int8, None),
        (2**15, pl.UInt16, pl.Int16, None),
        (2**31, pl.UInt32, pl.Int32, None),
        (2**63, pl.UInt64, pl.Int64, None),
        (2**7 - 1, pl.UInt8, pl.Int8, 2**7 - 1),
        (2**15 - 1, pl.UInt16, pl.Int16, 2**15 - 1),
        (2**31 - 1, pl.UInt32, pl.Int32, 2**31 - 1),
        (2**63 - 1, pl.UInt64, pl.Int64, 2**63 - 1),
    ],
)
def test_cast_int(
    value: int,
    from_dtype: PolarsDataType,
    to_dtype: PolarsDataType,
    expected_value: Any,
) -> None:
    args = [value, from_dtype, to_dtype, False]
    assert _cast_series(*args) == expected_value  # type: ignore[arg-type]
    assert _cast_expr(*args) == expected_value  # type: ignore[arg-type]
    assert _cast_lit(*args) == expected_value  # type: ignore[arg-type]


def _cast_series_t(
    val: int | datetime | date | time | timedelta,
    dtype_in: PolarsDataType,
    dtype_out: PolarsDataType,
    strict: bool,
) -> pl.Series:
    return pl.Series("a", [val], dtype=dtype_in).cast(dtype_out, strict=strict)


def _cast_expr_t(
    val: int | datetime | date | time | timedelta,
    dtype_in: PolarsDataType,
    dtype_out: PolarsDataType,
    strict: bool,
) -> pl.Series:
    return (
        pl.Series("a", [val], dtype=dtype_in)
        .to_frame()
        .select(pl.col("a").cast(dtype_out, strict=strict))
        .to_series()
    )


def _cast_lit_t(
    val: int | datetime | date | time | timedelta,
    dtype_in: PolarsDataType,
    dtype_out: PolarsDataType,
    strict: bool,
) -> pl.Series:
    return pl.select(
        pl.lit(val, dtype=dtype_in).cast(dtype_out, strict=strict)
    ).to_series()


@pytest.mark.parametrize(
    (
        "value",
        "from_dtype",
        "to_dtype",
        "should_succeed",
        "expected_value",
    ),
    [
        # fmt: off
        # date to datetime
        (date(1970, 1, 1), pl.Date, pl.Datetime("ms"), True, datetime(1970, 1, 1)),
        (date(1970, 1, 1), pl.Date, pl.Datetime("us"), True, datetime(1970, 1, 1)),
        (date(1970, 1, 1), pl.Date, pl.Datetime("ns"), True, datetime(1970, 1, 1)),
        # datetime to date
        (datetime(1970, 1, 1), pl.Datetime("ms"), pl.Date, True, date(1970, 1, 1)),
        (datetime(1970, 1, 1), pl.Datetime("us"), pl.Date, True, date(1970, 1, 1)),
        (datetime(1970, 1, 1), pl.Datetime("ns"), pl.Date, True, date(1970, 1, 1)),
        # datetime to time
        (datetime(2000, 1, 1, 1, 0, 0), pl.Datetime("ms"), pl.Time, True, time(hour=1)),
        (datetime(2000, 1, 1, 1, 0, 0), pl.Datetime("us"), pl.Time, True, time(hour=1)),
        (datetime(2000, 1, 1, 1, 0, 0), pl.Datetime("ns"), pl.Time, True, time(hour=1)),
        # duration to int
        (timedelta(seconds=1), pl.Duration("ms"), pl.Int32, True, MS_PER_SECOND),
        (timedelta(seconds=1), pl.Duration("us"), pl.Int64, True, US_PER_SECOND),
        (timedelta(seconds=1), pl.Duration("ns"), pl.Int64, True, NS_PER_SECOND),
        # time to duration
        (time(hour=1), pl.Time, pl.Duration("ms"), True, timedelta(hours=1)),
        (time(hour=1), pl.Time, pl.Duration("us"), True, timedelta(hours=1)),
        (time(hour=1), pl.Time, pl.Duration("ns"), True, timedelta(hours=1)),
        # int to date
        (100, pl.UInt8, pl.Date, True, date(1970, 4, 11)),
        (100, pl.UInt16, pl.Date, True, date(1970, 4, 11)),
        (100, pl.UInt32, pl.Date, True, date(1970, 4, 11)),
        (100, pl.UInt64, pl.Date, True, date(1970, 4, 11)),
        (100, pl.Int8, pl.Date, True, date(1970, 4, 11)),
        (100, pl.Int16, pl.Date, True, date(1970, 4, 11)),
        (100, pl.Int32, pl.Date, True, date(1970, 4, 11)),
        (100, pl.Int64, pl.Date, True, date(1970, 4, 11)),
        # failures
        (2**63 - 1, pl.Int64, pl.Date, False, None),
        (-(2**62), pl.Int64, pl.Date, False, None),
        (date(1970, 5, 10), pl.Date, pl.Int8, False, None),
        (date(2149, 6, 7), pl.Date, pl.Int16, False, None),
        (datetime(9999, 12, 31), pl.Datetime, pl.Int8, False, None),
        (datetime(9999, 12, 31), pl.Datetime, pl.Int16, False, None),
        # fmt: on
    ],
)
def test_strict_cast_temporal(
    value: int,
    from_dtype: PolarsDataType,
    to_dtype: PolarsDataType,
    should_succeed: bool,
    expected_value: Any,
) -> None:
    args = [value, from_dtype, to_dtype, True]
    if should_succeed:
        out = _cast_series_t(*args)  # type: ignore[arg-type]
        assert out.item() == expected_value
        assert out.dtype == to_dtype
        out = _cast_expr_t(*args)  # type: ignore[arg-type]
        assert out.item() == expected_value
        assert out.dtype == to_dtype
        out = _cast_lit_t(*args)  # type: ignore[arg-type]
        assert out.item() == expected_value
        assert out.dtype == to_dtype
    else:
        with pytest.raises(InvalidOperationError):
            _cast_series_t(*args)  # type: ignore[arg-type]
        with pytest.raises(InvalidOperationError):
            _cast_expr_t(*args)  # type: ignore[arg-type]
        with pytest.raises(InvalidOperationError):
            _cast_lit_t(*args)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    (
        "value",
        "from_dtype",
        "to_dtype",
        "expected_value",
    ),
    [
        # fmt: off
        # date to datetime
        (date(1970, 1, 1), pl.Date, pl.Datetime("ms"), datetime(1970, 1, 1)),
        (date(1970, 1, 1), pl.Date, pl.Datetime("us"), datetime(1970, 1, 1)),
        (date(1970, 1, 1), pl.Date, pl.Datetime("ns"), datetime(1970, 1, 1)),
        # datetime to date
        (datetime(1970, 1, 1), pl.Datetime("ms"), pl.Date, date(1970, 1, 1)),
        (datetime(1970, 1, 1), pl.Datetime("us"), pl.Date, date(1970, 1, 1)),
        (datetime(1970, 1, 1), pl.Datetime("ns"), pl.Date, date(1970, 1, 1)),
        # datetime to time
        (datetime(2000, 1, 1, 1, 0, 0), pl.Datetime("ms"), pl.Time, time(hour=1)),
        (datetime(2000, 1, 1, 1, 0, 0), pl.Datetime("us"), pl.Time, time(hour=1)),
        (datetime(2000, 1, 1, 1, 0, 0), pl.Datetime("ns"), pl.Time, time(hour=1)),
        # duration to int
        (timedelta(seconds=1), pl.Duration("ms"), pl.Int32, MS_PER_SECOND),
        (timedelta(seconds=1), pl.Duration("us"), pl.Int64, US_PER_SECOND),
        (timedelta(seconds=1), pl.Duration("ns"), pl.Int64, NS_PER_SECOND),
        # time to duration
        (time(hour=1), pl.Time, pl.Duration("ms"), timedelta(hours=1)),
        (time(hour=1), pl.Time, pl.Duration("us"), timedelta(hours=1)),
        (time(hour=1), pl.Time, pl.Duration("ns"), timedelta(hours=1)),
        # int to date
        (100, pl.UInt8, pl.Date, date(1970, 4, 11)),
        (100, pl.UInt16, pl.Date, date(1970, 4, 11)),
        (100, pl.UInt32, pl.Date, date(1970, 4, 11)),
        (100, pl.UInt64, pl.Date, date(1970, 4, 11)),
        (100, pl.Int8, pl.Date, date(1970, 4, 11)),
        (100, pl.Int16, pl.Date, date(1970, 4, 11)),
        (100, pl.Int32, pl.Date, date(1970, 4, 11)),
        (100, pl.Int64, pl.Date, date(1970, 4, 11)),
        # failures
        (2**63 - 1, pl.Int64, pl.Date, None),
        (-(2**62), pl.Int64, pl.Date, None),
        (date(1970, 5, 10), pl.Date, pl.Int8, None),
        (date(2149, 6, 7), pl.Date, pl.Int16, None),
        (datetime(9999, 12, 31), pl.Datetime, pl.Int8, None),
        (datetime(9999, 12, 31), pl.Datetime, pl.Int16, None),
        # fmt: on
    ],
)
def test_cast_temporal(
    value: int,
    from_dtype: PolarsDataType,
    to_dtype: PolarsDataType,
    expected_value: Any,
) -> None:
    args = [value, from_dtype, to_dtype, False]
    out = _cast_series_t(*args)  # type: ignore[arg-type]
    if expected_value is None:
        assert out.item() is None
    else:
        assert out.item() == expected_value
        assert out.dtype == to_dtype

    out = _cast_expr_t(*args)  # type: ignore[arg-type]
    if expected_value is None:
        assert out.item() is None
    else:
        assert out.item() == expected_value
        assert out.dtype == to_dtype

    out = _cast_lit_t(*args)  # type: ignore[arg-type]
    if expected_value is None:
        assert out.item() is None
    else:
        assert out.item() == expected_value
        assert out.dtype == to_dtype


@pytest.mark.parametrize(
    (
        "value",
        "from_dtype",
        "to_dtype",
        "expected_value",
    ),
    [
        (str(2**7 - 1).encode(), pl.Binary, pl.Int8, 2**7 - 1),
        (str(2**15 - 1).encode(), pl.Binary, pl.Int16, 2**15 - 1),
        (str(2**31 - 1).encode(), pl.Binary, pl.Int32, 2**31 - 1),
        (str(2**63 - 1).encode(), pl.Binary, pl.Int64, 2**63 - 1),
        (b"1.0", pl.Binary, pl.Float32, 1.0),
        (b"1.0", pl.Binary, pl.Float64, 1.0),
        (str(2**7 - 1), pl.String, pl.Int8, 2**7 - 1),
        (str(2**15 - 1), pl.String, pl.Int16, 2**15 - 1),
        (str(2**31 - 1), pl.String, pl.Int32, 2**31 - 1),
        (str(2**63 - 1), pl.String, pl.Int64, 2**63 - 1),
        ("1.0", pl.String, pl.Float32, 1.0),
        ("1.0", pl.String, pl.Float64, 1.0),
        # overflow
        (str(2**7), pl.String, pl.Int8, None),
        (str(2**15), pl.String, pl.Int16, None),
        (str(2**31), pl.String, pl.Int32, None),
        (str(2**63), pl.String, pl.Int64, None),
        (str(2**7).encode(), pl.Binary, pl.Int8, None),
        (str(2**15).encode(), pl.Binary, pl.Int16, None),
        (str(2**31).encode(), pl.Binary, pl.Int32, None),
        (str(2**63).encode(), pl.Binary, pl.Int64, None),
    ],
)
def test_cast_string_and_binary(
    value: int,
    from_dtype: PolarsDataType,
    to_dtype: PolarsDataType,
    expected_value: Any,
) -> None:
    args = [value, from_dtype, to_dtype, False]
    out = _cast_series_t(*args)  # type: ignore[arg-type]
    if expected_value is None:
        assert out.item() is None
    else:
        assert out.item() == expected_value
        assert out.dtype == to_dtype

    out = _cast_expr_t(*args)  # type: ignore[arg-type]
    if expected_value is None:
        assert out.item() is None
    else:
        assert out.item() == expected_value
        assert out.dtype == to_dtype

    out = _cast_lit_t(*args)  # type: ignore[arg-type]
    if expected_value is None:
        assert out.item() is None
    else:
        assert out.item() == expected_value
        assert out.dtype == to_dtype


@pytest.mark.parametrize(
    (
        "value",
        "from_dtype",
        "to_dtype",
        "should_succeed",
        "expected_value",
    ),
    [
        (str(2**7 - 1).encode(), pl.Binary, pl.Int8, True, 2**7 - 1),
        (str(2**15 - 1).encode(), pl.Binary, pl.Int16, True, 2**15 - 1),
        (str(2**31 - 1).encode(), pl.Binary, pl.Int32, True, 2**31 - 1),
        (str(2**63 - 1).encode(), pl.Binary, pl.Int64, True, 2**63 - 1),
        (b"1.0", pl.Binary, pl.Float32, True, 1.0),
        (b"1.0", pl.Binary, pl.Float64, True, 1.0),
        (str(2**7 - 1), pl.String, pl.Int8, True, 2**7 - 1),
        (str(2**15 - 1), pl.String, pl.Int16, True, 2**15 - 1),
        (str(2**31 - 1), pl.String, pl.Int32, True, 2**31 - 1),
        (str(2**63 - 1), pl.String, pl.Int64, True, 2**63 - 1),
        ("1.0", pl.String, pl.Float32, True, 1.0),
        ("1.0", pl.String, pl.Float64, True, 1.0),
        # overflow
        (str(2**7), pl.String, pl.Int8, False, None),
        (str(2**15), pl.String, pl.Int16, False, None),
        (str(2**31), pl.String, pl.Int32, False, None),
        (str(2**63), pl.String, pl.Int64, False, None),
        (str(2**7).encode(), pl.Binary, pl.Int8, False, None),
        (str(2**15).encode(), pl.Binary, pl.Int16, False, None),
        (str(2**31).encode(), pl.Binary, pl.Int32, False, None),
        (str(2**63).encode(), pl.Binary, pl.Int64, False, None),
    ],
)
def test_strict_cast_string_and_binary(
    value: int,
    from_dtype: PolarsDataType,
    to_dtype: PolarsDataType,
    should_succeed: bool,
    expected_value: Any,
) -> None:
    args = [value, from_dtype, to_dtype, True]
    if should_succeed:
        out = _cast_series_t(*args)  # type: ignore[arg-type]
        assert out.item() == expected_value
        assert out.dtype == to_dtype
        out = _cast_expr_t(*args)  # type: ignore[arg-type]
        assert out.item() == expected_value
        assert out.dtype == to_dtype
        out = _cast_lit_t(*args)  # type: ignore[arg-type]
        assert out.item() == expected_value
        assert out.dtype == to_dtype
    else:
        with pytest.raises(InvalidOperationError):
            _cast_series_t(*args)  # type: ignore[arg-type]
        with pytest.raises(InvalidOperationError):
            _cast_expr_t(*args)  # type: ignore[arg-type]
        with pytest.raises(InvalidOperationError):
            _cast_lit_t(*args)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "dtype_in",
    [(pl.Categorical), (pl.Enum(["1"]))],
)
@pytest.mark.parametrize(
    "dtype_out",
    [
        (pl.UInt8),
        (pl.Int8),
        (pl.UInt16),
        (pl.Int16),
        (pl.UInt32),
        (pl.Int32),
        (pl.UInt64),
        (pl.Int64),
        (pl.Date),
        (pl.Datetime),
        (pl.Time),
        (pl.Duration),
        (pl.String),
        (pl.Categorical),
        (pl.Enum(["1", "2"])),
    ],
)
def test_cast_categorical_name_retention(
    dtype_in: PolarsDataType, dtype_out: PolarsDataType
) -> None:
    assert pl.Series("a", ["1"], dtype=dtype_in).cast(dtype_out).name == "a"


def test_cast_date_to_time() -> None:
    s = pl.Series([date(1970, 1, 1), date(2000, 12, 31)])
    msg = "casting from Date to Time not supported"
    with pytest.raises(InvalidOperationError, match=msg):
        s.cast(pl.Time)


def test_cast_time_to_date() -> None:
    s = pl.Series([time(0, 0), time(20, 00)])
    msg = "casting from Time to Date not supported"
    with pytest.raises(InvalidOperationError, match=msg):
        s.cast(pl.Date)


def test_cast_decimal_to_boolean() -> None:
    s = pl.Series("s", [Decimal("0.0"), Decimal("1.5"), Decimal("-1.5")])
    assert_series_equal(s.cast(pl.Boolean), pl.Series("s", [False, True, True]))

    df = s.to_frame()
    assert_frame_equal(
        df.select(pl.col("s").cast(pl.Boolean)),
        pl.DataFrame({"s": [False, True, True]}),
    )


def test_cast_array_to_different_width() -> None:
    s = pl.Series([[1, 2], [3, 4]], dtype=pl.Array(pl.Int8, 2))
    with pytest.raises(
        InvalidOperationError, match="cannot cast Array to a different width"
    ):
        s.cast(pl.Array(pl.Int16, 3))


def test_cast_decimal_to_decimal_high_precision() -> None:
    precision = 22
    values = [Decimal("9" * precision)]
    s = pl.Series(values, dtype=pl.Decimal(None, 0))

    target_dtype = pl.Decimal(precision, 0)
    result = s.cast(target_dtype)

    assert result.dtype == target_dtype
    assert result.to_list() == values


def test_err_on_time_datetime_cast() -> None:
    s = pl.Series([time(10, 0, 0), time(11, 30, 59)])
    with pytest.raises(
        InvalidOperationError,
        match="casting from Time to Datetime\\(Microseconds, None\\) not supported; consider using `dt.combine`",
    ):
        s.cast(pl.Datetime)


def test_err_on_invalid_time_zone_cast() -> None:
    s = pl.Series([datetime(2021, 1, 1)])
    with pytest.raises(ComputeError, match=r"unable to parse time zone: 'qwerty'"):
        s.cast(pl.Datetime("us", "qwerty"))


def test_invalid_inner_type_cast_list() -> None:
    s = pl.Series([[-1, 1]])
    with pytest.raises(
        InvalidOperationError,
        match=r"cannot cast List inner type: 'Int64' to Categorical",
    ):
        s.cast(pl.List(pl.Categorical))


def test_all_null_cast_5826() -> None:
    df = pl.DataFrame(data=[pl.Series("a", [None], dtype=pl.String)])
    out = df.with_columns(pl.col("a").cast(pl.Boolean))
    assert out.dtypes == [pl.Boolean]
    assert out.item() is None


@pytest.mark.parametrize(
    "dtype",
    [pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, pl.Int8, pl.Int16, pl.Int32, pl.Int64],
)
def test_bool_numeric_supertype(dtype: PolarsDataType) -> None:
    df = pl.DataFrame({"v": [1, 2, 3, 4, 5, 6]})
    result = df.select((pl.col("v") < 3).sum().cast(dtype) / pl.len())
    assert result.item() - 0.3333333 <= 0.00001
