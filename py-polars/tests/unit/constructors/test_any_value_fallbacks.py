# TODO: Replace direct calls to fallback constructors with calls to the Series
# constructor once the Python-side logic has been updated
from __future__ import annotations

from datetime import date, datetime, time, timedelta
from decimal import Decimal as D
from typing import TYPE_CHECKING, Any

import pytest

import polars as pl
from polars._utils.wrap import wrap_s
from polars.polars import PySeries

if TYPE_CHECKING:
    from polars._typing import PolarsDataType


@pytest.mark.parametrize(
    ("dtype", "values"),
    [
        (pl.Int64, [-1, 0, 100_000, None]),
        (pl.Float64, [-1.5, 0.0, 10.0, None]),
        (pl.Boolean, [True, False, None]),
        (pl.Binary, [b"123", b"xyz", None]),
        (pl.String, ["123", "xyz", None]),
        (pl.Date, [date(1970, 1, 1), date(2020, 12, 31), None]),
        (pl.Time, [time(0, 0), time(23, 59, 59), None]),
        (pl.Datetime, [datetime(1970, 1, 1), datetime(2020, 12, 31, 23, 59, 59), None]),
        (pl.Duration, [timedelta(hours=0), timedelta(seconds=100), None]),
        (pl.Categorical, ["a", "b", "a", None]),
        (pl.Enum(["a", "b"]), ["a", "b", "a", None]),
        (pl.Decimal(10, 3), [D("12.345"), D("0.789"), None]),
        (
            pl.Struct({"a": pl.Int8, "b": pl.String}),
            [{"a": 1, "b": "foo"}, {"a": -1, "b": "bar"}],
        ),
    ],
)
@pytest.mark.parametrize("strict", [True, False])
def test_fallback_with_dtype_strict(
    dtype: PolarsDataType, values: list[Any], strict: bool
) -> None:
    result = wrap_s(
        PySeries.new_from_any_values_and_dtype("", values, dtype, strict=strict)
    )
    assert result.to_list() == values


@pytest.mark.parametrize(
    ("dtype", "values"),
    [
        (pl.Int64, [1.0, 2.0]),
        (pl.Float64, [1, 2]),
        (pl.Boolean, [0, 1]),
        (pl.Binary, ["123", "xyz"]),
        (pl.String, [b"123", b"xyz"]),
        (pl.Date, [datetime(1970, 1, 1), datetime(2020, 12, 31, 23, 59, 59)]),
        (pl.Time, [datetime(1970, 1, 1), datetime(2020, 12, 31, 23, 59, 59)]),
        (pl.Datetime, [date(1970, 1, 1), date(2020, 12, 31)]),
        (pl.Datetime("ms"), [datetime(1970, 1, 1), datetime(2020, 12, 31, 23, 59, 59)]),
        (pl.Datetime("ns"), [datetime(1970, 1, 1), datetime(2020, 12, 31, 23, 59, 59)]),
        (pl.Duration, [0, 1200]),
        (pl.Duration("ms"), [timedelta(hours=0), timedelta(seconds=100)]),
        (pl.Duration("ns"), [timedelta(hours=0), timedelta(seconds=100)]),
        (pl.Categorical, [0, 1, 0]),
        (pl.Enum(["a", "b"]), [0, 1, 0]),
        (pl.Decimal(10, 3), [100, 200]),
        (pl.Decimal(5, 3), [D("1.2345")]),
        (
            pl.Struct({"a": pl.Int8, "b": pl.String}),
            [{"a": 1, "b": "foo"}, {"a": 2.0, "b": "bar"}],
        ),
    ],
)
def test_fallback_with_dtype_strict_failure(
    dtype: PolarsDataType, values: list[Any]
) -> None:
    with pytest.raises(TypeError, match="unexpected value"):
        PySeries.new_from_any_values_and_dtype("", values, dtype, strict=True)


@pytest.mark.parametrize(
    ("dtype", "values", "expected"),
    [
        (
            pl.Int64,
            [False, True, 0, -1, 0.0, 2.5, date(1970, 1, 2), "5", "xyz"],
            [0, 1, 0, -1, 0, 2, 1, 5, None],
        ),
        (
            pl.Float64,
            [False, True, 0, -1, 0.0, 2.5, date(1970, 1, 2), "5", "xyz"],
            [0.0, 1.0, 0.0, -1.0, 0.0, 2.5, 1.0, 5.0, None],
        ),
        (
            pl.Boolean,
            [False, True, 0, -1, 0.0, 2.5, date(1970, 1, 1), "true"],
            [False, True, False, True, False, True, None, None],
        ),
        (
            pl.Binary,
            [b"123", "xyz", 100, True, None],
            [b"123", b"xyz", None, None, None],
        ),
        (
            pl.String,
            ["xyz", 1, 2.5, date(1970, 1, 1), True, b"123", None],
            ["xyz", "1", "2.5", "1970-01-01", "true", None, None],
        ),
        (
            pl.Date,
            ["xyz", 1, 2.5, date(1970, 1, 1), datetime(2000, 1, 1, 12), True, None],
            [
                None,
                date(1970, 1, 2),
                date(1970, 1, 3),
                date(1970, 1, 1),
                date(2000, 1, 1),
                None,
                None,
            ],
        ),
        (
            pl.Time,
            [
                "xyz",
                1,
                2.5,
                date(1970, 1, 1),
                time(12, 0),
                datetime(2000, 1, 1, 12),
                timedelta(hours=5),
                True,
                None,
            ],
            [
                None,
                time(0, 0),
                time(0, 0),
                None,
                time(12, 0),
                time(12, 0),
                None,
                None,
                None,
            ],
        ),
        (
            pl.Datetime,
            [
                "xyz",
                1,
                2.5,
                date(1970, 1, 1),
                time(12, 0),
                datetime(2000, 1, 1, 12),
                timedelta(hours=5),
                True,
                None,
            ],
            [
                None,
                datetime(1970, 1, 1, microsecond=1),
                datetime(1970, 1, 1, microsecond=2),
                datetime(1970, 1, 1),
                None,
                datetime(2000, 1, 1, 12, 0),
                None,
                None,
                None,
            ],
        ),
        (
            pl.Duration,
            [
                "xyz",
                1,
                2.5,
                date(1970, 1, 1),
                time(12, 0),
                datetime(2000, 1, 1, 12),
                timedelta(hours=5),
                True,
                None,
            ],
            [
                None,
                timedelta(microseconds=1),
                timedelta(microseconds=2),
                None,
                timedelta(hours=12),
                None,
                timedelta(hours=5),
                None,
                None,
            ],
        ),
        (
            pl.Categorical,
            ["xyz", 1, 2.5, date(1970, 1, 1), True, b"123", None],
            ["xyz", "1", "2.5", "1970-01-01", "true", None, None],
        ),
        (
            pl.Enum(["a", "b"]),
            ["a", "b", "c", 1, 2, None],
            ["a", "b", None, None, None, None],
        ),
        (
            pl.Decimal(5, 3),
            [
                D("12"),
                D("1.2345"),
                # D("123456"),
                False,
                True,
                0,
                -1,
                0.0,
                2.5,
                date(1970, 1, 2),
                "5",
                "xyz",
            ],
            [
                D("12.000"),
                None,
                # None,
                None,
                None,
                D("0.000"),
                D("-1.000"),
                None,
                None,
                None,
                None,
                None,
            ],
        ),
        (
            pl.Struct({"a": pl.Int8, "b": pl.String}),
            [{"a": 1, "b": "foo"}, {"a": 1_000, "b": 2.0}],
            [{"a": 1, "b": "foo"}, {"a": None, "b": "2.0"}],
        ),
    ],
)
def test_fallback_with_dtype_nonstrict(
    dtype: PolarsDataType, values: list[Any], expected: list[Any]
) -> None:
    result = wrap_s(
        PySeries.new_from_any_values_and_dtype("", values, dtype, strict=False)
    )
    assert result.to_list() == expected


@pytest.mark.parametrize(
    ("expected_dtype", "values"),
    [
        (pl.Int64, [-1, 0, 100_000, None]),
        (pl.Float64, [-1.5, 0.0, 10.0, None]),
        (pl.Boolean, [True, False, None]),
        (pl.Binary, [b"123", b"xyz", None]),
        (pl.String, ["123", "xyz", None]),
        (pl.Date, [date(1970, 1, 1), date(2020, 12, 31), None]),
        (pl.Time, [time(0, 0), time(23, 59, 59), None]),
        (
            pl.Datetime("us"),
            [datetime(1970, 1, 1), datetime(2020, 12, 31, 23, 59, 59), None],
        ),
        (pl.Duration("us"), [timedelta(hours=0), timedelta(seconds=100), None]),
        (pl.Decimal(None, 3), [D("12.345"), D("0.789"), None]),
        (pl.Decimal(None, 0), [D("12"), D("56789"), None]),
        (
            pl.Struct({"a": pl.Int64, "b": pl.String, "c": pl.Float64}),
            [{"a": 1, "b": "foo", "c": None}, {"a": -1, "b": "bar", "c": 3.0}],
        ),
    ],
)
@pytest.mark.parametrize("strict", [True, False])
def test_fallback_without_dtype(
    expected_dtype: PolarsDataType, values: list[Any], strict: bool
) -> None:
    result = wrap_s(PySeries.new_from_any_values("", values, strict=strict))
    assert result.to_list() == values
    assert result.dtype == expected_dtype


@pytest.mark.parametrize(
    "values",
    [
        [1.0, 2],
        [1, 2.0],
        [False, 1],
        [b"123", "xyz"],
        ["123", b"xyz"],
        [date(1970, 1, 1), datetime(2020, 12, 31)],
        [time(0, 0), 1_000],
        [datetime(1970, 1, 1), date(2020, 12, 31)],
        [timedelta(hours=0), 1_000],
        [D("12.345"), 100],
        [D("12.345"), 3.14],
        [{"a": 1, "b": "foo"}, {"a": -1, "b": date(2020, 12, 31)}],
        [{"a": None}, {"a": 1.0}, {"a": 1}],
    ],
)
def test_fallback_without_dtype_strict_failure(values: list[Any]) -> None:
    with pytest.raises(TypeError, match="unexpected value"):
        PySeries.new_from_any_values("", values, strict=True)


@pytest.mark.parametrize(
    ("values", "expected", "expected_dtype"),
    [
        ([True, 2], [1, 2], pl.Int64),
        ([1, 2.0], [1.0, 2.0], pl.Float64),
        ([2.0, "c"], ["2.0", "c"], pl.String),
        (
            [date(1970, 1, 1), datetime(2022, 12, 31)],
            [datetime(1970, 1, 1), datetime(2022, 12, 31)],
            pl.Datetime("us"),
        ),
        ([D("3.1415"), 2.51], [3.1415, 2.51], pl.Float64),
        ([D("3.1415"), 100], [D("3.1415"), D("100")], pl.Decimal(None, 4)),
        ([1, 2.0, b"d", date(2022, 1, 1)], [1, 2.0, b"d", date(2022, 1, 1)], pl.Object),
        (
            [
                {"a": 1, "b": "foo", "c": None},
                {"a": 2.0, "b": date(2020, 12, 31), "c": None},
            ],
            [
                {"a": 1.0, "b": "foo", "c": None},
                {"a": 2.0, "b": "2020-12-31", "c": None},
            ],
            pl.Struct({"a": pl.Float64, "b": pl.String, "c": pl.Null}),
        ),
        (
            [{"a": None}, {"a": 1.0}, {"a": 1}],
            [{"a": None}, {"a": 1.0}, {"a": 1.0}],
            pl.Struct({"a": pl.Float64}),
        ),
    ],
)
def test_fallback_without_dtype_nonstrict_mixed_types(
    values: list[Any],
    expected_dtype: PolarsDataType,
    expected: list[Any],
) -> None:
    result = wrap_s(PySeries.new_from_any_values("", values, strict=False))
    assert result.dtype == expected_dtype
    assert result.to_list() == expected


def test_fallback_without_dtype_large_int() -> None:
    values = [1, 2**64, None]
    with pytest.raises(
        OverflowError,
        match="int value too large for Polars integer types: 18446744073709551616",
    ):
        PySeries.new_from_any_values("", values, strict=True)

    result = wrap_s(PySeries.new_from_any_values("", values, strict=False))
    assert result.dtype == pl.Float64
    assert result.to_list() == [1.0, 1.8446744073709552e19, None]


def test_fallback_with_dtype_large_int() -> None:
    values = [1, 2**64, None]
    with pytest.raises(OverflowError):
        PySeries.new_from_any_values_and_dtype("", values, dtype=pl.Int64, strict=True)

    result = wrap_s(
        PySeries.new_from_any_values_and_dtype("", values, dtype=pl.Int64, strict=False)
    )
    assert result.dtype == pl.Int64
    assert result.to_list() == [1, None, None]


def test_fallback_with_dtype_strict_failure_enum_casting() -> None:
    dtype = pl.Enum(["a", "b"])
    values = ["a", "b", "c", None]

    with pytest.raises(TypeError, match="conversion from `str` to `enum` failed"):
        PySeries.new_from_any_values_and_dtype("", values, dtype, strict=True)


def test_fallback_with_dtype_strict_failure_decimal_precision() -> None:
    dtype = pl.Decimal(3, 0)
    values = [D("12345")]

    with pytest.raises(
        TypeError, match="decimal precision 3 can't fit values with 5 digits"
    ):
        PySeries.new_from_any_values_and_dtype("", values, dtype, strict=True)
