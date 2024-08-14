from __future__ import annotations

import enum
from datetime import datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from hypothesis import given

import polars as pl
from polars.testing import assert_frame_equal
from polars.testing.parametric.strategies import series
from polars.testing.parametric.strategies.data import datetimes

if TYPE_CHECKING:
    from polars._typing import PolarsDataType


@pytest.mark.parametrize(
    "input",
    [
        [[1, 2], [3, 4, 5]],
        [1, 2, 3],
    ],
)
def test_lit_list_input(input: list[Any]) -> None:
    df = pl.DataFrame({"a": [1, 2]})
    result = df.with_columns(pl.lit(input))
    expected = pl.DataFrame({"a": [1, 2], "literal": [input, input]})
    assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "input",
    [
        ([1, 2], [3, 4, 5]),
        (1, 2, 3),
    ],
)
def test_lit_tuple_input(input: tuple[Any, ...]) -> None:
    df = pl.DataFrame({"a": [1, 2]})
    result = df.with_columns(pl.lit(input))

    expected = pl.DataFrame({"a": [1, 2], "literal": [list(input), list(input)]})
    assert_frame_equal(result, expected)


def test_lit_numpy_array_input() -> None:
    df = pl.DataFrame({"a": [1, 2]})
    input = np.array([3, 4])

    result = df.with_columns(pl.lit(input, dtype=pl.Int64))

    expected = pl.DataFrame({"a": [1, 2], "literal": [3, 4]})
    assert_frame_equal(result, expected)


def test_lit_ambiguous_datetimes_11379() -> None:
    df = pl.DataFrame(
        {
            "ts": pl.datetime_range(
                datetime(2020, 10, 25),
                datetime(2020, 10, 25, 2),
                "1h",
                time_zone="Europe/London",
                eager=True,
            )
        }
    )
    for i in range(len(df)):
        result = df.filter(pl.col("ts") >= df["ts"][i])
        expected = df[i:]
        assert_frame_equal(result, expected)


def test_list_datetime_11571() -> None:
    sec_np_ns = np.timedelta64(1_000_000_000, "ns")
    sec_np_us = np.timedelta64(1_000_000, "us")
    assert pl.select(pl.lit(sec_np_ns))[0, 0] == timedelta(seconds=1)
    assert pl.select(pl.lit(sec_np_us))[0, 0] == timedelta(seconds=1)


@pytest.mark.parametrize(
    ("input", "dtype"),
    [
        pytest.param(-(2**31), pl.Int32, id="i32 min"),
        pytest.param(-(2**31) - 1, pl.Int64, id="below i32 min"),
        pytest.param(2**31 - 1, pl.Int32, id="i32 max"),
        pytest.param(2**31, pl.Int64, id="above i32 max"),
        pytest.param(2**63 - 1, pl.Int64, id="i64 max"),
        pytest.param(2**63, pl.UInt64, id="above i64 max"),
    ],
)
def test_lit_int_return_type(input: int, dtype: PolarsDataType) -> None:
    assert pl.select(pl.lit(input)).to_series().dtype == dtype


def test_lit_unsupported_type() -> None:
    with pytest.raises(
        TypeError,
        match="cannot create expression literal for value of type LazyFrame: ",
    ):
        pl.lit(pl.LazyFrame({"a": [1, 2, 3]}))


def test_lit_enum_input_16668() -> None:
    # https://github.com/pola-rs/polars/issues/16668

    class State(str, enum.Enum):
        VIC = "victoria"
        NSW = "new south wales"

    value = State.VIC

    result = pl.lit(value)
    assert pl.select(result).dtypes[0] == pl.Enum(["victoria", "new south wales"])
    assert pl.select(result).item() == "victoria"

    result = pl.lit(value, dtype=pl.String)
    assert pl.select(result).dtypes[0] == pl.String
    assert pl.select(result).item() == "victoria"


def test_lit_enum_input_non_string() -> None:
    # https://github.com/pola-rs/polars/issues/16668

    class State(int, enum.Enum):
        ONE = 1
        TWO = 2

    value = State.ONE

    result = pl.lit(value)
    assert pl.select(result).dtypes[0] == pl.Int32
    assert pl.select(result).item() == 1

    result = pl.lit(value, dtype=pl.Int8)
    assert pl.select(result).dtypes[0] == pl.Int8
    assert pl.select(result).item() == 1


@given(value=datetimes("ns"))
def test_datetime_ns(value: datetime) -> None:
    result = pl.select(pl.lit(value, dtype=pl.Datetime("ns")))["literal"][0]
    assert result == value


@given(value=datetimes("us"))
def test_datetime_us(value: datetime) -> None:
    result = pl.select(pl.lit(value, dtype=pl.Datetime("us")))["literal"][0]
    assert result == value
    result = pl.select(pl.lit(value, dtype=pl.Datetime))["literal"][0]
    assert result == value


@given(value=datetimes("ms"))
def test_datetime_ms(value: datetime) -> None:
    result = pl.select(pl.lit(value, dtype=pl.Datetime("ms")))["literal"][0]
    expected_microsecond = value.microsecond // 1000 * 1000
    assert result == value.replace(microsecond=expected_microsecond)


def test_lit_decimal() -> None:
    value = Decimal("0.1")

    expr = pl.lit(value)
    df = pl.select(expr)
    result = df.item()

    assert df.dtypes[0] == pl.Decimal(None, 1)
    assert result == value


def test_lit_string_float() -> None:
    value = 3.2

    expr = pl.lit(value, dtype=pl.Utf8)
    df = pl.select(expr)
    result = df.item()

    assert df.dtypes[0] == pl.String
    assert result == str(value)


@given(s=series(min_size=1, max_size=1, allow_null=False, allowed_dtypes=pl.Decimal))
def test_lit_decimal_parametric(s: pl.Series) -> None:
    scale = s.dtype.scale  # type: ignore[attr-defined]
    value = s.item()

    expr = pl.lit(value)
    df = pl.select(expr)
    result = df.item()

    assert df.dtypes[0] == pl.Decimal(None, scale)
    assert result == value
