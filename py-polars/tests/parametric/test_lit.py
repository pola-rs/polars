from datetime import datetime

from hypothesis import given

import polars as pl
from polars.testing.parametric.strategies import (
    strategy_datetime_ms,
    strategy_datetime_ns,
    strategy_datetime_us,
)


@given(value=strategy_datetime_ns)
def test_datetime_ns(value: datetime) -> None:
    result = pl.select(pl.lit(value, dtype=pl.Datetime("ns")))["literal"][0]
    assert result == value


@given(value=strategy_datetime_us)
def test_datetime_us(value: datetime) -> None:
    result = pl.select(pl.lit(value, dtype=pl.Datetime("us")))["literal"][0]
    assert result == value
    result = pl.select(pl.lit(value, dtype=pl.Datetime))["literal"][0]
    assert result == value


@given(value=strategy_datetime_ms)
def test_datetime_ms(value: datetime) -> None:
    result = pl.select(pl.lit(value, dtype=pl.Datetime("ms")))["literal"][0]
    expected_microsecond = value.microsecond // 1000 * 1000
    assert result == value.replace(microsecond=expected_microsecond)
