from datetime import time

import pytest

import polars as pl


def test_time_to_string_cast() -> None:
    assert pl.Series([time(12, 1, 1)]).cast(str).to_list() == ["12:01:01"]


def test_time_zero_3828() -> None:
    assert pl.Series(values=[time(0)], dtype=pl.Time).to_list() == [time(0)]


def test_time_microseconds_3843() -> None:
    in_val = [time(0, 9, 11, 558332)]
    s = pl.Series(in_val)
    assert s.to_list() == in_val


def test_invalid_casts() -> None:
    with pytest.raises(pl.exceptions.InvalidOperationError):
        pl.DataFrame({"a": []}).with_columns(a=pl.lit(-1).cast(pl.Time))

    with pytest.raises(pl.exceptions.InvalidOperationError):
        pl.Series([-1]).cast(pl.Time)

    with pytest.raises(pl.exceptions.InvalidOperationError):
        pl.Series([24 * 60 * 60 * 1_000_000_000]).cast(pl.Time)

    largest_value = pl.Series([24 * 60 * 60 * 1_000_000_000 - 1]).cast(pl.Time)
    assert "23:59:59.999999999" in str(largest_value)
