from datetime import time

import polars as pl


def test_time_to_string_cast() -> None:
    assert pl.Series([time(12, 1, 1)]).cast(str).to_list() == ["12:01:01"]


def test_time_zero_3828() -> None:
    assert pl.Series(values=[time(0)], dtype=pl.Time).to_list() == [time(0)]


def test_time_microseconds_3843() -> None:
    in_val = [time(0, 9, 11, 558332)]
    s = pl.Series(in_val)
    assert s.to_list() == in_val
