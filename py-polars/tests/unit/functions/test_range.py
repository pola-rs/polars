from datetime import time

import polars as pl


def test_time_range_name():
    expected_name = "time"
    result_eager = pl.time_range(time(10), time(12), eager=True)
    assert result_eager.name == expected_name

    result_lazy = pl.select(pl.time_range(time(10), time(12), eager=False)).to_series()
    assert result_lazy.name == expected_name
