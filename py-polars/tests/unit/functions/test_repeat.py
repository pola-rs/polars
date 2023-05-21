from datetime import datetime

import polars as pl


def test_repeat_lazy() -> None:
    s = pl.select(pl.repeat(2**31 - 1, 3)).to_series()
    assert s.dtype == pl.Int32
    assert s.len() == 3
    assert s.to_list() == [2**31 - 1] * 3
    s = pl.select(pl.repeat(-(2**31), 4)).to_series()
    assert s.dtype == pl.Int32
    assert s.len() == 4
    assert s.to_list() == [-(2**31)] * 4
    s = pl.select(pl.repeat(2**31, 5)).to_series()
    assert s.dtype == pl.Int64
    assert s.len() == 5
    assert s.to_list() == [2**31] * 5
    s = pl.select(pl.repeat(-(2**31) - 1, 3)).to_series()
    assert s.dtype == pl.Int64
    assert s.len() == 3
    assert s.to_list() == [-(2**31) - 1] * 3
    s = pl.select(pl.repeat("foo", 2)).to_series()
    assert s.dtype == pl.Utf8
    assert s.len() == 2
    assert s.to_list() == ["foo"] * 2
    s = pl.select(pl.repeat(1.0, 5)).to_series()
    assert s.dtype == pl.Float64
    assert s.len() == 5
    assert s.to_list() == [1.0] * 5
    s = pl.select(pl.repeat(True, 4)).to_series()
    assert s.dtype == pl.Boolean
    assert s.len() == 4
    assert s.to_list() == [True] * 4
    s = pl.select(pl.repeat(None, 7)).to_series()
    assert s.dtype == pl.Null
    assert s.len() == 7
    assert s.to_list() == [None] * 7
    s = pl.select(pl.repeat(0, 0)).to_series()
    assert s.dtype == pl.Int32
    assert s.len() == 0


def test_repeat_lazy_dtype() -> None:
    s = pl.select(pl.repeat(1, n=3, dtype=pl.Int8)).to_series()
    assert s.dtype == pl.Int8
    assert s.len() == 3


def test_repeat_eager() -> None:
    s = pl.repeat(2**31 - 1, 3, eager=True)
    assert s.dtype == pl.Int32
    assert s.len() == 3
    assert s.to_list() == [2**31 - 1] * 3
    s = pl.repeat(-(2**31), 4, eager=True)
    assert s.dtype == pl.Int32
    assert s.len() == 4
    assert s.to_list() == [-(2**31)] * 4
    s = pl.repeat(2**31, 5, eager=True)
    assert s.dtype == pl.Int64
    assert s.len() == 5
    assert s.to_list() == [2**31] * 5
    s = pl.repeat(-(2**31) - 1, 3, eager=True)
    assert s.dtype == pl.Int64
    assert s.len() == 3
    assert s.to_list() == [-(2**31) - 1] * 3
    s = pl.repeat("foo", 2, eager=True)
    assert s.dtype == pl.Utf8
    assert s.len() == 2
    assert s.to_list() == ["foo"] * 2
    s = pl.repeat(1.0, 5, eager=True)
    assert s.dtype == pl.Float64
    assert s.len() == 5
    assert s.to_list() == [1.0] * 5
    s = pl.repeat(True, 4, eager=True)
    assert s.dtype == pl.Boolean
    assert s.len() == 4
    assert s.to_list() == [True] * 4
    s = pl.repeat(None, 7, eager=True)
    assert s.dtype == pl.Null
    assert s.len() == 7
    assert s.to_list() == [None] * 7
    s = pl.repeat(0, 0, eager=True)
    assert s.dtype == pl.Int32
    assert s.len() == 0
    assert pl.repeat(datetime(2023, 2, 2), 3, eager=True).to_list() == [
        datetime(2023, 2, 2, 0, 0),
        datetime(2023, 2, 2, 0, 0),
        datetime(2023, 2, 2, 0, 0),
    ]


def test_repeat_eager_dtype() -> None:
    s = pl.repeat(1, n=3, eager=True, dtype=pl.Int8)
    assert s.dtype == pl.Int8
    assert s.len() == 3
