from __future__ import annotations

import polars as pl
from polars.testing import assert_series_equal


def test_all_horizontal_expansion() -> None:
    df = pl.select(x=True, y=False)
    out = df.select(z=pl.all_horizontal(pl.all())).to_series()

    assert_series_equal(out, pl.Series("z", [False]))


def test_any_horizontal_expansion() -> None:
    df = pl.select(x=True, y=False)
    out = df.select(z=pl.any_horizontal(pl.all())).to_series()

    assert_series_equal(out, pl.Series("z", [True]))


def test_coalesce_expansion() -> None:
    df = pl.select(x=None, y=1).cast(pl.Int64)
    out = df.select(z=pl.coalesce(pl.all())).to_series()

    assert_series_equal(out, pl.Series("z", [1], dtype=pl.Int64))


def test_concat_list_expansion() -> None:
    df = pl.select(x=[1], y=[2])
    out = df.select(z=pl.concat_list(pl.all())).to_series()

    assert_series_equal(out, pl.Series("z", [[1, 2]], dtype=pl.List(pl.Int64)))


def test_concat_str_expansion() -> None:
    df = pl.select(x=pl.lit("polar"), y=pl.lit("bear"))
    out = df.select(z=pl.concat_str(pl.all(), separator=" ")).to_series()

    assert_series_equal(out, pl.Series("z", ["polar bear"]))


def test_cum_sum_horizontal_expansion() -> None:
    df = pl.select(x=1, y=2).cast(pl.Int64)
    out = df.select(z=pl.cum_sum_horizontal(pl.all())).to_series()

    assert_series_equal(out, pl.Series("z", [{"x": 1, "y": 3}]))


def test_max_horizontal_expansion() -> None:
    df = pl.select(x=1, y=2).cast(pl.Int64)
    out = df.select(z=pl.max_horizontal(pl.all())).to_series()

    assert_series_equal(out, pl.Series("z", [2], dtype=pl.Int64))


def test_mean_horizontal_expansion() -> None:
    df = pl.select(x=1, y=2)
    out = df.select(z=pl.mean_horizontal(pl.all())).to_series()

    assert_series_equal(out, pl.Series("z", [1.5]))


def test_min_horizontal_expansion() -> None:
    df = pl.select(x=1, y=2).cast(pl.Int64)
    out = df.select(z=pl.min_horizontal(pl.all())).to_series()

    assert_series_equal(out, pl.Series("z", [1], dtype=pl.Int64))


def test_sum_horizontal_expansion() -> None:
    df = pl.select(x=1, y=2).cast(pl.Int64)
    out = df.select(z=pl.sum_horizontal(pl.all())).to_series()

    assert_series_equal(out, pl.Series("z", [3], dtype=pl.Int64))
