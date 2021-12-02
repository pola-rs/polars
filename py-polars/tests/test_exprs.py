import numpy as np

import polars as pl
from polars import testing


def test_horizontal_agg(fruits_cars: pl.DataFrame) -> None:
    df = fruits_cars
    out = df.select(pl.max([pl.col("A"), pl.col("B")]))
    assert out[:, 0].to_list() == [5, 4, 3, 4, 5]

    out = df.select(pl.min([pl.col("A"), pl.col("B")]))
    assert out[:, 0].to_list() == [1, 2, 3, 2, 1]


def test_suffix(fruits_cars: pl.DataFrame) -> None:
    df = fruits_cars
    out = df.select([pl.all().suffix("_reverse")])
    assert out.columns == ["A_reverse", "fruits_reverse", "B_reverse", "cars_reverse"]


def test_prefix(fruits_cars: pl.DataFrame) -> None:
    df = fruits_cars
    out = df.select([pl.all().prefix("reverse_")])
    assert out.columns == ["reverse_A", "reverse_fruits", "reverse_B", "reverse_cars"]


def test_cumcount() -> None:
    df = pl.DataFrame([["a"], ["a"], ["a"], ["b"], ["b"], ["a"]], columns=["A"])

    out = df.groupby("A", maintain_order=True).agg(
        [pl.col("A").cumcount(reverse=False).alias("foo")]
    )

    assert out["foo"][0].to_list() == [0, 1, 2, 3]
    assert out["foo"][1].to_list() == [0, 1]


def test_log_exp() -> None:
    a = pl.Series("a", [1, 100, 1000])
    out = pl.select(a.log10()).to_series()
    expected = pl.Series("a", [0.0, 2.0, 3.0])
    testing.assert_series_equal(out, expected)

    out = pl.select(a.log()).to_series()
    expected = pl.Series("a", np.log(a.to_numpy()))
    testing.assert_series_equal(out, expected)

    out = pl.select(a.exp()).to_series()
    expected = pl.Series("a", np.exp(a.to_numpy()))
    testing.assert_series_equal(out, expected)
