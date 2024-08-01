from datetime import date

import pytest

import polars as pl

# Calling `plot` the first time is slow
# https://github.com/pola-rs/polars/issues/13500
pytestmark = pytest.mark.slow


def test_dataframe_scatter() -> None:
    df = pl.DataFrame(
        {
            "length": [1, 4, 6],
            "width": [4, 5, 6],
            "species": ["setosa", "setosa", "versicolor"],
        }
    )
    df.plot.point(x="length", y="width", color="species")


def test_dataframe_line() -> None:
    df = pl.DataFrame(
        {
            "date": [date(2020, 1, 2), date(2020, 1, 3), date(2020, 1, 4)] * 2,
            "price": [1, 4, 6, 1, 5, 2],
            "stock": ["a", "a", "a", "b", "b", "b"],
        }
    )
    df.plot.line(x="date", y="price", color="stock")


def test_empty_dataframe() -> None:
    pl.DataFrame({"a": [], "b": []}).plot.point(x="a", y="b")
