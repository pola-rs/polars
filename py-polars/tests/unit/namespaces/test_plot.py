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
    df.plot.scatter(x="length", y="width", by="species")


def test_dataframe_line() -> None:
    df = pl.DataFrame(
        {
            "date": [date(2020, 1, 2), date(2020, 1, 3), date(2020, 1, 3)],
            "stock_1": [1, 4, 6],
            "stock_2": [1, 5, 2],
        }
    )
    df.plot.line(x="date", y=["stock_1", "stock_2"])


def test_series_hist() -> None:
    s = pl.Series("values", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    s.plot.hist()


def test_empty_dataframe() -> None:
    pl.DataFrame({"a": [], "b": []}).plot.scatter(x="a", y="b")
