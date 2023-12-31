from datetime import date

import pytest

import polars as pl
from polars.exceptions import PolarsPanicError


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


def test_unsupported_dtype() -> None:
    with pytest.raises(PolarsPanicError):
        pl.DataFrame({"a": [{1, 2}], "b": [4]}).plot.scatter(x="a", y="b")
