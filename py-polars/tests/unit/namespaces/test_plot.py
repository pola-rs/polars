from datetime import date

import polars as pl


def test_plot_namespace() -> None:
    df = pl.DataFrame(
        {
            "length": [1, 4, 6],
            "width": [4, 5, 6],
            "species": ["setosa", "setosa", "versicolor"],
        }
    )
    df.plot.scatter(x="length", y="width", by="species")

    df = pl.DataFrame(
        {
            "date": [date(2020, 1, 2), date(2020, 1, 3), date(2020, 1, 3)],
            "stock_1": [1, 4, 6],
            "stock_2": [1, 5, 2],
        }
    )
    df.plot.line(x="date", y=["stock_1", "stock_2"])
