import polars as pl


def test_date_datetime() -> None:
    df = pl.DataFrame(
        {
            "year": [2001, 2002, 2003],
            "month": [1, 2, 3],
            "day": [1, 2, 3],
            "hour": [23, 12, 8],
        }
    )

    out = df.select(
        [
            pl.all(),  # type: ignore
            pl.datetime("year", "month", "day", "hour").dt.hour().alias("h2"),  # type: ignore
            pl.date("year", "month", "day").dt.day().alias("date"),  # type: ignore
        ]
    )

    assert out["date"].series_equal(df["day"])
    assert out["h2"].series_equal(df["hour"])
