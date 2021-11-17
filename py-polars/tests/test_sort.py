import polars as pl


def test_sort_dates_multiples() -> None:
    df = pl.DataFrame(
        [
            pl.Series(
                "date",
                [
                    "2021-01-01 00:00:00",
                    "2021-01-01 00:00:00",
                    "2021-01-02 00:00:00",
                    "2021-01-02 00:00:00",
                    "2021-01-03 00:00:00",
                ],
            ).str.strptime(pl.Datetime, "%Y-%m-%d %T"),
            pl.Series("values", [5, 4, 3, 2, 1]),
        ]
    )

    expected = [4, 5, 2, 3, 1]

    # datetime
    out: pl.DataFrame = df.sort(["date", "values"])  # type: ignore
    assert out["values"].to_list() == expected

    # Date
    out = df.with_column(pl.col("date").cast(pl.Date)).sort(["date", "values"])  # type: ignore
    assert out["values"].to_list() == expected
