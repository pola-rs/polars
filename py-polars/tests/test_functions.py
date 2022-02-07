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
            pl.all(),
            pl.datetime("year", "month", "day", "hour").dt.hour().cast(int).alias("h2"),
            pl.date("year", "month", "day").dt.day().cast(int).alias("date"),
        ]
    )

    assert out["date"].series_equal(df["day"].rename("date"))
    assert out["h2"].series_equal(df["hour"].rename("h2"))


def test_diag_concat() -> None:
    a = pl.DataFrame({"a": [1, 2]})
    b = pl.DataFrame({"b": ["a", "b"], "c": [1, 2]})
    c = pl.DataFrame({"a": [5, 7], "c": [1, 2], "d": [1, 2]})

    out = pl.concat([a, b, c], how="diagonal")
    expected = pl.DataFrame(
        {
            "a": [1, 2, None, None, 5, 7],
            "b": [None, None, "a", "b", None, None],
            "c": [None, None, 1, 2, 1, 2],
            "d": [None, None, None, None, 1, 2],
        }
    )

    assert out.frame_equal(expected, null_equal=True)


def test_concat_horizontal() -> None:
    a = pl.DataFrame({"a": ["a", "b"], "b": [1, 2]})
    b = pl.DataFrame({"c": [5, 7, 8, 9], "d": [1, 2, 1, 2], "e": [1, 2, 1, 2]})

    out = pl.concat([a, b], how="horizontal")
    expected = pl.DataFrame(
        {
            "a": ["a", "b", None, None],
            "b": [1, 2, None, None],
            "c": [5, 7, 8, 9],
            "d": [1, 2, 1, 2],
            "e": [1, 2, 1, 2],
        }
    )
    assert out.frame_equal(expected)
