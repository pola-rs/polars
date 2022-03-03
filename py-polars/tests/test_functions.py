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


def test_all_any_horizontally() -> None:
    df = pl.DataFrame(
        [
            [False, False, True],
            [False, False, True],
            [True, False, False],
            [False, None, True],
            [None, None, False],
        ],
        columns=["var1", "var2", "var3"],
    )

    expected = pl.DataFrame(
        {
            "any": [True, True, False, True, None],
            "all": [False, False, False, None, False],
        }
    )

    assert df.select(
        [
            pl.any([pl.col("var2"), pl.col("var3")]),
            pl.all([pl.col("var2"), pl.col("var3")]),
        ]
    ).frame_equal(expected)
