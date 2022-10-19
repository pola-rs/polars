from datetime import date, timedelta

import polars as pl


def test_predicate_4906() -> None:
    one_day = timedelta(days=1)

    ldf = pl.DataFrame(
        {
            "dt": [
                date(2022, 9, 1),
                date(2022, 9, 10),
                date(2022, 9, 20),
            ]
        }
    ).lazy()

    assert ldf.filter(
        pl.min([(pl.col("dt") + one_day), date(2022, 9, 30)]) > date(2022, 9, 10)
    ).collect().to_dict(False) == {"dt": [date(2022, 9, 10), date(2022, 9, 20)]}


def test_when_then_implicit_none() -> None:
    df = pl.DataFrame(
        {
            "team": ["A", "A", "A", "B", "B", "C"],
            "points": [11, 8, 10, 6, 6, 5],
        }
    )

    assert df.select(
        [
            pl.when(pl.col("points") > 7).then("Foo"),
            pl.when(pl.col("points") > 7).then("Foo").alias("bar"),
        ]
    ).to_dict(False) == {
        "literal": ["Foo", "Foo", "Foo", None, None, None],
        "bar": ["Foo", "Foo", "Foo", None, None, None],
    }
