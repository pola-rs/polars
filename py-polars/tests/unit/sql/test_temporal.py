from __future__ import annotations

from datetime import date

import polars as pl
from polars.testing import assert_frame_equal


def test_date() -> None:
    df = pl.DataFrame(
        {
            "date": [
                date(2021, 3, 15),
                date(2021, 3, 28),
                date(2021, 4, 4),
            ],
            "version": ["0.0.1", "0.7.3", "0.7.4"],
        }
    )
    with pl.SQLContext(df=df, eager_execution=True) as ctx:
        result = ctx.execute("SELECT date < DATE('2021-03-20') from df")

    expected = pl.DataFrame({"date": [True, False, False]})
    assert_frame_equal(result, expected)

    result = pl.select(pl.sql_expr("""CAST(DATE('2023-03', '%Y-%m') as STRING)"""))
    expected = pl.DataFrame({"literal": ["2023-03-01"]})
    assert_frame_equal(result, expected)
