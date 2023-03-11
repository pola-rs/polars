from datetime import datetime

import numpy as np

import polars as pl
from polars.testing import assert_frame_equal


def test_asof_join_inline_cast_6438() -> None:
    df_trades = pl.DataFrame(
        {
            "time": [
                datetime(2020, 1, 1, 9, 1, 0),
                datetime(2020, 1, 1, 9, 1, 0),
                datetime(2020, 1, 1, 9, 3, 0),
                datetime(2020, 1, 1, 9, 6, 0),
            ],
            "stock": ["A", "B", "B", "C"],
            "trade": [101, 299, 301, 500],
        }
    )

    df_quotes = pl.DataFrame(
        {
            "time": [
                datetime(2020, 1, 1, 9, 0, 0),
                datetime(2020, 1, 1, 9, 2, 0),
                datetime(2020, 1, 1, 9, 3, 0),
                datetime(2020, 1, 1, 9, 6, 0),
            ],
            "stock": ["A", "B", "C", "A"],
            "quote": [100, 300, 501, 102],
        }
    ).with_columns([pl.col("time").dt.cast_time_unit("ns")])

    assert df_trades.join_asof(
        df_quotes, on=pl.col("time").cast(pl.Datetime("ns")), by="stock"
    ).to_dict(False) == {
        "time": [
            datetime(2020, 1, 1, 9, 1),
            datetime(2020, 1, 1, 9, 1),
            datetime(2020, 1, 1, 9, 3),
            datetime(2020, 1, 1, 9, 6),
        ],
        "stock": ["A", "B", "B", "C"],
        "trade": [101, 299, 301, 500],
        "quote": [100, None, 300, 501],
    }


def test_asof_join_projection_resolution_4606() -> None:
    a = pl.DataFrame({"a": [1], "b": [2], "c": [3]}).lazy()
    b = pl.DataFrame({"a": [1], "b": [2], "d": [4]}).lazy()
    joined_tbl = a.join_asof(b, on="a", by="b")
    assert joined_tbl.groupby("a").agg(
        [pl.col("c").sum().alias("c")]
    ).collect().columns == ["a", "c"]


def test_asof_join_schema_5211() -> None:
    df1 = pl.DataFrame({"today": [1, 2]})

    df2 = pl.DataFrame({"next_friday": [1, 2]})

    assert (
        df1.lazy()
        .join_asof(
            df2.lazy(), left_on="today", right_on="next_friday", strategy="forward"
        )
        .schema
    ) == {"today": pl.Int64, "next_friday": pl.Int64}


def test_asof_join_schema_5684() -> None:
    df_a = pl.DataFrame(
        {
            "id": [1],
            "a": [1],
            "b": [1],
        }
    ).lazy()

    df_b = pl.DataFrame(
        {
            "id": [1, 1, 2],
            "b": [3, -3, 6],
        }
    ).lazy()

    q = (
        df_a.join_asof(df_b, by="id", left_on="a", right_on="b")
        .drop("b")
        .join_asof(df_b, by="id", left_on="a", right_on="b")
        .drop("b")
    )

    projected_result = q.select(pl.all()).collect()
    result = q.collect()

    assert_frame_equal(projected_result, result)
    assert (
        q.schema
        == projected_result.schema
        == {"id": pl.Int64, "a": pl.Int64, "b_right": pl.Int64}
    )


def test_join_asof_floats() -> None:
    df1 = pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": ["lrow1", "lrow2", "lrow3"]})
    df2 = pl.DataFrame({"a": [0.59, 1.49, 2.89], "b": ["rrow1", "rrow2", "rrow3"]})
    assert df1.join_asof(df2, on="a", strategy="backward").to_dict(False) == {
        "a": [1.0, 2.0, 3.0],
        "b": ["lrow1", "lrow2", "lrow3"],
        "b_right": ["rrow1", "rrow2", "rrow3"],
    }

    # with by argument
    # 5740
    df1 = pl.DataFrame(
        {"b": np.linspace(0, 5, 7), "c": ["x" if i < 4 else "y" for i in range(7)]}
    )
    df2 = pl.DataFrame(
        {
            "val": [0, 2.5, 2.6, 2.7, 3.4, 4, 5],
            "c": ["x", "x", "x", "y", "y", "y", "y"],
        }
    ).with_columns(pl.col("val").alias("b"))
    assert df1.join_asof(df2, on="b", by="c").to_dict(False) == {
        "b": [
            0.0,
            0.8333333333333334,
            1.6666666666666667,
            2.5,
            3.3333333333333335,
            4.166666666666667,
            5.0,
        ],
        "c": ["x", "x", "x", "x", "y", "y", "y"],
        "val": [0.0, 0.0, 0.0, 2.5, 2.7, 4.0, 5.0],
    }


def test_join_asof_tolerance() -> None:
    df_trades = pl.DataFrame(
        {
            "time": [
                datetime(2020, 1, 1, 9, 0, 1),
                datetime(2020, 1, 1, 9, 0, 1),
                datetime(2020, 1, 1, 9, 0, 3),
                datetime(2020, 1, 1, 9, 0, 6),
            ],
            "stock": ["A", "B", "B", "C"],
            "trade": [101, 299, 301, 500],
        }
    )

    df_quotes = pl.DataFrame(
        {
            "time": [
                datetime(2020, 1, 1, 9, 0, 0),
                datetime(2020, 1, 1, 9, 0, 2),
                datetime(2020, 1, 1, 9, 0, 4),
                datetime(2020, 1, 1, 9, 0, 6),
            ],
            "stock": ["A", "B", "C", "A"],
            "quote": [100, 300, 501, 102],
        }
    )

    assert df_trades.join_asof(
        df_quotes, on="time", by="stock", tolerance="2s"
    ).to_dict(False) == {
        "time": [
            datetime(2020, 1, 1, 9, 0, 1),
            datetime(2020, 1, 1, 9, 0, 1),
            datetime(2020, 1, 1, 9, 0, 3),
            datetime(2020, 1, 1, 9, 0, 6),
        ],
        "stock": ["A", "B", "B", "C"],
        "trade": [101, 299, 301, 500],
        "quote": [100, None, 300, 501],
    }

    assert df_trades.join_asof(
        df_quotes, on="time", by="stock", tolerance="1s"
    ).to_dict(False) == {
        "time": [
            datetime(2020, 1, 1, 9, 0, 1),
            datetime(2020, 1, 1, 9, 0, 1),
            datetime(2020, 1, 1, 9, 0, 3),
            datetime(2020, 1, 1, 9, 0, 6),
        ],
        "stock": ["A", "B", "B", "C"],
        "trade": [101, 299, 301, 500],
        "quote": [100, None, 300, None],
    }


def test_join_asof_tolerance_forward() -> None:
    df_quotes = pl.DataFrame(
        {
            "time": [
                datetime(2020, 1, 1, 9, 0, 0),
                datetime(2020, 1, 1, 9, 0, 2),
                datetime(2020, 1, 1, 9, 0, 4),
                datetime(2020, 1, 1, 9, 0, 6),
                datetime(2020, 1, 1, 9, 0, 7),
            ],
            "stock": ["A", "B", "C", "A", "D"],
            "quote": [100, 300, 501, 102, 10],
        }
    )

    df_trades = pl.DataFrame(
        {
            "time": [
                datetime(2020, 1, 1, 9, 0, 2),
                datetime(2020, 1, 1, 9, 0, 1),
                datetime(2020, 1, 1, 9, 0, 3),
                datetime(2020, 1, 1, 9, 0, 6),
                datetime(2020, 1, 1, 9, 0, 7),
            ],
            "stock": ["A", "B", "B", "C", "D"],
            "trade": [101, 299, 301, 500, 10],
        }
    )

    assert df_quotes.join_asof(
        df_trades, on="time", by="stock", tolerance="2s", strategy="forward"
    ).to_dict(False) == {
        "time": [
            datetime(2020, 1, 1, 9, 0, 0),
            datetime(2020, 1, 1, 9, 0, 2),
            datetime(2020, 1, 1, 9, 0, 4),
            datetime(2020, 1, 1, 9, 0, 6),
            datetime(2020, 1, 1, 9, 0, 7),
        ],
        "stock": ["A", "B", "C", "A", "D"],
        "quote": [100, 300, 501, 102, 10],
        "trade": [101, 301, 500, None, 10],
    }

    assert df_quotes.join_asof(
        df_trades, on="time", by="stock", tolerance="1s", strategy="forward"
    ).to_dict(False) == {
        "time": [
            datetime(2020, 1, 1, 9, 0, 0),
            datetime(2020, 1, 1, 9, 0, 2),
            datetime(2020, 1, 1, 9, 0, 4),
            datetime(2020, 1, 1, 9, 0, 6),
            datetime(2020, 1, 1, 9, 0, 7),
        ],
        "stock": ["A", "B", "C", "A", "D"],
        "quote": [100, 300, 501, 102, 10],
        "trade": [None, 301, None, None, 10],
    }

    # Sanity check that this gives us equi-join
    assert df_quotes.join_asof(
        df_trades, on="time", by="stock", tolerance="0s", strategy="forward"
    ).to_dict(False) == {
        "time": [
            datetime(2020, 1, 1, 9, 0, 0),
            datetime(2020, 1, 1, 9, 0, 2),
            datetime(2020, 1, 1, 9, 0, 4),
            datetime(2020, 1, 1, 9, 0, 6),
            datetime(2020, 1, 1, 9, 0, 7),
        ],
        "stock": ["A", "B", "C", "A", "D"],
        "quote": [100, 300, 501, 102, 10],
        "trade": [None, None, None, None, 10],
    }


def test_join_asof_projection() -> None:
    df1 = pl.DataFrame(
        {
            "df1_date": [20221011, 20221012, 20221013, 20221014, 20221016],
            "df1_col1": ["foo", "bar", "foo", "bar", "foo"],
            "key": ["a", "b", "b", "a", "b"],
        }
    )

    df2 = pl.DataFrame(
        {
            "df2_date": [20221012, 20221015, 20221018],
            "df2_col1": ["1", "2", "3"],
            "key": ["a", "b", "b"],
        }
    )

    assert (
        (
            df1.lazy().join_asof(df2.lazy(), left_on="df1_date", right_on="df2_date")
        ).select([pl.col("df2_date"), "df1_date"])
    ).collect().to_dict(False) == {
        "df2_date": [None, 20221012, 20221012, 20221012, 20221015],
        "df1_date": [20221011, 20221012, 20221013, 20221014, 20221016],
    }
    assert (
        df1.lazy().join_asof(
            df2.lazy(), by="key", left_on="df1_date", right_on="df2_date"
        )
    ).select(["df2_date", "df1_date"]).collect().to_dict(False) == {
        "df2_date": [None, None, None, 20221012, 20221015],
        "df1_date": [20221011, 20221012, 20221013, 20221014, 20221016],
    }


def test_asof_join_by_logical_types() -> None:
    dates = (
        pl.date_range(datetime(2022, 1, 1), datetime(2022, 1, 2), interval="2h")
        .cast(pl.Datetime("ns"))
        .head(9)
    )
    x = pl.DataFrame({"a": dates, "b": map(float, range(9)), "c": ["1", "2", "3"] * 3})
    assert x.join_asof(x, on="b", by=["c", "a"]).to_dict(False) == {
        "a": [
            datetime(2022, 1, 1, 0, 0),
            datetime(2022, 1, 1, 2, 0),
            datetime(2022, 1, 1, 4, 0),
            datetime(2022, 1, 1, 6, 0),
            datetime(2022, 1, 1, 8, 0),
            datetime(2022, 1, 1, 10, 0),
            datetime(2022, 1, 1, 12, 0),
            datetime(2022, 1, 1, 14, 0),
            datetime(2022, 1, 1, 16, 0),
        ],
        "b": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "c": ["1", "2", "3", "1", "2", "3", "1", "2", "3"],
    }


def test_join_asof_projection_7481() -> None:
    ldf1 = pl.DataFrame({"a": [1, 2, 2], "b": "bleft"}).lazy()
    ldf2 = pl.DataFrame({"a": 2, "b": [1, 2, 2]}).lazy()

    assert (
        ldf1.join_asof(ldf2, left_on="a", right_on="b").select("a", "b")
    ).collect().to_dict(False) == {"a": [1, 2, 2], "b": ["bleft", "bleft", "bleft"]}
