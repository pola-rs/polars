from datetime import date, datetime
from typing import Any

import numpy as np
import pytest

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
        df_quotes, on=pl.col("time").cast(pl.Datetime("ns")).set_sorted(), by="stock"
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
    joined_tbl = a.join_asof(b, on=pl.col("a").set_sorted(), by="b")
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
    df_a = (
        pl.DataFrame(
            {
                "id": [1],
                "a": [1],
                "b": [1],
            }
        )
        .lazy()
        .set_sorted("a")
    )

    df_b = (
        pl.DataFrame(
            {
                "id": [1, 1, 2],
                "b": [-3, -3, 6],
            }
        )
        .lazy()
        .set_sorted("b")
    )

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
    assert df1.join_asof(df2, on=pl.col("a").set_sorted(), strategy="backward").to_dict(
        False
    ) == {
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
    assert df1.join_asof(df2, on=pl.col("b").set_sorted(), by="c").to_dict(False) == {
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
    ).set_sorted("time")

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
    ).set_sorted("time")

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
    ).set_sorted("time")

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
    ).set_sorted("time")

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
    ).set_sorted("df1_date")

    df2 = pl.DataFrame(
        {
            "df2_date": [20221012, 20221015, 20221018],
            "df2_col1": ["1", "2", "3"],
            "key": ["a", "b", "b"],
        }
    ).set_sorted("df2_date")

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
        pl.date_range(
            datetime(2022, 1, 1), datetime(2022, 1, 2), interval="2h", eager=True
        )
        .cast(pl.Datetime("ns"))
        .head(9)
    )
    x = pl.DataFrame({"a": dates, "b": map(float, range(9)), "c": ["1", "2", "3"] * 3})
    assert x.join_asof(x, on=pl.col("b").set_sorted(), by=["c", "a"]).to_dict(
        False
    ) == {
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
    ldf1 = pl.DataFrame({"a": [1, 2, 2], "b": "bleft"}).lazy().set_sorted("a")
    ldf2 = pl.DataFrame({"a": 2, "b": [1, 2, 2]}).lazy().set_sorted("b")

    assert (
        ldf1.join_asof(ldf2, left_on="a", right_on="b").select("a", "b")
    ).collect().to_dict(False) == {"a": [1, 2, 2], "b": ["bleft", "bleft", "bleft"]}


def test_asof_join_sorted_by_group(capsys: Any) -> None:
    df1 = pl.DataFrame(
        {
            "key": ["a", "a", "a", "b", "b", "b"],
            "asof_key": [2.0, 1.0, 3.0, 1.0, 2.0, 3.0],
            "a": [102, 101, 103, 104, 105, 106],
        }
    ).sort(by=["key", "asof_key"])

    df2 = pl.DataFrame(
        {
            "key": ["a", "a", "a", "b", "b", "b"],
            "asof_key": [0.9, 1.9, 2.9, 0.9, 1.9, 2.9],
            "b": [201, 202, 203, 204, 205, 206],
        }
    ).sort(by=["key", "asof_key"])

    expected = pl.DataFrame(
        [
            pl.Series("key", ["a", "a", "a", "b", "b", "b"], dtype=pl.Utf8),
            pl.Series("asof_key", [1.0, 2.0, 3.0, 1.0, 2.0, 3.0], dtype=pl.Float64),
            pl.Series("a", [101, 102, 103, 104, 105, 106], dtype=pl.Int64),
            pl.Series("b", [201, 202, 203, 204, 205, 206], dtype=pl.Int64),
        ]
    )

    out = df1.join_asof(df2, on="asof_key", by="key")
    assert_frame_equal(out, expected)

    _, err = capsys.readouterr()
    assert "is not explicitly sorted" not in err


def test_asof_join_nearest() -> None:
    df1 = pl.DataFrame(
        {
            "asof_key": [-1, 1, 2, 4, 6],
            "a": [1, 2, 3, 4, 5],
        }
    ).sort(by="asof_key")

    df2 = pl.DataFrame(
        {
            "asof_key": [1, 2, 4, 5],
            "b": [1, 2, 3, 4],
        }
    ).sort(by="asof_key")

    expected = pl.DataFrame(
        {"asof_key": [-1, 1, 2, 4, 6], "a": [1, 2, 3, 4, 5], "b": [1, 1, 2, 3, 4]}
    )

    out = df1.join_asof(df2, on="asof_key", strategy="nearest")
    assert_frame_equal(out, expected)


def test_asof_join_nearest_by() -> None:
    df1 = pl.DataFrame(
        {
            "asof_key": [-1, 1, 2, 6, 1],
            "group": [1, 1, 1, 2, 2],
            "a": [1, 2, 3, 2, 5],
        }
    ).sort(by=["group", "asof_key"])

    df2 = pl.DataFrame(
        {
            "asof_key": [1, 2, 5, 1],
            "group": [1, 1, 2, 2],
            "b": [1, 2, 3, 4],
        }
    ).sort(by=["group", "asof_key"])

    expected = pl.DataFrame(
        {
            "asof_key": [-1, 1, 2, 6, 1],
            "group": [1, 1, 1, 2, 2],
            "a": [1, 2, 3, 2, 5],
            "b": [1, 1, 2, 3, 4],
        }
    ).sort(by=["group", "asof_key"])

    out = df1.join_asof(df2, on="asof_key", by="group", strategy="nearest")
    assert_frame_equal(out, expected)


def test_asof_join_nearest_by_date() -> None:
    df1 = pl.DataFrame(
        {
            "asof_key": [
                date(2019, 12, 30),
                date(2020, 1, 1),
                date(2020, 1, 2),
                date(2020, 1, 6),
                date(2020, 1, 1),
            ],
            "group": [1, 1, 1, 2, 2],
            "a": [1, 2, 3, 2, 5],
        }
    ).sort(by=["group", "asof_key"])

    df2 = pl.DataFrame(
        {
            "asof_key": [
                date(2020, 1, 1),
                date(2020, 1, 2),
                date(2020, 1, 5),
                date(2020, 1, 1),
            ],
            "group": [1, 1, 2, 2],
            "b": [1, 2, 3, 4],
        }
    ).sort(by=["group", "asof_key"])

    expected = pl.DataFrame(
        {
            "asof_key": [
                date(2019, 12, 30),
                date(2020, 1, 1),
                date(2020, 1, 2),
                date(2020, 1, 6),
                date(2020, 1, 1),
            ],
            "group": [1, 1, 1, 2, 2],
            "a": [1, 2, 3, 2, 5],
            "b": [1, 1, 2, 3, 4],
        }
    ).sort(by=["group", "asof_key"])

    out = df1.join_asof(df2, on="asof_key", by="group", strategy="nearest")
    assert_frame_equal(out, expected)


def test_asof_join_string_err() -> None:
    left = pl.DataFrame({"date_str": ["2023/02/15"]}).sort("date_str")
    right = pl.DataFrame(
        {"date_str": ["2023/01/31", "2023/02/28"], "value": [0, 1]}
    ).sort("date_str")
    with pytest.raises(pl.InvalidOperationError):
        left.join_asof(right, on="date_str")
