from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import pytest

import polars as pl
from polars.testing import assert_frame_equal


def test_asof_join_singular_right_11966() -> None:
    df = pl.DataFrame({"id": [1, 2, 3], "time": [0.9, 2.1, 2.8]}).sort("time")
    lookup = pl.DataFrame({"time": [2.0], "value": [100]}).sort("time")
    joined = df.join_asof(lookup, on="time", strategy="nearest")
    expected = pl.DataFrame(
        {"id": [1, 2, 3], "time": [0.9, 2.1, 2.8], "value": [100, 100, 100]}
    )
    assert_frame_equal(joined, expected)


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
    ).to_dict(as_series=False) == {
        "time": [
            datetime(2020, 1, 1, 9, 1),
            datetime(2020, 1, 1, 9, 1),
            datetime(2020, 1, 1, 9, 3),
            datetime(2020, 1, 1, 9, 6),
        ],
        "time_right": [
            datetime(2020, 1, 1, 9, 0),
            None,
            datetime(2020, 1, 1, 9, 2),
            datetime(2020, 1, 1, 9, 3),
        ],
        "stock": ["A", "B", "B", "C"],
        "trade": [101, 299, 301, 500],
        "quote": [100, None, 300, 501],
    }


def test_asof_join_projection_resolution_4606() -> None:
    a = pl.DataFrame({"a": [1], "b": [2], "c": [3]}).lazy()
    b = pl.DataFrame({"a": [1], "b": [2], "d": [4]}).lazy()
    joined_tbl = a.join_asof(b, on=pl.col("a").set_sorted(), by="b")
    assert joined_tbl.group_by("a").agg(
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
        .collect_schema()
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
        q.collect_schema()
        == projected_result.schema
        == {"id": pl.Int64, "a": pl.Int64, "b_right": pl.Int64}
    )


def test_join_asof_mismatched_dtypes() -> None:
    # test 'on' dtype mismatch
    df1 = pl.DataFrame(
        {"a": pl.Series([1, 2, 3], dtype=pl.Int64), "b": ["a", "b", "c"]}
    )
    df2 = pl.DataFrame(
        {"a": pl.Series([1, 2, 3], dtype=pl.Int32), "c": ["d", "e", "f"]}
    )

    with pytest.raises(
        pl.exceptions.ComputeError, match="datatypes of join keys don't match"
    ):
        df1.join_asof(df2, on="a", strategy="forward")

    # test 'by' dtype mismatch
    df1 = pl.DataFrame(
        {
            "time": pl.date_range(date(2018, 1, 1), date(2018, 1, 8), eager=True),
            "group": pl.Series([1, 1, 1, 1, 2, 2, 2, 2], dtype=pl.Int32),
            "value": [0, 0, None, None, 2, None, 1, None],
        }
    )
    df2 = pl.DataFrame(
        {
            "time": pl.date_range(date(2018, 1, 1), date(2018, 1, 8), eager=True),
            "group": pl.Series([1, 1, 1, 1, 2, 2, 2, 2], dtype=pl.Int64),
            "value": [0, 0, None, None, 2, None, 1, None],
        }
    )

    with pytest.raises(
        pl.exceptions.ComputeError, match="mismatching dtypes in 'by' parameter"
    ):
        df1.join_asof(df2, on="time", by="group", strategy="forward")


def test_join_asof_floats() -> None:
    df1 = pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": ["lrow1", "lrow2", "lrow3"]})
    df2 = pl.DataFrame({"a": [0.59, 1.49, 2.89], "b": ["rrow1", "rrow2", "rrow3"]})

    result = df1.join_asof(df2, on=pl.col("a").set_sorted(), strategy="backward")
    expected = {
        "a": [1.0, 2.0, 3.0],
        "b": ["lrow1", "lrow2", "lrow3"],
        "a_right": [0.59, 1.49, 2.89],
        "b_right": ["rrow1", "rrow2", "rrow3"],
    }
    assert result.to_dict(as_series=False) == expected

    # with by argument
    # 5740
    df1 = pl.DataFrame(
        {"b": np.linspace(0, 5, 7), "c": ["x" if i < 4 else "y" for i in range(7)]}
    )
    df2 = pl.DataFrame(
        {
            "val": [0.0, 2.5, 2.6, 2.7, 3.4, 4.0, 5.0],
            "c": ["x", "x", "x", "y", "y", "y", "y"],
        }
    ).with_columns(pl.col("val").alias("b").set_sorted())
    assert df1.set_sorted("b").join_asof(df2, on=pl.col("b"), by="c").to_dict(
        as_series=False
    ) == {
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
    ).to_dict(as_series=False) == {
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
    ).to_dict(as_series=False) == {
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
    ).to_dict(as_series=False) == {
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
    ).to_dict(as_series=False) == {
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
    ).to_dict(as_series=False) == {
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
    ).collect().to_dict(as_series=False) == {
        "df2_date": [None, 20221012, 20221012, 20221012, 20221015],
        "df1_date": [20221011, 20221012, 20221013, 20221014, 20221016],
    }
    assert (
        df1.lazy().join_asof(
            df2.lazy(), by="key", left_on="df1_date", right_on="df2_date"
        )
    ).select(["df2_date", "df1_date"]).collect().to_dict(as_series=False) == {
        "df2_date": [None, None, None, 20221012, 20221015],
        "df1_date": [20221011, 20221012, 20221013, 20221014, 20221016],
    }


def test_asof_join_by_logical_types() -> None:
    dates = (
        pl.datetime_range(
            datetime(2022, 1, 1), datetime(2022, 1, 2), interval="2h", eager=True
        )
        .cast(pl.Datetime("ns"))
        .head(9)
    )
    x = pl.DataFrame({"a": dates, "b": map(float, range(9)), "c": ["1", "2", "3"] * 3})

    result = x.join_asof(x, on=pl.col("b").set_sorted(), by=["c", "a"])

    expected = {
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
        "b_right": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    }
    assert result.to_dict(as_series=False) == expected


def test_join_asof_projection_7481() -> None:
    ldf1 = pl.DataFrame({"a": [1, 2, 2], "b": "bleft"}).lazy().set_sorted("a")
    ldf2 = pl.DataFrame({"a": 2, "b": [1, 2, 2]}).lazy().set_sorted("b")

    assert (
        ldf1.join_asof(ldf2, left_on="a", right_on="b").select("a", "b")
    ).collect().to_dict(as_series=False) == {
        "a": [1, 2, 2],
        "b": ["bleft", "bleft", "bleft"],
    }


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
            pl.Series("key", ["a", "a", "a", "b", "b", "b"], dtype=pl.String),
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
    # Generic join_asof
    df1 = pl.DataFrame(
        {
            "asof_key": [-1, 1, 2, 4, 6],
            "a": [1, 2, 3, 4, 5],
        }
    ).sort(by="asof_key")

    df2 = pl.DataFrame(
        {
            "asof_key": [-1, 2, 4, 5],
            "b": [1, 2, 3, 4],
        }
    ).sort(by="asof_key")

    expected = pl.DataFrame(
        {"asof_key": [-1, 1, 2, 4, 6], "a": [1, 2, 3, 4, 5], "b": [1, 2, 2, 3, 4]}
    )

    out = df1.join_asof(df2, on="asof_key", strategy="nearest")
    assert_frame_equal(out, expected)

    # Edge case: last item of right matches multiples on left
    df1 = pl.DataFrame(
        {
            "asof_key": [9, 9, 10, 10, 10],
            "a": [1, 2, 3, 4, 5],
        }
    ).set_sorted("asof_key")
    df2 = pl.DataFrame(
        {
            "asof_key": [1, 2, 3, 10],
            "b": [1, 2, 3, 4],
        }
    ).set_sorted("asof_key")
    expected = pl.DataFrame(
        {
            "asof_key": [9, 9, 10, 10, 10],
            "a": [1, 2, 3, 4, 5],
            "b": [4, 4, 4, 4, 4],
        }
    )

    out = df1.join_asof(df2, on="asof_key", strategy="nearest")
    assert_frame_equal(out, expected)


def test_asof_join_nearest_with_tolerance() -> None:
    a = b = [1, 2, 3, 4, 5]

    nones = pl.Series([None, None, None, None, None], dtype=pl.Int64)

    # Case 1: complete miss
    df1 = pl.DataFrame({"asof_key": [1, 2, 3, 4, 5], "a": a}).set_sorted("asof_key")
    df2 = pl.DataFrame(
        {
            "asof_key": [7, 8, 9, 10, 11],
            "b": b,
        }
    ).set_sorted("asof_key")
    expected = df1.with_columns(nones.alias("b"))
    out = df1.join_asof(df2, on="asof_key", strategy="nearest", tolerance=1)
    assert_frame_equal(out, expected)

    # Case 2: complete miss in other direction
    df1 = pl.DataFrame({"asof_key": [7, 8, 9, 10, 11], "a": a}).set_sorted("asof_key")
    df2 = pl.DataFrame(
        {
            "asof_key": [1, 2, 3, 4, 5],
            "b": b,
        }
    ).set_sorted("asof_key")
    expected = df1.with_columns(nones.alias("b"))
    out = df1.join_asof(df2, on="asof_key", strategy="nearest", tolerance=1)
    assert_frame_equal(out, expected)

    # Case 3: match first item
    df1 = pl.DataFrame({"asof_key": [1, 2, 3, 4, 5], "a": a}).set_sorted("asof_key")
    df2 = pl.DataFrame(
        {
            "asof_key": [6, 7, 8, 9, 10],
            "b": b,
        }
    ).set_sorted("asof_key")
    out = df1.join_asof(df2, on="asof_key", strategy="nearest", tolerance=1)
    expected = df1.with_columns(pl.Series([None, None, None, None, 1]).alias("b"))
    assert_frame_equal(out, expected)

    # Case 4: match last item
    df1 = pl.DataFrame({"asof_key": [1, 2, 3, 4, 5], "a": a}).set_sorted("asof_key")
    df2 = pl.DataFrame(
        {
            "asof_key": [-4, -3, -2, -1, 0],
            "b": b,
        }
    ).set_sorted("asof_key")
    out = df1.join_asof(df2, on="asof_key", strategy="nearest", tolerance=1)
    expected = df1.with_columns(pl.Series([5, None, None, None, None]).alias("b"))
    assert_frame_equal(out, expected)

    # Case 5: match multiples, pick closer
    df1 = pl.DataFrame(
        {"asof_key": pl.Series([1, 2, 3, 4, 5], dtype=pl.Float64), "a": a}
    ).set_sorted("asof_key")
    df2 = pl.DataFrame(
        {
            "asof_key": [0.0, 2.0, 2.4, 3.4, 10.0],
            "b": b,
        }
    ).set_sorted("asof_key")
    out = df1.join_asof(df2, on="asof_key", strategy="nearest", tolerance=1)
    expected = df1.with_columns(pl.Series([2, 2, 4, 4, None]).alias("b"))
    assert_frame_equal(out, expected)

    # Case 6: use 0 tolerance
    df1 = pl.DataFrame(
        {"asof_key": pl.Series([1, 2, 3, 4, 5], dtype=pl.Float64), "a": a}
    ).set_sorted("asof_key")
    df2 = pl.DataFrame(
        {
            "asof_key": [0.0, 2.0, 2.4, 3.4, 10.0],
            "b": b,
        }
    ).set_sorted("asof_key")
    out = df1.join_asof(df2, on="asof_key", strategy="nearest", tolerance=0)
    expected = df1.with_columns(pl.Series([None, 2, None, None, None]).alias("b"))
    assert_frame_equal(out, expected)

    # Case 7: test with datetime
    df1 = pl.DataFrame(
        {
            "asof_key": pl.Series(
                [
                    datetime(2023, 1, 1),
                    datetime(2023, 1, 2),
                    datetime(2023, 1, 3),
                    datetime(2023, 1, 4),
                    datetime(2023, 1, 6),
                ]
            ),
            "a": a,
        }
    ).set_sorted("asof_key")
    df2 = pl.DataFrame(
        {
            "asof_key": pl.Series(
                [
                    datetime(2022, 1, 1),
                    datetime(2022, 1, 2),
                    datetime(2022, 1, 3),
                    datetime(
                        2023, 1, 2, 21, 30, 0
                    ),  # should match with 2023-01-02, 2023-01-03, and 2021-01-04
                    datetime(2023, 1, 7),
                ]
            ),
            "b": b,
        }
    ).set_sorted("asof_key")
    out = df1.join_asof(df2, on="asof_key", strategy="nearest", tolerance="1d4h")
    expected = df1.with_columns(pl.Series([None, 4, 4, 4, 5]).alias("b"))
    assert_frame_equal(out, expected)

    # Case 8: test using timedelta tolerance
    out = df1.join_asof(
        df2, on="asof_key", strategy="nearest", tolerance=timedelta(days=1, hours=4)
    )
    assert_frame_equal(out, expected)

    # Case #9: last item is closest match
    df1 = pl.DataFrame(
        {
            "asof_key_left": [10.00001, 20.0, 30.0],
        }
    ).set_sorted("asof_key_left")
    df2 = pl.DataFrame(
        {
            "asof_key_right": [10.00001, 20.0001, 29.0],
        }
    ).set_sorted("asof_key_right")
    out = df1.join_asof(
        df2,
        left_on="asof_key_left",
        right_on="asof_key_right",
        strategy="nearest",
        tolerance=0.5,
    )
    expected = pl.DataFrame(
        {
            "asof_key_left": [10.00001, 20.0, 30.0],
            "asof_key_right": [10.00001, 20.0001, None],
        }
    )
    assert_frame_equal(out, expected)


def test_asof_join_nearest_by() -> None:
    # Generic join_asof
    df1 = pl.DataFrame(
        {
            "asof_key": [-1, 1, 2, 6, 1],
            "group": [1, 1, 1, 2, 2],
            "a": [1, 2, 3, 2, 5],
        }
    ).sort(by=["group", "asof_key"])

    df2 = pl.DataFrame(
        {
            "asof_key": [-1, 2, 5, 1],
            "group": [1, 1, 2, 2],
            "b": [1, 2, 3, 4],
        }
    ).sort(by=["group", "asof_key"])

    expected = pl.DataFrame(
        {
            "asof_key": [-1, 1, 2, 6, 1],
            "group": [1, 1, 1, 2, 2],
            "a": [1, 2, 3, 5, 2],
            "b": [1, 2, 2, 4, 3],
        }
    ).sort(by=["group", "asof_key"])

    # Edge case: last item of right matches multiples on left
    df1 = pl.DataFrame(
        {
            "asof_key": [9, 9, 10, 10, 10],
            "group": [1, 1, 1, 2, 2],
            "a": [1, 2, 3, 2, 5],
        }
    ).sort(by=["group", "asof_key"])

    df2 = pl.DataFrame(
        {
            "asof_key": [-1, 1, 1, 10],
            "group": [1, 1, 2, 2],
            "b": [1, 2, 3, 4],
        }
    ).sort(by=["group", "asof_key"])

    expected = pl.DataFrame(
        {
            "asof_key": [9, 9, 10, 10, 10],
            "group": [1, 1, 1, 2, 2],
            "a": [1, 2, 3, 2, 5],
            "b": [2, 2, 2, 4, 4],
        }
    )

    out = df1.join_asof(df2, on="asof_key", by="group", strategy="nearest")
    assert_frame_equal(out, expected)

    a = pl.DataFrame(
        {
            "code": [676, 35, 676, 676, 676],
            "time": [364360, 364370, 364380, 365400, 367440],
        }
    )
    b = pl.DataFrame(
        {
            "code": [676, 676, 35, 676, 676],
            "time": [364000, 365000, 365000, 366000, 367000],
            "price": [1.0, 2.0, 50, 3.0, None],
        }
    )

    expected = pl.DataFrame(
        {
            "code": [676, 35, 676, 676, 676],
            "time": [364360, 364370, 364380, 365400, 367440],
            "price": [1.0, 50.0, 1.0, 2.0, None],
        }
    )

    out = a.join_asof(b, by="code", on="time", strategy="nearest")
    assert_frame_equal(out, expected)

    # last item is closest match
    df1 = pl.DataFrame(
        {
            "a": [1, 1, 1],
            "asof_key_left": [10.00001, 20.0, 30.0],
        }
    ).set_sorted("asof_key_left")
    df2 = pl.DataFrame(
        {
            "a": [1, 1, 1],
            "asof_key_right": [10.00001, 20.0001, 29.0],
        }
    ).set_sorted("asof_key_right")
    out = df1.join_asof(
        df2,
        left_on="asof_key_left",
        right_on="asof_key_right",
        by="a",
        strategy="nearest",
    )
    expected = pl.DataFrame(
        {
            "a": [1, 1, 1],
            "asof_key_left": [10.00001, 20.0, 30.0],
            "asof_key_right": [10.00001, 20.0001, 29.0],
        }
    )
    assert_frame_equal(out, expected)


def test_asof_join_nearest_by_with_tolerance() -> None:
    df1 = pl.DataFrame(
        {
            "group": [
                1,
                1,
                1,
                1,
                1,
                2,
                2,
                2,
                2,
                2,
                3,
                3,
                3,
                3,
                3,
                4,
                4,
                4,
                4,
                4,
                5,
                5,
                5,
                5,
                5,
                6,
                6,
                6,
                6,
                6,
            ],
            "asof_key": pl.Series(
                [
                    1,
                    2,
                    3,
                    4,
                    5,
                    7,
                    8,
                    9,
                    10,
                    11,
                    1,
                    2,
                    3,
                    4,
                    5,
                    1,
                    2,
                    3,
                    4,
                    5,
                    1,
                    2,
                    3,
                    4,
                    5,
                    1,
                    2,
                    3,
                    4,
                    5,
                ],
                dtype=pl.Float32,
            ),
            "a": [
                1,
                2,
                3,
                4,
                5,
                1,
                2,
                3,
                4,
                5,
                1,
                2,
                3,
                4,
                5,
                1,
                2,
                3,
                4,
                5,
                1,
                2,
                3,
                4,
                5,
                1,
                2,
                3,
                4,
                5,
            ],
        }
    )

    df2 = pl.DataFrame(
        {
            "group": [
                1,
                1,
                1,
                1,
                1,
                2,
                2,
                2,
                2,
                2,
                3,
                3,
                3,
                3,
                3,
                4,
                4,
                4,
                4,
                4,
                5,
                5,
                5,
                5,
                5,
                6,
                6,
                6,
                6,
                6,
            ],
            "asof_key": pl.Series(
                [
                    7,
                    8,
                    9,
                    10,
                    11,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    5,
                    -3,
                    -2,
                    -1,
                    0,
                    0,
                    2,
                    2.4,
                    3.4,
                    10,
                    -3,
                    3,
                    8,
                    9,
                    10,
                ],
                dtype=pl.Float32,
            ),
            "b": [
                1,
                2,
                3,
                4,
                5,
                1,
                2,
                3,
                4,
                5,
                1,
                2,
                3,
                4,
                5,
                1,
                2,
                3,
                4,
                5,
                1,
                2,
                3,
                4,
                5,
                1,
                2,
                3,
                4,
                5,
            ],
        }
    )

    expected = df1.with_columns(
        pl.Series(
            [
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                1,
                5,
                None,
                None,
                1,
                1,
                2,
                2,
                4,
                4,
                None,
                None,
                2,
                2,
                2,
                None,
            ]
        ).alias("b")
    )
    df1 = df1.sort(by=["group", "asof_key"])
    df2 = df2.sort(by=["group", "asof_key"])
    expected = expected.sort(by=["group", "a"])

    out = df1.join_asof(
        df2, by="group", on="asof_key", strategy="nearest", tolerance=1.0
    ).sort(by=["group", "a"])
    assert_frame_equal(out, expected)

    # last item is closest match
    df1 = pl.DataFrame(
        {
            "a": [1, 1, 1],
            "asof_key_left": [10.00001, 20.0, 30.0],
        }
    ).set_sorted("asof_key_left")
    df2 = pl.DataFrame(
        {
            "a": [1, 1, 1],
            "asof_key_right": [10.00001, 20.0001, 29.0],
        }
    ).set_sorted("asof_key_right")
    out = df1.join_asof(
        df2,
        left_on="asof_key_left",
        right_on="asof_key_right",
        by="a",
        strategy="nearest",
        tolerance=0.5,
    )
    expected = pl.DataFrame(
        {
            "a": [1, 1, 1],
            "asof_key_left": [10.00001, 20.0, 30.0],
            "asof_key_right": [10.00001, 20.0001, None],
        }
    )
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


def test_asof_join_string() -> None:
    left = pl.DataFrame({"x": [None, "a", "b", "c", None, "d", None]}).set_sorted("x")
    right = pl.DataFrame({"x": ["apple", None, "chutney"], "y": [0, 1, 2]}).set_sorted(
        "x"
    )
    forward = left.join_asof(right, on="x", strategy="forward")
    backward = left.join_asof(right, on="x", strategy="backward")
    forward_expected = pl.DataFrame(
        {
            "x": [None, "a", "b", "c", None, "d", None],
            "y": [None, 0, 2, 2, None, None, None],
        }
    )
    backward_expected = pl.DataFrame(
        {
            "x": [None, "a", "b", "c", None, "d", None],
            "y": [None, None, 0, 0, None, 2, None],
        }
    )
    assert_frame_equal(forward, forward_expected)
    assert_frame_equal(backward, backward_expected)


def test_join_asof_by_argument_parsing() -> None:
    df1 = pl.DataFrame(
        {
            "n": [10, 20, 30, 40, 50, 60],
            "id1": [0, 0, 3, 3, 5, 5],
            "id2": [1, 2, 1, 2, 1, 2],
            "x": ["a", "b", "c", "d", "e", "f"],
        }
    ).sort(by="n")

    df2 = pl.DataFrame(
        {
            "n": [25, 8, 5, 23, 15, 35],
            "id1": [0, 0, 3, 3, 5, 5],
            "id2": [1, 2, 1, 2, 1, 2],
            "y": ["A", "B", "C", "D", "E", "F"],
        }
    ).sort(by="n")

    # any sequency for by argument is allowed, so we should see the same results here
    by_list = df1.join_asof(df2, on="n", by=["id1", "id2"])
    by_tuple = df1.join_asof(df2, on="n", by=("id1", "id2"))
    assert_frame_equal(by_list, by_tuple)

    # same for using the by_left and by_right kwargs
    by_list2 = df1.join_asof(
        df2, on="n", by_left=["id1", "id2"], by_right=["id1", "id2"]
    )
    by_tuple2 = df1.join_asof(
        df2, on="n", by_left=("id1", "id2"), by_right=("id1", "id2")
    )
    assert_frame_equal(by_list2, by_list)
    assert_frame_equal(by_tuple2, by_list)


def test_join_asof_invalid_args() -> None:
    df1 = pl.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [1, 2, 3],
        }
    ).set_sorted("a")
    df2 = pl.DataFrame(
        {
            "a": [1, 2, 3],
            "c": [1, 2, 3],
        }
    ).set_sorted("a")

    with pytest.raises(TypeError, match="expected `on` to be str or Expr, got 'list'"):
        df1.join_asof(df2, on=["a"])  # type: ignore[arg-type]
    with pytest.raises(
        TypeError, match="expected `left_on` to be str or Expr, got 'list'"
    ):
        df1.join_asof(df2, left_on=["a"], right_on="a")  # type: ignore[arg-type]
    with pytest.raises(
        TypeError, match="expected `right_on` to be str or Expr, got 'list'"
    ):
        df1.join_asof(df2, left_on="a", right_on=["a"])  # type: ignore[arg-type]


def test_join_as_of_by_schema() -> None:
    a = pl.DataFrame({"a": [1], "b": [2], "c": [3]}).lazy()
    b = pl.DataFrame({"a": [1], "b": [2], "d": [4]}).lazy()
    q = a.join_asof(b, on=pl.col("a").set_sorted(), by="b")
    assert q.collect_schema().names() == q.collect().columns


def test_asof_join_by_schema() -> None:
    # different `by` names.
    df1 = pl.DataFrame({"on1": 0, "by1": 0})
    df2 = pl.DataFrame({"on1": 0, "by2": 0})

    q = df1.lazy().join_asof(
        df2.lazy(),
        on="on1",
        by_left="by1",
        by_right="by2",
    )

    assert q.collect_schema() == q.collect().schema
