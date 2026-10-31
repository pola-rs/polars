from __future__ import annotations

import functools
import io
import itertools
import math
import random
import warnings
from datetime import date, datetime, time, timedelta
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given

import polars as pl
import polars.selectors as cs
from polars.exceptions import DuplicateError, InvalidOperationError
from polars.testing import assert_frame_equal
from polars.testing.parametric.strategies.core import dataframes

if TYPE_CHECKING:
    from pathlib import Path

    from polars._typing import AsofJoinStrategy, EngineType, PolarsIntegerType
    from tests.conftest import PlMonkeyPatch


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
        {"a": pl.Series([1.0, 2.0, 3.0], dtype=pl.Float64), "c": ["d", "e", "f"]}
    )

    with pytest.raises(
        pl.exceptions.SchemaError, match="datatypes of join keys don't match"
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

    for invalid_tolerance, match in [
        ("foo", "expected leading integer"),
        ([None], "could not extract number"),
    ]:
        with pytest.raises(pl.exceptions.PolarsError, match=match):
            df_trades.join_asof(
                df_quotes,
                on="time",
                by="stock",
                tolerance=invalid_tolerance,  # type: ignore[arg-type]
            )


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


@pytest.mark.slow
@pytest.mark.parametrize("allow_exact_matches", [True, False])
def test_asof_join_nearest_reference(allow_exact_matches: bool) -> None:
    def asof_join_nearest_reference(
        df_left: pl.DataFrame, df_right: pl.DataFrame
    ) -> pl.DataFrame:
        schema = {"key": pl.Int32, "value": pl.Int32, "value_right": pl.Int32}
        result = pl.DataFrame(schema=schema)
        for left_row in df_left.iter_rows():
            cross_product_sorted = (
                pl.DataFrame(
                    {
                        "key": pl.Series([left_row[0]] * len(df_right), dtype=pl.Int32),
                        "value": pl.Series(
                            [left_row[1]] * len(df_right), dtype=pl.Int32
                        ),
                        "key_right": df_right["key"],
                        "value_right": df_right["value"],
                    },
                )
                .with_row_index()
                .filter(
                    pl.when(allow_exact_matches)
                    .then(pl.lit(True))
                    .otherwise(pl.col("key") != pl.col("key_right"))
                )
                .sort(
                    (pl.col("key") - pl.col("key_right")).abs(),
                    -pl.col("index").cast(pl.Int32),
                )
                .drop("index", "key_right")
            )
            if len(cross_product_sorted) == 0:
                result = result.vstack(
                    pl.DataFrame([left_row + (None,)], schema=schema, orient="row"),
                )
            else:
                best_match = cross_product_sorted[0]
                result = result.vstack(best_match)
        return result

    test_dfs = []
    rng = random.Random(0)
    for n_a, n_b, n_c, n_d in itertools.product([0, 1, 2], repeat=4):
        a = rng.randint(0, 10)
        b = rng.randint(0, 10)
        c = rng.randint(0, 10)
        d = rng.randint(0, 10)
        keys = [a] * n_a + [b] * n_b + [c] * n_c + [d] * n_d
        values = [rng.randint(0, 100000) for _ in keys]

        df = pl.DataFrame(
            {"key": keys, "value": values}, schema={"key": pl.Int32, "value": pl.Int32}
        ).sort(by="key")
        test_dfs.append(df)

    for df_left, df_right in itertools.product(test_dfs, repeat=2):
        expected = asof_join_nearest_reference(df_left, df_right)
        actual = df_left.join_asof(
            df_right,
            on="key",
            strategy="nearest",
            allow_exact_matches=allow_exact_matches,
        )
        assert_frame_equal(actual, expected)


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
    # These set_sorted() calls are invalid, so the code is technically incorrect.
    # However, this code used to work in the past and we'd like to know when it
    # breaks. (See also: #18927)
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


def test_join_asof_selector() -> None:
    data1 = {"time": [1, 2, 3], "value": [10, 20, 30]}
    left = pl.DataFrame(data1).set_sorted("time")

    data2 = {"time": [1, 3], "value": [100, 300]}
    right = pl.DataFrame(data2).set_sorted("time")

    expected = left.join_asof(right, on="time")
    assert_frame_equal(left.join_asof(right, on=cs.matches(r"^t\w+e$")), expected)

    res = left.lazy().join_asof(right.lazy(), on=cs.starts_with("t"))
    assert res.collect_schema() == expected.schema
    assert_frame_equal(res.collect(engine="in-memory"), expected)
    assert_frame_equal(res.collect(engine="streaming"), expected)


@pytest.mark.parametrize(
    ("selector", "match"),
    [
        (cs.float(), "left join keys expanded to zero expressions"),
        (cs.integer(), "expects exactly one join key.*got 2"),
    ],
)
def test_join_asof_selector_cardinality(selector: cs.Selector, match: str) -> None:
    left = pl.LazyFrame({"time": [1, 2], "left_value": [10, 20]}).set_sorted("time")
    right = pl.LazyFrame({"time": [1, 2], "right_value": [10, 20]}).set_sorted("time")

    with pytest.raises(InvalidOperationError, match=match):
        left.join_asof(right, on=selector).collect_schema()


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


def test_raise_invalid_by_arg_13020() -> None:
    df1 = pl.DataFrame({"asOfDate": [date(2020, 1, 1)]})
    df2 = pl.DataFrame(
        {
            "endityId": [date(2020, 1, 1)],
            "eventDate": ["A"],
        }
    )
    with pytest.raises(pl.exceptions.InvalidOperationError, match="expected both"):
        df1.sort("asOfDate").join_asof(
            df2.sort("eventDate"),
            left_on="asOfDate",
            right_on="eventDate",
            by_left=None,
            by_right=["entityId"],
        )


def test_join_asof_no_exact_matches() -> None:
    trades = pl.DataFrame(
        {
            "time": [
                "2016-05-25 13:30:00.023",
                "2016-05-25 13:30:00.038",
                "2016-05-25 13:30:00.048",
                "2016-05-25 13:30:00.048",
                "2016-05-25 13:30:00.048",
            ],
            "ticker": ["MSFT", "MSFT", "GOOG", "GOOG", "AAPL"],
            "price": [51.95, 51.95, 720.77, 720.92, 98.0],
            "quantity": [75, 155, 100, 100, 100],
        }
    ).with_columns(pl.col("time").str.to_datetime())

    quotes = pl.DataFrame(
        {
            "time": [
                "2016-05-25 13:30:00.023",
                "2016-05-25 13:30:00.023",
                "2016-05-25 13:30:00.030",
                "2016-05-25 13:30:00.041",
                "2016-05-25 13:30:00.048",
                "2016-05-25 13:30:00.049",
                "2016-05-25 13:30:00.072",
                "2016-05-25 13:30:00.075",
            ],
            "ticker": ["GOOG", "MSFT", "MSFT", "MSFT", "GOOG", "AAPL", "GOOG", "MSFT"],
            "bid": [720.50, 51.95, 51.97, 51.99, 720.50, 97.99, 720.50, 52.01],
            "ask": [720.93, 51.96, 51.98, 52.00, 720.93, 98.01, 720.88, 52.03],
        }
    ).with_columns(pl.col("time").str.to_datetime())

    assert trades.join_asof(
        quotes, on="time", by="ticker", tolerance="10ms", allow_exact_matches=False
    ).to_dict(as_series=False) == {
        "time": [
            datetime(2016, 5, 25, 13, 30, 0, 23000),
            datetime(2016, 5, 25, 13, 30, 0, 38000),
            datetime(2016, 5, 25, 13, 30, 0, 48000),
            datetime(2016, 5, 25, 13, 30, 0, 48000),
            datetime(2016, 5, 25, 13, 30, 0, 48000),
        ],
        "ticker": ["MSFT", "MSFT", "GOOG", "GOOG", "AAPL"],
        "price": [51.95, 51.95, 720.77, 720.92, 98.0],
        "quantity": [75, 155, 100, 100, 100],
        "bid": [None, 51.97, None, None, None],
        "ask": [None, 51.98, None, None, None],
    }


@pytest.mark.parametrize("strategy", ["backward", "forward", "nearest"])
@given(
    df_left=dataframes(cols=1, allowed_dtypes=pl.Int32),
    df_right=dataframes(cols=1, allowed_dtypes=pl.Int32),
)
def test_join_asof_no_exact_matches_parametric(
    strategy: AsofJoinStrategy, df_left: pl.DataFrame, df_right: pl.DataFrame
) -> None:
    df_left = df_left.sort("col0")
    df_right = df_right.sort("col0")

    out = df_left.join_asof(
        df_right,
        on="col0",
        strategy=strategy,
        suffix="_right",
        coalesce=False,
        allow_exact_matches=False,
    )

    for l_val, r_val in zip(
        out["col0"],
        out["col0_right"],
        strict=False,
    ):
        assert l_val != r_val or r_val is None


def test_join_asof_not_sorted() -> None:
    df = pl.DataFrame({"a": [1, 1, 1, 2, 2, 2], "b": [2, 1, 3, 1, 2, 3]})
    with pytest.raises(
        InvalidOperationError,
        match="argument in operation 'asof_join' is not sorted, please sort the 'expr/series/column' first",
    ):
        df.join_asof(df, on="b")

    # When 'by' is provided, we do not check sortedness, but a warning is received
    with pytest.warns(
        UserWarning,
        match="Sortedness of columns cannot be checked when 'by' groups provided",
    ):
        df.join_asof(df, on="b", by="a")

    # When sortedness is False, we should get no warning
    with warnings.catch_warnings(record=True) as w:
        df.join_asof(df, on="b", check_sortedness=False)
        df.join_asof(df, on="b", by="a", check_sortedness=False)
        assert len(w) == 0  # no warnings caught


@pytest.mark.parametrize("swap", [False, True])
def test_join_asof_not_sorted_streaming_27457(
    swap: bool, plmonkeypatch: PlMonkeyPatch
) -> None:
    plmonkeypatch.setenv("POLARS_IDEAL_MORSEL_SIZE", "3")
    for n in range(4, 10):
        df = pl.DataFrame({"a": (10 * [1, 2, 3])[:n]})
        df2 = df.sort("a")
        if swap:
            (df, df2) = df2, df
        with pytest.raises(
            InvalidOperationError,
            match="argument in operation 'asof_join' is not sorted, please sort the 'expr/series/column' first",
        ):
            df.lazy().join_asof(df2.lazy(), on="a").collect(engine="streaming")


@pytest.mark.parametrize("swap", [False, True])
def test_join_asof_not_sorted_streaming_grouped_27457(
    swap: bool, plmonkeypatch: PlMonkeyPatch
) -> None:
    plmonkeypatch.setenv("POLARS_IDEAL_MORSEL_SIZE", "3")
    for n in range(4, 10):
        df = pl.DataFrame({"group": (10 * [1, 2, 3])[:n]})
        df2 = df.sort("group")
        lf = df.lazy().set_sorted("group").with_row_index("a")  # Deliberately invalid
        lf2 = df2.lazy().set_sorted("group").with_row_index("a")
        if swap:
            (lf, lf2) = lf2, lf
        with pytest.raises(
            InvalidOperationError,
            match="argument in operation 'asof_join' is not sorted, please sort the 'expr/series/column' first",
        ):
            lf.join_asof(lf2, on="a", by="group").collect(engine="streaming")


@pytest.mark.parametrize(
    ("nulls_last", "groups", "expected_v"),
    [
        (False, [None, 0, 1], [None, 8, 9]),
        (True, [0, 1, None], [7, 8, None]),
    ],
)
def test_join_asof_streaming_by_set_sorted_28538(
    nulls_last: bool, groups: list[int | None], expected_v: list[int | None]
) -> None:
    left = pl.LazyFrame({"group": groups, "time": [1, 1, 1]}).with_columns(
        pl.col("group").set_sorted(nulls_last=nulls_last)
    )
    right = pl.LazyFrame(
        {"group": groups, "time": [0, 0, 0], "v": [7, 8, 9]}
    ).with_columns(pl.col("group").set_sorted(nulls_last=nulls_last))
    out = left.join_asof(right, on="time", by="group").collect()
    expected = pl.DataFrame({"group": groups, "time": [1, 1, 1], "v": expected_v})
    assert_frame_equal(out, expected)


@pytest.mark.parametrize(
    "dtypes",
    [
        (pl.Int64, pl.Int64),
        (pl.Int64, pl.UInt64),
        (pl.Int64, pl.Int128),
        (pl.UInt64, pl.Int64),
        (pl.UInt64, pl.UInt64),
        (pl.UInt64, pl.Int128),
        (pl.UInt64, pl.UInt128),
        (pl.Int128, pl.Int64),
        (pl.Int128, pl.UInt64),
        (pl.Int128, pl.Int128),
        (pl.UInt128, pl.UInt64),
        (pl.UInt128, pl.UInt128),
    ],
)
@pytest.mark.parametrize("swap", [False, True])
@pytest.mark.parametrize("strategy", ["backward", "forward", "nearest"])
def test_join_asof_large_int_21276(
    dtypes: tuple[PolarsIntegerType, PolarsIntegerType],
    swap: bool,
    strategy: AsofJoinStrategy,
) -> None:
    left_dtype, right_dtype = reversed(dtypes) if swap else dtypes
    large_int64 = 1608129000134000123  # it only happen when "on" column is large
    left = pl.DataFrame({"ts": pl.Series([large_int64 + 2], dtype=left_dtype)})
    right = pl.DataFrame(
        {
            "ts": pl.Series([large_int64 + 1, large_int64 + 3], dtype=right_dtype),
            "value": [111, 333],
        }
    )
    result = left.join_asof(right, on="ts", strategy=strategy)
    idx = 0 if strategy == "backward" else 1
    expected = pl.DataFrame(
        {
            "ts": left["ts"],
            "value": right["value"].gather(idx),
        }
    )
    assert_frame_equal(result, expected)


@pytest.mark.parametrize("by", ["constant", None])
def test_join_asof_slice_23583(by: str | None) -> None:
    lhs = pl.LazyFrame(
        {
            "index": [0],
            "constant": 0,
            "date": [date(2025, 1, 1)],
        },
    ).set_sorted("date")

    rhs = pl.LazyFrame(
        {
            "index": [0, 1],
            "constant": 0,
            "date": [date(1970, 1, 1), date(2025, 1, 1)],
        },
    ).set_sorted("date")

    q = (
        lhs.join_asof(rhs, on="date", by=by, check_sortedness=False)
        .head(1)
        .select(pl.exclude("constant_right"))
    )

    expect = pl.DataFrame(
        {
            "index": [0],
            "constant": 0,
            "date": [date(2025, 1, 1)],
            "index_right": [1],
        },
    )

    assert_frame_equal(q.collect(optimizations=pl.QueryOptFlags.none()), expect)
    assert_frame_equal(q.collect(), expect)


def test_join_asof_23751() -> None:
    a = pl.DataFrame(
        [
            pl.Series([1, 2, 3, 4, 5]).alias("index") * int(1e10),
            pl.Series([1, -1, 1, 1, -1]).alias("side"),
        ]
    )

    b = pl.DataFrame(
        [
            pl.Series([0, 1, 1, 3, 3, 5]).alias("index_right").cast(pl.UInt64)
            * int(1e10),
            pl.Series([-1, 1, -1, 1, 1, -1]).alias("side"),
            pl.Series([0, 10, 20, 30, 40, 50]).alias("value"),
        ]
    )

    assert a.join_asof(b, left_on="index", right_on="index_right", by="side").to_dict(
        as_series=False
    ) == {
        "index": [10000000000, 20000000000, 30000000000, 40000000000, 50000000000],
        "side": [1, -1, 1, 1, -1],
        "index_right": [
            10000000000,
            10000000000,
            30000000000,
            30000000000,
            50000000000,
        ],
        "value": [10, 20, 40, 40, 50],
    }


def test_join_asof_nosuffix_dup_col_23834() -> None:
    a = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    b = pl.DataFrame({"b": [1, 2, 3], "c": [9, 10, 11]})
    with pytest.raises(DuplicateError):
        a.join_asof(b, left_on="a", right_on="b", suffix="")


def test_join_asof_planner_schema_24000() -> None:
    a = pl.DataFrame([pl.Series("index", [1, 2, 3]) * 10])
    b = pl.DataFrame(
        [
            pl.Series("value", [10, 20, 30]),
            pl.Series("index_right", [1, 2, 3]).cast(pl.UInt64) * 10,
        ]
    )
    q = a.lazy().join_asof(b.lazy(), left_on="index", right_on="index_right")

    assert q.collect().schema == q.collect_schema()

    b = pl.DataFrame(
        [
            pl.Series("index_right", [1, 2, 3]).cast(pl.UInt64) * 10,
            pl.Series("value", [10, 20, 30]),
        ]
    )
    q = a.lazy().join_asof(b.lazy(), left_on="index", right_on="index_right")

    assert q.collect().schema == q.collect_schema()


@pytest.mark.parametrize(
    "dtype",
    [pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, pl.Int8, pl.Int16, pl.Int32, pl.Int64],
)
def test_join_asof_int_dtypes_24383(dtype: PolarsIntegerType) -> None:
    lf1 = pl.LazyFrame(
        {
            "id": pl.Series([1], dtype=dtype),
            "date": pl.Series([date(2025, 12, 31)], dtype=pl.Date),
        }
    )

    lf2 = pl.LazyFrame(
        {
            "id": pl.Series([1], dtype=dtype),
            "date": pl.Series([date(2025, 12, 31)], dtype=pl.Date),
            "value": pl.Series([2.5], dtype=pl.Float32),
        }
    )

    result = lf1.join_asof(
        other=lf2,
        on="date",
        by="id",
        check_sortedness=False,
    )
    expected = pl.DataFrame(
        {
            "id": pl.Series([1], dtype=dtype),
            "date": pl.Series([date(2025, 12, 31)], dtype=pl.Date),
            "value": pl.Series([2.5], dtype=pl.Float32),
        }
    )
    assert result.collect_schema() == expected.schema
    assert_frame_equal(result.collect(), expected)


@pytest.mark.parametrize("by", [None, "one"])
def test_join_asof_on_cast_expr_24999(by: str | None) -> None:
    q = pl.LazyFrame({"x": date(2025, 10, 13), "one": 1}).join_asof(
        pl.LazyFrame({"x_right": datetime(2025, 10, 10, 2), "one": 1}),
        left_on=pl.col("x").cast(pl.Datetime),
        right_on=pl.col("x_right"),
        tolerance=timedelta(days=100),
        by=by,
    )

    expect = pl.DataFrame(
        {
            "x": date(2025, 10, 13),
            "one": 1,
            "x_right": datetime(2025, 10, 10, 2),
        }
    )

    if by is None:
        expect = expect.with_columns(pl.col("one").alias("one_right"))

    assert_frame_equal(q.collect(), expect)


@pytest.mark.parametrize(
    ("left", "right", "expected_right"),
    [
        ([], [], []),
        ([1], [2], [2]),
        ([], [1, 2], []),
        ([1, 2], [], [None, None]),
        ([1, 2], [1, 2], [2, 1]),
        ([1, 1], [1, 1], [None, None]),
        ([1, 2, 2, 2, 3], [1, 2, 2, 2, 3], [2, 3, 3, 3, 2]),
    ],
)
def test_join_asof_nearest_no_exact_matches_25468(
    left: list[int],
    right: list[int],
    expected_right: list[int],
) -> None:
    df_left = pl.DataFrame({"a": left}, schema={"a": pl.Int32})
    df_right = pl.DataFrame({"a": right}, schema={"a": pl.Int32})
    expected_df = pl.DataFrame(
        {"a": left, "a_right": expected_right},
        schema={"a": pl.Int32, "a_right": pl.Int32},
    )
    result = df_left.join_asof(
        df_right, on="a", strategy="nearest", coalesce=False, allow_exact_matches=False
    )
    assert_frame_equal(result, expected_df)


@pytest.mark.parametrize("strategy", ["backward", "nearest"])
def test_join_asof_nans(strategy: AsofJoinStrategy) -> None:
    df_left = pl.LazyFrame(
        {
            "time": [0.0, 6.0, math.nan, math.nan],
            "value": [100, 101, 102, 103],
        }
    )
    df_right = pl.LazyFrame(
        {
            "time": [0.0, 5.0, 10.0, 42.0],
            "value": [100, 200, 300, 400],
        }
    )
    actual = df_left.join_asof(
        df_right,
        on="time",
        strategy=strategy,
    ).collect()
    expected = pl.DataFrame(
        {
            "time": [0.0, 6.0, math.nan, math.nan],
            "value": [100, 101, 102, 103],
            "value_right": [100, 200, 400, 400],
        }
    )
    assert_frame_equal(actual, expected)


@pytest.mark.filterwarnings(
    "ignore:Sortedness of columns cannot be checked when 'by' groups provided"
)
@pytest.mark.parametrize("strategy", ["backward", "forward", "nearest"])
@pytest.mark.parametrize("allow_exact_matches", [False, True])
@given(
    df_left=dataframes(min_cols=1, max_cols=1, allowed_dtypes=[pl.Int32]),
    df_right=dataframes(min_cols=1, max_cols=1, allowed_dtypes=[pl.Int32]),
    data=st.data(),
)
def test_join_asof_group_matches_nogroup(
    df_left: pl.DataFrame,
    df_right: pl.DataFrame,
    strategy: AsofJoinStrategy,
    allow_exact_matches: bool,
    data: st.DataObject,
) -> None:
    def groups(n: int) -> st.SearchStrategy[list[int | None]]:
        return st.lists(st.sampled_from([None, 0]), min_size=n, max_size=n)

    df_left = df_left.rename({"col0": "key"}).sort("key")
    groups_left = pl.Series(data.draw(groups(len(df_left))), dtype=pl.Int32)
    df_left = df_left.with_columns(group=groups_left).with_row_index()
    df_right = df_right.rename({"col0": "key"}).sort("key")
    groups_right = pl.Series(data.draw(groups(len(df_right))), dtype=pl.Int32)
    df_right = df_right.with_columns(group=groups_right).with_row_index()

    q_grouped = df_left.lazy().join_asof(
        df_right.lazy(),
        on="key",
        by="group",
        strategy=strategy,
        allow_exact_matches=allow_exact_matches,
    )
    q_ungrouped = (
        df_left.join_asof(
            df_right.filter(pl.col("group").is_not_null()),
            on="key",
            strategy=strategy,
            allow_exact_matches=allow_exact_matches,
        )
        .drop("group_right")
        .with_columns(
            pl.when(pl.col("group").is_not_null())
            .then(pl.col("index_right"))
            .alias("index_right")
        )
    )

    grouped = q_grouped.collect()
    ungrouped = q_ungrouped
    assert_frame_equal(grouped, ungrouped)


def test_join_asof_by_nulls_27165() -> None:
    dataframe = functools.partial(
        pl.DataFrame,
        schema_overrides={
            "group": pl.Int32,
            "index": pl.get_index_type(),
            "index_right": pl.get_index_type(),
        },
    )
    left = dataframe({"key": [1.0], "group": [0]}).with_row_index()
    right = dataframe({"key": [0.0], "group": [None]}).with_row_index()
    actual = left.join_asof(right, on="key", by="group", strategy="nearest")
    expected = dataframe(
        {"index": 0, "key": [1.0], "group": [0], "index_right": [None]}
    )
    assert_frame_equal(actual, expected)


def test_join_asof_by_nulls_27165_2() -> None:
    dataframe = functools.partial(
        pl.DataFrame,
        schema_overrides={
            "group": pl.Int32,
            "index": pl.get_index_type(),
            "index_right": pl.get_index_type(),
        },
    )
    left = dataframe({"index": [0, 1], "key": [None, 0.0], "group": [None, 0]})
    right = dataframe(
        {
            "index": [0, 1, 2, 3],
            "key": [None, None, None, 1.0],
            "group": [None, None, None, 0],
        }
    )

    actual = (
        left.lazy()
        .join_asof(
            right.lazy(),
            on="key",
            by="group",
            strategy="nearest",
            allow_exact_matches=False,
        )
        .collect()
    )
    expected = (
        left.lazy()
        .join_asof(
            right.lazy(), on="key", strategy="nearest", allow_exact_matches=False
        )
        .collect()
        .drop("group_right")
    )
    assert_frame_equal(actual, expected)


@pytest.mark.slow
def test_join_asof_maintain_order_left_27526() -> None:
    N_IDS = 20
    N_TS = 100_000
    data = io.BytesIO()
    metadata = io.BytesIO()

    pl.DataFrame(
        {
            "id": [i for i in range(N_IDS) for _ in range(N_TS)],
            "ts": [j for _ in range(N_IDS) for j in range(N_TS)],
        }
    ).sort("id", "ts").write_parquet(data)
    pl.DataFrame(
        {
            "id": range(N_IDS),
            "name": [f"name_{i}" for i in range(N_IDS)],
        }
    ).write_parquet(metadata)

    joined = pl.scan_parquet(data.getvalue()).join(
        pl.scan_parquet(metadata.getvalue()),
        on="id",
        how="left",
        maintain_order="left_right",
    )
    q = joined.set_sorted("id", "ts").join_asof(
        pl.scan_parquet(data.getvalue()).set_sorted("id", "ts"),
        on="ts",
        by="id",
        check_sortedness=False,
    )
    assert q.collect(engine="streaming")["id"].is_sorted(), (
        "expected asof_join to maintain left ordering"
    )


def test_join_asof_buffer_ordering_27619() -> None:
    n = 10
    for n_dataframes in range(1, 20):
        linspace = pl.select(pl.linear_space(0, 1, n).alias("a"))
        left_dfs = pl.concat(
            [linspace.lazy()]
            + [linspace.slice(0, 0).lazy() for _ in range(n_dataframes - 1)]
        )
        left_lazy = left_dfs.set_sorted("a")
        right_lazy = left_dfs.select(pl.col("a") - 0.5).set_sorted("a")
        q = left_lazy.join_asof(
            right_lazy,
            on="a",
            check_sortedness=False,
        )
        assert_frame_equal(q.collect(engine="streaming"), q.collect(engine="in-memory"))


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
@pytest.mark.parametrize("nulls_last", [False, True])
@pytest.mark.parametrize(
    ("strategy", "tolerance", "expected"),
    [
        ("backward", None, [None, 1, 1, 2]),
        ("forward", None, [0, 1, 2, None]),
        ("nearest", None, [0, 1, 2, 2]),
        ("nearest", 0, [None, 1, None, None]),
    ],
)
def test_join_asof_null_right_keys(
    engine: EngineType,
    nulls_last: bool,
    strategy: AsofJoinStrategy,
    tolerance: int | None,
    expected: list[int | None],
) -> None:
    right = pl.DataFrame({"t": [1, 2, 6], "b": [0, 1, 2]})
    nulls = pl.DataFrame({"t": [None, None], "b": [None, None]}, schema=right.schema)
    right = pl.concat([right, nulls] if nulls_last else [nulls, right])
    # Split the right side over several morsels.
    right_lf = pl.concat([right[:2].lazy(), right[2:4].lazy(), right[4:].lazy()])

    out = (
        pl.LazyFrame({"t": [0, 2, 5, 7]})
        .join_asof(
            right_lf,
            on="t",
            strategy=strategy,
            tolerance=tolerance,
            check_sortedness=False,
        )
        .collect(engine=engine)
    )
    assert out["b"].to_list() == expected


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
@pytest.mark.parametrize("strategy", ["backward", "nearest"])
def test_join_asof_by_null_right_keys_in_group(
    engine: EngineType, strategy: AsofJoinStrategy
) -> None:
    right = pl.DataFrame(
        {
            "g": [1, 1, 1, 2, 2, 2],
            "t": [1, 4, None, 2, 5, None],
            "b": [0, 1, 2, 3, 4, 5],
        }
    )
    right_lf = pl.concat([right[:3].lazy(), right[3:].lazy()]).set_sorted("g")

    out = (
        pl.LazyFrame({"g": [1, 1, 2, 2], "t": [4, 6, 2, 6]})
        .set_sorted("g")
        .join_asof(right_lf, on="t", by="g", strategy=strategy, check_sortedness=False)
        .collect(engine=engine)
    )
    assert out["b"].to_list() == [1, 1, 3, 4]


def _asof_right_filter(q: pl.LazyFrame) -> str | None:
    right_plan = q.explain().split("RIGHT PLAN")[1].splitlines()
    return right_plan[1].strip() if "FILTER" in right_plan[1] else None


def _assert_asof_optimized_equal(q: pl.LazyFrame) -> None:
    expected = q.collect(optimizations=pl.QueryOptFlags.none())
    engines: list[EngineType] = ["in-memory", "streaming"]
    for engine in engines:
        assert_frame_equal(q.collect(engine=engine), expected, check_row_order=False)


@pytest.mark.parametrize(
    ("strategy", "tolerance", "expected_filter"),
    [
        ("backward", 2, 'FILTER (col("t") >= 2) & (col("t") <= 6)'),
        ("backward", None, 'FILTER col("t") <= 6'),
        ("forward", 2, 'FILTER (col("t") >= 4) & (col("t") <= 8)'),
        ("forward", None, 'FILTER col("t") >= 4'),
        ("nearest", 2, 'FILTER (col("t") >= 2) & (col("t") <= 8)'),
        ("nearest", -2, 'FILTER (col("t") >= 2) & (col("t") <= 8)'),
        ("nearest", None, None),
    ],
)
def test_join_asof_key_filter_pushdown(
    strategy: AsofJoinStrategy, tolerance: int | None, expected_filter: str | None
) -> None:
    left = pl.LazyFrame({"t": range(10), "a": range(10)})
    right = pl.LazyFrame({"t": range(0, 10, 3), "b": range(4)})
    q = left.join_asof(right, on="t", strategy=strategy, tolerance=tolerance).filter(
        pl.col("t").is_between(4, 6)
    )

    assert _asof_right_filter(q) == expected_filter
    _assert_asof_optimized_equal(q)


@pytest.mark.parametrize(
    ("predicate", "expected_filter"),
    [
        (
            (pl.col("t") < 2) | (pl.col("t") > 12),
            'FILTER (col("t") <= 2) | (col("t") >= 10)',
        ),
        # Overlapping ranges are merged.
        (
            (pl.col("t") < 6) | (pl.col("t").is_between(7, 9)),
            'FILTER col("t") <= 9',
        ),
        (pl.col("t").is_in([5, 9]), 'FILTER (col("t") >= 3) & (col("t") <= 9)'),
        (pl.col("t") == 5, 'FILTER (col("t") >= 3) & (col("t") <= 5)'),
        # Each OR branch also bounds the key, so the filter is kept above the join
        # and a weaker one goes to the right.
        (
            ((pl.col("t") < 2) & (pl.col("b") > 0))
            | ((pl.col("t") > 12) & (pl.col("a") < 9)),
            'FILTER (col("t") <= 2) | (col("t") >= 10)',
        ),
        # A branch without a bound on the key.
        ((pl.col("t") < 2) | (pl.col("b") > 2), None),
        ((pl.col("t") >= 5) | (pl.col("b") > pl.col("b").mean()), None),
        (
            (pl.col("t") >= 5) & (pl.col("b") > pl.col("b").mean()),
            None,
        ),
        (pl.col("t") > pl.col("t").max() - 3, None),
    ],
)
def test_join_asof_key_filter_pushdown_shapes(
    predicate: pl.Expr, expected_filter: str | None
) -> None:
    left = pl.LazyFrame({"t": range(15), "a": range(15)})
    right = pl.LazyFrame({"t": range(0, 15, 2), "b": range(8)})
    q = left.join_asof(right, on="t", tolerance=2).filter(predicate)

    assert _asof_right_filter(q) == expected_filter
    _assert_asof_optimized_equal(q)


def test_join_asof_key_filter_pushdown_chained_filters() -> None:
    left = pl.LazyFrame({"t": range(30)})
    right = pl.LazyFrame({"t": range(0, 30, 2), "b": range(15)})
    q = (
        left.join_asof(right, on="t", tolerance=2)
        .filter((pl.col("t") < 5) | (pl.col("t") > 20))
        .filter((pl.col("t") < 3) | (pl.col("t") > 25))
    )
    right_filter = _asof_right_filter(q)
    assert right_filter is not None
    assert '(col("t") <= 5) | (col("t") >= 18)' in right_filter
    assert '(col("t") <= 3) | (col("t") >= 23)' in right_filter
    _assert_asof_optimized_equal(q)


@pytest.mark.parametrize(
    ("dtype", "left_t", "right_t", "tolerance", "predicate", "expected_filter"),
    [
        (
            pl.Date,
            [date(2024, 1, d) for d in range(1, 10)],
            [date(2024, 1, d) for d in range(1, 10, 2)],
            # Truncated to 1 day.
            "36h",
            pl.col("t") >= date(2024, 1, 5),
            'FILTER col("t") >= 2024-01-04',
        ),
        (
            pl.Datetime("us", "Europe/Amsterdam"),
            [datetime(2024, 1, 1, h) for h in range(10)],
            [datetime(2024, 1, 1, h) for h in range(0, 10, 3)],
            timedelta(hours=2),
            pl.col("t") >= datetime(2024, 1, 1, 5, tzinfo=ZoneInfo("Europe/Amsterdam")),
            'FILTER col("t") >= 2024-01-01 03:00:00 CET',
        ),
        # Bounds past midnight are dropped.
        (
            pl.Time,
            [time(0, 30), time(12), time(23, 30)],
            [time(0, 10), time(11), time(22)],
            "2h",
            pl.col("t").is_between(time(0), time(1)) | (pl.col("t") >= time(23)),
            'FILTER (col("t") <= 01:00:00) | (col("t") >= 21:00:00)',
        ),
        (
            pl.UInt8,
            [0, 1, 2, 250, 255],
            [0, 2, 249, 254],
            3,
            (pl.col("t") <= 1) | (pl.col("t") >= 254),
            'FILTER (col("t") <= 1) | (col("t") >= 251)',
        ),
        (
            pl.Int64,
            [-(2**63), 0, 2**63 - 1],
            [-(2**63), 1, 2**63 - 1],
            -(2**63),
            pl.col("t") >= 0,
            None,
        ),
        (
            pl.Duration("ms"),
            [timedelta(milliseconds=i) for i in range(10)],
            [timedelta(milliseconds=i) for i in range(0, 10, 3)],
            "2ms",
            pl.col("t") >= timedelta(milliseconds=5),
            'FILTER col("t") >= 3ms',
        ),
        (
            pl.Float64,
            [1.0, 2.0, 3.0],
            [1.0, 2.5],
            1.0,
            pl.col("t") >= 2.0,
            None,
        ),
    ],
)
def test_join_asof_key_filter_pushdown_dtypes(
    dtype: pl.DataType,
    left_t: list[Any],
    right_t: list[Any],
    tolerance: Any,
    predicate: pl.Expr,
    expected_filter: str | None,
) -> None:
    left = pl.LazyFrame({"t": left_t}, schema={"t": dtype})
    right = pl.LazyFrame(
        {"t": right_t, "b": range(len(right_t))}, schema={"t": dtype, "b": pl.Int64}
    )
    q = left.join_asof(right, on="t", tolerance=tolerance).filter(predicate)

    assert _asof_right_filter(q) == expected_filter
    _assert_asof_optimized_equal(q)


@pytest.mark.parametrize(
    ("right_on", "coalesce", "allow_exact_matches"),
    [("s", True, True), ("t", False, True), ("t", True, False)],
)
def test_join_asof_key_filter_pushdown_names(
    right_on: str, coalesce: bool, allow_exact_matches: bool
) -> None:
    left = pl.LazyFrame({"t": range(10), "g": [0, 1] * 5})
    right = pl.LazyFrame({right_on: range(0, 10, 3), "g": [0, 1, 0, 1], "b": range(4)})
    q = left.join_asof(
        right,
        left_on="t",
        right_on=right_on,
        by="g",
        tolerance=2,
        coalesce=coalesce,
        allow_exact_matches=allow_exact_matches,
    ).filter(pl.col("t") >= 5)

    assert _asof_right_filter(q) == f'FILTER col("{right_on}") >= 3'
    _assert_asof_optimized_equal(q)


def test_join_asof_key_filter_pushdown_scan(tmp_path: Path) -> None:
    path = tmp_path / "right.parquet"
    pl.DataFrame({"t": range(100), "b": range(100)}).write_parquet(
        path, row_group_size=10
    )
    q = (
        pl.LazyFrame({"t": range(0, 100, 5)})
        .join_asof(pl.scan_parquet(path), on="t", tolerance=7)
        .filter(pl.col("t").is_between(50, 60))
    )

    right_plan = q.explain().split("RIGHT PLAN")[1]
    assert 'SELECTION: (col("t") >= 43) & (col("t") <= 60)' in right_plan
    _assert_asof_optimized_equal(q)


def test_join_asof_key_filter_pushdown_random() -> None:
    rng = random.Random(0)
    for _ in range(40):
        left = pl.LazyFrame({"t": sorted(rng.choices(range(20), k=8))})
        right = pl.LazyFrame({"t": sorted(rng.choices(range(20), k=8)), "b": range(8)})
        strategy: AsofJoinStrategy = rng.choice(["backward", "forward", "nearest"])
        tolerance = rng.choice([None, 0, 1, 3])
        lo, hi = sorted(rng.choices(range(-2, 22), k=2))
        predicate = pl.col("t").is_between(lo, hi)
        if rng.random() < 0.5:
            predicate = predicate | (pl.col("t") == rng.randint(0, 20))
        q = left.join_asof(
            right,
            on="t",
            strategy=strategy,
            tolerance=tolerance,
            allow_exact_matches=rng.random() < 0.8,
            check_sortedness=False,
        ).filter(predicate)
        _assert_asof_optimized_equal(q)


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
@pytest.mark.parametrize(
    ("dtype", "tolerance", "expected"),
    [
        # Larger than any distance between two keys.
        (pl.Int8, 2**40, [0, 0, 2]),
        (pl.Int64, -(2**63), [0, 0, 2]),
        (pl.UInt64, 2**63 + 5, [0, 0, 2]),
        # Same as a tolerance of 1.
        (pl.UInt32, -1, [0, None, 2]),
        (pl.Int16, -1.7, [0, None, 2]),
    ],
)
def test_join_asof_tolerance_normalized(
    engine: EngineType,
    dtype: pl.DataType,
    tolerance: int | float,
    expected: list[int | None],
) -> None:
    left = pl.LazyFrame({"t": [0, 5, 100]}, schema={"t": dtype})
    right = pl.LazyFrame(
        {"t": [0, 50, 99], "b": [0, 1, 2]}, schema_overrides={"t": dtype}
    )
    out = left.join_asof(right, on="t", tolerance=tolerance).collect(engine=engine)
    assert out["b"].to_list() == expected


@pytest.mark.parametrize("strategy", ["backward", "forward", "nearest"])
@pytest.mark.parametrize("allow_exact_matches", [True, False])
def test_join_asof_streaming_right_morsel_edges(
    strategy: AsofJoinStrategy, allow_exact_matches: bool
) -> None:
    schema = {"t": pl.Int64, "b": pl.Int64}
    parts = [
        pl.DataFrame({"t": [], "b": []}, schema=schema),
        pl.DataFrame({"t": [1, 3, 3], "b": [0, 1, 2]}, schema=schema),
        pl.DataFrame({"t": [3, 3, 7], "b": [3, 4, 5]}, schema=schema),
        pl.DataFrame({"t": [], "b": []}, schema=schema),
        pl.DataFrame({"t": [None, None], "b": [6, 7]}, schema=schema),
        pl.DataFrame({"t": [None], "b": [8]}, schema=schema),
    ]
    right = pl.concat([p.lazy() for p in parts])
    left = pl.LazyFrame({"t": [0, 3, 4, 7, 9]})

    def join(right: pl.LazyFrame) -> pl.LazyFrame:
        return left.join_asof(
            right,
            on="t",
            strategy=strategy,
            allow_exact_matches=allow_exact_matches,
        )

    expected = join(right.drop_nulls()).collect(engine="in-memory")
    assert_frame_equal(join(right).collect(engine="streaming"), expected)
    assert_frame_equal(
        join(right).head(2).collect(engine="streaming"), expected.head(2)
    )


@pytest.mark.parametrize("strategy", ["backward", "forward"])
@pytest.mark.parametrize(
    ("dtype", "value", "tolerance"),
    [
        (pl.Date, date(2024, 1, 5), 2**31 - 20_000),
        (pl.Datetime("ms"), datetime(2024, 1, 5), 2**62),
        (pl.Datetime("us", "Europe/Amsterdam"), datetime(2024, 1, 5), 9 * 10**18),
    ],
)
def test_join_asof_key_filter_pushdown_calendar_range(
    strategy: AsofJoinStrategy, dtype: pl.DataType, value: Any, tolerance: int
) -> None:
    left = pl.DataFrame({"t": [value]}).cast({"t": dtype}).lazy()
    right = pl.DataFrame({"t": [value], "b": [1]}).cast({"t": dtype}).lazy()
    q = left.join_asof(right, on="t", strategy=strategy, tolerance=tolerance).filter(
        pl.col("t") == pl.lit(value).cast(dtype)
    )
    # The shifted bound lies outside the calendar and is dropped.
    right_filter = _asof_right_filter(q)
    assert right_filter is not None
    assert ("<=" in right_filter) != (">=" in right_filter)
    _assert_asof_optimized_equal(q)


def test_join_asof_key_filter_pushdown_128_bit_keys() -> None:
    left = pl.LazyFrame({"t": [1, 5, 9]}, schema={"t": pl.Int128})
    right = pl.LazyFrame(
        {"t": [0, 4, 8], "b": [0, 1, 2]}, schema_overrides={"t": pl.Int128}
    )
    q = left.join_asof(right, on="t", tolerance=2).filter(pl.col("t") >= 5)
    assert _asof_right_filter(q) is None
    _assert_asof_optimized_equal(q)


def test_join_asof_key_filter_pushdown_hive_pruning(tmp_path: Path) -> None:
    for t in range(10):
        part = tmp_path / f"t={t}"
        part.mkdir()
        if t >= 8:
            # Only readable if the scan does not prune this partition.
            (part / "0.parquet").write_bytes(b"not a parquet file")
        else:
            pl.DataFrame({"b": [t]}).write_parquet(part / "0.parquet")

    q = (
        pl.LazyFrame({"t": [2, 3, 4, 5]})
        .join_asof(
            pl.scan_parquet(tmp_path, hive_partitioning=True), on="t", tolerance=2
        )
        .filter(pl.col("t").is_between(3, 4))
    )
    expected = pl.DataFrame({"t": [3, 4], "b": [3, 4]})
    assert_frame_equal(q.collect(engine="in-memory"), expected)
    assert_frame_equal(q.collect(engine="streaming"), expected)
