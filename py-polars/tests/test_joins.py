from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

import polars as pl


def test_semi_anti_join() -> None:
    df_a = pl.DataFrame({"key": [1, 2, 3], "payload": ["f", "i", None]})

    df_b = pl.DataFrame({"key": [3, 4, 5, None]})

    assert df_a.join(df_b, on="key", how="anti").to_dict(False) == {
        "key": [1, 2],
        "payload": ["f", "i"],
    }
    assert df_a.join(df_b, on="key", how="semi").to_dict(False) == {
        "key": [3],
        "payload": [None],
    }

    # lazy
    assert df_a.lazy().join(df_b.lazy(), on="key", how="anti").collect().to_dict(
        False
    ) == {
        "key": [1, 2],
        "payload": ["f", "i"],
    }
    assert df_a.lazy().join(df_b.lazy(), on="key", how="semi").collect().to_dict(
        False
    ) == {
        "key": [3],
        "payload": [None],
    }

    df_a = pl.DataFrame(
        {"a": [1, 2, 3, 1], "b": ["a", "b", "c", "a"], "payload": [10, 20, 30, 40]}
    )

    df_b = pl.DataFrame({"a": [3, 3, 4, 5], "b": ["c", "c", "d", "e"]})

    assert df_a.join(df_b, on=["a", "b"], how="anti").to_dict(False) == {
        "a": [1, 2, 1],
        "b": ["a", "b", "a"],
        "payload": [10, 20, 40],
    }
    assert df_a.join(df_b, on=["a", "b"], how="semi").to_dict(False) == {
        "a": [3],
        "b": ["c"],
        "payload": [30],
    }


def test_join_same_cat_src() -> None:
    df = pl.DataFrame(
        data={"column": ["a", "a", "b"], "more": [1, 2, 3]},
        columns=[("column", pl.Categorical), ("more", pl.Int32)],
    )
    df_agg = df.groupby("column").agg(pl.col("more").mean())
    assert df.join(df_agg, on="column").to_dict(False) == {
        "column": ["a", "a", "b"],
        "more": [1, 2, 3],
        "more_right": [1.5, 1.5, 3.0],
    }


def test_sorted_merge_joins() -> None:
    for reverse in [False, True]:
        n = 30
        df_a = pl.DataFrame(
            {"a": np.sort(np.random.randint(0, n // 2, n))}
        ).with_row_count("row_a")

        df_b = pl.DataFrame(
            {"a": np.sort(np.random.randint(0, n // 2, n // 2))}
        ).with_row_count("row_b")

        if reverse:
            df_a = df_a.select(pl.all().reverse())
            df_b = df_b.select(pl.all().reverse())

        for how in ["left", "inner"]:
            # hash join
            out_hash_join = df_a.join(df_b, on="a", how=how)

            # sorted merge join
            out_sorted_merge_join = df_a.with_column(
                pl.col("a").set_sorted(reverse)
            ).join(df_b.with_column(pl.col("a").set_sorted(reverse)), on="a", how=how)

            assert out_hash_join.frame_equal(out_sorted_merge_join)


def test_join_negative_integers() -> None:
    expected = {"a": [-6, -1, 0], "b": [-6, -1, 0]}

    df1 = pl.DataFrame(
        {
            "a": [-1, -6, -3, 0],
        }
    )

    df2 = pl.DataFrame(
        {
            "a": [-6, -1, -4, -2, 0],
            "b": [-6, -1, -4, -2, 0],
        }
    )

    for dt in [pl.Int8, pl.Int16, pl.Int32, pl.Int64]:
        assert (
            df1.with_column(pl.all().cast(dt))
            .join(df2.with_column(pl.all().cast(dt)), on="a", how="inner")
            .to_dict(False)
            == expected
        )


def test_join_asof_floats() -> None:
    df1 = pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": ["lrow1", "lrow2", "lrow3"]})
    df2 = pl.DataFrame({"a": [0.59, 1.49, 2.89], "b": ["rrow1", "rrow2", "rrow3"]})
    assert df1.join_asof(df2, on="a", strategy="backward").to_dict(False) == {
        "a": [1.0, 2.0, 3.0],
        "b": ["lrow1", "lrow2", "lrow3"],
        "b_right": ["rrow1", "rrow2", "rrow3"],
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


def test_deprecated() -> None:
    df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    other = pl.DataFrame({"a": [1, 2], "c": [3, 4]})
    result = pl.DataFrame({"a": [1, 2], "b": [3, 4], "c": [3, 4]})

    with pytest.deprecated_call():
        df.join(df=other, on="a")
    with pytest.deprecated_call():
        df.lazy().join(ldf=other.lazy(), on="a").collect()

    np.testing.assert_equal(df.join(other=other, on="a").to_numpy(), result.to_numpy())
    np.testing.assert_equal(
        df.lazy().join(other=other.lazy(), on="a").collect().to_numpy(),
        result.to_numpy(),
    )
