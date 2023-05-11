import typing

import polars as pl
from polars.testing import assert_frame_equal


def test_comparison_nulls_single() -> None:
    df1 = pl.DataFrame(
        {
            "a": pl.Series([None], dtype=pl.Utf8),
            "b": pl.Series([None], dtype=pl.Int64),
            "c": pl.Series([None], dtype=pl.Boolean),
        }
    )
    df2 = pl.DataFrame(
        {
            "a": pl.Series([None], dtype=pl.Utf8),
            "b": pl.Series([None], dtype=pl.Int64),
            "c": pl.Series([None], dtype=pl.Boolean),
        }
    )
    assert (df1 == df2).row(0) == (True, True, True)
    assert (df1 != df2).row(0) == (False, False, False)


def test_comparison_series_expr() -> None:
    df = pl.DataFrame({"a": pl.Series([1, 2, 3]), "b": pl.Series([2, 1, 3])})

    assert_frame_equal(
        df.select(
            (df["a"] == pl.col("b")).alias("eq"),  # False, False, True
            (df["a"] != pl.col("b")).alias("ne"),  # True, True, False
            (df["a"] < pl.col("b")).alias("lt"),  # True, False, False
            (df["a"] <= pl.col("b")).alias("le"),  # True, False, True
            (df["a"] > pl.col("b")).alias("gt"),  # False, True, False
            (df["a"] >= pl.col("b")).alias("ge"),  # False, True, True
        ),
        pl.DataFrame(
            {
                "eq": [False, False, True],
                "ne": [True, True, False],
                "lt": [True, False, False],
                "le": [True, False, True],
                "gt": [False, True, False],
                "ge": [False, True, True],
            }
        ),
    )


def test_comparison_expr_expr() -> None:
    df = pl.DataFrame({"a": pl.Series([1, 2, 3]), "b": pl.Series([2, 1, 3])})

    assert_frame_equal(
        df.select(
            (pl.col("a") == pl.col("b")).alias("eq"),  # False, False, True
            (pl.col("a") != pl.col("b")).alias("ne"),  # True, True, False
            (pl.col("a") < pl.col("b")).alias("lt"),  # True, False, False
            (pl.col("a") <= pl.col("b")).alias("le"),  # True, False, True
            (pl.col("a") > pl.col("b")).alias("gt"),  # False, True, False
            (pl.col("a") >= pl.col("b")).alias("ge"),  # False, True, True
        ),
        pl.DataFrame(
            {
                "eq": [False, False, True],
                "ne": [True, True, False],
                "lt": [True, False, False],
                "le": [True, False, True],
                "gt": [False, True, False],
                "ge": [False, True, True],
            }
        ),
    )


def test_comparison_expr_series() -> None:
    df = pl.DataFrame({"a": pl.Series([1, 2, 3]), "b": pl.Series([2, 1, 3])})

    assert_frame_equal(
        df.select(
            (pl.col("a") == df["b"]).alias("eq"),  # False, False, True
            (pl.col("a") != df["b"]).alias("ne"),  # True, True, False
            (pl.col("a") < df["b"]).alias("lt"),  # True, False, False
            (pl.col("a") <= df["b"]).alias("le"),  # True, False, True
            (pl.col("a") > df["b"]).alias("gt"),  # False, True, False
            (pl.col("a") >= df["b"]).alias("ge"),  # False, True, True
        ),
        pl.DataFrame(
            {
                "eq": [False, False, True],
                "ne": [True, True, False],
                "lt": [True, False, False],
                "le": [True, False, True],
                "gt": [False, True, False],
                "ge": [False, True, True],
            }
        ),
    )


@typing.no_type_check
def test_offset_handling_arg_where_7863() -> None:
    df_check = pl.DataFrame({"a": [0, 1]})
    df_check.select((pl.lit(0).append(pl.col("a")).append(0)) != 0)
    assert (
        df_check.select((pl.lit(0).append(pl.col("a")).append(0)) != 0)
        .select(pl.col("literal").arg_true())
        .item()
        == 2
    )
