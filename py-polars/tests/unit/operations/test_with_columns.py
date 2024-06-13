import pytest

import polars as pl
from polars.testing import assert_frame_equal


def test_with_columns() -> None:
    import datetime

    df = pl.DataFrame(
        {
            "a": [1, 2, 3, 4],
            "b": [0.5, 4, 10, 13],
            "c": [True, True, False, True],
        }
    )
    srs_named = pl.Series("f", [3, 2, 1, 0])
    srs_unnamed = pl.Series(values=[3, 2, 1, 0])

    expected = pl.DataFrame(
        {
            "a": [1, 2, 3, 4],
            "b": [0.5, 4, 10, 13],
            "c": [True, True, False, True],
            "d": [0.5, 8.0, 30.0, 52.0],
            "e": [False, False, True, False],
            "f": [3, 2, 1, 0],
            "g": True,
            "h": pl.Series(values=[1, 1, 1, 1], dtype=pl.Int32),
            "i": 3.2,
            "j": [1, 2, 3, 4],
            "k": pl.Series(values=[None, None, None, None], dtype=pl.Null),
            "l": datetime.datetime(2001, 1, 1, 0, 0),
        }
    )

    # as exprs list
    dx = df.with_columns(
        (pl.col("a") * pl.col("b")).alias("d"),
        ~pl.col("c").alias("e"),
        srs_named,
        pl.lit(True).alias("g"),
        pl.lit(1).alias("h"),
        pl.lit(3.2).alias("i"),
        pl.col("a").alias("j"),
        pl.lit(None).alias("k"),
        pl.lit(datetime.datetime(2001, 1, 1, 0, 0)).alias("l"),
    )
    assert_frame_equal(dx, expected)

    # as positional arguments
    dx = df.with_columns(
        (pl.col("a") * pl.col("b")).alias("d"),
        ~pl.col("c").alias("e"),
        srs_named,
        pl.lit(True).alias("g"),
        pl.lit(1).alias("h"),
        pl.lit(3.2).alias("i"),
        pl.col("a").alias("j"),
        pl.lit(None).alias("k"),
        pl.lit(datetime.datetime(2001, 1, 1, 0, 0)).alias("l"),
    )
    assert_frame_equal(dx, expected)

    # as keyword arguments
    dx = df.with_columns(
        d=pl.col("a") * pl.col("b"),
        e=~pl.col("c"),
        f=srs_unnamed,
        g=True,
        h=1,
        i=3.2,
        j="a",  # Note: string interpreted as column name, resolves to `pl.col("a")`
        k=None,
        l=datetime.datetime(2001, 1, 1, 0, 0),
    )
    assert_frame_equal(dx, expected)

    # mixed
    dx = df.with_columns(
        (pl.col("a") * pl.col("b")).alias("d"),
        ~pl.col("c").alias("e"),
        f=srs_unnamed,
        g=True,
        h=1,
        i=3.2,
        j="a",  # Note: string interpreted as column name, resolves to `pl.col("a")`
        k=None,
        l=datetime.datetime(2001, 1, 1, 0, 0),
    )
    assert_frame_equal(dx, expected)

    # automatically upconvert multi-output expressions to struct
    with pl.Config() as cfg:
        cfg.set_auto_structify(True)

        ldf = (
            pl.DataFrame({"x1": [1, 2, 6], "x2": [1, 2, 3]})
            .lazy()
            .with_columns(
                pl.col(["x1", "x2"]).pct_change().alias("pct_change"),
                maxes=pl.all().max().name.suffix("_max"),
                xcols=pl.col("^x.*$"),
            )
        )
        # ┌─────┬─────┬─────────────┬───────────┬───────────┐
        # │ x1  ┆ x2  ┆ pct_change  ┆ maxes     ┆ xcols     │
        # │ --- ┆ --- ┆ ---         ┆ ---       ┆ ---       │
        # │ i64 ┆ i64 ┆ struct[2]   ┆ struct[2] ┆ struct[2] │
        # ╞═════╪═════╪═════════════╪═══════════╪═══════════╡
        # │ 1   ┆ 1   ┆ {null,null} ┆ {6,3}     ┆ {1,1}     │
        # │ 2   ┆ 2   ┆ {1.0,1.0}   ┆ {6,3}     ┆ {2,2}     │
        # │ 6   ┆ 3   ┆ {2.0,0.5}   ┆ {6,3}     ┆ {6,3}     │
        # └─────┴─────┴─────────────┴───────────┴───────────┘
        assert ldf.collect().to_dicts() == [
            {
                "x1": 1,
                "x2": 1,
                "pct_change": {"x1": None, "x2": None},
                "maxes": {"x1_max": 6, "x2_max": 3},
                "xcols": {"x1": 1, "x2": 1},
            },
            {
                "x1": 2,
                "x2": 2,
                "pct_change": {"x1": 1.0, "x2": 1.0},
                "maxes": {"x1_max": 6, "x2_max": 3},
                "xcols": {"x1": 2, "x2": 2},
            },
            {
                "x1": 6,
                "x2": 3,
                "pct_change": {"x1": 2.0, "x2": 0.5},
                "maxes": {"x1_max": 6, "x2_max": 3},
                "xcols": {"x1": 6, "x2": 3},
            },
        ]


def test_with_columns_empty() -> None:
    df = pl.DataFrame({"a": [1, 2]})
    result = df.with_columns()
    assert_frame_equal(result, df)


def test_with_columns_single_series() -> None:
    ldf = pl.LazyFrame({"a": [1, 2]})
    result = ldf.with_columns(pl.Series("b", [3, 4]))

    expected = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    assert_frame_equal(result.collect(), expected)


def test_with_columns_seq() -> None:
    df = pl.DataFrame({"a": [1, 2]})
    result = df.with_columns_seq(
        pl.lit(5).alias("b"),
        pl.lit("foo").alias("c"),
    )
    expected = pl.DataFrame(
        {
            "a": [1, 2],
            "b": pl.Series([5, 5], dtype=pl.Int32),
            "c": ["foo", "foo"],
        }
    )
    assert_frame_equal(result, expected)


# https://github.com/pola-rs/polars/issues/15588
def test_with_columns_invalid_type() -> None:
    lf = pl.LazyFrame({"a": [1, 2, 3]})
    with pytest.raises(
        TypeError, match="cannot create expression literal for value of type LazyFrame"
    ):
        lf.with_columns(lf)  # type: ignore[arg-type]
