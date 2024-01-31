import polars as pl
from polars.testing import assert_frame_equal


def test_null_index() -> None:
    df = pl.DataFrame({"a": [[1, 2], [3, 4], [5, 6]], "b": [[1, 2], [1, 2], [4, 5]]})

    result = df.with_columns(pl.lit(None).alias("null_col"))[-1]

    expected = pl.DataFrame(
        {"a": [[5, 6]], "b": [[4, 5]], "null_col": [None]},
        schema_overrides={"null_col": pl.Null},
    )
    assert_frame_equal(result, expected)


def test_null_grouping_12950() -> None:
    assert pl.DataFrame({"x": None}).unique().to_dict(as_series=False) == {"x": [None]}
    assert pl.DataFrame({"x": [None, None]}).unique().to_dict(as_series=False) == {
        "x": [None]
    }
    assert pl.DataFrame({"x": None}).slice(0, 0).unique().to_dict(as_series=False) == {
        "x": []
    }


def test_null_comp_14118() -> None:
    df = pl.DataFrame(
        {
            "a": [None, None],
            "b": [None, None],
        }
    )

    output_df = df.select(
        gt=pl.col("a") > pl.col("b"),
        lt=pl.col("a") < pl.col("b"),
        gt_eq=pl.col("a") >= pl.col("b"),
        lt_eq=pl.col("a") <= pl.col("b"),
        eq=pl.col("a") == pl.col("b"),
        eq_missing=pl.col("a").eq_missing(pl.col("b")),
        ne=pl.col("a") != pl.col("b"),
        ne_missing=pl.col("a").ne_missing(pl.col("b")),
    )

    expected_df = pl.DataFrame(
        {
            "gt": [None, None],
            "lt": [None, None],
            "gt_eq": [None, None],
            "lt_eq": [None, None],
            "eq": [None, None],
            "eq_missing": [True, True],
            "ne": [None, None],
            "ne_missing": [False, False],
        },
        schema={
            "gt": pl.Boolean,
            "lt": pl.Boolean,
            "gt_eq": pl.Boolean,
            "lt_eq": pl.Boolean,
            "eq": pl.Boolean,
            "eq_missing": pl.Boolean,
            "ne": pl.Boolean,
            "ne_missing": pl.Boolean,
        },
    )
    assert_frame_equal(output_df, expected_df)
