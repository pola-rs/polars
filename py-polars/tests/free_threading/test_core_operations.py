from __future__ import annotations


def test_dataframe_select_filter_with_columns() -> None:
    import polars as pl
    from polars.testing import assert_frame_equal

    df = pl.DataFrame(
        {
            "g": ["a", "a", "b", "b"],
            "x": [1, 2, 3, 4],
            "flag": [True, False, True, False],
        }
    )
    out = (
        df.with_columns((pl.col("x") * 2).alias("y"))
        .filter(pl.col("flag"))
        .select("g", "y")
        .sort("g")
    )

    expected = pl.DataFrame({"g": ["a", "b"], "y": [2, 6]})
    assert_frame_equal(out, expected)


def test_series_basic_ops() -> None:
    import polars as pl
    from polars.testing import assert_series_equal

    s = pl.Series("x", [1, 2, 3, 4])
    out = (s * 2).filter(s > 2)

    assert_series_equal(out, pl.Series("x", [6, 8]))
    assert s.sum() == 10
    assert s.min() == 1
    assert s.max() == 4


def test_group_by_builtin_aggregations() -> None:
    import polars as pl
    from polars.testing import assert_frame_equal

    df = pl.DataFrame(
        {
            "g": ["a", "a", "b", "b"],
            "x": [1, 3, 5, 7],
        }
    )
    out = (
        df.group_by("g")
        .agg(
            pl.col("x").sum().alias("sum_x"),
            pl.col("x").mean().alias("mean_x"),
            pl.len().alias("count"),
            pl.col("x").min().alias("min_x"),
            pl.col("x").max().alias("max_x"),
        )
        .sort("g")
    )

    expected = pl.DataFrame(
        {
            "g": ["a", "b"],
            "sum_x": [4, 12],
            "mean_x": [2.0, 6.0],
            "count": pl.Series([2, 2], dtype=pl.UInt32),
            "min_x": [1, 5],
            "max_x": [3, 7],
        }
    )
    assert_frame_equal(out, expected)


def test_concat_eager_frames() -> None:
    import polars as pl
    from polars.testing import assert_frame_equal

    frames = [
        pl.DataFrame({"x": [1, 2], "label": ["a", "b"]}),
        pl.DataFrame({"x": [3, 4], "label": ["c", "d"]}),
    ]
    out = pl.concat(frames)

    expected = pl.DataFrame({"x": [1, 2, 3, 4], "label": ["a", "b", "c", "d"]})
    assert_frame_equal(out, expected)
    assert out["x"].sum() == 10
