from __future__ import annotations


def test_lazy_select_filter_collect() -> None:
    import polars as pl
    from polars.testing import assert_frame_equal

    lf = (
        pl.DataFrame({"x": [1, 2, 3, 4], "y": [10, 20, 30, 40]})
        .lazy()
        .filter(pl.col("x") > 2)
        .select((pl.col("x") + pl.col("y")).alias("z"))
    )
    assert lf.explain(optimized=False)
    assert lf.explain(optimized=True)
    out = lf.collect()

    assert_frame_equal(out, pl.DataFrame({"z": [33, 44]}))


def test_lazy_join_collect() -> None:
    import polars as pl
    from polars.testing import assert_frame_equal

    left = pl.DataFrame({"k": [1, 2, 3], "x": [10, 20, 30]}).lazy()
    right = pl.DataFrame({"k": [2, 3, 4], "y": [200, 300, 400]}).lazy()
    out = (
        left.join(right, on="k", how="inner")
        .with_columns((pl.col("x") + pl.col("y")).alias("z"))
        .select("k", "z")
        .sort("k")
        .collect()
    )

    assert_frame_equal(pl.DataFrame({"k": [2, 3], "z": [220, 330]}), out)


def test_lazy_group_by_collect() -> None:
    import polars as pl
    from polars.testing import assert_frame_equal

    out = (
        pl.DataFrame({"g": ["a", "a", "b", "b"], "x": [1, 2, 3, 4]})
        .lazy()
        .group_by("g")
        .agg(pl.col("x").sum().alias("sum_x"))
        .sort("g")
        .collect()
    )

    assert_frame_equal(pl.DataFrame({"g": ["a", "b"], "sum_x": [3, 7]}), out)
