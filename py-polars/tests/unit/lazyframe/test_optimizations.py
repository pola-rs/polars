import polars as pl
from polars.testing import assert_frame_equal


def test_is_null_followed_by_all() -> None:
    lf = pl.LazyFrame({"group": [0, 0, 0, 1], "val": [6, 0, None, None]})

    expected_df = pl.DataFrame({"group": [0, 1], "val": [False, True]})
    result_lf = lf.group_by("group", maintain_order=True).agg(
        pl.col("val").is_null().all()
    )

    assert (
        r'[[(col("val").count()) == (col("val").null_count())]]' in result_lf.explain()
    )
    assert "is_null" not in result_lf
    assert_frame_equal(expected_df, result_lf.collect())

    # verify we don't optimize on chained expressions when last one is not col
    non_optimized_result_plan = (
        lf.group_by("group", maintain_order=True)
        .agg(pl.col("val").abs().is_null().all())
        .explain()
    )
    assert "null_count" not in non_optimized_result_plan
    assert "is_null" in non_optimized_result_plan

    # edge case of empty series
    lf = pl.LazyFrame({"val": []}, schema={"val": pl.Int32})

    expected_df = pl.DataFrame({"val": [True]})
    result_df = lf.select(pl.col("val").is_null().all()).collect()
    assert_frame_equal(expected_df, result_df)


def test_is_null_followed_by_any() -> None:
    lf = pl.LazyFrame({"group": [0, 0, 0, 1, 2], "val": [6, 0, None, None, 5]})

    expected_df = pl.DataFrame({"group": [0, 1, 2], "val": [True, True, False]})
    result_lf = lf.group_by("group", maintain_order=True).agg(
        pl.col("val").is_null().any()
    )
    assert_frame_equal(expected_df, result_lf.collect())

    # edge case of empty series
    lf = pl.LazyFrame({"val": []}, schema={"val": pl.Int32})

    expected_df = pl.DataFrame({"val": [False]})
    result_df = lf.select(pl.col("val").is_null().any()).collect()
    assert_frame_equal(expected_df, result_df)


def test_is_not_null_followed_by_all() -> None:
    lf = pl.LazyFrame({"group": [0, 0, 0, 1], "val": [6, 0, 5, None]})

    expected_df = pl.DataFrame({"group": [0, 1], "val": [True, False]})
    result_df = (
        lf.group_by("group", maintain_order=True)
        .agg(pl.col("val").is_not_null().all())
        .collect()
    )

    assert_frame_equal(expected_df, result_df)

    # edge case of empty series
    lf = pl.LazyFrame({"val": []}, schema={"val": pl.Int32})

    expected_df = pl.DataFrame({"val": [True]})
    result_df = lf.select(pl.col("val").is_not_null().all()).collect()
    assert_frame_equal(expected_df, result_df)


def test_is_not_null_followed_by_any() -> None:
    lf = pl.LazyFrame({"group": [0, 0, 0, 1, 2], "val": [6, 0, None, None, 5]})

    expected_df = pl.DataFrame({"group": [0, 1, 2], "val": [True, False, True]})
    result_lf = lf.group_by("group", maintain_order=True).agg(
        pl.col("val").is_not_null().any()
    )

    assert (
        r'[[(col("val").null_count()) < (col("val").count())]]' in result_lf.explain()
    )
    assert "is_not_null" not in result_lf.explain()
    assert_frame_equal(expected_df, result_lf.collect())

    # verify we don't optimize on chained expressions when last one is not col
    non_optimized_result_plan = (
        lf.group_by("group", maintain_order=True)
        .agg(pl.col("val").abs().is_not_null().any())
        .explain()
    )
    assert "null_count" not in non_optimized_result_plan
    assert "is_not_null" in non_optimized_result_plan

    # edge case of empty series
    lf = pl.LazyFrame({"val": []}, schema={"val": pl.Int32})

    expected_df = pl.DataFrame({"val": [False]})
    result_df = lf.select(pl.col("val").is_not_null().any()).collect()
    assert_frame_equal(expected_df, result_df)


def test_is_null_followed_by_sum() -> None:
    lf = pl.LazyFrame({"group": [0, 0, 0, 1, 2], "val": [6, 0, None, None, 5]})

    expected_df = pl.DataFrame(
        {"group": [0, 1, 2], "val": [1, 1, 0]}, schema_overrides={"val": pl.UInt32}
    )
    result_lf = lf.group_by("group", maintain_order=True).agg(
        pl.col("val").is_null().sum()
    )

    assert r'[col("val").null_count()]' in result_lf.explain()
    assert "is_null" not in result_lf.explain()
    assert_frame_equal(expected_df, result_lf.collect())

    # edge case of empty series
    lf = pl.LazyFrame({"val": []}, schema={"val": pl.Int32})

    expected_df = pl.DataFrame({"val": [0]}, schema={"val": pl.UInt32})
    result_df = lf.select(pl.col("val").is_null().sum()).collect()
    assert_frame_equal(expected_df, result_df)


def test_is_not_null_followed_by_sum() -> None:
    lf = pl.LazyFrame({"group": [0, 0, 0, 1, 2], "val": [6, 0, None, None, 5]})

    expected_df = pl.DataFrame(
        {"group": [0, 1, 2], "val": [2, 0, 1]}, schema_overrides={"val": pl.UInt32}
    )
    result_lf = lf.group_by("group", maintain_order=True).agg(
        pl.col("val").is_not_null().sum()
    )

    assert (
        r'[[(col("val").count()) - (col("val").null_count())]]' in result_lf.explain()
    )
    assert "is_not_null" not in result_lf.explain()
    assert_frame_equal(expected_df, result_lf.collect())

    # verify we don't optimize on chained expressions when last one is not col
    non_optimized_result_lf = lf.group_by("group", maintain_order=True).agg(
        pl.col("val").abs().is_not_null().sum()
    )
    assert "null_count" not in non_optimized_result_lf.explain()
    assert "is_not_null" in non_optimized_result_lf.explain()

    # edge case of empty series
    lf = pl.LazyFrame({"val": []}, schema={"val": pl.Int32})

    expected_df = pl.DataFrame({"val": [0]}, schema={"val": pl.UInt32})
    result_df = lf.select(pl.col("val").is_not_null().sum()).collect()
    assert_frame_equal(expected_df, result_df)


def test_drop_nulls_followed_by_len() -> None:
    lf = pl.LazyFrame({"group": [0, 0, 0, 1, 2], "val": [6, 0, None, None, 5]})

    expected_df = pl.DataFrame(
        {"group": [0, 1, 2], "val": [2, 0, 1]}, schema_overrides={"val": pl.UInt32}
    )
    result_lf = lf.group_by("group", maintain_order=True).agg(
        pl.col("val").drop_nulls().len()
    )

    assert (
        r'[[(col("val").count()) - (col("val").null_count())]]' in result_lf.explain()
    )
    assert "drop_nulls" not in result_lf.explain()
    assert_frame_equal(expected_df, result_lf.collect())

    # verify we don't optimize on chained expressions when last one is not col
    non_optimized_result_plan = (
        lf.group_by("group", maintain_order=True)
        .agg(pl.col("val").abs().drop_nulls().len())
        .explain()
    )
    assert "null_count" not in non_optimized_result_plan
    assert "drop_nulls" in non_optimized_result_plan


def test_drop_nulls_followed_by_count() -> None:
    lf = pl.LazyFrame({"group": [0, 0, 0, 1, 2], "val": [6, 0, None, None, 5]})

    expected_df = pl.DataFrame(
        {"group": [0, 1, 2], "val": [2, 0, 1]}, schema_overrides={"val": pl.UInt32}
    )
    result_lf = lf.group_by("group", maintain_order=True).agg(
        pl.col("val").drop_nulls().count()
    )

    assert (
        r'[[(col("val").count()) - (col("val").null_count())]]' in result_lf.explain()
    )
    assert "drop_nulls" not in result_lf.explain()
    assert_frame_equal(expected_df, result_lf.collect())

    # verify we don't optimize on chained expressions when last one is not col
    non_optimized_result_plan = (
        lf.group_by("group", maintain_order=True)
        .agg(pl.col("val").abs().drop_nulls().count())
        .explain()
    )
    assert "null_count" not in non_optimized_result_plan
    assert "drop_nulls" in non_optimized_result_plan
