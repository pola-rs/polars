import polars as pl
from polars.testing import assert_frame_equal


def test_is_null_followed_by_all() -> None:
    lf = pl.LazyFrame({"group": [0, 0, 0, 1], "val": [6, 0, None, None]})

    expected_df = pl.DataFrame({"group": [0, 1], "val": [False, True]})
    result_df = lf.group_by("group").agg(pl.col("val").is_null().all()).collect()

    assert_frame_equal(expected_df, result_df, check_row_order=False)


def test_is_null_followed_by_any() -> None:
    lf = pl.LazyFrame({"group": [0, 0, 0, 1, 2], "val": [6, 0, None, None, 5]})

    expected_df = pl.DataFrame({"group": [0, 1, 2], "val": [True, True, False]})
    result_df = lf.group_by("group").agg(pl.col("val").is_null().any()).collect()

    assert_frame_equal(expected_df, result_df, check_row_order=False)


def test_is_not_null_followed_by_all() -> None:
    lf = pl.LazyFrame({"group": [0, 0, 0, 1], "val": [6, 0, 5, None]})

    expected_df = pl.DataFrame({"group": [0, 1], "val": [True, False]})
    result_df = lf.group_by("group").agg(pl.col("val").is_not_null().all()).collect()

    assert_frame_equal(expected_df, result_df, check_row_order=False)


def test_is_not_null_followed_by_any() -> None:
    lf = pl.LazyFrame({"group": [0, 0, 0, 1, 2], "val": [6, 0, None, None, 5]})

    expected_df = pl.DataFrame({"group": [0, 1, 2], "val": [True, False, True]})
    result_df = lf.group_by("group").agg(pl.col("val").is_not_null().any()).collect()

    assert_frame_equal(expected_df, result_df, check_row_order=False)


def test_is_null_followed_by_sum() -> None:
    lf = pl.LazyFrame({"group": [0, 0, 0, 1, 2], "val": [6, 0, None, None, 5]})

    expected_df = pl.DataFrame(
        {"group": [0, 1, 2], "val": [1, 1, 0]}, schema_overrides={"val": pl.UInt32}
    )
    result_df = lf.group_by("group").agg(pl.col("val").is_null().sum()).collect()

    assert_frame_equal(expected_df, result_df, check_row_order=False)


def test_is_not_null_followed_by_sum() -> None:
    lf = pl.LazyFrame({"group": [0, 0, 0, 1, 2], "val": [6, 0, None, None, 5]})

    expected_df = pl.DataFrame(
        {"group": [0, 1, 2], "val": [2, 0, 1]}, schema_overrides={"val": pl.UInt32}
    )
    result_df = lf.group_by("group").agg(pl.col("val").is_not_null().sum()).collect()

    assert_frame_equal(expected_df, result_df, check_row_order=False)


def test_drop_nulls_followed_by_len() -> None:
    lf = pl.LazyFrame({"group": [0, 0, 0, 1, 2], "val": [6, 0, None, None, 5]})

    expected_df = pl.DataFrame(
        {"group": [0, 1, 2], "val": [2, 0, 1]}, schema_overrides={"val": pl.UInt32}
    )
    result_df = lf.group_by("group").agg(pl.col("val").drop_nulls().len()).collect()

    assert_frame_equal(expected_df, result_df, check_row_order=False)


def test_drop_nulls_followed_by_count() -> None:
    lf = pl.LazyFrame({"group": [0, 0, 0, 1, 2], "val": [6, 0, None, None, 5]})

    expected_df = pl.DataFrame(
        {"group": [0, 1, 2], "val": [2, 0, 1]}, schema_overrides={"val": pl.UInt32}
    )
    result_df = lf.group_by("group").agg(pl.col("val").drop_nulls().count()).collect()

    assert_frame_equal(expected_df, result_df, check_row_order=False)
