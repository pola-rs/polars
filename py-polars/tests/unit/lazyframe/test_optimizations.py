import itertools

import pytest

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


def test_collapse_joins() -> None:
    a = pl.LazyFrame({"a": [1, 2, 3], "b": [2, 2, 2]})
    b = pl.LazyFrame({"x": [7, 1, 2]})

    cross = a.join(b, how="cross")

    inner_join = cross.filter(pl.col.a == pl.col.x)
    e = inner_join.explain()
    assert "INNER JOIN" in e
    assert "FILTER" not in e
    assert_frame_equal(inner_join.collect(collapse_joins=False), inner_join.collect())

    inner_join = cross.filter(pl.col.x == pl.col.a)
    e = inner_join.explain()
    assert "INNER JOIN" in e
    assert "FILTER" not in e
    assert_frame_equal(
        inner_join.collect(collapse_joins=False),
        inner_join.collect(),
        check_row_order=False,
    )

    double_inner_join = cross.filter(pl.col.x == pl.col.a).filter(pl.col.x == pl.col.b)
    e = double_inner_join.explain()
    assert "INNER JOIN" in e
    assert "FILTER" not in e
    assert_frame_equal(
        double_inner_join.collect(collapse_joins=False),
        double_inner_join.collect(),
        check_row_order=False,
    )

    dont_mix = cross.filter(pl.col.x + pl.col.a != 0)
    e = dont_mix.explain()
    assert "NESTED LOOP JOIN" in e
    assert "FILTER" not in e
    assert_frame_equal(
        dont_mix.collect(collapse_joins=False),
        dont_mix.collect(),
        check_row_order=False,
    )

    no_literals = cross.filter(pl.col.x == 2)
    e = no_literals.explain()
    assert "NESTED LOOP JOIN" in e
    assert_frame_equal(
        no_literals.collect(collapse_joins=False),
        no_literals.collect(),
        check_row_order=False,
    )

    iejoin = cross.filter(pl.col.x >= pl.col.a)
    e = iejoin.explain()
    assert "IEJOIN" in e
    assert "NESTED LOOP JOIN" not in e
    assert "CROSS JOIN" not in e
    assert "FILTER" not in e
    assert_frame_equal(
        iejoin.collect(collapse_joins=False),
        iejoin.collect(),
        check_row_order=False,
    )

    iejoin = cross.filter(pl.col.x >= pl.col.a).filter(pl.col.x <= pl.col.b)
    e = iejoin.explain()
    assert "IEJOIN" in e
    assert "CROSS JOIN" not in e
    assert "NESTED LOOP JOIN" not in e
    assert "FILTER" not in e
    assert_frame_equal(
        iejoin.collect(collapse_joins=False), iejoin.collect(), check_row_order=False
    )


@pytest.mark.slow
def test_collapse_joins_combinations() -> None:
    # This just tests all possible combinations for expressions on a cross join.

    a = pl.LazyFrame({"a": [1, 2, 3], "x": [7, 2, 1]})
    b = pl.LazyFrame({"b": [2, 2, 2], "x": [7, 1, 3]})

    cross = a.join(b, how="cross")

    exprs = []

    for lhs in [pl.col.a, pl.col.b, pl.col.x, pl.lit(1), pl.col.a + pl.col.b]:
        for rhs in [pl.col.a, pl.col.b, pl.col.x, pl.lit(1), pl.col.a * pl.col.x]:
            for cmp in ["__eq__", "__ge__", "__lt__"]:
                e = (getattr(lhs, cmp))(rhs)
                exprs.append(e)

    for amount in range(3):
        for merge in itertools.product(["__and__", "__or__"] * (amount - 1)):
            for es in itertools.product(*([exprs] * amount)):
                e = es[0]
                for i in range(amount - 1):
                    e = (getattr(e, merge[i]))(es[i + 1])

                # NOTE: We need to sort because the order of the cross-join &
                # IE-join is unspecified. Therefore, this might not necessarily
                # create the exact same dataframe.
                optimized = cross.filter(e).sort(pl.all()).collect()
                unoptimized = cross.filter(e).collect(collapse_joins=False)

                try:
                    assert_frame_equal(optimized, unoptimized, check_row_order=False)
                except:
                    print(e)
                    print()
                    print("Optimized")
                    print(cross.filter(e).explain())
                    print(optimized)
                    print()
                    print("Unoptimized")
                    print(cross.filter(e).explain(collapse_joins=False))
                    print(unoptimized)
                    print()

                    raise


def test_select_after_join_where_20831() -> None:
    left = pl.LazyFrame(
        {
            "a": [1, 2, 3, 1, None],
            "b": [1, 2, 3, 4, 5],
            "c": [2, 3, 4, 5, 6],
        }
    )

    right = pl.LazyFrame(
        {
            "a": [1, 4, 3, 7, None, None, 1],
            "c": [2, 3, 4, 5, 6, 7, 8],
            "d": [6, None, 7, 8, -1, 2, 4],
        }
    )

    q = left.join_where(
        right, pl.col("b") * 2 <= pl.col("a_right"), pl.col("a") < pl.col("c_right")
    )

    assert_frame_equal(
        q.select("d").collect().sort("d"),
        pl.Series("d", [None, None, 7, 8, 8, 8]).to_frame(),
    )

    assert q.select(pl.len()).collect().item() == 6

    q = (
        left.join(right, how="cross")
        .filter(pl.col("b") * 2 <= pl.col("a_right"))
        .filter(pl.col("a") < pl.col("c_right"))
    )

    assert_frame_equal(
        q.select("d").collect().sort("d"),
        pl.Series("d", [None, None, 7, 8, 8, 8]).to_frame(),
    )

    assert q.select(pl.len()).collect().item() == 6
