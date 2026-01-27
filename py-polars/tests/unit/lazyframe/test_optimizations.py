import datetime as dt
import io
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

    assert r'[[(col("val").len()) == (col("val").null_count())]]' in result_lf.explain()
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

    assert r'[[(col("val").null_count()) < (col("val").len())]]' in result_lf.explain()
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
        {"group": [0, 1, 2], "val": [1, 1, 0]},
        schema_overrides={"val": pl.get_index_type()},
    )
    result_lf = lf.group_by("group", maintain_order=True).agg(
        pl.col("val").is_null().sum()
    )

    assert r'[col("val").null_count()]' in result_lf.explain()
    assert "is_null" not in result_lf.explain()
    assert_frame_equal(expected_df, result_lf.collect())

    # edge case of empty series
    lf = pl.LazyFrame({"val": []}, schema={"val": pl.Int32})

    expected_df = pl.DataFrame({"val": [0]}, schema={"val": pl.get_index_type()})
    result_df = lf.select(pl.col("val").is_null().sum()).collect()
    assert_frame_equal(expected_df, result_df)


def test_is_not_null_followed_by_sum() -> None:
    lf = pl.LazyFrame({"group": [0, 0, 0, 1, 2], "val": [6, 0, None, None, 5]})

    expected_df = pl.DataFrame(
        {"group": [0, 1, 2], "val": [2, 0, 1]},
        schema_overrides={"val": pl.get_index_type()},
    )
    result_lf = lf.group_by("group", maintain_order=True).agg(
        pl.col("val").is_not_null().sum()
    )

    assert r'[[(col("val").len()) - (col("val").null_count())]]' in result_lf.explain()
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

    expected_df = pl.DataFrame({"val": [0]}, schema={"val": pl.get_index_type()})
    result_df = lf.select(pl.col("val").is_not_null().sum()).collect()
    assert_frame_equal(expected_df, result_df)


def test_drop_nulls_followed_by_len() -> None:
    lf = pl.LazyFrame({"group": [0, 0, 0, 1, 2], "val": [6, 0, None, None, 5]})

    expected_df = pl.DataFrame(
        {"group": [0, 1, 2], "val": [2, 0, 1]},
        schema_overrides={"val": pl.get_index_type()},
    )
    result_lf = lf.group_by("group", maintain_order=True).agg(
        pl.col("val").drop_nulls().len()
    )

    assert r'[[(col("val").len()) - (col("val").null_count())]]' in result_lf.explain()
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
        {"group": [0, 1, 2], "val": [2, 0, 1]},
        schema_overrides={"val": pl.get_index_type()},
    )
    result_lf = lf.group_by("group", maintain_order=True).agg(
        pl.col("val").drop_nulls().count()
    )

    assert r'[[(col("val").len()) - (col("val").null_count())]]' in result_lf.explain()
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
    assert_frame_equal(
        inner_join.collect(optimizations=pl.QueryOptFlags.none()),
        inner_join.collect(),
        check_row_order=False,
    )

    inner_join = cross.filter(pl.col.x == pl.col.a)
    e = inner_join.explain()
    assert "INNER JOIN" in e
    assert "FILTER" not in e
    assert_frame_equal(
        inner_join.collect(optimizations=pl.QueryOptFlags.none()),
        inner_join.collect(),
        check_row_order=False,
    )

    double_inner_join = cross.filter(pl.col.x == pl.col.a).filter(pl.col.x == pl.col.b)
    e = double_inner_join.explain()
    assert "INNER JOIN" in e
    assert "FILTER" not in e
    assert_frame_equal(
        double_inner_join.collect(optimizations=pl.QueryOptFlags.none()),
        double_inner_join.collect(),
        check_row_order=False,
    )

    dont_mix = cross.filter(pl.col.x + pl.col.a != 0)
    e = dont_mix.explain()
    assert "NESTED LOOP JOIN" in e
    assert "FILTER" not in e
    assert_frame_equal(
        dont_mix.collect(optimizations=pl.QueryOptFlags.none()),
        dont_mix.collect(),
        check_row_order=False,
    )

    iejoin = cross.filter(pl.col.x >= pl.col.a)
    e = iejoin.explain()
    assert "IEJOIN" in e
    assert "NESTED LOOP JOIN" not in e
    assert "CROSS JOIN" not in e
    assert "FILTER" not in e
    assert_frame_equal(
        iejoin.collect(optimizations=pl.QueryOptFlags.none()),
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
        iejoin.collect(optimizations=pl.QueryOptFlags.none()),
        iejoin.collect(),
        check_row_order=False,
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
                unoptimized = cross.filter(e).collect(
                    optimizations=pl.QueryOptFlags.none()
                )

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
                    print(
                        cross.filter(e).explain(optimizations=pl.QueryOptFlags.none())
                    )
                    print(unoptimized)
                    print()

                    raise


def test_order_observe_sort_before_unique_22485() -> None:
    lf = pl.LazyFrame(
        {
            "order": [3, 2, 1],
            "id": ["A", "A", "B"],
        }
    )

    expect = pl.DataFrame({"order": [1, 3], "id": ["B", "A"]})

    q = lf.sort("order").unique(["id"], keep="last").sort("order")

    plan = q.explain()
    assert "SORT BY" in plan[plan.index("UNIQUE") :]

    assert_frame_equal(q.collect(), expect)

    q = lf.sort("order").unique(["id"], keep="last", maintain_order=True)

    plan = q.explain()
    assert "SORT BY" in plan[plan.index("UNIQUE") :]

    assert_frame_equal(q.collect(), expect)


def test_order_observe_group_by() -> None:
    q = (
        pl.LazyFrame({"a": range(5)})
        .group_by("a", maintain_order=True)
        .agg(b=1)
        .sort("b")
    )

    plan = q.explain()
    assert "AGGREGATE[maintain_order: false]" in plan

    q = (
        pl.LazyFrame({"a": range(5)})
        .group_by("a", maintain_order=True)
        .agg(b=1)
        .sort("b", maintain_order=True)
    )

    plan = q.explain()
    assert "AGGREGATE[maintain_order: true]" in plan


def test_fused_correct_name() -> None:
    df = pl.DataFrame({"x": [1, 2, 3]})

    lf = df.lazy().select(
        (pl.col.x.alias("a") * pl.col.x.alias("b")) + pl.col.x.alias("c")
    )

    no_opts = lf.collect(optimizations=pl.QueryOptFlags.none())
    opts = lf.collect()
    assert_frame_equal(
        no_opts,
        opts,
    )
    assert_frame_equal(opts, pl.DataFrame({"a": [2, 6, 12]}))


def test_slice_pushdown_within_concat_24734() -> None:
    q = pl.concat(
        [
            pl.LazyFrame({"x": [0, 1, 2, 3, 4]}).head(2),
            pl.LazyFrame(schema={"x": pl.Int64}),
        ]
    )

    plan = q.explain()
    assert "SLICE" not in plan

    assert_frame_equal(q, pl.LazyFrame({"x": [0, 1]}))

    q = pl.concat(
        [
            pl.LazyFrame({"x": [0, 1, 2, 3, 4]}).select(pl.col("x").reverse()),
            pl.LazyFrame(schema={"x": pl.Int64}),
        ]
    ).slice(1, 2)

    plan = q.explain()
    assert plan.index("SLICE[offset: 0, len: 3]") > plan.index("PLAN 0:")

    assert_frame_equal(q, pl.LazyFrame({"x": [3, 2]}))


def test_is_between_pushdown_25499() -> None:
    f = io.BytesIO()
    pl.LazyFrame(
        {"a": [0, 1, 2, 3, 4]}, schema_overrides={"a": pl.UInt32}
    ).sink_parquet(f)
    parquet = f.getvalue()

    expr = pl.lit(3, dtype=pl.UInt32).is_between(
        pl.lit(1, dtype=pl.UInt32), pl.col("a")
    )

    df1 = pl.scan_parquet(parquet).filter(expr).collect()
    df2 = pl.scan_parquet(parquet).collect().filter(expr)
    assert_frame_equal(df1, df2)


def test_slice_pushdown_expr_25473() -> None:
    lf = pl.LazyFrame({"a": [0, 1, 2, 3, 4]})

    assert_frame_equal(
        lf.select((pl.col("a") + 1).slice(-4, 2)).collect(), pl.DataFrame({"a": [2, 3]})
    )

    assert_frame_equal(
        lf.select(
            a=(
                pl.when(pl.col("a") == 1).then(pl.lit("one")).otherwise(pl.lit("other"))
            ).slice(-4, 2)
        ).collect(),
        pl.DataFrame({"a": ["one", "other"]}),
    )

    assert_frame_equal(
        lf.select(a=pl.col("a").is_in(pl.Series([1]).implode()).slice(-4, 2)).collect(),
        pl.DataFrame({"a": [True, False]}),
    )

    q = pl.LazyFrame().select(
        pl.lit(pl.Series([0, 1, 2, 3, 4])).is_in(pl.Series([[3], [1]])).slice(-2, 1)
    )

    with pytest.raises(pl.exceptions.ShapeError, match=r"lengths.*5 != 2"):
        q.collect()


def test_lazy_groupby_maintain_order_after_asof_join_25973() -> None:
    # Small target times: 00:00, 00:10, 00:20, 00:30
    targettime = (
        pl.DataFrame(
            {
                "targettime": pl.time_range(
                    dt.time(0, 0),
                    dt.time(0, 30),
                    interval="10m",
                    closed="both",
                    eager=True,
                )
            }
        )
        .with_columns(
            targettime=pl.lit(dt.date(2026, 1, 1)).dt.combine(pl.col("targettime")),
            grp=pl.lit(1),
        )
        .lazy()
    )

    # Small input times: every second from 00:00 to 00:30
    df = (
        pl.DataFrame(
            {
                "time": pl.time_range(
                    dt.time(0, 0),
                    dt.time(0, 30),
                    interval="1s",
                    closed="both",
                    eager=True,
                )
            }
        )
        .with_row_index("value")
        .with_columns(
            time=pl.lit(dt.date(2026, 1, 1)).dt.combine(pl.col("time")),
            grp=pl.lit(1),
        )
        .lazy()
    )

    # This used to produce out-of-order results.
    # The optimizer previously cleared maintain_order.
    q = (
        df.join_asof(
            targettime,
            left_on="time",
            right_on="targettime",
            strategy="forward",
        )
        .drop_nulls("targettime")
        .group_by("targettime", maintain_order=True)
        .agg(pl.col("value").last())
    )

    # Verify optimizer preserves maintain_order on UNIQUE
    plan = q.explain()
    assert "AGGREGATE[maintain_order: true" in plan

    result = q.collect()

    idx_dtype = pl.get_index_type()

    expected = pl.DataFrame(
        {
            "targettime": [
                dt.datetime(2026, 1, 1, 0, 0),
                dt.datetime(2026, 1, 1, 0, 10),
                dt.datetime(2026, 1, 1, 0, 20),
                dt.datetime(2026, 1, 1, 0, 30),
            ],
            "value": pl.Series("value", [0, 600, 1200, 1800], dtype=idx_dtype),
        }
    )

    assert_frame_equal(result, expected)


def test_fast_count_alias_18581() -> None:
    f = io.BytesIO()
    f.write(b"a,b,c\n1,2,3\n4,5,6")
    f.flush()
    f.seek(0)

    df = pl.scan_csv(f).select(pl.len().alias("weird_name")).collect()

    # Just check the value, let assert_frame_equal handle dtype matching
    expected = pl.DataFrame(
        {"weird_name": [2]}, schema={"weird_name": pl.get_index_type()}
    )
    assert_frame_equal(expected, df)


def test_flatten_alias() -> None:
    assert (
        """len().alias("bar")"""
        in pl.LazyFrame({"a": [1, 2]})
        .select(pl.len().alias("foo").alias("bar"))
        .explain()
    )
