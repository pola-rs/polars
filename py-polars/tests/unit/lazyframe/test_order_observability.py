from __future__ import annotations

import io
from typing import Any

import pytest

import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal


def test_order_observability() -> None:
    q = pl.LazyFrame({"a": [1, 2, 3], "b": [1, 2, 3]}).sort("a")

    opts = pl.QueryOptFlags(check_order_observe=True)

    assert "SORT" not in q.group_by("a").sum().explain(optimizations=opts)
    assert "SORT" not in q.group_by("a").min().explain(optimizations=opts)
    assert "SORT" not in q.group_by("a").max().explain(optimizations=opts)
    assert "SORT" in q.group_by("a").last().explain(optimizations=opts)
    assert "SORT" in q.group_by("a").first().explain(optimizations=opts)

    # (sort on column: keys) -- missed optimization opportunity for now
    # assert "SORT" not in q.group_by("a").agg(pl.col("b")).explain(optimizations=opts)

    # (sort on columns: agg) -- sort cannot be dropped
    assert "SORT" in q.group_by("b").agg(pl.col("a")).explain(optimizations=opts)


def test_order_observability_group_by_dynamic() -> None:
    assert (
        pl.LazyFrame(
            {"REGIONID": [1, 23, 4], "INTERVAL_END": [32, 43, 12], "POWER": [12, 3, 1]}
        )
        .sort("REGIONID", "INTERVAL_END")
        .group_by_dynamic(index_column="INTERVAL_END", every="1i", group_by="REGIONID")
        .agg(pl.col("POWER").sum())
        .sort("POWER")
        .head()
        .explain()
    ).count("SORT") == 2


def test_remove_double_sort() -> None:
    assert (
        pl.LazyFrame({"a": [1, 2, 3, 3]}).sort("a").sort("a").explain().count("SORT")
        == 1
    )


def test_double_sort_maintain_order_18558() -> None:
    df = pl.DataFrame(
        {
            "col1": [1, 2, 2, 4, 5, 6],
            "col2": [2, 2, 0, 0, 2, None],
        }
    )

    lf = df.lazy().sort("col2").sort("col1", maintain_order=True)

    expect = pl.DataFrame(
        [
            pl.Series("col1", [1, 2, 2, 4, 5, 6], dtype=pl.Int64),
            pl.Series("col2", [2, 0, 2, 0, 2, None], dtype=pl.Int64),
        ]
    )

    assert_frame_equal(lf.collect(), expect)


def test_sort_on_agg_maintain_order() -> None:
    lf = pl.DataFrame(
        {
            "grp": [10, 10, 10, 30, 30, 30, 20, 20, 20],
            "val": [1, 33, 2, 7, 99, 8, 4, 66, 5],
        }
    ).lazy()
    opts = pl.QueryOptFlags(check_order_observe=True)

    out = lf.sort(pl.col("val")).group_by("grp").agg(pl.col("val"))
    assert "SORT" in out.explain(optimizations=opts)

    expected = pl.DataFrame(
        {
            "grp": [10, 20, 30],
            "val": [[1, 2, 33], [4, 5, 66], [7, 8, 99]],
        }
    )
    assert_frame_equal(out.collect(optimizations=opts), expected, check_row_order=False)


@pytest.mark.parametrize(
    ("func", "result"),
    [
        (pl.col("val").cum_sum(), 16),  # (3  + (3+10)) after sort
        (pl.col("val").cum_prod(), 33),  # (3  + (3*10)) after sort
        (pl.col("val").cum_min(), 6),  # (3  + 3) after sort
        (pl.col("val").cum_max(), 13),  # (3  + 10) after sort
    ],
)
def test_sort_agg_with_nested_windowing_22918(func: pl.Expr, result: int) -> None:
    # target pattern: df.sort().group_by().agg(_fooexpr()._barexpr())
    # where _fooexpr is order dependent (e.g., cum_sum)
    # and _barexpr is not order dependent (e.g., sum)

    lf = pl.DataFrame(
        data=[
            {"val": 10, "id": 1, "grp": 0},
            {"val": 3, "id": 0, "grp": 0},
        ]
    ).lazy()

    out = lf.sort("id").group_by("grp").agg(func.sum())
    expected = pl.DataFrame({"grp": 0, "val": result})  # (3  + (3+10)) after sort

    assert_frame_equal(out.collect(), expected)
    assert "SORT" in out.explain()


def test_remove_sorts_on_unordered() -> None:
    lf = pl.LazyFrame({"a": [1, 2, 3]}).sort("a").sort("a").sort("a")
    explain = lf.explain()
    assert explain.count("SORT") == 1

    lf = (
        pl.LazyFrame({"a": [1, 2, 3]})
        .sort("a")
        .group_by("a")
        .agg([])
        .sort("a")
        .group_by("a")
        .agg([])
        .sort("a")
        .group_by("a")
        .agg([])
    )
    explain = lf.explain()
    assert explain.count("SORT") == 0

    lf = (
        pl.LazyFrame({"a": [1, 2, 3]})
        .sort("a")
        .join(pl.LazyFrame({"b": [1, 2, 3]}), on=pl.lit(1))
    )
    explain = lf.explain()
    assert explain.count("SORT") == 0

    lf = pl.LazyFrame({"a": [1, 2, 3]}).sort("a").unique()
    explain = lf.explain()
    assert explain.count("SORT") == 0


def test_merge_sorted_to_union() -> None:
    lf1 = pl.LazyFrame({"a": [1, 2, 3]})
    lf2 = pl.LazyFrame({"a": [2, 3, 4]})

    lf = lf1.merge_sorted(lf2, "a").unique()

    explain = lf.explain(optimizations=pl.QueryOptFlags(check_order_observe=False))
    assert "MERGE_SORTED" in explain
    assert "UNION" not in explain

    explain = lf.explain()
    assert "MERGE_SORTED" not in explain
    assert "UNION" in explain


@pytest.mark.parametrize(
    "order_sensitive_expr",
    [
        pl.arange(0, pl.len()),
        pl.int_range(pl.len()),
        pl.row_index().cast(pl.Int64),
        pl.lit([0, 1, 2, 3, 4], dtype=pl.List(pl.Int64)).explode(),
        pl.lit(pl.Series([0, 1, 2, 3, 4])),
        pl.lit(pl.Series([[0], [1], [2], [3], [4]])).explode(),
        pl.col("y").sort(),
        pl.col("y").sort_by(pl.col("y"), maintain_order=True),
        pl.col("y").sort_by(pl.col("y"), maintain_order=False),
        pl.col("x").gather(pl.col("x")),
    ],
)
def test_order_sensitive_exprs_24335(order_sensitive_expr: pl.Expr) -> None:
    expect = pl.DataFrame(
        {
            "x": [0, 1, 2, 3, 4],
            "y": [3, 4, 0, 1, 2],
            "out": [0, 1, 2, 3, 4],
        }
    )

    q = (
        pl.LazyFrame({"x": [0, 1, 2, 3, 4], "y": [3, 4, 0, 1, 2]})
        .unique(maintain_order=True)
        .with_columns(order_sensitive_expr.alias("out"))
        .unique()
    )

    plan = q.explain()

    assert plan.index("UNIQUE[maintain_order: true") > plan.index("WITH_COLUMNS")

    assert_frame_equal(q.collect().sort(pl.all()), expect)


def assert_correct_ordering(
    lf: pl.LazyFrame,
    expr: pl.Expr,
    *,
    expected: pl.Series | None,
    is_order_observing: bool,
    pad_exprs: list[pl.Expr] | None = None,
) -> None:
    if pad_exprs is None:
        pad_exprs = []
    q = lf.unique(maintain_order=True).select(pad_exprs + [expr]).unique()
    assert ("UNIQUE[maintain_order: true" in q.explain()) == is_order_observing

    result = q.collect()
    if expected is not None:
        unoptimized_result = q.collect(optimizations=pl.QueryOptFlags.none())

        assert_series_equal(
            result.to_series(len(pad_exprs)), expected, check_order=False
        )
        assert_frame_equal(
            result,
            unoptimized_result,
            check_row_order=False,
        )


c = pl.col.a


@pytest.mark.parametrize(
    ("is_order_observing", "agg", "output", "output_dtype"),
    [
        (False, c.min(), 1, pl.Int64()),
        (False, c.count(), 3, pl.get_index_type()),
        (False, c.len(), 3, pl.get_index_type()),
        (False, c.product(), 6, pl.Int64()),
        (False, c.bitwise_or(), 3, pl.Int64()),
        (False, (c == 1).any(), True, pl.Boolean()),
        (False, pl.when(c != 1).then(c).null_count(), 1, pl.get_index_type()),
        (True, c.first(), 2, pl.Int64()),
        (True, c.implode(), [2, 1, 3], pl.List(pl.Int64())),
        (True, c.arg_min(), 1, pl.get_index_type()),
    ],
)
def test_order_sensitive_aggregations_parametric(
    is_order_observing: bool, agg: pl.Expr, output: Any, output_dtype: pl.DataType
) -> None:
    assert_correct_ordering(
        pl.LazyFrame({"a": [2, 1, 3]}),
        agg.alias("agg"),
        expected=pl.Series("agg", [output] * 3, output_dtype),
        is_order_observing=is_order_observing,
        pad_exprs=[pl.col.a],
    )


lf1 = pl.LazyFrame({"a": [3, 1, 2]})
lf2 = pl.LazyFrame({"a": [2, 1, 3]})
lf3 = pl.LazyFrame({"a": [[1, 2], [3]], "b": [[3], [4, 5]]})
lf4 = pl.LazyFrame({"a": [2, 1, 3], "b": [4, 6, 5]})
lf5 = pl.LazyFrame({"a": [2, None, 3]})
lf6 = pl.LazyFrame({"a": [[1], [2]], "b": [[3], [4]]})


@pytest.mark.parametrize(
    ("lf", "expr", "expected", "is_order_observing"),
    [
        (lf1, pl.col.a.sort() * pl.col.a, [3, 2, 6], True),
        (lf1, pl.col.a * pl.col.a, [1, 4, 9], False),
        (
            lf2,
            pl.lit(pl.Series("a", [2, 1, 3, 4])).gather(
                pl.col.a.filter(pl.col.a > 1) - 1
            ),
            [1, 3],
            False,
        ),
        (lf1, pl.col.a.mode(), [1, 2, 3], False),
        (lf2, pl.col.a.gather([0, 2]), [2, 3], True),
        (lf2, pl.col.a, [2, 1, 3], False),
        (lf2, pl.col.a + 1, [3, 2, 4], False),
        (lf2, pl.lit(pl.Series("a", [2, 1, 3, 4])).gather([0, 2]), [2, 3], False),
        (lf2, pl.col.a.filter(pl.col.a != 1), [2, 3], False),
        (lf3, pl.col.a.explode() * pl.col.b.explode(), [3, 8, 15], True),
        (lf4, pl.col.a.sort() + pl.col.b, [5, 8], True),
        (lf4, pl.col.a.sort() + pl.col.b.sort(), [5, 7, 9], False),
        (lf4, pl.col.a + pl.col.b, pl.Series("a", [6, 7, 8]), False),
        (lf4, pl.col.a.unique() * pl.col.b.unique(), None, False),
        (lf5, pl.col.a.drop_nulls(), [2, 3], False),
    ],
)
def test_order_sensitive_paramateric(
    lf: pl.LazyFrame,
    expr: pl.Expr,
    expected: pl.Series | list[Any] | None,
    is_order_observing: bool,
) -> None:
    if isinstance(expected, pl.Series):
        expected = expected.rename("a")
    elif isinstance(expected, list):
        expected = pl.Series("a", expected)

    assert_correct_ordering(
        lf,
        expr.alias("a"),
        expected=expected,
        is_order_observing=is_order_observing,
    )


def test_with_columns_implicit_columns() -> None:
    # Test that overwriting all columns in `with_columns` does not require ordering to
    # be preserved.
    q = (
        lf6.select("a")
        .unique(maintain_order=True)
        .with_columns(pl.col.a.explode())
        .unique()
    )
    assert "UNIQUE[maintain_order: true" not in q.explain()
    assert_series_equal(
        q.collect().to_series(), pl.Series("a", [1, 2]), check_order=False
    )
    q = lf6.unique(maintain_order=True).with_columns(pl.col.a.explode()).unique()
    assert "UNIQUE[maintain_order: true" in q.explain()
    assert_frame_equal(
        q.collect(),
        pl.DataFrame(
            {
                "a": [1, 2],
                "b": [[3], [4]],
            }
        ),
        check_row_order=False,
    )
    q = lf6.unique(maintain_order=True).with_columns(pl.col.a.alias("c")).unique()
    assert "UNIQUE[maintain_order: true" not in q.explain()
    assert_frame_equal(
        q.collect(),
        pl.DataFrame(
            {
                "a": [[1], [2]],
                "b": [[3], [4]],
                "c": [[1], [2]],
            }
        ),
        check_row_order=False,
    )


@pytest.mark.parametrize(
    ("expr", "values", "is_ordered", "is_output_ordered"),
    [
        (pl.col.a, [1, 2, 3], False, False),
        (pl.col.a.map_batches(lambda x: x), [1, 2, 3], True, False),
        (
            pl.col.a.map_batches(lambda x: x, is_elementwise=True),
            [1, 2, 3],
            False,
            False,
        ),
        (
            pl.col.a.cast(pl.List(pl.Int64))
            .map_batches(lambda x: x, is_elementwise=True)
            .explode(),
            [1, 2, 3],
            True,
            False,
        ),
        (pl.col.a.sort(), [1, 2, 3], True, True),
        (pl.col.a.sort() + pl.col.a, None, True, True),
        (pl.col.a.min() + pl.col.a, [2, 3, 4], False, False),
        (pl.col.a.first() + pl.col.a, None, False, False),
    ],
)
def test_group_by_key_sensitivity(
    expr: pl.Expr, values: list[int] | None, is_ordered: bool, is_output_ordered: bool
) -> None:
    lf = pl.LazyFrame({"a": [2, 2, 1, 3], "b": ["A", "B", "C", "D"]}).unique()

    q = lf.group_by(expr.alias("a"), maintain_order=True).agg("b")
    df = q.collect()
    assert ("AGGREGATE[maintain_order: true]" in q.explain()) is is_ordered

    print(q.explain())
    print(df)

    expected_values = pl.Series("a", values)

    if values is not None:
        assert_series_equal(df["a"], expected_values, check_order=is_output_ordered)


@pytest.mark.parametrize(
    ("expr", "is_ordered"),
    [
        (pl.col.a, False),
        (pl.col.a.map_batches(lambda x: x), True),
        (pl.col.a.map_batches(lambda x: x, is_elementwise=True), False),
        (
            pl.col.a.cast(pl.List(pl.Int64))
            .map_batches(lambda x: x, is_elementwise=True)
            .explode(),
            True,
        ),
        (pl.col.a.cum_prod(), True),
        (pl.col.a.cum_prod() + pl.col.a, True),
        (pl.col.a.min() + pl.col.a, False),
        (pl.col.a.first() + pl.col.a, True),
    ],
)
def test_sort_key_sensitivity(expr: pl.Expr, is_ordered: bool) -> None:
    lf = pl.LazyFrame({"a": [2, 2, 1, 3], "b": ["A", "B", "C", "D"]}).sort(pl.all())
    q = lf.sort(expr)
    assert (q.explain().count("SORT BY") == 2) is is_ordered
    assert_frame_equal(q.collect(), lf.sort("a").collect())


@pytest.mark.parametrize(
    ("expr", "is_ordered"),
    [
        (pl.col.a, False),
        (pl.col.a.map_batches(lambda x: x), True),
        (pl.col.a.map_batches(lambda x: x, is_elementwise=True), False),
        (
            pl.col.a.cast(pl.List(pl.Int64))
            .map_batches(lambda x: x, is_elementwise=True)
            .explode(),
            True,
        ),
        (pl.col.a.cum_prod(), True),
        (pl.col.a.cum_prod() + pl.col.a, True),
        (pl.col.a.min() + pl.col.a, False),
        (pl.col.a.first() + pl.col.a, True),
    ],
)
def test_filter_sensitivity(expr: pl.Expr, is_ordered: bool) -> None:
    lf = pl.LazyFrame({"a": [2, 2, 1, 3], "b": ["A", "B", "C", "D"]}).sort(pl.all())
    q = lf.filter(expr > 0).unique()
    assert ("SORT BY" in q.explain()) is is_ordered
    assert_frame_equal(q.collect(), lf.collect(), check_row_order=False)


@pytest.mark.parametrize(
    ("exprs", "is_ordered", "unordered_columns"),
    [
        ([pl.col.a], True, None),
        ([pl.col.a, pl.col.b], True, None),
        ([pl.col.a.unique()], True, ["a"]),
        ([pl.col.a.min()], True, None),
        ([pl.col.a.product()], True, None),
        ([pl.col.a.unique(), pl.col.b], True, ["a"]),
        ([pl.col.a.unique(), pl.col.b.unique()], False, ["a", "b"]),
        ([pl.col.a.min(), pl.col.b.min()], False, None),
        ([pl.col.a.product(), pl.col.b.null_count()], False, None),
        ([pl.col.b.unique()], True, ["b"]),
        ([pl.col.a.unique(), pl.col.b.unique(), pl.col.a.alias("c")], True, ["a", "b"]),
        (
            [pl.col.a.unique(), pl.col.b.unique(), (pl.col.a + 1).unique().alias("c")],
            False,
            ["a", "b", "c"],
        ),
        (
            [pl.col.a.min(), pl.col.b.min(), (pl.col.a + 1).min().alias("c")],
            False,
            None,
        ),
        (
            [
                pl.col.a.product(),
                pl.col.b.null_count(),
                (pl.col.a + 1).product().alias("c"),
            ],
            False,
            None,
        ),
    ],
)
def test_with_columns_sensitivity(
    exprs: list[pl.Expr], is_ordered: bool, unordered_columns: list[str] | None
) -> None:
    lf = (
        pl.LazyFrame({"a": [2, 4, 1, 3], "b": ["A", "C", "B", "D"]})
        .sort("a")
        .with_columns(*exprs)
        .unique(maintain_order=True)
    )
    assert ("UNIQUE[maintain_order: true" in lf.explain()) is is_ordered

    df_opt = lf.collect()
    df_unopt = lf.collect(optimizations=pl.QueryOptFlags(check_order_observe=False))

    if unordered_columns is None:
        assert_frame_equal(df_opt, df_unopt)
    else:
        assert_frame_equal(
            df_opt.drop(unordered_columns), df_unopt.drop(unordered_columns)
        )
        for c in unordered_columns:
            assert_series_equal(df_opt[c], df_unopt[c], check_order=False)


def test_partition_sink_sensitivity() -> None:
    q = (
        pl.LazyFrame({"a": [1, 2, 3]})
        .unique(maintain_order=True)
        .sink_csv(
            pl.PartitionByKey(".", file_path=lambda _: io.BytesIO(), by=pl.col.a),
            lazy=True,
            maintain_order=False,
        )
    )

    assert "UNIQUE[maintain_order: false" in q.explain()

    q = (
        pl.LazyFrame({"a": [1, 2, 3]})
        .unique(maintain_order=True)
        .sink_csv(
            pl.PartitionByKey(
                ".", file_path=lambda _: io.BytesIO(), by=pl.col.a.cum_sum()
            ),
            lazy=True,
            maintain_order=False,
        )
    )

    assert "UNIQUE[maintain_order: true" in q.explain()
