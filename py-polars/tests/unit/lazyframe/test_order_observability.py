import pytest

import polars as pl
from polars.testing import assert_frame_equal


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
