import polars as pl
from polars.testing import assert_frame_equal


def test_order_observability() -> None:
    q = pl.LazyFrame({"a": [1, 2, 3], "b": [1, 2, 3]}).sort("a")

    assert "SORT" not in q.group_by("a").sum().explain(_check_order=True)
    assert "SORT" not in q.group_by("a").min().explain(_check_order=True)
    assert "SORT" not in q.group_by("a").max().explain(_check_order=True)
    assert "SORT" in q.group_by("a").last().explain(_check_order=True)
    assert "SORT" in q.group_by("a").first().explain(_check_order=True)


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
