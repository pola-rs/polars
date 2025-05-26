import pytest

import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal


def test_implode_explode_over_22188() -> None:
    df = pl.DataFrame(
        {
            "x": [1, 2, 3, 1, 2, 3, 1, 2, 3],
            "y": [2, 2, 2, 3, 3, 3, 4, 4, 4],
        }
    )
    result = df.select(
        (pl.col.x * (pl.lit(pl.Series([1, 1, 1])).implode().explode())).over(pl.col.y),
    )

    assert_series_equal(result.to_series(), df.get_column("x"))


def test_implode_in_over_22188() -> None:
    df = pl.DataFrame(
        {
            "x": [[1], [2], [3]],
            "y": [2, 3, 4],
        }
    ).select(pl.col.x.list.set_union(pl.lit(pl.Series([1])).implode()).over(pl.col.y))
    assert_series_equal(df.to_series(), pl.Series("x", [[1], [2, 1], [3, 1]]))


def test_over_no_partition_by() -> None:
    df = pl.DataFrame({"a": [1, 1, 2], "i": [2, 1, 3]})
    result = df.with_columns(b=pl.col("a").cum_sum().over(order_by="i"))
    expected = pl.DataFrame({"a": [1, 1, 2], "i": [2, 1, 3], "b": [2, 1, 4]})
    assert_frame_equal(result, expected)


def test_over_no_partition_by_no_over() -> None:
    df = pl.DataFrame({"a": [1, 1, 2], "i": [2, 1, 3]})
    with pytest.raises(pl.exceptions.InvalidOperationError):
        df.with_columns(b=pl.col("a").cum_sum().over())


def test_over_explode_22770() -> None:
    df = pl.DataFrame({"x": [[1.0], [2.0]], "idx": [1, 2]})
    e = pl.col("x").list.explode().over("idx", mapping_strategy="join")

    assert_frame_equal(
        df.select(pl.col("x").list.diff()),
        df.select(e.list.diff()),
    )


def test_over_replace_strict_22870() -> None:
    lookup = pl.DataFrame(
        {
            "cat": ["a", "b", "c"],
            "val": [102, 100, 101],
        }
    )

    df = pl.DataFrame(
        {
            "cat": ["a", "b", "a", "a", "b"],
            "data": [2, 3, 4, 5, 6],
            "a": ["a", "b", "c", "d", "e"],
            "b": [102, 100, 101, 109, 110],
        }
    )

    out = (
        df.lazy()
        .select(
            pl.col("cat")
            .replace_strict(lookup["cat"], lookup["val"], default=-1)
            .alias("val"),
            pl.col("cat")
            .replace_strict(lookup["cat"], lookup["val"], default=-1)
            .over("cat")
            .alias("val_over"),
        )
        .collect()
    )
    assert_series_equal(
        out.get_column("val"), out.get_column("val_over"), check_names=False
    )

    out = (
        df.lazy()
        .select(
            pl.col("cat").replace_strict(pl.col.a, pl.col.b, default=-1).alias("val"),
            pl.col("cat")
            .replace_strict(pl.col.a, pl.col.b, default=-1)
            .over("cat")
            .alias("val_over"),
        )
        .collect()
    )
    assert_series_equal(
        out.get_column("val"), out.get_column("val_over"), check_names=False
    )
