import polars as pl
from polars.testing import assert_series_equal


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
