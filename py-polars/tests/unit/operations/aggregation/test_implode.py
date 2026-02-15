import polars as pl
from polars.testing import assert_frame_equal


def test_implode_22192_22191() -> None:
    df = pl.DataFrame({"x": [5, 6, 7, 8, 9], "g": [1, 2, 3, 3, 3]})
    assert df.group_by("g").agg(pl.col.x.implode()).sort("x").to_dict(
        as_series=False
    ) == {"g": [1, 2, 3], "x": [[5], [6], [7, 8, 9]]}
    assert df.select(pl.col.x.implode().over("g")).to_dict(as_series=False) == {
        "x": [[5], [6], [7, 8, 9], [7, 8, 9], [7, 8, 9]]
    }


def test_implode_agg_lit() -> None:
    assert_frame_equal(
        pl.DataFrame({"a": [pl.lit(1)]})
        .group_by(pl.lit(1, pl.Int64))
        .agg(x=pl.lit([3]).list.set_union(pl.lit(1).implode())),
        pl.DataFrame({"literal": [1], "x": [[3, 1]]}),
    )


def test_implode_explode_agg() -> None:
    assert_frame_equal(
        pl.DataFrame({"a": [1, 2]})
        .group_by(pl.lit(1, pl.Int64))
        .agg(pl.col.a.implode().explode().sum()),
        pl.DataFrame({"literal": [1], "a": [3]}),
    )
