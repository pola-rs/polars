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
        pl.DataFrame()
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


def test_implode_unordered() -> None:
    df = pl.DataFrame({"x": [5, 6, 7, 8, 9], "g": [1, 2, 3, 3, 3]})
    out = (
        df.group_by("g")
        .agg(pl.col.x.implode(maintain_order=False))
        .sort("g")
        .to_dict(as_series=False)
    )
    out["x"] = [sorted(v) for v in out["x"]]
    assert out == {"g": [1, 2, 3], "x": [[5], [6], [7, 8, 9]]}

    df = pl.DataFrame({"x": [1, 2, 5], "y": [3, 4, 3]})
    out = (
        df.group_by("y")
        .agg(pl.struct(pl.col.x, pl.col.y).implode(maintain_order=False))
        .sort("y")
        .to_dict(as_series=False)
    )
    out["x"] = [sorted(list(d.items()) for d in v) for v in out["x"]]
    assert out == {
        "y": [3, 4],
        "x": [[[("x", 1), ("y", 3)], [("x", 5), ("y", 3)]], [[("x", 2), ("y", 4)]]],
    }


def test_implode_and_list_sort_aggregated_27252() -> None:
    df = pl.DataFrame(
        {"a": [1, 1, 1, 2, 2, 2, 3, 3, 3], "b": [1, 2, 1, 3, 1, 2, 3, 3, 1]}
    )

    q = (
        df.lazy()
        .group_by("a")
        .agg(pl.col.b.implode().list.unique(maintain_order=True).sort())
    )

    expected = pl.DataFrame({"a": [1, 2, 3], "b": [[1, 2], [3, 1, 2], [3, 1]]})
    assert_frame_equal(q.collect(), expected, check_row_order=False)
