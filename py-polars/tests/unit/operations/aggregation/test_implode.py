import polars as pl


def test_implode_22192_22191() -> None:
    df = pl.DataFrame({"x": [5, 6, 7, 8, 9], "g": [1, 2, 3, 3, 3]})
    assert df.group_by("g").agg(pl.col.x.implode()).sort("x").to_dict(
        as_series=False
    ) == {"g": [1, 2, 3], "x": [[5], [6], [7, 8, 9]]}
    assert df.select(pl.col.x.implode().over("g")).to_dict(as_series=False) == {
        "x": [[5], [6], [7, 8, 9], [7, 8, 9], [7, 8, 9]]
    }


def test_implode_agg_lit() -> None:
    assert (
        pl.DataFrame()
        .group_by(1)
        .agg(
            pl.lit(pl.Series("x", [[3]])).list.set_union(
                pl.lit(pl.Series([1])).implode()
            )
        )
    ).to_dict(as_series=False) == {"literal": [1], "x": [[[3, 1]]]}
