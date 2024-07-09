import polars as pl


def test_expression_15183() -> None:
    assert (
        pl.DataFrame(
            {"a": [1, 2, 3, 4, 5, 2, 3, 5, 1], "b": [1, 2, 3, 1, 2, 3, 1, 2, 3]}
        )
        .group_by("a")
        .agg(pl.col.b.unique().sort().str.join("-").str.split("-"))
        .sort("a")
    ).to_dict(as_series=False) == {
        "a": [1, 2, 3, 4, 5],
        "b": [["1", "3"], ["2", "3"], ["1", "3"], ["1"], ["2"]],
    }
