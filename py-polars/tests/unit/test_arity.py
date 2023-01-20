import polars as pl


def test_nested_when_then_and_wildcard_expansion_6284() -> None:
    df = pl.DataFrame(
        {
            "1": ["a", "b"],
            "2": ["c", "d"],
        }
    )

    out0 = df.with_column(
        pl.when(pl.any(pl.all() == "a"))
        .then("a")
        .otherwise(pl.when(pl.any(pl.all() == "d")).then("d").otherwise(None))
        .alias("result")
    )

    out1 = df.with_column(
        pl.when(pl.any(pl.all() == "a"))
        .then("a")
        .when(pl.any(pl.all() == "d"))
        .then("d")
        .otherwise(None)
        .alias("result")
    )

    assert out0.frame_equal(out1)
    assert out0.to_dict(False) == {
        "1": ["a", "b"],
        "2": ["c", "d"],
        "result": ["a", "d"],
    }
