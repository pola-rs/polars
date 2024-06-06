import polars as pl


def test_negative_index() -> None:
    df = pl.DataFrame({"a": [1, 2, 3, 4, 5, 6]})
    assert df.select(pl.col("a").gather([0, -1])).to_dict(as_series=False) == {
        "a": [1, 6]
    }
    assert df.group_by(pl.col("a") % 2).agg(b=pl.col("a").gather([0, -1])).sort(
        "a"
    ).to_dict(as_series=False) == {"a": [0, 1], "b": [[2, 6], [1, 5]]}


def test_gather_agg_schema() -> None:
    df = pl.DataFrame(
        {
            "group": [
                "one",
                "one",
                "one",
                "two",
                "two",
                "two",
            ],
            "value": [1, 98, 2, 3, 99, 4],
        }
    )
    assert (
        df.lazy()
        .group_by("group", maintain_order=True)
        .agg(pl.col("value").get(1))
        .schema["value"]
        == pl.Int64
    )


def test_gather_lit_single_16535() -> None:
    df = pl.DataFrame({"x": [1, 2, 2, 1], "y": [1, 2, 3, 4]})

    assert df.group_by(["x"], maintain_order=True).agg(pl.all().gather([1])).to_dict(
        as_series=False
    ) == {"x": [1, 2], "y": [[4], [3]]}
