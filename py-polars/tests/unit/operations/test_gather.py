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
        .collect_schema()["value"]
        == pl.Int64
    )


def test_gather_lit_single_16535() -> None:
    df = pl.DataFrame({"x": [1, 2, 2, 1], "y": [1, 2, 3, 4]})

    assert df.group_by(["x"], maintain_order=True).agg(pl.all().gather([1])).to_dict(
        as_series=False
    ) == {"x": [1, 2], "y": [[4], [3]]}


def test_list_get_null_offset_17248() -> None:
    df = pl.DataFrame({"material": [["PB", "PVC", "CI"], ["CI"], ["CI"]]})

    assert df.select(
        result=pl.when(pl.col.material.list.len() == 1).then("material").list.get(0),
    )["result"].to_list() == [None, "CI", "CI"]


def test_list_get_null_oob_17252() -> None:
    df = pl.DataFrame(
        {
            "name": ["BOB-3", "BOB", None],
        }
    )

    split = df.with_columns(pl.col("name").str.split("-"))
    assert split.with_columns(pl.col("name").list.get(0))["name"].to_list() == [
        "BOB",
        "BOB",
        None,
    ]
