import polars as pl


def test_pass_name_alias_18914() -> None:
    df = pl.DataFrame({"id": [1], "value": [2]})

    assert df.select(
        pl.all()
        .map_elements(
            lambda x: x,
            skip_nulls=False,
            pass_name=True,
            return_dtype=pl.List(pl.Int64),
        )
        .over("id")
    ).to_dict(as_series=False) == {"id": [1], "value": [2]}
