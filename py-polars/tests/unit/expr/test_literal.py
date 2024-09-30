import polars as pl


def test_literal_scalar_list_18686() -> None:
    df = pl.DataFrame({"column1": [1, 2], "column2": ["A", "B"]})
    out = df.with_columns(lit1=pl.lit([]).cast(pl.List(pl.String)), lit2=pl.lit([]))

    assert out.to_dict(as_series=False) == {
        "column1": [1, 2],
        "column2": ["A", "B"],
        "lit1": [[], []],
        "lit2": [[], []],
    }
    assert out.schema == pl.Schema(
        [
            ("column1", pl.Int64),
            ("column2", pl.String),
            ("lit1", pl.List(pl.String)),
            ("lit2", pl.List(pl.Null)),
        ]
    )
