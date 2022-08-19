import polars as pl


def test_simplify_expression_lit_true_4376() -> None:
    df = pl.DataFrame([[1, 4, 7], [2, 5, 8], [3, 6, 9]])
    assert df.lazy().filter(pl.lit(True) | (pl.col("column_0") == 1)).collect(
        simplify_expression=True
    ).shape == (3, 3)
    assert df.lazy().filter((pl.col("column_0") == 1) | pl.lit(True)).collect(
        simplify_expression=True
    ).shape == (3, 3)


def test_melt_values_predicate_pushdown() -> None:
    lf = pl.DataFrame(
        {
            "id": [1],
            "asset_key_1": ["123"],
            "asset_key_2": ["456"],
            "asset_key_3": ["abc"],
        }
    ).lazy()

    assert (
        lf.melt("id", ["asset_key_1", "asset_key_2", "asset_key_3"])
        .filter(pl.col("value") == pl.lit("123"))
        .collect()
    ).to_dict(False) == {"id": [1], "variable": ["asset_key_1"], "value": ["123"]}
