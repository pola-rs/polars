import polars as pl


def test_df_item_negative_row_index() -> None:
    df = pl.DataFrame({"a": [1, 2, 3]})
    assert df.item(-1, 0) == 3
