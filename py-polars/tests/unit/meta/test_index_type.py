import polars as pl


def test_get_index_type() -> None:
    len_type = pl.DataFrame({"a": []}).select(pl.len()).schema["len"]
    index_type = pl.DataFrame({"a": []}).with_row_index().schema["index"]
    assert pl.get_index_type() == pl.Int64
    assert pl.get_index_type() == index_type
