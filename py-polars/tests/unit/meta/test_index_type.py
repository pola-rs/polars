import polars as pl


def test_get_index_type() -> None:
    assert pl.get_index_type() == pl.UInt32()
