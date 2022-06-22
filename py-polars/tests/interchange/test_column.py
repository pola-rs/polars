import polars as pl
from polars.interchange.column import _IXColumn

SERIES = pl.Series([None, 1, 0])


def test_column() -> None:
    column = _IXColumn(SERIES)
    assert column.null_count == 1
    # assert column.dtype == TODO
    assert column.offset == 0
    # polars does not make use of metadata for ix
    assert not column.metadata
    # assert column.size
