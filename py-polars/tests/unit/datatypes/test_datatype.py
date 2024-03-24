import copy

import polars as pl


# https://github.com/pola-rs/polars/issues/14771
def test_datatype_copy() -> None:
    dtype = pl.Int64()
    result = copy.deepcopy(dtype)
    assert dtype == dtype
    assert isinstance(result, pl.Int64)
