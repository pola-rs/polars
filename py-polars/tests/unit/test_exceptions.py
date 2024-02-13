import pytest

import polars as pl


def test_base_class() -> None:
    assert isinstance(pl.ComputeError("msg"), pl.PolarsError)
    msg = "msg"
    with pytest.raises(pl.PolarsError, match=msg):
        raise pl.OutOfBoundsError(msg)
