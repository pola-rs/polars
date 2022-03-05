import io

import pytest

import polars as pl


def test_not_found_error() -> None:
    csv = "a,b,c\n2,1,1"
    df = pl.read_csv(io.StringIO(csv))
    with pytest.raises(pl.NotFoundError):
        df.select("d")
