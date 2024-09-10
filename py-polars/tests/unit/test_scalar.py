import pytest

import polars as pl


def test_invalid_broadcast() -> None:
    df = pl.DataFrame(
        {
            "a": [100, 103],
            "group": [0, 1],
        }
    )
    with pytest.raises(pl.exceptions.InvalidOperationError):
        df.select(pl.col("group").filter(pl.col("group") == 0), "a")
