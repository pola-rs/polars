from datetime import timedelta

import polars as pl


def test_duration_cumsum() -> None:
    df = pl.DataFrame({"A": [timedelta(days=1), timedelta(days=2)]})
    assert df.select(pl.col("A").cumsum()).to_dict(False) == {
        "A": [timedelta(days=1), timedelta(days=3)]
    }
