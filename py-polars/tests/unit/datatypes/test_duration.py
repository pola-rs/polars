from datetime import timedelta

import polars as pl


def test_duration_cumsum() -> None:
    df = pl.DataFrame({"A": [timedelta(days=1), timedelta(days=2)]})

    assert df.select(pl.col("A").cumsum()).to_dict(False) == {
        "A": [timedelta(days=1), timedelta(days=3)]
    }
    assert df.schema["A"].is_(pl.Duration(time_unit="us"))
    for duration_dtype in (
        pl.Duration,
        pl.Duration(time_unit="ms"),
        pl.Duration(time_unit="ns"),
    ):
        assert df.schema["A"].is_not(duration_dtype)  # type: ignore[arg-type]
