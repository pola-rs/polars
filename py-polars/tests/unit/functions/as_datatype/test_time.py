import polars as pl
from polars.testing import assert_series_equal


def test_time() -> None:
    df = pl.DataFrame(
        {
            "hour": [7, 14, 21],
            "min": [10, 20, 30],
            "sec": [15, 30, 45],
            "micro": [123456, 555555, 987654],
        }
    )
    out = df.select(
        pl.all(),
        pl.time("hour", "min", "sec", "micro").dt.hour().cast(int).alias("h2"),
        pl.time("hour", "min", "sec", "micro").dt.minute().cast(int).alias("m2"),
        pl.time("hour", "min", "sec", "micro").dt.second().cast(int).alias("s2"),
        pl.time("hour", "min", "sec", "micro").dt.microsecond().cast(int).alias("ms2"),
    )
    assert_series_equal(out["h2"], df["hour"].rename("h2"))
    assert_series_equal(out["m2"], df["min"].rename("m2"))
    assert_series_equal(out["s2"], df["sec"].rename("s2"))
    assert_series_equal(out["ms2"], df["micro"].rename("ms2"))
