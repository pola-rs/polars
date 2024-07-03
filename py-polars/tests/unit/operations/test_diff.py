import polars as pl
from polars.testing import assert_frame_equal


def test_diff_duration_dtype() -> None:
    data = ["2022-01-01", "2022-01-02", "2022-01-03", "2022-01-03"]
    df = pl.Series("date", data).str.to_date("%Y-%m-%d").to_frame()

    result = df.select(pl.col("date").diff() < pl.duration(days=1))

    expected = pl.Series("date", [None, False, False, True]).to_frame()
    assert_frame_equal(result, expected)
