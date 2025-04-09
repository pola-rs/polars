import pytest

import polars as pl
from polars.exceptions import ComputeError
from polars.testing import assert_frame_equal


def test_diff_duration_dtype() -> None:
    data = ["2022-01-01", "2022-01-02", "2022-01-03", "2022-01-03"]
    df = pl.Series("date", data).str.to_date("%Y-%m-%d").to_frame()

    result = df.select(pl.col("date").diff() < pl.duration(days=1))

    expected = pl.Series("date", [None, False, False, True]).to_frame()
    assert_frame_equal(result, expected)


def test_diff_scalarity() -> None:
    df = pl.DataFrame(
        {
            "a": [1, 2, 3, 2, 2, 3, 0],
            "n": [1, 3, 2, 4, 5, 1, 1],
        }
    )

    with pytest.raises(ComputeError, match="'n' must be scalar value"):
        df.select(pl.col("a").diff("n"))

    result = df.select(pl.col("a").diff(pl.col("n").mean().cast(pl.Int32)))
    expected = pl.DataFrame({"a": [None, None, 2, 0, -1, 1, -2]})
    assert_frame_equal(result, expected)

    result = df.select(pl.col("a").diff(2))
    assert_frame_equal(result, expected)
