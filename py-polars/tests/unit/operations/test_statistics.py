from datetime import timedelta

import polars as pl
from polars.testing import assert_frame_equal


def test_corr() -> None:
    df = pl.DataFrame(
        {
            "a": [1, 2, 4],
            "b": [-1, 23, 8],
        }
    )
    result = df.corr()
    expected = pl.DataFrame(
        {
            "a": [1.0, 0.18898223650461357],
            "b": [0.1889822365046136, 1.0],
        }
    )
    assert_frame_equal(result, expected)


def test_hist() -> None:
    a = pl.Series("a", [1, 3, 8, 8, 2, 1, 3])
    assert (
        str(a.hist(bin_count=4).to_dict(as_series=False))
        == "{'break_point': [0.0, 2.25, 4.5, 6.75, inf], 'category': ['(-inf, 0.0]', '(0.0, 2.25]', '(2.25, 4.5]', '(4.5, 6.75]', '(6.75, inf]'], 'a_count': [0, 3, 2, 0, 2]}"
    )


def test_median_quantile_duration() -> None:
    df = pl.DataFrame({"A": [timedelta(days=0), timedelta(days=1)]})

    result = df.select(pl.col("A").median())
    expected = pl.DataFrame({"A": [timedelta(seconds=43200)]})
    assert_frame_equal(result, expected)

    result = df.select(pl.col("A").quantile(0.5, interpolation="linear"))
    expected = pl.DataFrame({"A": [timedelta(seconds=43200)]})
    assert_frame_equal(result, expected)


def test_correlation_cast_supertype() -> None:
    df = pl.DataFrame({"a": [1, 8, 3], "b": [4.0, 5.0, 2.0]})
    df = df.with_columns(pl.col("b"))
    assert df.select(pl.corr("a", "b")).to_dict(as_series=False) == {
        "a": [0.5447047794019223]
    }
