from __future__ import annotations

from datetime import datetime

import pytest

import polars as pl
from polars.testing.asserts.series import assert_series_equal


def test_clip() -> None:
    clip_exprs = [
        pl.col("a").clip(pl.col("min"), pl.col("max")).alias("clip"),
        pl.col("a").clip(lower_bound=pl.col("min")).alias("clip_min"),
        pl.col("a").clip(upper_bound=pl.col("max")).alias("clip_max"),
    ]

    df = pl.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "min": [0, -1, 4, None, 4],
            "max": [2, 1, 8, 5, None],
        }
    )

    assert df.select(clip_exprs).to_dict(as_series=False) == {
        "clip": [1, 1, 4, None, None],
        "clip_min": [1, 2, 4, None, 5],
        "clip_max": [1, 1, 3, 4, None],
    }

    df = pl.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "min": [0, -1.0, 4.0, None, 4.0],
            "max": [2.0, 1.0, 8.0, 5.0, None],
        }
    )

    assert df.select(clip_exprs).to_dict(as_series=False) == {
        "clip": [1.0, 1.0, 4.0, None, None],
        "clip_min": [1.0, 2.0, 4.0, None, 5.0],
        "clip_max": [1.0, 1.0, 3.0, 4.0, None],
    }

    df = pl.DataFrame(
        {
            "a": [
                datetime(1995, 6, 5, 10, 30),
                datetime(1995, 6, 5),
                datetime(2023, 10, 20, 18, 30, 6),
                None,
                datetime(2023, 9, 24),
                datetime(2000, 1, 10),
            ],
            "min": [
                datetime(1995, 6, 5, 10, 29),
                datetime(1996, 6, 5),
                datetime(2020, 9, 24),
                datetime(2020, 1, 1),
                None,
                datetime(2000, 1, 1),
            ],
            "max": [
                datetime(1995, 7, 21, 10, 30),
                datetime(2000, 1, 1),
                datetime(2023, 9, 20, 18, 30, 6),
                datetime(2000, 1, 1),
                datetime(1993, 3, 13),
                None,
            ],
        }
    )

    assert df.select(clip_exprs).to_dict(as_series=False) == {
        "clip": [
            datetime(1995, 6, 5, 10, 30),
            datetime(1996, 6, 5),
            datetime(2023, 9, 20, 18, 30, 6),
            None,
            None,
            None,
        ],
        "clip_min": [
            datetime(1995, 6, 5, 10, 30),
            datetime(1996, 6, 5),
            datetime(2023, 10, 20, 18, 30, 6),
            None,
            None,
            datetime(2000, 1, 10),
        ],
        "clip_max": [
            datetime(1995, 6, 5, 10, 30),
            datetime(1995, 6, 5),
            datetime(2023, 9, 20, 18, 30, 6),
            None,
            datetime(1993, 3, 13),
            None,
        ],
    }


def test_clip_min_max_deprecated() -> None:
    s = pl.Series([-1, 0, 1])

    with pytest.deprecated_call():
        result = s.clip_min(0)
    expected = pl.Series([0, 0, 1])
    assert_series_equal(result, expected)

    with pytest.deprecated_call():
        result = s.clip_max(0)
    expected = pl.Series([-1, 0, 0])
    assert_series_equal(result, expected)
