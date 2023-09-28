from __future__ import annotations

from datetime import datetime
from typing import Any, Sequence

import pytest

import polars as pl
from polars.testing import assert_frame_equal


@pytest.mark.parametrize(
    "sequence",
    [
        [[1, 2], [3, 4, 5]],
        (1, 2, 3),
    ],
)
def test_lit_deprecated_sequence_input(sequence: Sequence[Any]) -> None:
    with pytest.deprecated_call():
        pl.lit(sequence)


def test_lit_ambiguous_datetimes_11379() -> None:
    df = pl.DataFrame(
        {
            "ts": pl.datetime_range(
                datetime(2020, 10, 25),
                datetime(2020, 10, 25, 2),
                "1h",
                time_zone="Europe/London",
                eager=True,
            )
        }
    )
    for i in range(len(df)):
        result = df.filter(pl.col("ts") >= df["ts"][i])
        expected = df[i:]
        assert_frame_equal(result, expected)
