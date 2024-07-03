import sys
from datetime import datetime

import pytest

from polars.dependencies import _ZONEINFO_AVAILABLE

if sys.version_info >= (3, 9):
    from zoneinfo import ZoneInfo
elif _ZONEINFO_AVAILABLE:
    # Import from submodule due to typing issue with backports.zoneinfo package:
    # https://github.com/pganssle/zoneinfo/issues/125
    from backports.zoneinfo._zoneinfo import ZoneInfo

import polars as pl


def test_cross_join_predicate_pushdown_block_16956() -> None:
    lf = pl.LazyFrame(
        [
            [1718085600000, 1718172000000, 1718776800000],
            [1718114400000, 1718200800000, 1718805600000],
        ],
        schema=["start_datetime", "end_datetime"],
    ).cast(pl.Datetime("ms", "Europe/Amsterdam"))

    assert (
        lf.join(lf, on="start_datetime", how="cross")
        .filter(
            pl.col.end_datetime_right.is_between(
                pl.col.start_datetime, pl.col.start_datetime.dt.offset_by("132h")
            )
        )
        .select("start_datetime", "end_datetime_right")
    ).collect(predicate_pushdown=True).to_dict(as_series=False) == {
        "start_datetime": [
            datetime(2024, 6, 11, 8, 0, tzinfo=ZoneInfo(key="Europe/Amsterdam")),
            datetime(2024, 6, 11, 8, 0, tzinfo=ZoneInfo(key="Europe/Amsterdam")),
            datetime(2024, 6, 12, 8, 0, tzinfo=ZoneInfo(key="Europe/Amsterdam")),
            datetime(2024, 6, 19, 8, 0, tzinfo=ZoneInfo(key="Europe/Amsterdam")),
        ],
        "end_datetime_right": [
            datetime(2024, 6, 11, 16, 0, tzinfo=ZoneInfo(key="Europe/Amsterdam")),
            datetime(2024, 6, 12, 16, 0, tzinfo=ZoneInfo(key="Europe/Amsterdam")),
            datetime(2024, 6, 12, 16, 0, tzinfo=ZoneInfo(key="Europe/Amsterdam")),
            datetime(2024, 6, 19, 16, 0, tzinfo=ZoneInfo(key="Europe/Amsterdam")),
        ],
    }


def test_cross_join_raise_on_keys() -> None:
    df = pl.DataFrame({"a": [0, 1], "b": ["x", "y"]})

    with pytest.raises(ValueError):
        df.join(df, how="cross", left_on="a", right_on="b")
