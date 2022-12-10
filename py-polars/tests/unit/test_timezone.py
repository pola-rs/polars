from datetime import datetime, timedelta, timezone

import pytest

import polars as pl


@pytest.mark.parametrize(
    "tz_string,timedelta",
    [("+01:00", timedelta(minutes=60)), ("-01:30", timedelta(hours=-1, minutes=-30))],
)
def test_timezone_aware_strptime(tz_string: str, timedelta: timedelta) -> None:
    times = pl.DataFrame(
        {
            "delivery_datetime": [
                "2021-12-05 06:00:00" + tz_string,
                "2021-12-05 07:00:00" + tz_string,
                "2021-12-05 08:00:00" + tz_string,
            ]
        }
    )
    assert times.with_column(
        pl.col("delivery_datetime").str.strptime(pl.Datetime, fmt="%Y-%m-%d %H:%M:%S%z")
    ).to_dict(False) == {
        "delivery_datetime": [
            datetime(2021, 12, 5, 6, 0, tzinfo=timezone(timedelta)),
            datetime(2021, 12, 5, 7, 0, tzinfo=timezone(timedelta)),
            datetime(2021, 12, 5, 8, 0, tzinfo=timezone(timedelta)),
        ]
    }
