from datetime import datetime

import pytz

import polars as pl


def test_timezone_aware_strptime() -> None:
    times = pl.DataFrame(
        {
            "delivery_datetime": [
                "2021-12-05 06:00:00+01:00",
                "2021-12-05 07:00:00+01:00",
                "2021-12-05 08:00:00+01:00",
            ]
        }
    )
    assert times.with_column(
        pl.col("delivery_datetime").str.strptime(pl.Datetime, fmt="%Y-%m-%d %H:%M:%S%z")
    ).to_dict(False) == {
        "delivery_datetime": [
            datetime(2021, 12, 5, 6, 0, tzinfo=pytz.FixedOffset(60)),
            datetime(2021, 12, 5, 7, 0, tzinfo=pytz.FixedOffset(60)),
            datetime(2021, 12, 5, 8, 0, tzinfo=pytz.FixedOffset(60)),
        ]
    }
