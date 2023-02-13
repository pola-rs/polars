from datetime import datetime, time

import polars as pl


def test_clip() -> None:
    s = pl.Series("foo", [-50, 5, None, 50])
    assert s.clip(1, 10).to_list() == [1, 5, None, 10]


def test_clip_exprs() -> None:
    clipped_exprs = [
        pl.col("src").clip(pl.col("min"), pl.col("max")).alias("clipped"),
        pl.col("src").clip_min(pl.col("min")).alias("clipped_min"),
        pl.col("src").clip_max(pl.col("max")).alias("clipped_max"),
    ]
    df_datetime = pl.DataFrame(
        {
            "src": [
                datetime(2001, 1, 1),
                datetime(2002, 1, 1),
                None,
                datetime(2003, 1, 1),
                datetime(2004, 1, 1),
            ],
            "min": [
                datetime(1999, 1, 1),
                datetime(2010, 1, 1),
                datetime(2020, 1, 1),
                None,
                datetime(2020, 1, 1),
            ],
            "max": [
                datetime(2100, 1, 1),
                datetime(2000, 1, 1),
                datetime(2100, 1, 1),
                datetime(2000, 1, 1),
                None,
            ],
        }
    )

    assert df_datetime.select(clipped_exprs).to_dict(False) == {
        "clipped": [
            datetime(2001, 1, 1),
            None,
            None,
            datetime(2000, 1, 1),
            datetime(2020, 1, 1),
        ],
        "clipped_min": [
            datetime(2001, 1, 1),
            datetime(2010, 1, 1),
            None,
            datetime(2003, 1, 1),
            datetime(2020, 1, 1),
        ],
        "clipped_max": [
            datetime(2001, 1, 1),
            datetime(2000, 1, 1),
            None,
            datetime(2000, 1, 1),
            datetime(2004, 1, 1),
        ],
    }

    df_time = pl.DataFrame(
        {
            "src": [time(1), time(2), None, time(3), time(4)],
            "min": [time(0), time(10), time(20), None, time(20)],
            "max": [time(21), time(0), time(21), time(0), None],
        }
    )

    assert df_time.select(clipped_exprs).to_dict(False) == {
        "clipped": [time(1, 0), None, None, time(0, 0), time(20, 0)],
        "clipped_min": [time(1, 0), time(10, 0), None, time(3, 0), time(20, 0)],
        "clipped_max": [time(1, 0), time(0, 0), None, time(0, 0), time(4, 0)],
    }

    df_float = pl.DataFrame(
        {
            "src": [1.5, 2.5, None, 3.5, 4.5],
            "min": [0, 10, 20, None, 20],
            "max": [21.5, 0.5, 21.5, 0.5, None],
        }
    )

    assert df_float.select(clipped_exprs).to_dict(False) == {
        "clipped": [1.5, None, None, 0.5, 20.0],
        "clipped_min": [1.5, 10.0, None, 3.5, 20.0],
        "clipped_max": [1.5, 0.5, None, 0.5, 4.5],
    }
