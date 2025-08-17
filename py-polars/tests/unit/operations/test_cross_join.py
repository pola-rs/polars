from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

import polars as pl
from polars.testing import assert_frame_equal


def test_cross_join_predicate_pushdown_block_16956() -> None:
    lf = pl.LazyFrame(
        [
            [1718085600000, 1718172000000, 1718776800000],
            [1718114400000, 1718200800000, 1718805600000],
        ],
        schema=["start_datetime", "end_datetime"],
    ).cast(pl.Datetime("ms", "Europe/Amsterdam"))

    assert (
        lf.join(lf, how="cross")
        .filter(
            pl.col.end_datetime_right.is_between(
                pl.col.start_datetime, pl.col.start_datetime.dt.offset_by("132h")
            )
        )
        .select("start_datetime", "end_datetime_right")
    ).collect(optimizations=pl.QueryOptFlags(predicate_pushdown=True)).to_dict(
        as_series=False
    ) == {
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


def test_nested_loop_join() -> None:
    left = pl.LazyFrame(
        {
            "a": [1, 2, 1, 3],
            "b": [1, 2, 3, 4],
        }
    )
    right = pl.LazyFrame(
        {
            "c": [4, 1, 2],
            "d": [1, 2, 3],
        }
    )

    actual = left.join_where(right, pl.col("a") != pl.col("c"))
    plan = actual.explain()
    assert "NESTED LOOP JOIN" in plan
    expected = pl.DataFrame(
        {
            "a": [1, 1, 2, 2, 1, 1, 3, 3, 3],
            "b": [1, 1, 2, 2, 3, 3, 4, 4, 4],
            "c": [4, 2, 4, 1, 4, 2, 4, 1, 2],
            "d": [1, 3, 1, 2, 1, 3, 1, 2, 3],
        }
    )
    assert_frame_equal(
        actual.collect(), expected, check_row_order=False, check_exact=True
    )


def test_cross_join_chunking_panic_22793() -> None:
    N = int(pl.thread_pool_size() ** 0.5) * 2
    df = pl.DataFrame(
        [pl.concat([pl.Series("a", [0]) for _ in range(N)]), pl.Series("b", [0] * N)],
    )
    assert_frame_equal(
        df.lazy()
        .join(pl.DataFrame().lazy(), how="cross")
        .filter(pl.col("a") == pl.col("a"))
        .collect(),
        df.schema.to_frame(),
    )
