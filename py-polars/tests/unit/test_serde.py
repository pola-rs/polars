from __future__ import annotations

import pickle
from datetime import datetime, timedelta

import polars as pl


def test_pickling_simple_expression() -> None:
    e = pl.col("foo").sum()
    buf = pickle.dumps(e)
    assert str(pickle.loads(buf)) == str(e)


def serde_lazy_frame_lp() -> None:
    lf = pl.DataFrame({"a": [1, 2, 3], "b": ["a", "b", "c"]}).lazy().select(pl.col("a"))
    json = lf.write_json(to_string=True)

    assert (
        pl.LazyFrame.from_json(json)
        .collect()
        .to_series()
        .series_equal(pl.Series("a", [1, 2, 3]))
    )


def test_serde_time_unit() -> None:
    assert pickle.loads(
        pickle.dumps(
            pl.Series(
                [datetime(2022, 1, 1) + timedelta(days=1) for _ in range(3)]
            ).cast(pl.Datetime("ns"))
        )
    ).dtype == pl.Datetime("ns")
