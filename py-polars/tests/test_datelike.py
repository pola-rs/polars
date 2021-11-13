from datetime import datetime

import polars as pl


def test_fill_null():
    dt = datetime.strptime("2021-01-01", "%Y-%m-%d")
    s = pl.Series("A", [dt, None])

    for fill_val in (dt, pl.lit(dt)):
        out = s.fill_null(fill_val)

        assert out.null_count() == 0
        assert out.dt[0] == dt
        assert out.dt[1] == dt
