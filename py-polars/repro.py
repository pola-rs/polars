from __future__ import annotations

import polars as pl
from polars.testing import assert_frame_equal

df = pl.DataFrame({"a": [1, 2]})
lf = df.lazy()

assert_frame_equal(df, df)
