from __future__ import annotations

import polars as pl
from polars.testing import assert_frame_equal


def test_array_to_string() -> None:
    df = pl.DataFrame({"values": [["aa", "bb"], [None, "cc"], ["dd", None]]})

    with pl.SQLContext(df=df, eager_execution=True) as ctx:
        res = ctx.execute(
            """
            SELECT
              ARRAY_TO_STRING(values, '') AS v1,
              ARRAY_TO_STRING(values, ':') AS v2,
              ARRAY_TO_STRING(values, ':', 'NA') AS v3
            FROM df
            """
        )
    assert_frame_equal(
        res,
        pl.DataFrame(
            {
                "v1": ["aabb", "cc", "dd"],
                "v2": ["aa:bb", "cc", "dd"],
                "v3": ["aa:bb", "NA:cc", "dd:NA"],
            }
        ),
    )
