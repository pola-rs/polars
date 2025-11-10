import pytest

import polars as pl
import polars.selectors as cs
from polars._typing import EngineType
from polars.testing import assert_frame_equal


@pytest.mark.ci_only
@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
def test_basic(engine: EngineType) -> None:
    from plugin_v1 import (
        byte_rev,
        count,
        horizontal_count,
        min_by,
        rolling_product,
        vertical_scan,
    )

    df = pl.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [6, 5, 4],
            "c": [1, None, 3],
        }
    )

    q = df.lazy().with_columns(
        byte_rev=byte_rev(pl.col.a),
        min_by=min_by(pl.col.a, by=pl.col.b),
        rolling_product=rolling_product(pl.col.a, n=2),
        vertical_scan=vertical_scan(pl.col.a, init=0),
        horizontal_count=horizontal_count(cs.all()),
        count_a=count(pl.col.a),
        count_c=count(pl.col.c),
    )

    expected = pl.DataFrame(
        [
            pl.Series("byte_rev", [x << 56 for x in df["a"]]),
            pl.Series("min_by", [3, 3, 3]),
            pl.Series("rolling_product", [1, 2, 6]),
            pl.Series("vertical_scan", [0, 1, 2]),
            pl.Series("horizontal_count", [3, 2, 3], pl.UInt32()),
            pl.Series("count_a", [3, 3, 3], pl.UInt64()),
            pl.Series("count_c", [2, 2, 2], pl.UInt64()),
        ]
    )

    assert_frame_equal(
        q.collect(engine=engine), pl.concat([df, expected], how="horizontal")
    )
