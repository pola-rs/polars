import io

import polars as pl
from polars.testing import assert_frame_equal


def test_fast_count_alias_18581() -> None:
    f = io.BytesIO()
    f.write(b"a,b,c\n1,2,3\n4,5,6")
    f.flush()
    f.seek(0)

    df = pl.scan_csv(f).select(pl.len().alias("weird_name")).collect()

    assert_frame_equal(pl.DataFrame({"weird_name": 2}), df)
