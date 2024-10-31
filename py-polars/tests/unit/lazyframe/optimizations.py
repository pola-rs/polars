import io

import polars as pl
from polars.testing import assert_frame_equal


def test_remove_double_sort() -> None:
    assert (
        pl.LazyFrame({"a": [1, 2, 3, 3]}).sort("a").sort("a").explain().count("SORT")
        == 1
    )


def test_double_sort_maintain_order_18558() -> None:
    df = pl.DataFrame(
        {
            "col1": [1, 2, 2, 4, 5, 6],
            "col2": [2, 2, 0, 0, 2, None],
        }
    )

    lf = df.lazy().sort("col2").sort("col1", maintain_order=True)

    expect = pl.DataFrame(
        [
            pl.Series("col1", [1, 2, 2, 4, 5, 6], dtype=pl.Int64),
            pl.Series("col2", [2, 0, 2, 0, 2, None], dtype=pl.Int64),
        ]
    )

    assert_frame_equal(lf.collect(), expect)


def test_fast_count_alias_18581() -> None:
    f = io.BytesIO()
    f.write(b"a,b,c\n1,2,3\n4,5,6")
    f.flush()
    f.seek(0)

    df = pl.scan_csv(f).select(pl.len().alias("weird_name")).collect()

    assert_frame_equal(pl.DataFrame({"weird_name": 2}), df)
