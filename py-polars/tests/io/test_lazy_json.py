from __future__ import annotations

import polars as pl


def test_scan_ndjson(foods_ndjson: str) -> None:
    df = pl.scan_ndjson(foods_ndjson, row_count_name="row_count").collect()
    assert df["row_count"].to_list() == list(range(27))

    df = (
        pl.scan_ndjson(foods_ndjson, row_count_name="row_count")
        .filter(pl.col("category") == pl.lit("vegetables"))
        .collect()
    )

    assert df["row_count"].to_list() == [0, 6, 11, 13, 14, 20, 25]

    df = (
        pl.scan_ndjson(foods_ndjson, row_count_name="row_count")
        .with_row_count("foo", 10)
        .filter(pl.col("category") == pl.lit("vegetables"))
        .collect()
    )

    assert df["foo"].to_list() == [10, 16, 21, 23, 24, 30, 35]
