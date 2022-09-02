from __future__ import annotations

import polars as pl


def test_row_count(foods_ipc: str) -> None:
    df = pl.read_ipc(foods_ipc, row_count_name="row_count", use_pyarrow=False)
    assert df["row_count"].to_list() == list(range(27))

    df = (
        pl.scan_ipc(foods_ipc, row_count_name="row_count")
        .filter(pl.col("category") == pl.lit("vegetables"))
        .collect()
    )

    assert df["row_count"].to_list() == [0, 6, 11, 13, 14, 20, 25]

    df = (
        pl.scan_ipc(foods_ipc, row_count_name="row_count")
        .with_row_count("foo", 10)
        .filter(pl.col("category") == pl.lit("vegetables"))
        .collect()
    )

    assert df["foo"].to_list() == [10, 16, 21, 23, 24, 30, 35]


def test_is_in_type_coercion(foods_ipc: str) -> None:
    out = (
        pl.scan_ipc(foods_ipc)
        .filter(pl.col("category").is_in(("vegetables", "ice cream")))
        .collect()
    )
    assert out.shape == (7, 4)
    out = (
        pl.scan_ipc(foods_ipc)
        .select(pl.col("category").alias("cat"))
        .filter(pl.col("cat").is_in(["vegetables"]))
        .collect()
    )
    assert out.shape == (7, 1)


def test_row_count_schema(foods_ipc: str) -> None:
    assert (
        pl.scan_ipc(foods_ipc, row_count_name="id").select(["id", "category"]).collect()
    ).dtypes == [pl.UInt32, pl.Utf8]
