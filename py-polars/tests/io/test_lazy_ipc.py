# flake8: noqa: W191,E101
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
