import pyarrow as pa

import polars as pl


def test_explode_empty_df_3402() -> None:
    df = pl.DataFrame({"a": pa.array([], type=pa.large_list(pa.int32()))})
    assert df.explode("a").dtypes == [pl.Int32]


def test_explode_empty_df_3460() -> None:
    df = pl.DataFrame({"a": pa.array([[]], type=pa.large_list(pa.int32()))})
    assert df.explode("a").dtypes == [pl.Int32]
