import numpy as np

import polars as pl


def test_chunks_align_16830() -> None:
    n = 2
    df = pl.DataFrame(
        {"index_1": np.repeat(np.arange(10), n), "index_2": np.repeat(np.arange(10), n)}
    )
    df = pl.concat([df[0:10], df[10:]], rechunk=False)
    df = df.filter(df["index_1"] == 0)  # filter chunks
    df = df.with_columns(
        index_2=pl.Series(values=[0] * n)
    )  # set a chunk of different size
    df.set_sorted("index_2")  # triggers `select_chunk`.
