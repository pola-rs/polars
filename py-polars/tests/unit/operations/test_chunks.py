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


def test_first_last_non_null_empty_leading_chunk_28495() -> None:
    # A series carrying a sorted flag whose *first* chunk is empty used to read out of
    # bounds while deciding whether the nulls sit at the start or at the end.
    def with_empty_leading_chunk(values: list[int | None]) -> pl.Series:
        s = pl.concat(
            [
                pl.Series("a", [0], dtype=pl.Int64),
                pl.Series("a", values, dtype=pl.Int64),
            ],
            rechunk=False,
        )
        s = s.filter(pl.Series([False] + [True] * len(values)))
        assert [len(c) for c in s.get_chunks()] == [0, len(values)]
        return s

    nulls_first = with_empty_leading_chunk([None, None, 1, 2, 3]).set_sorted()
    assert nulls_first.arg_min() == 2
    assert nulls_first.arg_max() == 4
    assert nulls_first.min() == 1
    assert nulls_first.max() == 3

    nulls_last = with_empty_leading_chunk([3, 2, 1, None, None]).set_sorted(
        descending=True
    )
    assert nulls_last.arg_min() == 2
    assert nulls_last.arg_max() == 0
    assert nulls_last.min() == 1
    assert nulls_last.max() == 3
