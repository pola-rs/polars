"""Chunking of the Arrow exports of DataFrame, Series and LazyFrame batches."""

from __future__ import annotations

import pyarrow as pa

import polars as pl


def multi_chunk_frame() -> pl.DataFrame:
    df = pl.concat([pl.DataFrame({"a": [1, 2], "s": ["x", "y"]})] * 3, rechunk=False)
    assert df.n_chunks("all") == [3, 3]
    return df


def batch_rows(table: pa.Table) -> list[int]:
    return [batch.num_rows for batch in table.to_batches()]


def test_arrow_c_stream_exports_an_aligned_frame_per_chunk() -> None:
    df = multi_chunk_frame()

    assert batch_rows(pa.table(df)) == [2, 2, 2]
    assert df.n_chunks("all") == [3, 3]


def test_to_arrow_and_to_pandas_keep_the_chunks_of_an_aligned_frame() -> None:
    df = multi_chunk_frame()

    assert batch_rows(df.to_arrow()) == [2, 2, 2]
    assert df.to_pandas()["a"].tolist() == [1, 2, 1, 2, 1, 2]
    assert df.n_chunks("all") == [3, 3]


def test_arrow_c_stream_rechunks_a_misaligned_frame_once() -> None:
    df = multi_chunk_frame().with_columns(b=pl.Series(range(6)))
    assert df.n_chunks("all") == [3, 3, 1]

    assert batch_rows(pa.table(df)) == [6]
    # The aligned frame is stored, so a second export doesn't rechunk again.
    assert df.n_chunks("all") == [1, 1, 1]


def test_collect_batches_arrow_c_stream_exports_each_chunk() -> None:
    lf = multi_chunk_frame().lazy()
    assert [b.n_chunks() for b in lf.collect_batches(chunk_size=6)] == [3]

    reader = pa.RecordBatchReader.from_stream(lf.collect_batches(chunk_size=6))

    assert [batch.num_rows for batch in reader] == [2, 2, 2]


def test_to_arrow_zero_width_keeps_the_height() -> None:
    df = pl.DataFrame({"a": range(5)}).drop("a")

    assert df.to_arrow().shape == (5, 0)
    assert pl.DataFrame().to_arrow().shape == (0, 0)


def test_to_pandas_categorical_has_signed_keys() -> None:
    df = pl.DataFrame(
        {
            "c": pl.Series(["a", "b"], dtype=pl.Categorical),
            "e": pl.Series(["x", "y"], dtype=pl.Enum(["x", "y"])),
        }
    )

    out = df.to_pandas()

    assert out["c"].cat.codes.dtype.kind == "i"
    assert out["e"].cat.ordered
    assert out["e"].tolist() == ["x", "y"]
