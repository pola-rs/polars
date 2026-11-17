"""Chunking of the Arrow exports of DataFrame, Series and LazyFrame batches."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pytest

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Callable


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
    # The aligned frame is stored back, so a second export doesn't rechunk again.
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


def test_to_pandas_dictionaries_have_signed_keys() -> None:
    df = pl.DataFrame({"c": pl.Series(["a", "b"], dtype=pl.Categorical)})

    schema = df._df.to_pandas().schema

    assert schema.field("c").type == pa.dictionary(pa.int64(), pa.large_string())


def test_categorical_frame_shares_one_dictionary_across_batches() -> None:
    a = pl.DataFrame(
        {"c": ["x", "y"], "i": [0, 1]}, schema_overrides={"c": pl.Categorical}
    )
    b = pl.DataFrame(
        {"c": ["z", "x"], "i": [2, 3]}, schema_overrides={"c": pl.Categorical}
    )
    df = pl.concat([a, b], rechunk=False)
    assert df.n_chunks("all") == [2, 2]

    for table in (pa.table(df), df.to_arrow()):
        batches = table.to_batches()
        assert [batch.num_rows for batch in batches] == [2, 2]
        assert batches[0]["c"].dictionary == batches[1]["c"].dictionary
        assert table.group_by("c").aggregate([]).num_rows == 3
        pa.ipc.new_file(io.BytesIO(), table.schema).write_table(table)
        assert table.column("c").to_pylist() == ["x", "y", "z", "x"]
    assert df.n_chunks("all") == [2, 2]


def test_enum_batches_share_one_dictionary_buffer() -> None:
    dtype = pl.Enum(["x", "y", "z"])
    df = pl.concat([pl.DataFrame({"e": ["x", "y"]}, schema={"e": dtype})] * 2)

    batches = df.to_arrow().to_batches()

    dicts = [batch["e"].dictionary for batch in batches]
    assert len(dicts) == 2
    assert [b.address for b in dicts[0].buffers()[1:]] == [
        b.address for b in dicts[1].buffers()[1:]
    ]


def test_arrow_c_stream_stores_back_a_frame_with_a_scalar_column() -> None:
    df = multi_chunk_frame().with_columns(b=pl.Series(range(6)), c=pl.lit(7))
    assert df.n_chunks("all") == [3, 3, 1, 1]

    assert batch_rows(pa.table(df)) == [6]

    assert df.n_chunks("all") == [1, 1, 1, 1]
    assert df["c"].to_list() == [7] * 6


@pytest.mark.parametrize(
    "export",
    [
        lambda df: df.to_arrow(),
        lambda df: df.to_pandas(),
        lambda df: pa.table(df),
    ],
    ids=["to_arrow", "to_pandas", "pa.table"],
)
def test_nul_in_a_column_name_raises(export: Callable[[pl.DataFrame], Any]) -> None:
    df = pl.DataFrame({"a\0b": [1]})

    with pytest.raises(OSError, match="panic in an Arrow C stream callback"):
        export(df)


def test_collect_batches_arrow_c_stream_aligns_misaligned_batches() -> None:
    df = multi_chunk_frame().select("a").with_columns(b=pl.Series(range(6)))
    lf = df.lazy()
    assert [b.n_chunks("all") for b in lf.collect_batches()] == [[3, 1]]

    table = pa.RecordBatchReader.from_stream(lf.collect_batches()).read_all()

    assert table.to_pydict() == df.to_dict(as_series=False)


def test_series_to_arrow_chunked_keeps_the_chunks() -> None:
    s = multi_chunk_frame()["a"]

    out = s.to_arrow(chunked=True)

    assert isinstance(out, pa.ChunkedArray)
    assert [len(chunk) for chunk in out.chunks] == [2, 2, 2]
    assert out.to_pylist() == [1, 2, 1, 2, 1, 2]
    assert s.n_chunks() == 3


def test_series_to_arrow_chunked_empty_keeps_the_type() -> None:
    out = pl.Series("x", [], dtype=pl.Int8).to_arrow(chunked=True)

    assert out.type == pa.int8()
    assert len(out) == 0


def test_series_to_pandas_keeps_the_chunks() -> None:
    s = multi_chunk_frame()["a"]

    assert s.to_pandas().tolist() == [1, 2, 1, 2, 1, 2]
    assert s.n_chunks() == 3
