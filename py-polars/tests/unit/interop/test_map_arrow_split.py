"""Exporting Map columns whose entries overflow the i32 offsets of an Arrow MAP."""

from __future__ import annotations

import io
from itertools import accumulate
from typing import TYPE_CHECKING

import pyarrow as pa
import pytest

import polars as pl
from polars.exceptions import InvalidOperationError

if TYPE_CHECKING:
    from collections.abc import Sequence

pytestmark = pytest.mark.slow

MAX_ENTRIES = 2**31 - 1
ROW = 1 << 29


def zeroed_bools(n: int) -> pa.Array:
    return pa.Array.from_buffers(
        pa.bool_(), n, [None, pa.py_buffer(bytearray((n + 7) // 8))]
    )


def map_chunk(row_lens: Sequence[int], keys: pa.Array | None = None) -> pa.MapArray:
    # Arrow import keeps duplicate keys, so all-false keys are a cheap stand-in for real ones.
    offsets = [0, *accumulate(row_lens)]
    n = offsets[-1]
    keys = zeroed_bools(n) if keys is None else keys
    return pa.MapArray.from_arrays(pa.array(offsets, pa.int32()), keys, pa.nulls(n))


def batch_rows(table: pa.Table) -> list[int]:
    return [batch.num_rows for batch in table.to_batches()]


def map_entries(table: pa.Table, name: str) -> list[int]:
    return [len(batch.column(name).keys) for batch in table.to_batches()]


@pytest.fixture(scope="module")
def df() -> pl.DataFrame:
    """Six rows of 2^29 entries: 3 * 2^30 in total, but every 3 rows fit."""
    chunk = map_chunk([ROW, ROW])
    table = pa.table(
        {
            "m": pa.chunked_array([chunk, chunk, chunk]),
            "i": pa.chunked_array([[0, 1], [2, 3], [4, 5]]),
        }
    )
    out = pl.from_arrow(table)
    assert isinstance(out, pl.DataFrame)
    return out.rechunk()


def test_to_arrow_splits_an_oversized_map_chunk(df: pl.DataFrame) -> None:
    assert df.n_chunks("all") == [1, 1]

    table = df.to_arrow()

    assert batch_rows(table) == [3, 3]
    assert map_entries(table, "m") == [3 * ROW, 3 * ROW]
    assert table.column("i").to_pylist() == [0, 1, 2, 3, 4, 5]


def test_to_arrow_splits_a_slice_by_its_own_entries(df: pl.DataFrame) -> None:
    table = df.slice(1, 4).to_arrow()

    assert batch_rows(table) == [3, 1]
    assert table.column("i").to_pylist() == [1, 2, 3, 4]


def test_arrow_c_streams_split_an_oversized_map_chunk(df: pl.DataFrame) -> None:
    assert batch_rows(pa.table(df)) == [3, 3]

    reader = pa.RecordBatchReader.from_stream(df.lazy().collect_batches())
    assert [batch.num_rows for batch in reader] == [3, 3]


@pytest.mark.skipif(
    int(pa.__version__.split(".")[0]) < 16,
    reason="pa.chunked_array reads the Arrow C stream from pyarrow 16",
)
def test_series_arrow_c_stream_splits_an_oversized_map_chunk(
    df: pl.DataFrame,
) -> None:
    arr = pa.chunked_array(df["m"])

    assert [len(chunk) for chunk in arr.chunks] == [3, 3]


def test_series_to_arrow_raises_for_an_oversized_map(df: pl.DataFrame) -> None:
    # `Series.to_arrow` returns a single array, see #30780.
    with pytest.raises(InvalidOperationError, match="overflow the i32 offsets"):
        df["m"].to_arrow()


@pytest.mark.parametrize(
    "expr",
    [
        pl.struct("m"),
        pl.concat_list("m"),
        pl.col("m").reshape((-1, 1)),
    ],
)
def test_nested_maps_are_split_on_their_entries(
    df: pl.DataFrame, expr: pl.Expr
) -> None:
    table = df.select(expr.alias("n"), "i").to_arrow()

    assert batch_rows(table) == [3, 3]
    assert table.column("i").to_pylist() == [0, 1, 2, 3, 4, 5]


def test_a_map_of_maps_is_split_on_the_inner_entries(df: pl.DataFrame) -> None:
    inner = df["m"].to_frame().slice(0, 2).to_arrow().column("m").chunk(0)
    outer = pa.MapArray.from_arrays(pa.array([0, 1, 2], pa.int32()), zeroed_bools(2), inner)
    nested = pl.from_arrow(pa.chunked_array([outer, outer, outer])).rechunk()
    assert isinstance(nested, pl.Series)

    assert batch_rows(nested.to_frame().to_arrow()) == [3, 3]


def test_two_map_columns_are_cut_at_the_same_rows(df: pl.DataFrame) -> None:
    # Alone, `m` cuts after row 3 and `b` after row 2; together they need [2, 3, 1].
    b = pl.from_arrow(
        pa.chunked_array([map_chunk([MAX_ENTRIES // 2] * 2), map_chunk([2] * 4)])
    )
    assert isinstance(b, pl.Series)
    two = df.with_columns(b=b.rechunk())

    table = two.to_arrow()

    assert batch_rows(table) == [2, 3, 1]
    assert all(n <= MAX_ENTRIES for n in map_entries(table, "m"))
    assert all(n <= MAX_ENTRIES for n in map_entries(table, "b"))


def test_a_single_row_with_too_many_map_entries_raises(df: pl.DataFrame) -> None:
    with pytest.raises(InvalidOperationError, match="a single row has 3221225472"):
        df.select(pl.col("m").implode()).to_arrow()


def test_ipc_writers_split_an_oversized_map_chunk(df: pl.DataFrame) -> None:
    for write in (
        lambda f: df.write_ipc(f, compression="zstd"),
        lambda f: df.lazy().sink_ipc(f, compression="zstd"),
    ):
        f = io.BytesIO()
        write(f)
        f.seek(0)
        table = pa.ipc.open_file(f).read_all()

        assert batch_rows(table) == [3, 3]
        assert table.column("i").to_pylist() == [0, 1, 2, 3, 4, 5]


def test_to_pandas_batches_share_one_schema(df: pl.DataFrame) -> None:
    # Pandas conversion of the map itself would materialize 3 * 2^30 Python objects.
    with_cat = df.with_columns(c=pl.lit("x", pl.Categorical))

    batches = with_cat._df.to_pandas()

    assert len(batches) == 2
    assert batches[0].schema == batches[1].schema
    assert pa.Table.from_batches(batches).num_rows == 6
