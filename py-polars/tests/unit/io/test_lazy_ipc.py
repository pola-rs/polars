from __future__ import annotations

import io
import json
import os
import sys
import typing
from typing import IO, TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.ipc
import pytest

import polars as pl
from polars.interchange.protocol import CompatLevel
from polars.testing.asserts.frame import assert_frame_equal

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from polars._typing import IpcCompression
    from tests.conftest import PlMonkeyPatch

COMPRESSIONS = ["uncompressed", "lz4", "zstd"]


@pytest.fixture
def foods_ipc_path(io_files_path: Path) -> Path:
    return io_files_path / "foods1.ipc"


def test_row_index(foods_ipc_path: Path) -> None:
    df = pl.read_ipc(foods_ipc_path, row_index_name="row_index", use_pyarrow=False)
    assert df["row_index"].to_list() == list(range(27))

    df = (
        pl.scan_ipc(foods_ipc_path, row_index_name="row_index")
        .filter(pl.col("category") == pl.lit("vegetables"))
        .collect()
    )

    assert df["row_index"].to_list() == [0, 6, 11, 13, 14, 20, 25]

    df = (
        pl.scan_ipc(foods_ipc_path, row_index_name="row_index")
        .with_row_index("foo", 10)
        .filter(pl.col("category") == pl.lit("vegetables"))
        .collect()
    )

    assert df["foo"].to_list() == [10, 16, 21, 23, 24, 30, 35]


def test_is_in_type_coercion(foods_ipc_path: Path) -> None:
    out = (
        pl.scan_ipc(foods_ipc_path)
        .filter(pl.col("category").is_in(("vegetables", "ice cream")))
        .collect()
    )
    assert out.shape == (7, 4)
    out = (
        pl.scan_ipc(foods_ipc_path)
        .select(pl.col("category").alias("cat"))
        .filter(pl.col("cat").is_in(["vegetables"]))
        .collect()
    )
    assert out.shape == (7, 1)


def test_row_index_schema(foods_ipc_path: Path) -> None:
    assert (
        pl.scan_ipc(foods_ipc_path, row_index_name="id")
        .select(["id", "category"])
        .collect()
    ).dtypes == [pl.get_index_type(), pl.String]


def test_glob_n_rows(io_files_path: Path) -> None:
    file_path = io_files_path / "foods*.ipc"
    df = pl.scan_ipc(file_path, n_rows=40).collect()

    # 27 rows from foods1.ipc and 13 from foods2.ipc
    assert df.shape == (40, 4)

    # take first and last rows
    assert df[[0, 39]].to_dict(as_series=False) == {
        "category": ["vegetables", "seafood"],
        "calories": [45, 146],
        "fats_g": [0.5, 6.0],
        "sugars_g": [2, 2],
    }


def test_ipc_list_arg(io_files_path: Path) -> None:
    first = io_files_path / "foods1.ipc"
    second = io_files_path / "foods2.ipc"

    df = pl.scan_ipc(source=[first, second]).collect()
    assert df.shape == (54, 4)
    assert df.row(-1) == ("seafood", 194, 12.0, 1)
    assert df.row(0) == ("vegetables", 45, 0.5, 2)


def test_scan_ipc_local_with_async(
    plmonkeypatch: PlMonkeyPatch,
    io_files_path: Path,
) -> None:
    plmonkeypatch.setenv("POLARS_VERBOSE", "1")
    plmonkeypatch.setenv("POLARS_FORCE_ASYNC", "1")

    assert_frame_equal(
        pl.scan_ipc(io_files_path / "foods1.ipc").head(1).collect(),
        pl.DataFrame(
            {
                "category": ["vegetables"],
                "calories": [45],
                "fats_g": [0.5],
                "sugars_g": [2],
            }
        ),
    )


def test_sink_ipc_compat_level_22930() -> None:
    df = pl.DataFrame({"a": ["foo"]})

    f1 = io.BytesIO()
    f2 = io.BytesIO()

    df.lazy().sink_ipc(f1, compat_level=CompatLevel.oldest(), engine="in-memory")
    df.lazy().sink_ipc(f2, compat_level=CompatLevel.oldest(), engine="streaming")

    f1.seek(0)
    f2.seek(0)

    t1 = pa.ipc.open_file(f1)
    assert "large_string" in str(t1.schema)
    assert_frame_equal(pl.DataFrame(t1.read_all()), df)

    t2 = pa.ipc.open_file(f2)
    assert "large_string" in str(t2.schema)
    assert_frame_equal(pl.DataFrame(t2.read_all()), df)


def test_scan_file_info_cache(
    capfd: Any, plmonkeypatch: PlMonkeyPatch, foods_ipc_path: Path
) -> None:
    plmonkeypatch.setenv("POLARS_VERBOSE", "1")
    a = pl.scan_ipc(foods_ipc_path)
    b = pl.scan_ipc(foods_ipc_path)

    a.join(b, how="cross").explain()

    captured = capfd.readouterr().err
    assert "FILE_INFO CACHE HIT" in captured


def test_scan_ipc_file_async(
    plmonkeypatch: PlMonkeyPatch,
    io_files_path: Path,
) -> None:
    plmonkeypatch.setenv("POLARS_FORCE_ASYNC", "1")

    foods1 = io_files_path / "foods1.ipc"

    df = pl.scan_ipc(foods1).collect()

    assert_frame_equal(
        pl.scan_ipc(foods1).select(pl.len()).collect(), df.select(pl.len())
    )

    assert_frame_equal(
        pl.scan_ipc(foods1).head(1).collect(),
        df.head(1),
    )

    assert_frame_equal(
        pl.scan_ipc(foods1).tail(1).collect(),
        df.tail(1),
    )

    assert_frame_equal(
        pl.scan_ipc(foods1).slice(-1, 1).collect(),
        df.slice(-1, 1),
    )

    assert_frame_equal(
        pl.scan_ipc(foods1).slice(7, 10).collect(),
        df.slice(7, 10),
    )

    assert_frame_equal(
        pl.scan_ipc(foods1).select(pl.col.calories).collect(),
        df.select(pl.col.calories),
    )

    assert_frame_equal(
        pl.scan_ipc(foods1).select([pl.col.calories, pl.col.category]).collect(),
        df.select([pl.col.calories, pl.col.category]),
    )

    assert_frame_equal(
        pl.scan_ipc([foods1, foods1]).collect(),
        pl.concat([df, df]),
    )

    assert_frame_equal(
        pl.scan_ipc(foods1).select(pl.col.calories.sum()).collect(),
        df.select(pl.col.calories.sum()),
    )

    assert_frame_equal(
        pl.scan_ipc(foods1, row_index_name="ri", row_index_offset=42)
        .slice(0, 1)
        .select(pl.col.ri)
        .collect(),
        df.with_row_index(name="ri", offset=42).slice(0, 1).select(pl.col.ri),
    )


def test_scan_ipc_file_async_dict(
    plmonkeypatch: PlMonkeyPatch,
) -> None:
    plmonkeypatch.setenv("POLARS_FORCE_ASYNC", "1")

    buf = io.BytesIO()
    lf = pl.LazyFrame(
        {"cat": ["A", "B", "C", "A", "C", "B"]}, schema={"cat": pl.Categorical}
    ).with_row_index()
    lf.sink_ipc(buf)
    buf.seek(0)

    out = pl.scan_ipc(buf).collect()
    expected = lf.collect()
    assert_frame_equal(out, expected)


def test_scan_ipc_file_async_multiple_record_batches(
    plmonkeypatch: PlMonkeyPatch,
) -> None:
    plmonkeypatch.setenv("POLARS_FORCE_ASYNC", "1")

    buf = io.BytesIO()
    lf = pl.LazyFrame({"a": list(range(100))})
    lf.sink_ipc(buf, record_batch_size=10)
    buf.seek(0)
    df = lf.collect()

    buffers = typing.cast("list[IO[bytes]]", [buf, buf])

    assert_frame_equal(
        pl.scan_ipc(buf).collect(),
        df,
    )

    assert_frame_equal(
        pl.scan_ipc(buf).head(15).collect(),
        df.head(15),
    )

    assert_frame_equal(
        pl.scan_ipc(buf).tail(15).collect(),
        df.tail(15),
    )

    assert_frame_equal(
        pl.scan_ipc(buf).slice(45, 20).collect(),
        df.slice(45, 20),
    )

    assert_frame_equal(
        pl.scan_ipc(buffers).slice(85, 30).collect(),
        pl.concat([df.slice(85, 15), df.slice(0, 15)]),
    )

    assert_frame_equal(
        pl.scan_ipc(buf).select(pl.col.a.sum()).collect(),
        df.select(pl.col.a.sum()),
    )

    assert_frame_equal(
        pl.scan_ipc(buffers, row_index_name="ri").tail(15).select(pl.col.ri).collect(),
        pl.concat([df, df]).with_row_index("ri").tail(15).select(pl.col.ri),
    )


@pytest.mark.parametrize("n_a", [1, 999])
@pytest.mark.parametrize("n_b", [1, 12, 13, 999])  # problem starts 13
@pytest.mark.parametrize("compression", COMPRESSIONS)
def test_scan_ipc_varying_block_metadata_len_c4812(
    n_a: int, n_b: int, compression: IpcCompression, plmonkeypatch: PlMonkeyPatch
) -> None:
    plmonkeypatch.setenv("POLARS_FORCE_ASYNC", "1")

    buf = io.BytesIO()
    df = pl.DataFrame({"a": [n_a * "A", n_b * "B"]})
    df.lazy().sink_ipc(buf, compression=compression, record_batch_size=1)
    buf.seek(0)

    with pa.ipc.open_file(buf) as reader:
        assert [
            reader.get_batch(i).num_rows for i in range(reader.num_record_batches)
        ] == [1, 1]

    buf.seek(0)
    assert_frame_equal(pl.scan_ipc(buf).collect(), df)


@pytest.mark.parametrize(
    "record_batch_size", [1, 2, 5, 7, 50, 99, 100, 101, 299, 300, 100_000]
)
@pytest.mark.parametrize("n_chunks", [1, 2, 3])
def test_sink_ipc_record_batch_size(record_batch_size: int, n_chunks: int) -> None:
    n_rows = 100
    buf = io.BytesIO()

    df0 = pl.DataFrame({"a": range(n_rows)})
    df = df0
    while n_chunks > 1:
        df = pl.concat([df, df0])
        n_chunks -= 1

    df.lazy().sink_ipc(buf, record_batch_size=record_batch_size)

    buf.seek(0)
    out = pl.scan_ipc(buf).collect()
    assert_frame_equal(out, df)

    buf.seek(0)
    with pa.ipc.open_file(buf) as reader:
        record_batch_lengths = [
            reader.get_batch(i).num_rows for i in range(reader.num_record_batches)
        ]

    for i, n_rows in enumerate(record_batch_lengths):
        assert n_rows == record_batch_size or (
            i + 1 == len(record_batch_lengths) and n_rows <= record_batch_size
        )


@pytest.mark.parametrize("record_batch_size", [None, 3])
@pytest.mark.parametrize("slice", [(0, 0), (0, 1), (0, 5), (4, 7), (-1, 1), (-5, 4)])
@pytest.mark.parametrize("compression", COMPRESSIONS)
def test_scan_ipc_compression_with_slice_26063(
    record_batch_size: int, slice: tuple[int, int], compression: IpcCompression
) -> None:
    n_rows = 15
    df = pl.DataFrame({"a": range(n_rows)}).with_columns(
        pl.col.a.pow(3).cast(pl.String).alias("b")
    )
    buf = io.BytesIO()

    df.lazy().sink_ipc(
        buf, compression=compression, record_batch_size=record_batch_size
    )
    buf.seek(0)
    out = pl.scan_ipc(buf).slice(slice[0], slice[1]).collect()
    expected = df.slice(slice[0], slice[1])
    assert_frame_equal(out, expected)


def test_sink_scan_ipc_round_trip_statistics() -> None:
    n_rows = 4_000  # must be higher than (n_vCPU)^2 to avoid sortedness inference
    buf = io.BytesIO()

    df = (
        pl.DataFrame({"a": range(n_rows)})
        .with_columns(pl.col.a.reverse().alias("b"))
        .with_columns(pl.col.a.shuffle().alias("d"))
        .with_columns(pl.col.a.shuffle().sort().alias("d"))
    )
    df.lazy().sink_ipc(buf, _record_batch_statistics=True)
    buf.seek(0)

    metadata = df._to_metadata()

    # baseline
    assert metadata.select(pl.col("sorted_asc").sum()).item() == 2
    assert metadata.select(pl.col("sorted_dsc").sum()).item() == 1

    # round-trip
    out = pl.scan_ipc(buf, _record_batch_statistics=True).collect()
    assert_frame_equal(metadata, out._to_metadata())

    # do not read unless requested
    out = pl.scan_ipc(buf).collect()
    assert out._to_metadata().select(pl.col("sorted_asc").sum()).item() == 0
    assert out._to_metadata().select(pl.col("sorted_dsc").sum()).item() == 0

    # remain pyarrow compatible
    out = pl.read_ipc(buf, use_pyarrow=True)
    assert_frame_equal(df, out)


def test_sink_ipc_custom_metadata() -> None:
    f = io.BytesIO()
    pl.LazyFrame({"a": range(37)}).sink_ipc(
        f,
        record_batch_size=10,
        _record_batch_statistics=True,
    )

    with pa.ipc.open_file(f) as reader:
        assert [
            reader.get_record_batch(i).num_rows
            for i in range(reader.num_record_batches)
        ] == [10, 10, 10, 7]
        assert json.loads(reader.metadata.get(b"__POLARS_IPC_METADATA")) == {
            "record_batch_cum_len": [10, 20, 30, 37]
        }

    f = io.BytesIO()
    pl.LazyFrame({"a": [0, 1, 2, 3, 4]}).sink_ipc(
        f,
        record_batch_size=3,
        _record_batch_statistics=False,
    )

    with pa.ipc.open_file(f) as reader:
        assert reader.metadata is None


def test_sink_ipc_custom_metadata_dictionary() -> None:
    df = pl.DataFrame({"a": ["x", "y", "z", "x", "w"]}, schema={"a": pl.Categorical})

    f = io.BytesIO()
    df.lazy().sink_ipc(f, record_batch_size=2, _record_batch_statistics=True)

    # Dictionary batches must not be counted.
    with pa.ipc.open_file(f) as reader:
        assert json.loads(reader.metadata.get(b"__POLARS_IPC_METADATA")) == {
            "record_batch_cum_len": [2, 4, 5]
        }

    buf = f.getvalue()
    assert_frame_equal(pl.scan_ipc(buf).tail(1).collect(), df.tail(1))
    assert_frame_equal(pl.scan_ipc(buf).slice(1, 3).collect(), df.slice(1, 3))
    assert pl.scan_ipc(buf).select(pl.len()).collect().item() == 5


@pytest.mark.parametrize(
    "custom_metadata",
    [
        # Older versions of Polars also counted dictionary batches.
        b'{"record_batch_cum_len": [3, 6]}',
        b"not json",
    ],
)
def test_scan_ipc_ignores_unusable_custom_metadata(custom_metadata: bytes) -> None:
    table = pa.table({"a": pa.array(["x", "y", "z"]).dictionary_encode()})
    metadata = {b"__POLARS_IPC_METADATA": custom_metadata}

    f = io.BytesIO()
    with pa.ipc.new_file(f, table.schema, metadata=metadata) as writer:
        writer.write_table(table)

    buf = f.getvalue()
    with pytest.warns(UserWarning, match="ignoring unusable Polars metadata"):
        assert pl.scan_ipc(buf).tail(1).collect()["a"].to_list() == ["z"]
    with pytest.warns(UserWarning, match="ignoring unusable Polars metadata"):
        assert pl.scan_ipc(buf).select(pl.len()).collect().item() == 3


def test_scan_ipc_slicing_and_count_with_custom_metadata(
    plmonkeypatch: PlMonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    df = pl.DataFrame({"a": range(37)})

    f = io.BytesIO()
    df.lazy().sink_ipc(
        f,
        record_batch_size=10,
        _record_batch_statistics=True,
    )

    buf = f.getvalue()
    q = pl.scan_ipc(buf).slice(10, 10)

    plmonkeypatch.setenv("POLARS_VERBOSE", "1")
    capfd.readouterr()
    out = q.collect()
    capture = capfd.readouterr().err
    plmonkeypatch.setenv("POLARS_VERBOSE", "0")

    assert (
        "rb_total_count: 4, rb_full_fetch_count: 1, rb_metadata_fetch_count: 0"
        in capture
    )

    assert_frame_equal(out, pl.DataFrame({"a": range(10, 20)}))

    footer_header_len = 10
    footer_md_and_header_len = footer_header_len + int.from_bytes(
        buf[-10:][:4], byteorder="little"
    )
    footer_md_only_buf = buf[-footer_md_and_header_len:]

    # All the following should pass without needing to access record batch data.

    plmonkeypatch.setenv("POLARS_VERBOSE", "1")
    assert pl.scan_ipc(footer_md_only_buf).select(pl.len()).collect().item() == 37
    capture = capfd.readouterr().err
    plmonkeypatch.setenv("POLARS_VERBOSE", "0")

    # 0 fetches; record batch row counts sourced from custom metadata.
    assert (
        "rb_total_count: 4, rb_full_fetch_count: 0, rb_metadata_fetch_count: 0"
        in capture
    )

    assert (
        pl.scan_ipc(3 * [footer_md_only_buf]).select(pl.len()).collect().item() == 111
    )

    for offset_len in [(0, 0), (-1, 0), (1, 0), (-999, 1), (999, 1)]:
        assert (
            pl.scan_ipc([footer_md_only_buf, footer_md_only_buf])
            .slice(*offset_len)
            .collect()
            .height
            == 0
        )

    assert (
        pl.scan_ipc([footer_md_only_buf, footer_md_only_buf])
        .slice(47, 1)
        .select(pl.len())
        .collect()
        .item()
        == 1
    )
    assert (
        pl.scan_ipc([footer_md_only_buf, footer_md_only_buf])
        .slice(47, 999)
        .select(pl.len())
        .collect()
        .item()
        == 27
    )


def test_scan_ipc_fast_count_does_not_read_row_values(
    plmonkeypatch: PlMonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    df = pl.DataFrame({"a": range(3)})
    f = io.BytesIO()
    df.lazy().sink_ipc(
        f,
        record_batch_size=999,
        _record_batch_statistics=False,
    )

    q = pl.scan_ipc(f.getvalue()).select(pl.len())

    plmonkeypatch.setenv("POLARS_VERBOSE", "1")
    capfd.readouterr()
    out = q.collect()
    capture = capfd.readouterr().err
    plmonkeypatch.setenv("POLARS_VERBOSE", "0")

    assert (
        "rb_total_count: 1, rb_full_fetch_count: 0, rb_metadata_fetch_count: 1"
        in capture
    )
    assert out.item() == 3


def test_scan_ipc_slicing_and_count(
    plmonkeypatch: PlMonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    df = pl.DataFrame({"a": range(3)})
    f = io.BytesIO()
    df.lazy().sink_ipc(
        f,
        record_batch_size=1,
        _record_batch_statistics=False,
    )

    buf = f.getvalue()
    q = pl.scan_ipc(buf).select(pl.len())

    plmonkeypatch.setenv("POLARS_VERBOSE", "1")
    capfd.readouterr()
    out = q.collect()
    capture = capfd.readouterr().err
    plmonkeypatch.setenv("POLARS_VERBOSE", "0")

    assert (
        "rb_total_count: 3, rb_full_fetch_count: 0, rb_metadata_fetch_count: 3"
        in capture
    )
    assert out.item() == 3

    q = pl.scan_ipc(buf).slice(1, 1)

    plmonkeypatch.setenv("POLARS_VERBOSE", "1")
    capfd.readouterr()
    out = q.collect()
    capture = capfd.readouterr().err
    plmonkeypatch.setenv("POLARS_VERBOSE", "0")

    # rb_metadata_fetch_count == rb_total_count, we fetched all record batch
    # metadatas to resolve slice.
    assert (
        "rb_total_count: 3, rb_full_fetch_count: 1, rb_metadata_fetch_count: 3"
        in capture
    )

    assert_frame_equal(
        out,
        pl.DataFrame({"a": 1}),
    )


@pytest.mark.parametrize(
    "selection",
    [["b"], ["a", "b", "c", "d"], ["d", "c", "a", "b"], ["d", "a", "b"]],
)
@pytest.mark.parametrize("record_batch_size", [None, 100])
def test_sink_scan_ipc_round_trip_statistics_projection(
    selection: list[str], record_batch_size: int
) -> None:
    n_rows = 4_000  # must be higher than (n_vCPU)^2 to avoid sortedness inference
    buf = io.BytesIO()

    df = (
        pl.DataFrame({"a": range(n_rows)})
        .with_columns(pl.col.a.reverse().alias("b"))
        .with_columns(pl.col.a.shuffle().alias("c"))
        .with_columns(pl.col.a.shuffle().sort().alias("d"))
    )
    df.lazy().sink_ipc(
        buf, record_batch_size=record_batch_size, _record_batch_statistics=True
    )
    buf.seek(0)

    # round-trip with projection
    df = df.select(selection)
    out = pl.scan_ipc(buf, _record_batch_statistics=True).select(selection).collect()
    assert_frame_equal(df, out)
    assert_frame_equal(df._to_metadata(), out._to_metadata())


def test_scan_ipc_slice_empty_file() -> None:
    dfs = [
        pl.DataFrame({"a": range(0)}),
        pl.DataFrame({"a": range(100)}),
        pl.DataFrame({"a": range(0)}),
        pl.DataFrame({"a": range(100, 200)}),
    ]

    bufs: list[IO[bytes]] = [io.BytesIO() for _ in range(len(dfs))]

    for i in range(len(dfs)):
        dfs[i].write_ipc(bufs[i])
        bufs[i].seek(0)

    expected = pl.concat(dfs).slice(50, 100)
    actual = pl.scan_ipc(bufs).slice(50, 100).collect()

    assert_frame_equal(expected, actual)


@pytest.mark.slow
@pytest.mark.write_disk
@pytest.mark.skipif(
    sys.platform == "win32",
    reason="needs unix-only `resource` module to measure memory usage",
)
@pytest.mark.parametrize("force_async", [True, False])
def test_sink_ipc_memory_usage(force_async: bool) -> None:
    import subprocess
    import sys

    def mem_usage(n_chunks: int) -> int:
        n_runs = 3

        return min(
            int(
                subprocess.check_output(
                    [
                        sys.executable,
                        "-c",
                        """\
import resource
import sys
import tempfile

import polars as pl

(_, n_chunks) = sys.argv


s = pl.Series([0], dtype=pl.UInt32).new_from_index(
    0,
    1_000_000,
)
df = pl.concat(s for _ in range(int(n_chunks))).to_frame()

with tempfile.NamedTemporaryFile() as f:
    df.write_ipc(f.name)

print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

""",
                        str(n_chunks),
                    ],
                    env={
                        "POLARS_FORCE_ASYNC": "1" if force_async else "0",
                        **os.environ,
                    },
                ).decode()
            )
            for _ in range(n_runs)
        )

    m1 = mem_usage(1)
    m10 = mem_usage(10)

    ratio = m10 / m1

    # Ratio
    # 1.42.1: ~1.17
    # Fixed branch (debug build): ~1.008
    assert ratio < 1.05


def test_row_count_estimate_ipc(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)
    path = tmp_path / "a.ipc"
    pl.DataFrame({"a": range(37)}).write_ipc(path)

    # Polars writes the row count into the footer.
    assert "ESTIMATED ROWS: 37" in pl.scan_ipc(path).explain()


def test_row_count_estimate_ipc_foreign_writer(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)
    path = tmp_path / "a.ipc"

    # Without the Polars footer the record batch lengths are summed.
    schema = pa.schema([("a", pa.int64())])
    with pa.ipc.new_file(path, schema) as writer:
        for _ in range(4):
            writer.write_batch(pa.record_batch([pa.array(range(10))], schema=schema))

    assert "ESTIMATED ROWS: 40" in pl.scan_ipc(path).explain()


def test_row_count_estimate_ipc_many_blocks(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)
    path = tmp_path / "a.ipc"

    # Too many record batches to walk, so the count is extrapolated from a sample.
    schema = pa.schema([("a", pa.int64())])
    with pa.ipc.new_file(path, schema) as writer:
        for _ in range(199):
            writer.write_batch(pa.record_batch([pa.array(range(10))], schema=schema))
        writer.write_batch(pa.record_batch([pa.array(range(7))], schema=schema))

    assert "ESTIMATED ROWS: 1997" in pl.scan_ipc(path).explain()


def test_row_count_estimate_ipc_multifile(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)
    for name in ("a.ipc", "b.ipc"):
        pl.DataFrame({"a": range(10)}).write_ipc(tmp_path / name)

    # Only the first source is read, so the rest is extrapolated.
    assert "ESTIMATED ROWS: 20" in pl.scan_ipc(tmp_path / "*.ipc").explain()


@pytest.mark.parametrize(
    ("query", "maintain_order", "check_row_order"),
    [
        (lambda lf: lf, True, True),
        (lambda lf: lf.head(3), True, True),
        (lambda lf: lf.select(pl.col("a").sum()), False, True),
        (lambda lf: lf.sort("a"), False, True),
        (lambda lf: lf.head(3).select(pl.col("a").sum()), False, True),
        # Row index without a predicate is currently applied post-scan, which forces
        # order. Pushing it into the scan would be equally valid, so the flag is not
        # asserted.
        (lambda lf: lf.with_row_index().sort("a"), None, True),
        (
            lambda lf: lf.with_row_index().filter(pl.col("b") == 1).sort("a"),
            False,
            True,
        ),
        # Sortedness hints assert an order on the scan output.
        (
            lambda lf: (
                lf.with_columns(pl.col("a").set_sorted()).group_by("a").agg(pl.len())
            ),
            True,
            False,
        ),
    ],
)
def test_scan_ipc_maintain_order_only_if_observed(
    query: Callable[[pl.LazyFrame], pl.LazyFrame],
    maintain_order: bool | None,
    check_row_order: bool,
    plmonkeypatch: PlMonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    f = io.BytesIO()
    pl.DataFrame({"a": range(10), "b": [0, 1] * 5}).write_ipc(f, record_batch_size=2)
    f.seek(0)

    q = query(pl.scan_ipc(f))

    with plmonkeypatch.context() as cx:
        cx.setenv("POLARS_VERBOSE", "1")
        capfd.readouterr()
        out = q.collect(engine="streaming")
        capture = capfd.readouterr().err

    if maintain_order is not None:
        reader_lines = [
            x
            for x in capture.splitlines()
            if x.startswith("[IpcFileReader]") and "maintain_order:" in x
        ]
        assert reader_lines
        assert all(
            f"maintain_order: {str(maintain_order).lower()}" in x for x in reader_lines
        )

    assert_frame_equal(
        out,
        q.collect(
            engine="streaming",
            optimizations=pl.QueryOptFlags(check_order_observe=False),
        ),
        check_row_order=check_row_order,
    )


@pytest.mark.slow
@pytest.mark.write_disk
@pytest.mark.parametrize("n_files", [1, 3])
@pytest.mark.parametrize("writer", ["polars_metadata", "pyarrow"])
def test_scan_ipc_unordered_record_batches(
    n_files: int, writer: str, tmp_path: Path
) -> None:
    n = 10_000
    a = pl.Series("a", range(n))
    df = pl.DataFrame(
        {
            "a": a,
            "b": a % 7,
            "c": pl.select(pl.when(a % 3 == 0).then(a).otherwise(None)).to_series(),
            "d": pl.select(pl.concat_list([a, a % 5])).to_series(),
        }
    )

    for i in range(n_files):
        start, end = n * i // n_files, n * (i + 1) // n_files
        part = df.slice(start, end - start)
        path = tmp_path / f"{i}.ipc"
        if writer == "polars_metadata":
            # Record batch lengths in the footer, so row offsets are known up front.
            part.lazy().sink_ipc(
                path, record_batch_size=100, _record_batch_statistics=True
            )
        else:
            # No Polars metadata, so row offsets are counted as batches arrive.
            table = part.to_arrow()
            with pa.ipc.new_file(path, table.schema) as w:
                w.write_table(table, max_chunksize=100)

    lf = pl.scan_ipc(tmp_path / "*.ipc")

    def collect(q: pl.LazyFrame) -> pl.DataFrame:
        return q.collect(engine="streaming")

    assert_frame_equal(collect(lf.sort("a")), df)
    assert_frame_equal(
        collect(lf.with_row_index(offset=5).sort("index")),
        df.with_row_index(offset=5),
    )
    assert_frame_equal(
        collect(lf.with_row_index().filter(pl.col("b") == 3).sort("a")),
        df.with_row_index().filter(pl.col("b") == 3),
    )
    assert_frame_equal(
        collect(lf.slice(1_234, 5_432).select(pl.col("a").sum(), pl.len())),
        df.slice(1_234, 5_432).select(pl.col("a").sum(), pl.len()),
    )
    assert_frame_equal(
        collect(lf.group_by("b").agg(pl.col("a").sum()).sort("b")),
        df.group_by("b").agg(pl.col("a").sum()).sort("b"),
    )
    assert_frame_equal(
        collect(lf.filter(pl.col("b") == 3).select(pl.col("a").sum(), pl.len())),
        df.filter(pl.col("b") == 3).select(pl.col("a").sum(), pl.len()),
    )
    assert_frame_equal(collect(lf.select(pl.len())), df.select(pl.len()))


@pytest.mark.write_disk
def test_scan_ipc_set_sorted_expr_keeps_order(
    tmp_path: Path, plmonkeypatch: PlMonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    n_groups = 500
    group_size = 20
    n = n_groups * group_size
    n_record_batches = 8
    a = pl.Series("a", range(n)) // group_size

    # Data intentionally skewed so that record batches arrive out of order on decode.
    k = n // n_record_batches
    skewed_payload = pl.concat(
        [
            pl.select(
                pl.int_range(k).cast(pl.String).str.pad_start(300, "0")
            ).to_series(),
            pl.repeat("", n - k, eager=True),
        ]
    ).alias("s")
    df = pl.DataFrame({"a": a, "s": skewed_payload})
    df.write_ipc(tmp_path / "sorted.ipc", compression="zstd", record_batch_size=k)

    lf = pl.scan_ipc(tmp_path / "sorted.ipc")

    # Read the payload so its decode cost applies.
    s_len = pl.col("s").str.len_bytes().sum()

    q_group_by = (
        lf.with_columns(pl.col("a").set_sorted())
        .group_by("a")
        .agg(pl.len(), s_len)
        .sort("a")
    )
    expected_group_by = df.group_by("a").agg(pl.len(), s_len).sort("a")

    right = df.group_by("a").agg(pl.len().alias("n")).sort("a")
    right.write_ipc(tmp_path / "right.ipc", record_batch_size=n_groups // 8)
    q_join = (
        lf.select(pl.col("a").set_sorted(), "s")
        .join(
            pl.scan_ipc(tmp_path / "right.ipc").select(
                pl.col("a").set_sorted(), pl.col("n")
            ),
            on="a",
            maintain_order="none",
        )
        .group_by("a")
        .agg(pl.len(), pl.col("n").first(), s_len)
        .sort("a")
    )
    expected_join = expected_group_by.join(right, on="a").select("a", "len", "n", "s")

    # Verify fast-path
    def physical_plan(q: pl.LazyFrame) -> str:
        return q.show_graph(engine="streaming", plan_stage="physical", raw_output=True)

    assert "sorted-group-by" in physical_plan(q_group_by)
    assert "merge-join" in physical_plan(q_join)

    # Every scan must keep its order; skewed decode alone may not reorder batches.
    def collect_ordered(q: pl.LazyFrame) -> pl.DataFrame:
        with plmonkeypatch.context() as cx:
            cx.setenv("POLARS_VERBOSE", "1")
            capfd.readouterr()
            out = q.collect(engine="streaming")
            capture = capfd.readouterr().err
        reader_lines = [
            x
            for x in capture.splitlines()
            if x.startswith("[IpcFileReader]") and "maintain_order:" in x
        ]
        assert reader_lines
        assert all("maintain_order: true" in x for x in reader_lines)
        return out

    assert_frame_equal(collect_ordered(q_group_by), expected_group_by)
    assert_frame_equal(collect_ordered(q_join), expected_join)
