from __future__ import annotations

import io
import typing
from typing import IO, TYPE_CHECKING, Any

import pyarrow.ipc
import pytest

import polars as pl
from polars.interchange.protocol import CompatLevel
from polars.testing.asserts.frame import assert_frame_equal

if TYPE_CHECKING:
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

    t1 = pyarrow.ipc.open_file(f1)
    assert "large_string" in str(t1.schema)
    assert_frame_equal(pl.DataFrame(t1.read_all()), df)

    t2 = pyarrow.ipc.open_file(f2)
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

    out = pl.scan_ipc(buf).collect()
    expected = lf.collect()
    assert_frame_equal(out, expected)


# TODO: create multiple record batches through API instead of env variable
def test_scan_ipc_file_async_multiple_record_batches(
    plmonkeypatch: PlMonkeyPatch,
) -> None:
    plmonkeypatch.setenv("POLARS_FORCE_ASYNC", "1")
    plmonkeypatch.setenv("POLARS_IDEAL_SINK_MORSEL_SIZE_ROWS", "10")

    buf = io.BytesIO()
    lf = pl.LazyFrame({"a": list(range(100))})
    lf.sink_ipc(buf)
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

    with pyarrow.ipc.open_file(buf) as reader:
        assert [
            reader.get_batch(i).num_rows for i in range(reader.num_record_batches)
        ] == [1, 1]

    assert_frame_equal(pl.scan_ipc(buf).collect(), df)


@pytest.mark.parametrize(
    "record_batch_size", [1, 2, 5, 7, 50, 99, 100, 101, 299, 300, 100_000]
)
@pytest.mark.parametrize("n_chunks", [1, 2, 3])
def test_sink_ipc_record_batch_size(record_batch_size: int, n_chunks: int) -> None:
    n_rows = 100
    buf = io.BytesIO()

    df0 = pl.DataFrame({"a": list(range(n_rows))})
    df = df0
    while n_chunks > 1:
        df = pl.concat([df, df0])
        n_chunks -= 1

    df.lazy().sink_ipc(buf, record_batch_size=record_batch_size)

    buf.seek(0)
    out = pl.scan_ipc(buf).collect()
    assert_frame_equal(out, df)

    buf.seek(0)
    reader = pyarrow.ipc.open_file(buf)
    n_batches = reader.num_record_batches
    for i in range(n_batches):
        n_rows = reader.get_batch(i).num_rows
        assert n_rows == record_batch_size or (
            i + 1 == n_batches and n_rows <= record_batch_size
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

    # round-trip with projection
    df = df.select(selection)
    out = pl.scan_ipc(buf, _record_batch_statistics=True).select(selection).collect()
    assert_frame_equal(df, out)
    assert_frame_equal(df._to_metadata(), out._to_metadata())
