from __future__ import annotations

import io
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

import pytest

import polars as pl
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from polars._typing import EngineType
    from tests.conftest import PlMonkeyPatch


SINKS = [
    (pl.scan_ipc, pl.LazyFrame.sink_ipc),
    (pl.scan_parquet, pl.LazyFrame.sink_parquet),
    (pl.scan_csv, pl.LazyFrame.sink_csv),
    (pl.scan_ndjson, pl.LazyFrame.sink_ndjson),
]


@pytest.mark.parametrize(("scan", "sink"), SINKS)
@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
@pytest.mark.write_disk
def test_mkdir(tmp_path: Path, scan: Any, sink: Any, engine: EngineType) -> None:
    df = pl.DataFrame(
        {
            "a": [1, 2, 3],
        }
    )

    with pytest.raises(FileNotFoundError):
        sink(df.lazy(), tmp_path / "a" / "b" / "c" / "file", engine=engine)

    f = tmp_path / "a" / "b" / "c" / "file2"
    sink(df.lazy(), f, mkdir=True)

    assert_frame_equal(scan(f).collect(), df)


def test_write_mkdir(tmp_path: Path) -> None:
    df = pl.DataFrame(
        {
            "a": [1, 2, 3],
        }
    )

    with pytest.raises(FileNotFoundError):
        df.write_parquet(tmp_path / "a" / "b" / "c" / "file")

    f = tmp_path / "a" / "b" / "c" / "file2"
    df.write_parquet(f, mkdir=True)

    assert_frame_equal(pl.read_parquet(f), df)


@pytest.mark.parametrize(("scan", "sink"), SINKS)
@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
@pytest.mark.write_disk
def test_lazy_sinks(tmp_path: Path, scan: Any, sink: Any, engine: EngineType) -> None:
    df = pl.DataFrame({"a": [1, 2, 3]})
    lf1 = sink(df.lazy(), tmp_path / "a", lazy=True)
    lf2 = sink(df.lazy(), tmp_path / "b", lazy=True)

    assert not Path(tmp_path / "a").exists()
    assert not Path(tmp_path / "b").exists()

    pl.collect_all([lf1, lf2], engine=engine)

    assert_frame_equal(scan(tmp_path / "a").collect(), df)
    assert_frame_equal(scan(tmp_path / "b").collect(), df)


@pytest.mark.parametrize(
    "sink",
    [
        pl.LazyFrame.sink_ipc,
        pl.LazyFrame.sink_parquet,
        pl.LazyFrame.sink_csv,
        pl.LazyFrame.sink_ndjson,
    ],
)
@pytest.mark.write_disk
def test_double_lazy_error(sink: Any) -> None:
    df = pl.DataFrame({})

    with pytest.raises(
        pl.exceptions.InvalidOperationError,
        match="cannot create a sink on top of another sink",
    ):
        sink(sink(df.lazy(), "a", lazy=True), "b")


@pytest.mark.parametrize(("scan", "sink"), SINKS)
def test_sink_to_memory(sink: Any, scan: Any) -> None:
    df = pl.DataFrame(
        {
            "a": [5, 10, 1996],
        }
    )

    f = io.BytesIO()
    sink(df.lazy(), f)

    f.seek(0)
    assert_frame_equal(
        scan(f).collect(),
        df,
    )


@pytest.mark.parametrize(("scan", "sink"), SINKS)
@pytest.mark.write_disk
def test_sink_to_file(tmp_path: Path, sink: Any, scan: Any) -> None:
    df = pl.DataFrame(
        {
            "a": [5, 10, 1996],
        }
    )

    with (tmp_path / "f").open("w+") as f:
        sink(df.lazy(), f, sync_on_close="all")
        f.seek(0)
        assert_frame_equal(
            scan(f).collect(),
            df,
        )


@pytest.mark.parametrize(("scan", "sink"), SINKS)
def test_sink_empty(sink: Any, scan: Any) -> None:
    df = pl.LazyFrame(data={"col1": ["a"]})

    df_empty = pl.LazyFrame(
        data={"col1": []},
        schema={"col1": str},
    )

    expected = df_empty.join(df, how="cross").collect()
    expected_schema = expected.schema

    kwargs = {}
    if scan == pl.scan_ndjson:
        kwargs["schema"] = expected_schema

    # right empty
    f = io.BytesIO()
    sink(df.join(df_empty, how="cross"), f)
    f.seek(0)
    assert_frame_equal(scan(f, **kwargs), expected.lazy())

    # left empty
    f.seek(0)
    sink(df_empty.join(df, how="cross"), f)
    f.truncate()
    f.seek(0)
    assert_frame_equal(scan(f, **kwargs), expected.lazy())

    # both empty
    f.seek(0)
    sink(df_empty.join(df_empty, how="cross"), f)
    f.truncate()
    f.seek(0)
    assert_frame_equal(scan(f, **kwargs), expected.lazy())


@pytest.mark.parametrize(("scan", "sink"), SINKS)
def test_sink_boolean_panic_25806(sink: Any, scan: Any) -> None:
    morsel_size = int(os.environ.get("POLARS_IDEAL_MORSEL_SIZE", 100_000))
    df = pl.select(bool=pl.repeat(True, 3 * morsel_size))

    f = io.BytesIO()
    sink(df.lazy(), f)

    assert_frame_equal(scan(f).collect(), df)


def test_collect_all_lazy() -> None:
    with TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        a = pl.LazyFrame({"a": [1, 2, 3, 4, 5, 6]})
        b = a.filter(pl.col("a") % 2 == 0).sink_csv(tmp_path / "b.csv", lazy=True)
        c = a.filter(pl.col("a") % 3 == 0).sink_csv(tmp_path / "c.csv", lazy=True)
        d = a.sink_csv(tmp_path / "a.csv", lazy=True)

        q = pl.collect_all([d, b, c], lazy=True)

        assert q._ldf._node_name() == "SinkMultiple"  # type: ignore[attr-defined]
        q.collect()
        df_a = pl.read_csv(tmp_path / "a.csv")
        df_b = pl.read_csv(tmp_path / "b.csv")
        df_c = pl.read_csv(tmp_path / "c.csv")

        assert_frame_equal(df_a, pl.DataFrame({"a": [1, 2, 3, 4, 5, 6]}))
        assert_frame_equal(df_b, pl.DataFrame({"a": [2, 4, 6]}))
        assert_frame_equal(df_c, pl.DataFrame({"a": [3, 6]}))

    with pytest.raises(ValueError, match="all LazyFrames must end with a sink to use"):
        pl.collect_all([a, a], lazy=True)


def check_compression(content: bytes, expected_format: str) -> None:
    if expected_format == "gzip":
        assert content[:2] == bytes([0x1F, 0x8B])
    elif expected_format == "zstd":
        assert content[:4] == bytes([0x28, 0xB5, 0x2F, 0xFD])
    else:
        pytest.fail("Unreachable")


def write_fn(df: pl.DataFrame, write_fn_name: str) -> Any:
    if write_fn_name == "write_csv":
        return df.write_csv
    elif write_fn_name == "sink_csv":
        return df.lazy().sink_csv
    if write_fn_name == "write_ndjson":
        return df.write_ndjson
    elif write_fn_name == "sink_ndjson":
        return df.lazy().sink_ndjson
    else:
        pytest.fail("unreachable")


def scan_fn(write_fn_name: str) -> Any:
    if "csv" in write_fn_name:
        return pl.scan_csv
    elif "ndjson" in write_fn_name:
        return pl.scan_ndjson
    else:
        pytest.fail("unreachable")


@pytest.mark.parametrize(
    "write_fn_name", ["write_csv", "sink_csv", "write_ndjson", "sink_ndjson"]
)
@pytest.mark.parametrize("fmt", ["gzip", "zstd"])
@pytest.mark.parametrize("level", [None, 0, 9])
def test_write_compressed(write_fn_name: str, fmt: str, level: int | None) -> None:
    original = pl.DataFrame([pl.Series("A", [3.2, 6.2]), pl.Series("B", ["a", "z"])])
    buf = io.BytesIO()
    write_fn(original, write_fn_name)(buf, compression=fmt, compression_level=level)
    buf.seek(0)
    check_compression(buf.read(), fmt)
    buf.seek(0)
    df = scan_fn(write_fn_name)(buf).collect()
    assert_frame_equal(df, original)


@pytest.mark.write_disk
@pytest.mark.parametrize(
    "write_fn_name", ["write_csv", "sink_csv", "write_ndjson", "sink_ndjson"]
)
@pytest.mark.parametrize(("fmt", "suffix"), [("gzip", ".gz"), ("zstd", ".zst")])
@pytest.mark.parametrize("with_suffix", [True, False])
def test_write_compressed_disk(
    tmp_path: Path, write_fn_name: str, fmt: str, suffix: str, with_suffix: bool
) -> None:
    original = pl.DataFrame([pl.Series("A", [3.2, 6.2]), pl.Series("B", ["a", "z"])])
    path = tmp_path / (f"test_file.{suffix}" if with_suffix else "test_file")
    write_fn(original, write_fn_name)(path, compression=fmt)
    with path.open("rb") as file:
        check_compression(file.read(), fmt)
    df = scan_fn(write_fn_name)(path).collect()
    assert_frame_equal(df, original)


@pytest.mark.write_disk
@pytest.mark.parametrize(
    "write_fn_name", ["write_csv", "sink_csv", "write_ndjson", "sink_ndjson"]
)
@pytest.mark.parametrize("fmt", ["gzip", "zstd"])
def test_write_uncommon_file_suffix_ignore(
    tmp_path: Path, write_fn_name: str, fmt: str
) -> None:
    path = tmp_path / "x"
    write_fn(pl.DataFrame(), write_fn_name)(
        path, compression=fmt, check_extension=False
    )
    with Path.open(path, "rb") as file:
        check_compression(file.read(), fmt)


@pytest.mark.parametrize(
    "write_fn_name", ["write_csv", "sink_csv", "write_ndjson", "sink_ndjson"]
)
@pytest.mark.parametrize("fmt", ["gzip", "zstd"])
def test_write_uncommon_file_suffix_raise(write_fn_name: str, fmt: str) -> None:
    with pytest.raises(pl.exceptions.InvalidOperationError):
        write_fn(pl.DataFrame(), write_fn_name)("x.csv", compression=fmt)


@pytest.mark.parametrize(
    "write_fn_name", ["write_csv", "sink_csv", "write_ndjson", "sink_ndjson"]
)
@pytest.mark.parametrize("extension", ["gz", "zst", "zstd"])
def test_write_intended_compression(write_fn_name: str, extension: str) -> None:
    with pytest.raises(
        pl.exceptions.InvalidOperationError, match="use the compression parameter"
    ):
        write_fn(pl.DataFrame(), write_fn_name)(f"x.csv.{extension}")


@pytest.mark.write_disk
@pytest.mark.parametrize(
    "write_fn_name", ["write_csv", "sink_csv", "write_ndjson", "sink_ndjson"]
)
@pytest.mark.parametrize("extension", ["tsv", "xslb", "cs"])
def test_write_alternative_extension(
    tmp_path: Path, write_fn_name: str, extension: str
) -> None:
    path = tmp_path / f"x.{extension}"
    write_fn(pl.DataFrame(), write_fn_name)(path)
    assert Path.exists(path)


@pytest.mark.parametrize(
    "write_fn_name", ["write_csv", "sink_csv", "write_ndjson", "sink_ndjson"]
)
@pytest.mark.parametrize("fmt", ["gzipd", "zs", ""])
def test_write_unsupported_compression(write_fn_name: str, fmt: str) -> None:
    with pytest.raises(pl.exceptions.InvalidOperationError):
        write_fn(pl.DataFrame(), write_fn_name)("x", compression=fmt)


@pytest.mark.write_disk
@pytest.mark.parametrize("file_name", ["凸变英雄X", "影分身の術"])
def test_sink_path_slicing_utf8_boundaries_26324(
    tmp_path: Path, file_name: str
) -> None:
    os.chdir(tmp_path)

    df = pl.DataFrame({"a": 1})
    df.write_parquet(file_name)

    assert_frame_equal(pl.scan_parquet(file_name).collect(), df)


@pytest.mark.parametrize("file_format", ["parquet", "ipc", "csv", "ndjson"])
@pytest.mark.parametrize("partitioned", [True, False])
@pytest.mark.write_disk
def test_sink_metrics(
    plmonkeypatch: PlMonkeyPatch,
    capfd: pytest.CaptureFixture[str],
    file_format: str,
    tmp_path: Path,
    partitioned: bool,
) -> None:
    path = tmp_path / "a"

    df = pl.DataFrame({"a": 1})

    with plmonkeypatch.context() as cx:
        cx.setenv("POLARS_LOG_METRICS", "1")
        cx.setenv("POLARS_FORCE_ASYNC", "1")
        capfd.readouterr()
        getattr(pl.LazyFrame, f"sink_{file_format}")(
            df.lazy(),
            path
            if not partitioned
            else pl.PartitionBy("", file_path_provider=(lambda _: path), key="a"),
        )
        capture = capfd.readouterr().err

    [line] = (x for x in capture.splitlines() if x.startswith("io-sink"))

    logged_bytes_sent = int(
        pl.select(pl.lit(line).str.extract(r"total_bytes_sent=(\d+)")).item()
    )

    assert logged_bytes_sent == path.stat().st_size

    assert_frame_equal(getattr(pl, f"scan_{file_format}")(path).collect(), df)
