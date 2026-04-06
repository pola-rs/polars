from __future__ import annotations

import datetime as dt
import decimal
import io
import os
from itertools import permutations
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import polars as pl
from polars.exceptions import ComputeError
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from polars._typing import EngineType
    from polars.io.partition import SinkedFilesCallbackArgs
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
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, file_name: str
) -> None:
    monkeypatch.chdir(tmp_path)

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


@pytest.mark.parametrize(
    ("base_path", "provided_path"),
    [
        *permutations(["/", "s3://", "file:///"], 2),
        ("/a/", "/b/"),
    ],
)
def test_sink_file_provider_absolute_path_not_under_base_path(
    base_path: str, provided_path: str
) -> None:
    df = pl.DataFrame({"a": 1})

    with pytest.raises(
        ComputeError,
        match=r"provided path.*is absolute but does not start with base path",
    ):
        df.lazy().sink_parquet(
            pl.PartitionBy(
                base_path,
                file_path_provider=lambda _: provided_path,
                max_rows_per_file=1,
            )
        )


@pytest.mark.parametrize(
    "s",
    ["/", "\\"],
)
def test_sink_file_provider_forbid_parent_dir_component(s: str) -> None:
    df = pl.DataFrame({"a": 1})

    err_cx = pytest.raises(
        ComputeError,
        match=r"provided path.*contained parent dir component",
    )

    def expect_err(p: str) -> None:
        with err_cx:
            df.lazy().sink_parquet(
                pl.PartitionBy(
                    "",
                    file_path_provider=lambda _: p,
                    max_rows_per_file=1,
                )
            )

    expect_err("..")
    expect_err(f"{s}..")
    expect_err(f"..{s}")
    expect_err(f"{s}..{s}")


@pytest.mark.write_disk
def test_sinked_files_callback_single(tmp_path: Path) -> None:
    lf = pl.LazyFrame({"a": [0, 1, 2, 3, 4]})

    out_path = tmp_path / "a.parquet"
    lst: list[SinkedFilesCallbackArgs] = []
    lf.sink_parquet(out_path, sinked_files_callback=lst.append)

    assert [Path(x) for x in lst[0].paths] == [out_path]
    assert len(lst[0].files) == 1

    info = lst[0].files[0]
    assert info.parquet is not None
    assert info.num_rows == lf.collect().height
    assert 0 < info.parquet.footer_size_bytes < info.file_size_bytes
    assert len(info.parquet.column_stats) == 0


@pytest.mark.write_disk
def test_sinked_files_callback_single_with_field_ids(tmp_path: Path) -> None:
    lf = pl.LazyFrame({"a": [0, 1, 2, 3, 4]})
    arrow_schema = pa.schema(
        [pa.field("a", pa.int64()).with_metadata({"PARQUET:field_id": "1"})]
    )

    lst: list[SinkedFilesCallbackArgs] = []
    lf.sink_parquet(
        tmp_path / "a.parquet",
        sinked_files_callback=lst.append,
        arrow_schema=arrow_schema,
    )

    info = lst[0].files[0]
    assert info.parquet is not None
    assert len(info.parquet.column_stats) == 1

    col_stats = info.parquet.column_stats[0]
    assert col_stats.field_id == 1
    assert col_stats.compressed_size_bytes > 0
    assert col_stats.null_count == 0
    assert col_stats.min_value == 0
    assert col_stats.max_value == 4


@pytest.mark.write_disk
def test_sinked_files_callback_multiple(tmp_path: Path) -> None:
    lf = pl.LazyFrame({"a": [0, 1, 2, 3, 4]})

    lst: list[SinkedFilesCallbackArgs] = []
    lf.sink_parquet(
        pl.PartitionBy(tmp_path, max_rows_per_file=1),
        sinked_files_callback=lst.append,
    )
    assert len(lst[0].files) == 5
    assert all(s.num_rows == 1 for s in lst[0].files)


@pytest.mark.write_disk
def test_sinked_files_callback_failure(tmp_path: Path) -> None:
    lf = pl.LazyFrame({"a": [0, 1, 2, 3, 4]})
    with pytest.raises(ComputeError, match="encountered non-path sink target"):
        lf.sink_parquet(
            pl.PartitionBy(
                tmp_path,
                file_path_provider=lambda _: io.BytesIO(),
                max_rows_per_file=1,
            ),
            sinked_files_callback=lambda _: None,
        )


@pytest.mark.write_disk
def test_sinked_files_callback_no_rows(tmp_path: Path) -> None:
    lf = pl.LazyFrame({"a": []}, schema={"a": pl.Int64})

    lst: list[SinkedFilesCallbackArgs] = []
    lf.sink_parquet(tmp_path / "a.parquet", sinked_files_callback=lst.append)

    assert len(lst[0].files) == 1
    assert sum(s.num_rows for s in lst[0].files) == 0


@pytest.mark.write_disk
def test_sinked_files_callback_no_columns(tmp_path: Path) -> None:
    lf = pl.LazyFrame()

    lst: list[SinkedFilesCallbackArgs] = []
    lf.sink_parquet(tmp_path / "a.parquet", sinked_files_callback=lst.append)

    assert len(lst[0].files) == 1
    assert sum(s.num_rows for s in lst[0].files) == 0
    assert lst[0].files[0].parquet is not None
    assert len(lst[0].files[0].parquet.column_stats) == 0


@pytest.mark.write_disk
@pytest.mark.parametrize(
    ("dtype", "values", "expected_min", "expected_max"),
    [
        (pl.UInt8(), [1, 3, 2, 4, 0], 0, 4),
        (pl.UInt16(), [1, 3, 2, 4, 0], 0, 4),
        (pl.UInt32(), [1, 3, 2, 4, 0], 0, 4),
        (pl.UInt64(), [1, 3, 2, 4, 0], 0, 4),
        (pl.Int8(), [1, 3, 2, 4, 0], 0, 4),
        (pl.Int16(), [1, 3, 2, 4, 0], 0, 4),
        (pl.Int32(), [1, 3, 2, 4, 0], 0, 4),
        (pl.Int64(), [1, 3, 2, 4, 0], 0, 4),
        (pl.Float32(), [1, 3, 2, 4, 0], 0, 4),
        (pl.Float64(), [1, 3, 2, 4, 0], 0, 4),
        (pl.String(), ["x", "q", "m", "o", "z"], "m", "z"),
        (pl.Boolean(), [True, False, True], False, True),
        (
            pl.Date(),
            [dt.date(2020, 1, 1), dt.date(2022, 7, 5), dt.date(2019, 9, 19)],
            dt.date(2019, 9, 19),
            dt.date(2022, 7, 5),
        ),
        (
            pl.Datetime(),
            [
                dt.datetime(2020, 1, 1),
                dt.datetime(2022, 7, 5),
                dt.datetime(2019, 9, 19),
            ],
            dt.datetime(2019, 9, 19),
            dt.datetime(2022, 7, 5),
        ),
        (
            pl.Time(),
            [dt.time(0), dt.time(18, 45), dt.time(12)],
            dt.time(0),
            dt.time(18, 45),
        ),
        (
            pl.Duration(),
            [dt.timedelta(days=1), dt.timedelta(days=5), dt.timedelta(days=3)],
            dt.timedelta(days=1),
            dt.timedelta(days=5),
        ),
        (
            pl.Decimal(precision=2, scale=1),
            [decimal.Decimal("1.1"), decimal.Decimal("3.3"), decimal.Decimal("2.2")],
            decimal.Decimal("1.1"),
            decimal.Decimal("3.3"),
        ),
        (pl.Int64(), [], None, None),
    ],
)
def test_sinked_files_callback_stats_aggregation(
    tmp_path: Path,
    dtype: pl.DataType,
    values: list[Any],
    expected_min: Any,
    expected_max: Any,
) -> None:
    lf = pl.LazyFrame({"a": values}, schema={"a": dtype})
    pa_dtype = lf.collect().to_arrow().schema.field("a").type
    arrow_schema = pa.schema(
        [pa.field("a", pa_dtype).with_metadata({"PARQUET:field_id": "1"})]
    )

    lst: list[SinkedFilesCallbackArgs] = []
    lf.sink_parquet(
        tmp_path / "a.parquet",
        sinked_files_callback=lst.append,
        row_group_size=2,
        arrow_schema=arrow_schema,
    )

    assert len(lst[0].files) == 1
    assert (
        pq.ParquetFile(tmp_path / "a.parquet").metadata.num_row_groups
        == (len(values) + 1) // 2
    )
    assert sum(s.num_rows for s in lst[0].files) == lf.collect().height

    info = lst[0].files[0]
    assert info.parquet is not None
    assert len(info.parquet.column_stats) == 1
    col_a = info.parquet.column_stats[0]
    assert col_a.field_id == 1
    assert col_a.min_value == expected_min
    assert col_a.max_value == expected_max


@pytest.mark.write_disk
def test_sinked_files_callback_stats_nested(tmp_path: Path) -> None:
    lf = pl.LazyFrame(
        {
            "s": [{"x": 1, "y": [2, 3]}, {"x": 4, "y": [5, 6]}],
            "i": [8, 9],
            "l": [[1, 2], [3, 4]],
        }
    )

    FIELD_ID = "PARQUET:field_id"
    arrow_schema = pa.schema(
        [
            pa.field(
                "s",
                pa.struct(
                    [
                        pa.field("x", pa.int64()).with_metadata({FIELD_ID: "2"}),
                        pa.field(
                            "y",
                            pa.large_list(
                                pa.field("element", pa.int64()).with_metadata(
                                    {FIELD_ID: "4"}
                                )
                            ),
                        ).with_metadata({FIELD_ID: "3"}),
                    ]
                ),
            ).with_metadata({FIELD_ID: "1"}),
            pa.field("i", pa.int64()).with_metadata({FIELD_ID: "5"}),
            pa.field(
                "l",
                pa.large_list(
                    pa.field("element", pa.int64()).with_metadata({FIELD_ID: "7"})
                ),
            ).with_metadata({FIELD_ID: "6"}),
        ]
    )

    lst: list[SinkedFilesCallbackArgs] = []
    lf.sink_parquet(
        tmp_path / "a.parquet",
        sinked_files_callback=lst.append,
        arrow_schema=arrow_schema,
    )

    assert len(lst[0].files) == 1
    info = lst[0].files[0]
    assert info.parquet is not None
    columns = info.parquet.column_stats
    assert {col.field_id for col in columns} == {2, 4, 5, 7}

    stats_s_x = next(c for c in columns if c.field_id == 2)
    stats_s_y = next(c for c in columns if c.field_id == 4)
    stats_i = next(c for c in columns if c.field_id == 5)
    stats_l = next(c for c in columns if c.field_id == 7)

    assert (stats_s_x.min_value, stats_s_x.max_value) == (1, 4)
    assert (stats_s_y.min_value, stats_s_y.max_value) == (2, 6)
    assert (stats_i.min_value, stats_i.max_value) == (8, 9)
    assert (stats_l.min_value, stats_l.max_value) == (1, 4)
