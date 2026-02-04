from __future__ import annotations

import io
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

import pytest
from hypothesis import example, given

import polars as pl
from polars.exceptions import InvalidOperationError
from polars.testing import assert_frame_equal, assert_series_equal
from polars.testing.parametric.strategies import dataframes
from tests.unit.io.conftest import format_file_uri

if TYPE_CHECKING:
    from polars._typing import EngineType
    from polars.io.partition import FileProviderArgs


class IOType(TypedDict):
    """A type of IO."""

    ext: str
    scan: Any
    sink: Any


io_types: list[IOType] = [
    {"ext": "csv", "scan": pl.scan_csv, "sink": pl.LazyFrame.sink_csv},
    {"ext": "jsonl", "scan": pl.scan_ndjson, "sink": pl.LazyFrame.sink_ndjson},
    {"ext": "parquet", "scan": pl.scan_parquet, "sink": pl.LazyFrame.sink_parquet},
    {"ext": "ipc", "scan": pl.scan_ipc, "sink": pl.LazyFrame.sink_ipc},
]

engines: list[EngineType] = [
    "streaming",
    "in-memory",
]


def test_partition_by_api() -> None:
    with pytest.raises(
        ValueError,
        match=r"at least one of \('key', 'max_rows_per_file', 'approximate_bytes_per_file'\) must be specified for PartitionBy",
    ):
        pl.PartitionBy("")

    error_cx = pytest.raises(
        ValueError, match="cannot use 'include_key' without specifying 'key'"
    )

    with error_cx:
        pl.PartitionBy("", include_key=True, max_rows_per_file=1)

    with error_cx:
        pl.PartitionBy("", include_key=False, max_rows_per_file=1)

    assert (
        pl.PartitionBy("", key="key")._pl_partition_by.approximate_bytes_per_file
        == 4_294_967_295
    )

    # If `max_rows_per_file` was given then `approximate_bytes_per_file` should
    # default to disabled (u64::MAX).
    assert (
        pl.PartitionBy(
            "", max_rows_per_file=1
        )._pl_partition_by.approximate_bytes_per_file
        == (1 << 64) - 1
    )

    assert (
        pl.PartitionBy(
            "", key="key", max_rows_per_file=1
        )._pl_partition_by.approximate_bytes_per_file
        == (1 << 64) - 1
    )

    assert (
        pl.PartitionBy(
            "", max_rows_per_file=1, approximate_bytes_per_file=1024
        )._pl_partition_by.approximate_bytes_per_file
        == 1024
    )


@pytest.mark.parametrize("io_type", io_types)
@pytest.mark.parametrize("engine", engines)
@pytest.mark.parametrize("length", [0, 1, 4, 5, 6, 7])
@pytest.mark.parametrize("max_size", [1, 2, 3])
@pytest.mark.write_disk
def test_max_size_partition(
    tmp_path: Path,
    io_type: IOType,
    engine: EngineType,
    length: int,
    max_size: int,
) -> None:
    lf = pl.Series("a", range(length), pl.Int64).to_frame().lazy()

    (io_type["sink"])(
        lf,
        pl.PartitionBy(tmp_path, max_rows_per_file=max_size),
        engine=engine,
        # We need to sync here because platforms do not guarantee that a close on
        # one thread is immediately visible on another thread.
        #
        # "Multithreaded processes and close()"
        # https://man7.org/linux/man-pages/man2/close.2.html
        sync_on_close="data",
    )

    i = 0
    while length > 0:
        assert (io_type["scan"])(tmp_path / f"{i:08}.{io_type['ext']}").select(
            pl.len()
        ).collect()[0, 0] == min(max_size, length)

        length -= max_size
        i += 1


def test_partition_by_max_rows_per_file() -> None:
    files = {}

    def file_path_provider(args: FileProviderArgs) -> Any:
        f = io.BytesIO()
        files[args.index_in_partition] = f
        return f

    df = pl.select(x=pl.int_range(0, 100))
    df.lazy().sink_parquet(
        pl.PartitionBy("", file_path_provider=file_path_provider, max_rows_per_file=10)
    )

    for f in files.values():
        f.seek(0)

    assert_frame_equal(
        pl.scan_parquet([files[i] for i in range(len(files))]).collect(),  # type: ignore[arg-type]
        df,
    )

    for f in files.values():
        f.seek(0)

    assert [
        pl.scan_parquet(files[i]).select(pl.len()).collect().item()
        for i in range(len(files))
    ] == [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]


@pytest.mark.parametrize("io_type", io_types)
@pytest.mark.parametrize("engine", engines)
def test_max_size_partition_lambda(
    tmp_path: Path, io_type: IOType, engine: EngineType
) -> None:
    length = 17
    max_size = 3
    lf = pl.Series("a", range(length), pl.Int64).to_frame().lazy()

    (io_type["sink"])(
        lf,
        pl.PartitionBy(
            tmp_path,
            file_path_provider=lambda args: tmp_path
            / f"abc-{args.index_in_partition:08}.{io_type['ext']}",
            max_rows_per_file=max_size,
        ),
        engine=engine,
        # We need to sync here because platforms do not guarantee that a close on
        # one thread is immediately visible on another thread.
        #
        # "Multithreaded processes and close()"
        # https://man7.org/linux/man-pages/man2/close.2.html
        sync_on_close="data",
    )

    i = 0
    while length > 0:
        assert (io_type["scan"])(tmp_path / f"abc-{i:08}.{io_type['ext']}").select(
            pl.len()
        ).collect()[0, 0] == min(max_size, length)

        length -= max_size
        i += 1


@pytest.mark.parametrize("io_type", io_types)
@pytest.mark.parametrize("engine", engines)
@pytest.mark.write_disk
def test_partition_by_key(
    tmp_path: Path,
    io_type: IOType,
    engine: EngineType,
) -> None:
    lf = pl.Series("a", [i % 4 for i in range(7)], pl.Int64).to_frame().lazy()

    (io_type["sink"])(
        lf,
        pl.PartitionBy(
            tmp_path,
            file_path_provider=lambda args: f"{args.partition_keys.item()}.{io_type['ext']}",
            key="a",
        ),
        engine=engine,
        # We need to sync here because platforms do not guarantee that a close on
        # one thread is immediately visible on another thread.
        #
        # "Multithreaded processes and close()"
        # https://man7.org/linux/man-pages/man2/close.2.html
        sync_on_close="data",
    )

    assert_series_equal(
        (io_type["scan"])(tmp_path / f"0.{io_type['ext']}").collect().to_series(),
        pl.Series("a", [0, 0], pl.Int64),
    )
    assert_series_equal(
        (io_type["scan"])(tmp_path / f"1.{io_type['ext']}").collect().to_series(),
        pl.Series("a", [1, 1], pl.Int64),
    )
    assert_series_equal(
        (io_type["scan"])(tmp_path / f"2.{io_type['ext']}").collect().to_series(),
        pl.Series("a", [2, 2], pl.Int64),
    )
    assert_series_equal(
        (io_type["scan"])(tmp_path / f"3.{io_type['ext']}").collect().to_series(),
        pl.Series("a", [3], pl.Int64),
    )

    scan_flags = (
        {"schema": pl.Schema({"a": pl.String()})} if io_type["ext"] == "csv" else {}
    )

    # Change the datatype.
    (io_type["sink"])(
        lf,
        pl.PartitionBy(
            tmp_path,
            file_path_provider=lambda args: f"{args.partition_keys.item()}.{io_type['ext']}",
            key=pl.col.a.cast(pl.String()),
        ),
        engine=engine,
        sync_on_close="data",
    )

    assert_series_equal(
        (io_type["scan"])(tmp_path / f"0.{io_type['ext']}", **scan_flags)
        .collect()
        .to_series(),
        pl.Series("a", ["0", "0"], pl.String),
    )
    assert_series_equal(
        (io_type["scan"])(tmp_path / f"1.{io_type['ext']}", **scan_flags)
        .collect()
        .to_series(),
        pl.Series("a", ["1", "1"], pl.String),
    )
    assert_series_equal(
        (io_type["scan"])(tmp_path / f"2.{io_type['ext']}", **scan_flags)
        .collect()
        .to_series(),
        pl.Series("a", ["2", "2"], pl.String),
    )
    assert_series_equal(
        (io_type["scan"])(tmp_path / f"3.{io_type['ext']}", **scan_flags)
        .collect()
        .to_series(),
        pl.Series("a", ["3"], pl.String),
    )


# We only deal with self-describing formats
@pytest.mark.parametrize("io_type", [io_types[2], io_types[3]])
@example(df=pl.DataFrame({"a": [0.0, -0.0]}, schema={"a": pl.Float16}))
@given(
    df=dataframes(
        min_cols=1,
        min_size=1,
        excluded_dtypes=[
            pl.Decimal,  # Bug see: https://github.com/pola-rs/polars/issues/21684
            pl.Duration,  # Bug see: https://github.com/pola-rs/polars/issues/21964
            pl.Categorical,  # We cannot ensure the string cache is properly held.
            # Generate invalid UTF-8
            pl.Binary,
            pl.Struct,
            pl.Array,
            pl.List,
            pl.Extension,  # Can't be cast to string
        ],
    )
)
def test_partition_by_key_parametric(
    io_type: IOType,
    df: pl.DataFrame,
) -> None:
    col1 = df.columns[0]

    output_files = []

    def file_path_provider(args: FileProviderArgs) -> io.BytesIO:
        f = io.BytesIO()
        output_files.append(f)
        return f

    (io_type["sink"])(
        df.lazy(),
        pl.PartitionBy(
            "",
            file_path_provider=file_path_provider,
            key=col1,
        ),
        # We need to sync here because platforms do not guarantee that a close on
        # one thread is immediately visible on another thread.
        #
        # "Multithreaded processes and close()"
        # https://man7.org/linux/man-pages/man2/close.2.html
        sync_on_close="data",
    )

    for f in output_files:
        f.seek(0)

    assert_frame_equal(
        io_type["scan"](output_files).collect(),
        df,
        check_row_order=False,
    )


def test_partition_by_file_naming_preserves_order(tmp_path: Path) -> None:
    df = pl.DataFrame({"x": range(100)})
    df.lazy().sink_parquet(pl.PartitionBy(tmp_path, max_rows_per_file=1))

    output_files = sorted(tmp_path.iterdir())
    assert len(output_files) == 100

    assert_frame_equal(pl.scan_parquet(output_files).collect(), df)


@pytest.mark.parametrize(("io_type"), io_types)
@pytest.mark.parametrize("engine", engines)
def test_partition_to_memory(io_type: IOType, engine: EngineType) -> None:
    df = pl.DataFrame(
        {
            "a": [5, 10, 1996],
        }
    )

    output_files = {}

    def file_path_provider(args: FileProviderArgs) -> io.BytesIO:
        f = io.BytesIO()
        output_files[args.index_in_partition] = f
        return f

    io_type["sink"](
        df.lazy(),
        pl.PartitionBy("", file_path_provider=file_path_provider, max_rows_per_file=1),
        engine=engine,
    )

    assert len(output_files) == df.height

    for f in output_files.values():
        f.seek(0)

    assert_frame_equal(
        io_type["scan"](output_files[0]).collect(), pl.DataFrame({"a": [5]})
    )
    assert_frame_equal(
        io_type["scan"](output_files[1]).collect(), pl.DataFrame({"a": [10]})
    )
    assert_frame_equal(
        io_type["scan"](output_files[2]).collect(), pl.DataFrame({"a": [1996]})
    )


@pytest.mark.write_disk
def test_partition_key_order_22645(tmp_path: Path) -> None:
    pl.LazyFrame({"a": [1]}).sink_parquet(
        pl.PartitionBy(
            tmp_path,
            key=[pl.col.a.alias("b"), (pl.col.a + 42).alias("c")],
        ),
    )

    assert_frame_equal(
        pl.scan_parquet(tmp_path / "b=1" / "c=43").collect(),
        pl.DataFrame({"a": [1], "b": [1], "c": [43]}),
    )


@pytest.mark.write_disk
def test_parquet_preserve_order_within_partition_23376(tmp_path: Path) -> None:
    ll = list(range(20))
    df = pl.DataFrame({"a": ll})
    df.lazy().sink_parquet(pl.PartitionBy(tmp_path, max_rows_per_file=1))
    out = pl.scan_parquet(tmp_path).collect().to_series().to_list()
    assert ll == out


@pytest.mark.write_disk
def test_file_path_cb_new_cloud_path(tmp_path: Path) -> None:
    i = 0

    def new_path(_: Any) -> str:
        nonlocal i
        p = format_file_uri(f"{tmp_path}/pms-{i:08}.parquet")
        i += 1
        return p

    df = pl.DataFrame({"a": [1, 2]})
    df.lazy().sink_csv(
        pl.PartitionBy(
            "s3://bucket-x", file_path_provider=new_path, max_rows_per_file=1
        )
    )

    assert_frame_equal(pl.scan_csv(tmp_path).collect(), df, check_row_order=False)


@pytest.mark.write_disk
def test_partition_empty_string_24545(tmp_path: Path) -> None:
    df = pl.DataFrame(
        {
            "a": ["", None, "abc", "xyz"],
            "b": [1, 2, 3, 4],
        }
    )

    df.write_parquet(tmp_path, partition_by="a")

    assert_frame_equal(pl.read_parquet(tmp_path), df)


@pytest.mark.write_disk
@pytest.mark.parametrize("dtype", [pl.Int64(), pl.Date(), pl.Datetime()])
def test_partition_empty_dtype_24545(tmp_path: Path, dtype: pl.DataType) -> None:
    df = pl.DataFrame({"b": [1, 2, 3, 4]}).with_columns(
        a=pl.col.b.cast(dtype),
    )

    df.write_parquet(tmp_path, partition_by="a")
    extra = pl.select(b=pl.lit(0, pl.Int64), a=pl.lit(None, dtype))
    extra.write_parquet(Path(tmp_path / "a=" / "000.parquet"), mkdir=True)

    assert_frame_equal(pl.read_parquet(tmp_path), pl.concat([extra, df]))


@pytest.mark.slow
@pytest.mark.write_disk
def test_partition_approximate_size(tmp_path: Path) -> None:
    n_rows = 500_000
    df = pl.select(a=pl.repeat(0, n_rows), b=pl.int_range(0, n_rows))

    root = tmp_path
    df.lazy().sink_parquet(
        pl.PartitionBy(root, approximate_bytes_per_file=200000),
        row_group_size=10_000,
    )

    files = sorted(root.iterdir())

    assert len(files) == 30

    assert [
        pl.scan_parquet(x).select(pl.len()).collect().item() for x in files
    ] == 29 * [16667] + [16657]

    assert_frame_equal(pl.scan_parquet(root).collect(), df)


def test_sink_partitioned_forbid_non_elementwise_key_expr_25535() -> None:
    with pytest.raises(
        InvalidOperationError,
        match="cannot use non-elementwise expressions for PartitionBy keys",
    ):
        pl.LazyFrame({"a": 1}).sink_parquet(pl.PartitionBy("", key=pl.col("a").sum()))


@pytest.mark.write_disk
@pytest.mark.parametrize(
    ("scan_func", "sink_func"),
    [
        (pl.scan_parquet, pl.LazyFrame.sink_parquet),
        (pl.scan_ipc, pl.LazyFrame.sink_ipc),
    ],
)
def test_sink_partitioned_no_columns_in_file_25535(
    tmp_path: Path, scan_func: Any, sink_func: Any
) -> None:
    df = pl.DataFrame({"x": [1, 1, 1, 1, 1]})
    partitioned_root = tmp_path / "partitioned"
    sink_func(
        df.lazy(),
        pl.PartitionBy(partitioned_root, key="x", include_key=False),
    )

    assert_frame_equal(scan_func(partitioned_root).collect(), df)

    max_size_root = tmp_path / "max-size"
    sink_func(
        pl.LazyFrame(height=10),
        pl.PartitionBy(max_size_root, max_rows_per_file=2),
    )

    assert sum(1 for _ in max_size_root.iterdir()) == 5
    assert scan_func(max_size_root).collect().shape == (10, 0)
    assert scan_func(max_size_root).select(pl.len()).collect().item() == 10


def test_partition_by_scalar_expr_26294(tmp_path: Path) -> None:
    pl.LazyFrame(height=5).sink_parquet(
        pl.PartitionBy(tmp_path, key=pl.lit(1, dtype=pl.Int64))
    )

    assert_frame_equal(
        pl.scan_parquet(tmp_path).collect(),
        pl.DataFrame({"literal": [1, 1, 1, 1, 1]}),
    )


def test_partition_by_diff_expr_26370(tmp_path: Path) -> None:
    q = pl.LazyFrame({"x": [1, 2]}).cast(pl.Decimal(precision=1))
    q = q.with_columns(pl.col("x").diff().alias("y"), pl.lit(1).alias("z"))

    q.sink_parquet(pl.PartitionBy(tmp_path, key="z"))

    assert_frame_equal(pl.scan_parquet(tmp_path).collect(), q.collect())
