from __future__ import annotations

import io
from typing import TYPE_CHECKING, Any, TypedDict

import pytest
from hypothesis import given

import polars as pl
from polars.io.partition import (
    PartitionByKey,
    PartitionMaxSize,
    PartitionParted,
)
from polars.testing import assert_frame_equal, assert_series_equal
from polars.testing.parametric.strategies import dataframes

if TYPE_CHECKING:
    from pathlib import Path

    from polars.io.partition import BasePartitionContext, KeyedPartitionContext


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


@pytest.mark.parametrize("io_type", io_types)
@pytest.mark.parametrize("length", [0, 1, 4, 5, 6, 7])
@pytest.mark.parametrize("max_size", [1, 2, 3])
@pytest.mark.write_disk
def test_max_size_partition(
    tmp_path: Path,
    io_type: IOType,
    length: int,
    max_size: int,
) -> None:
    lf = pl.Series("a", range(length), pl.Int64).to_frame().lazy()

    (io_type["sink"])(
        lf,
        PartitionMaxSize(tmp_path, max_size=max_size),
        engine="streaming",
        # We need to sync here because platforms do not guarantee that a close on
        # one thread is immediately visible on another thread.
        #
        # "Multithreaded processes and close()"
        # https://man7.org/linux/man-pages/man2/close.2.html
        sync_on_close="data",
    )

    i = 0
    while length > 0:
        assert (io_type["scan"])(tmp_path / f"{i}.{io_type['ext']}").select(
            pl.len()
        ).collect()[0, 0] == min(max_size, length)

        length -= max_size
        i += 1


@pytest.mark.parametrize("io_type", io_types)
def test_max_size_partition_lambda(
    tmp_path: Path,
    io_type: IOType,
) -> None:
    length = 17
    max_size = 3
    lf = pl.Series("a", range(length), pl.Int64).to_frame().lazy()

    (io_type["sink"])(
        lf,
        PartitionMaxSize(
            tmp_path,
            file_path=lambda ctx: ctx.file_path.with_name("abc-" + ctx.file_path.name),
            max_size=max_size,
        ),
        engine="streaming",
        # We need to sync here because platforms do not guarantee that a close on
        # one thread is immediately visible on another thread.
        #
        # "Multithreaded processes and close()"
        # https://man7.org/linux/man-pages/man2/close.2.html
        sync_on_close="data",
    )

    i = 0
    while length > 0:
        assert (io_type["scan"])(tmp_path / f"abc-{i}.{io_type['ext']}").select(
            pl.len()
        ).collect()[0, 0] == min(max_size, length)

        length -= max_size
        i += 1


@pytest.mark.parametrize("io_type", io_types)
@pytest.mark.write_disk
def test_partition_by_key(
    tmp_path: Path,
    io_type: IOType,
) -> None:
    lf = pl.Series("a", [i % 4 for i in range(7)], pl.Int64).to_frame().lazy()

    (io_type["sink"])(
        lf,
        PartitionByKey(
            tmp_path, file_path=lambda ctx: f"{ctx.file_idx}.{io_type['ext']}", by="a"
        ),
        engine="streaming",
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
        PartitionByKey(
            tmp_path,
            file_path=lambda ctx: f"{ctx.file_idx}.{io_type['ext']}",
            by=pl.col.a.cast(pl.String()),
        ),
        engine="streaming",
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


@pytest.mark.parametrize("io_type", io_types)
@pytest.mark.write_disk
def test_partition_parted(
    tmp_path: Path,
    io_type: IOType,
) -> None:
    s = pl.Series("a", [1, 1, 2, 3, 3, 4, 4, 4, 6], pl.Int64)
    lf = s.to_frame().lazy()

    (io_type["sink"])(
        lf,
        PartitionParted(
            tmp_path, file_path=lambda ctx: f"{ctx.file_idx}.{io_type['ext']}", by="a"
        ),
        engine="streaming",
        # We need to sync here because platforms do not guarantee that a close on
        # one thread is immediately visible on another thread.
        #
        # "Multithreaded processes and close()"
        # https://man7.org/linux/man-pages/man2/close.2.html
        sync_on_close="data",
    )

    rle = s.rle()

    for i, row in enumerate(rle.struct.unnest().rows(named=True)):
        assert_series_equal(
            (io_type["scan"])(tmp_path / f"{i}.{io_type['ext']}").collect().to_series(),
            pl.Series("a", [row["value"]] * row["len"], pl.Int64),
        )

    scan_flags = (
        {"schema_overrides": pl.Schema({"a_str": pl.String()})}
        if io_type["ext"] == "csv"
        else {}
    )

    # Change the datatype.
    (io_type["sink"])(
        lf,
        PartitionParted(
            tmp_path,
            file_path=lambda ctx: f"{ctx.file_idx}.{io_type['ext']}",
            by=[pl.col.a, pl.col.a.cast(pl.String()).alias("a_str")],
        ),
        engine="streaming",
        sync_on_close="data",
    )

    for i, row in enumerate(rle.struct.unnest().rows(named=True)):
        assert_frame_equal(
            (io_type["scan"])(
                tmp_path / f"{i}.{io_type['ext']}", **scan_flags
            ).collect(),
            pl.DataFrame(
                [
                    pl.Series("a", [row["value"]] * row["len"], pl.Int64),
                    pl.Series("a_str", [str(row["value"])] * row["len"], pl.String),
                ]
            ),
        )

    # No include key.
    (io_type["sink"])(
        lf,
        PartitionParted(
            tmp_path,
            file_path=lambda ctx: f"{ctx.file_idx}.{io_type['ext']}",
            by=[pl.col.a.cast(pl.String()).alias("a_str")],
            include_key=False,
        ),
        engine="streaming",
        sync_on_close="data",
    )

    for i, row in enumerate(rle.struct.unnest().rows(named=True)):
        assert_series_equal(
            (io_type["scan"])(tmp_path / f"{i}.{io_type['ext']}").collect().to_series(),
            pl.Series("a", [row["value"]] * row["len"], pl.Int64),
        )


# We only deal with self-describing formats
@pytest.mark.parametrize("io_type", [io_types[2], io_types[3]])
@pytest.mark.write_disk
@given(
    df=dataframes(
        min_cols=1,
        excluded_dtypes=[
            pl.Decimal,  # Bug see: https://github.com/pola-rs/polars/issues/21684
            pl.Duration,  # Bug see: https://github.com/pola-rs/polars/issues/21964
            pl.Categorical,  # We cannot ensure the string cache is properly held.
            # Generate invalid UTF-8
            pl.Binary,
            pl.Struct,
            pl.Array,
            pl.List,
        ],
    )
)
def test_partition_by_key_parametric(
    tmp_path_factory: pytest.TempPathFactory,
    io_type: IOType,
    df: pl.DataFrame,
) -> None:
    col1 = df.columns[0]

    tmp_path = tmp_path_factory.mktemp("data")

    dfs = df.partition_by(col1)
    (io_type["sink"])(
        df.lazy(),
        PartitionByKey(
            tmp_path, file_path=lambda ctx: f"{ctx.file_idx}.{io_type['ext']}", by=col1
        ),
        engine="streaming",
        # We need to sync here because platforms do not guarantee that a close on
        # one thread is immediately visible on another thread.
        #
        # "Multithreaded processes and close()"
        # https://man7.org/linux/man-pages/man2/close.2.html
        sync_on_close="data",
    )

    for i, df in enumerate(dfs):
        assert_frame_equal(
            df,
            (io_type["scan"])(
                tmp_path / f"{i}.{io_type['ext']}",
            ).collect(),
        )


def test_max_size_partition_collect_files(tmp_path: Path) -> None:
    length = 17
    max_size = 3
    lf = pl.Series("a", range(length), pl.Int64).to_frame().lazy()

    io_type = io_types[0]
    output_files = []

    def file_path_cb(ctx: BasePartitionContext) -> Path:
        print(ctx)
        print(ctx.full_path)
        output_files.append(ctx.full_path)
        print(ctx.file_path)
        return ctx.file_path

    (io_type["sink"])(
        lf,
        PartitionMaxSize(tmp_path, file_path=file_path_cb, max_size=max_size),
        engine="streaming",
        # We need to sync here because platforms do not guarantee that a close on
        # one thread is immediately visible on another thread.
        #
        # "Multithreaded processes and close()"
        # https://man7.org/linux/man-pages/man2/close.2.html
        sync_on_close="data",
    )

    assert output_files == [tmp_path / f"{i}.{io_type['ext']}" for i in range(6)]


@pytest.mark.parametrize(("io_type"), io_types)
def test_partition_to_memory(tmp_path: Path, io_type: IOType) -> None:
    df = pl.DataFrame(
        {
            "a": [5, 10, 1996],
        }
    )

    output_files = {}

    def file_path_cb(ctx: BasePartitionContext) -> io.BytesIO:
        f = io.BytesIO()
        output_files[ctx.file_path] = f
        return f

    io_type["sink"](df.lazy(), PartitionMaxSize("", file_path=file_path_cb, max_size=1))

    assert len(output_files) == df.height
    for i, (_, value) in enumerate(output_files.items()):
        value.seek(0)
        assert_frame_equal(io_type["scan"](value).collect(), df.slice(i, 1))


def test_partition_key_order_22645() -> None:
    paths = []

    def cb(ctx: KeyedPartitionContext) -> io.BytesIO:
        paths.append(ctx.file_path.parent)
        return io.BytesIO()  # return an dummy output

    pl.LazyFrame({"a": [1, 2, 3]}).sink_parquet(
        pl.io.PartitionByKey(
            "",
            file_path=cb,
            by=[pl.col.a.alias("b"), (pl.col.a + 42).alias("c")],
        ),
    )

    paths.sort()
    assert [p.parts for p in paths] == [
        ("b=1", "c=43"),
        ("b=2", "c=44"),
        ("b=3", "c=45"),
    ]
