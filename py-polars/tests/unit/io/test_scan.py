from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from math import ceil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import pytest

import polars as pl
from polars.testing.asserts.frame import assert_frame_equal

if TYPE_CHECKING:
    from polars._typing import SchemaDict


@dataclass
class _RowIndex:
    name: str = "index"
    offset: int = 0


def _enable_force_async(monkeypatch: pytest.MonkeyPatch) -> None:
    """Modifies the provided monkeypatch context."""
    monkeypatch.setenv("POLARS_VERBOSE", "1")
    monkeypatch.setenv("POLARS_FORCE_ASYNC", "1")


def _assert_force_async(capfd: Any, data_file_extension: str) -> None:
    """Calls `capfd.readouterr`, consuming the captured output so far."""
    if data_file_extension == ".ndjson":
        return

    captured = capfd.readouterr().err
    assert captured.count("ASYNC READING FORCED") == 1


def _scan(
    file_path: Path,
    schema: SchemaDict | None = None,
    row_index: _RowIndex | None = None,
) -> pl.LazyFrame:
    suffix = file_path.suffix
    row_index_name = None if row_index is None else row_index.name
    row_index_offset = 0 if row_index is None else row_index.offset

    if (
        scan_func := {
            ".ipc"     : pl.scan_ipc,
            ".parquet" : pl.scan_parquet,
            ".csv"     : pl.scan_csv,
            ".ndjson"  : pl.scan_ndjson,
        }.get(suffix)
    ) is not None:  # fmt: skip
        result = scan_func(
            file_path,
            row_index_name=row_index_name,
            row_index_offset=row_index_offset,
        )  # type: ignore[operator]

    else:
        msg = f"Unknown suffix {suffix}"
        raise NotImplementedError(msg)

    return result  # type: ignore[no-any-return]


def _write(df: pl.DataFrame, file_path: Path) -> None:
    suffix = file_path.suffix

    if (
        write_func := {
            ".ipc"     : pl.DataFrame.write_ipc,
            ".parquet" : pl.DataFrame.write_parquet,
            ".csv"     : pl.DataFrame.write_csv,
            ".ndjson"  : pl.DataFrame.write_ndjson,
        }.get(suffix)
    ) is not None:  # fmt: skip
        return write_func(df, file_path)  # type: ignore[operator, no-any-return]

    msg = f"Unknown suffix {suffix}"
    raise NotImplementedError(msg)


@pytest.fixture(
    scope="session",
    params=["csv", "ipc", "parquet", "ndjson"],
)
def data_file_extension(request: pytest.FixtureRequest) -> str:
    return f".{request.param}"


@pytest.fixture(scope="session")
def session_tmp_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("polars-test")


@pytest.fixture(
    params=[False, True],
    ids=["sync", "async"],
)
def force_async(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> bool:
    value: bool = request.param
    return value


@dataclass
class _DataFile:
    path: Path
    df: pl.DataFrame


def df_with_chunk_size_limit(df: pl.DataFrame, limit: int) -> pl.DataFrame:
    return pl.concat(
        (
            df.slice(i * limit, min(limit, df.height - i * limit))
            for i in range(ceil(df.height / limit))
        ),
        rechunk=False,
    )


@pytest.fixture(scope="session")
def data_file_single(session_tmp_dir: Path, data_file_extension: str) -> _DataFile:
    max_rows_per_batch = 727
    file_path = (session_tmp_dir / "data").with_suffix(data_file_extension)
    df = pl.DataFrame(
        {
            "sequence": range(10000),
        }
    )
    assert max_rows_per_batch < df.height
    _write(df_with_chunk_size_limit(df, max_rows_per_batch), file_path)
    return _DataFile(path=file_path, df=df)


@pytest.fixture(scope="session")
def data_file_glob(session_tmp_dir: Path, data_file_extension: str) -> _DataFile:
    max_rows_per_batch = 200
    row_counts = [
        100, 186, 95, 185, 90, 84, 115, 81, 87, 217, 126, 85, 98, 122, 129, 122, 1089, 82,
        234, 86, 93, 90, 91, 263, 87, 126, 86, 161, 191, 1368, 403, 192, 102, 98, 115, 81,
        111, 305, 92, 534, 431, 150, 90, 128, 152, 118, 127, 124, 229, 368, 81,
    ]  # fmt: skip
    assert sum(row_counts) == 10000

    # Make sure we pad file names with enough zeros to ensure correct
    # lexographical ordering.
    assert len(row_counts) < 100

    # Make sure that some of our data frames consist of multiple chunks which
    # affects the output of certain file formats.
    assert any(row_count > max_rows_per_batch for row_count in row_counts)
    df = pl.DataFrame(
        {
            "sequence": range(10000),
        }
    )

    row_offset = 0
    for index, row_count in enumerate(row_counts):
        file_path = (session_tmp_dir / f"data_{index:02}").with_suffix(
            data_file_extension
        )
        _write(
            df_with_chunk_size_limit(
                df.slice(row_offset, row_count), max_rows_per_batch
            ),
            file_path,
        )
        row_offset += row_count
    return _DataFile(
        path=(session_tmp_dir / "data_*").with_suffix(data_file_extension), df=df
    )


@pytest.fixture(scope="session", params=["single", "glob"])
def data_file(
    request: pytest.FixtureRequest,
    data_file_single: _DataFile,
    data_file_glob: _DataFile,
) -> _DataFile:
    if request.param == "single":
        return data_file_single
    if request.param == "glob":
        return data_file_glob
    raise NotImplementedError()


@pytest.mark.write_disk()
def test_scan(
    capfd: Any, monkeypatch: pytest.MonkeyPatch, data_file: _DataFile, force_async: bool
) -> None:
    if force_async:
        _enable_force_async(monkeypatch)

    df = _scan(data_file.path, data_file.df.schema).collect()

    if force_async:
        _assert_force_async(capfd, data_file.path.suffix)

    assert_frame_equal(df, data_file.df)


@pytest.mark.write_disk()
def test_scan_with_limit(
    capfd: Any, monkeypatch: pytest.MonkeyPatch, data_file: _DataFile, force_async: bool
) -> None:
    if force_async:
        _enable_force_async(monkeypatch)

    df = _scan(data_file.path, data_file.df.schema).limit(4483).collect()

    if force_async:
        _assert_force_async(capfd, data_file.path.suffix)

    assert_frame_equal(
        df,
        pl.DataFrame(
            {
                "sequence": range(4483),
            }
        ),
    )


@pytest.mark.write_disk()
def test_scan_with_filter(
    capfd: Any, monkeypatch: pytest.MonkeyPatch, data_file: _DataFile, force_async: bool
) -> None:
    if force_async:
        _enable_force_async(monkeypatch)

    df = (
        _scan(data_file.path, data_file.df.schema)
        .filter(pl.col("sequence") % 2 == 0)
        .collect()
    )

    if force_async:
        _assert_force_async(capfd, data_file.path.suffix)

    assert_frame_equal(
        df,
        pl.DataFrame(
            {
                "sequence": (2 * x for x in range(5000)),
            }
        ),
    )


@pytest.mark.write_disk()
def test_scan_with_filter_and_limit(
    capfd: Any, monkeypatch: pytest.MonkeyPatch, data_file: _DataFile, force_async: bool
) -> None:
    if force_async:
        _enable_force_async(monkeypatch)

    df = (
        _scan(data_file.path, data_file.df.schema)
        .filter(pl.col("sequence") % 2 == 0)
        .limit(4483)
        .collect()
    )

    if force_async:
        _assert_force_async(capfd, data_file.path.suffix)

    assert_frame_equal(
        df,
        pl.DataFrame(
            {
                "sequence": (2 * x for x in range(4483)),
            },
        ),
    )


@pytest.mark.write_disk()
def test_scan_with_limit_and_filter(
    capfd: Any, monkeypatch: pytest.MonkeyPatch, data_file: _DataFile, force_async: bool
) -> None:
    if force_async:
        _enable_force_async(monkeypatch)

    df = (
        _scan(data_file.path, data_file.df.schema)
        .limit(4483)
        .filter(pl.col("sequence") % 2 == 0)
        .collect()
    )

    if force_async:
        _assert_force_async(capfd, data_file.path.suffix)

    assert_frame_equal(
        df,
        pl.DataFrame(
            {
                "sequence": (2 * x for x in range(2242)),
            },
        ),
    )


@pytest.mark.write_disk()
def test_scan_with_row_index_and_limit(
    capfd: Any, monkeypatch: pytest.MonkeyPatch, data_file: _DataFile, force_async: bool
) -> None:
    if force_async:
        _enable_force_async(monkeypatch)

    df = (
        _scan(data_file.path, data_file.df.schema, row_index=_RowIndex())
        .limit(4483)
        .collect()
    )

    if force_async:
        _assert_force_async(capfd, data_file.path.suffix)

    assert_frame_equal(
        df,
        pl.DataFrame(
            {
                "index": range(4483),
                "sequence": range(4483),
            },
            schema_overrides={"index": pl.UInt32},
        ),
    )


@pytest.mark.write_disk()
def test_scan_with_row_index_and_filter(
    capfd: Any, monkeypatch: pytest.MonkeyPatch, data_file: _DataFile, force_async: bool
) -> None:
    if force_async:
        _enable_force_async(monkeypatch)

    df = (
        _scan(data_file.path, data_file.df.schema, row_index=_RowIndex())
        .filter(pl.col("sequence") % 2 == 0)
        .collect()
    )

    if force_async:
        _assert_force_async(capfd, data_file.path.suffix)

    assert_frame_equal(
        df,
        pl.DataFrame(
            {
                "index": (2 * x for x in range(5000)),
                "sequence": (2 * x for x in range(5000)),
            },
            schema_overrides={"index": pl.UInt32},
        ),
    )


@pytest.mark.write_disk()
def test_scan_with_row_index_limit_and_filter(
    capfd: Any, monkeypatch: pytest.MonkeyPatch, data_file: _DataFile, force_async: bool
) -> None:
    if force_async:
        _enable_force_async(monkeypatch)

    df = (
        _scan(data_file.path, data_file.df.schema, row_index=_RowIndex())
        .limit(4483)
        .filter(pl.col("sequence") % 2 == 0)
        .collect()
    )

    if force_async:
        _assert_force_async(capfd, data_file.path.suffix)

    assert_frame_equal(
        df,
        pl.DataFrame(
            {
                "index": (2 * x for x in range(2242)),
                "sequence": (2 * x for x in range(2242)),
            },
            schema_overrides={"index": pl.UInt32},
        ),
    )


@pytest.mark.write_disk()
def test_scan_with_row_index_projected_out(
    capfd: Any, monkeypatch: pytest.MonkeyPatch, data_file: _DataFile, force_async: bool
) -> None:
    if data_file.path.suffix == ".csv" and force_async:
        pytest.skip(reason="async reading of .csv not yet implemented")

    if force_async:
        _enable_force_async(monkeypatch)

    subset = next(iter(data_file.df.schema.keys()))
    df = (
        _scan(data_file.path, data_file.df.schema, row_index=_RowIndex())
        .select(subset)
        .collect()
    )

    if force_async:
        _assert_force_async(capfd, data_file.path.suffix)

    assert_frame_equal(df, data_file.df.select(subset))


@pytest.mark.write_disk()
def test_scan_with_row_index_filter_and_limit(
    capfd: Any, monkeypatch: pytest.MonkeyPatch, data_file: _DataFile, force_async: bool
) -> None:
    if data_file.path.suffix == ".csv" and force_async:
        pytest.skip(reason="async reading of .csv not yet implemented")

    if force_async:
        _enable_force_async(monkeypatch)

    df = (
        _scan(data_file.path, data_file.df.schema, row_index=_RowIndex())
        .filter(pl.col("sequence") % 2 == 0)
        .limit(4483)
        .collect()
    )

    if force_async:
        _assert_force_async(capfd, data_file.path.suffix)

    assert_frame_equal(
        df,
        pl.DataFrame(
            {
                "index": (2 * x for x in range(4483)),
                "sequence": (2 * x for x in range(4483)),
            },
            schema_overrides={"index": pl.UInt32},
        ),
    )


@pytest.mark.write_disk()
@pytest.mark.parametrize(
    ("scan_func", "write_func"),
    [
        (pl.scan_parquet, pl.DataFrame.write_parquet),
        (pl.scan_ipc, pl.DataFrame.write_ipc),
        (pl.scan_csv, pl.DataFrame.write_csv),
        (pl.scan_ndjson, pl.DataFrame.write_ndjson),
    ],
)
@pytest.mark.parametrize(
    "streaming",
    [True, False],
)
def test_scan_limit_0_does_not_panic(
    tmp_path: Path,
    scan_func: Callable[[Any], pl.LazyFrame],
    write_func: Callable[[pl.DataFrame, Path], None],
    streaming: bool,
) -> None:
    tmp_path.mkdir(exist_ok=True)
    path = tmp_path / "data.bin"
    df = pl.DataFrame({"x": 1})
    write_func(df, path)
    assert_frame_equal(scan_func(path).head(0).collect(streaming=streaming), df.clear())


@pytest.mark.write_disk()
@pytest.mark.parametrize(
    ("scan_func", "write_func"),
    [
        (pl.scan_csv, pl.DataFrame.write_csv),
        (pl.scan_parquet, pl.DataFrame.write_parquet),
        (pl.scan_ipc, pl.DataFrame.write_ipc),
        (pl.scan_ndjson, pl.DataFrame.write_ndjson),
    ],
)
@pytest.mark.parametrize(
    "glob",
    [True, False],
)
def test_scan_directory(
    tmp_path: Path,
    scan_func: Callable[..., pl.LazyFrame],
    write_func: Callable[[pl.DataFrame, Path], None],
    glob: bool,
) -> None:
    tmp_path.mkdir(exist_ok=True)

    dfs: list[pl.DataFrame] = [
        pl.DataFrame({"a": [0, 0, 0, 0, 0]}),
        pl.DataFrame({"a": [1, 1, 1, 1, 1]}),
        pl.DataFrame({"a": [2, 2, 2, 2, 2]}),
    ]

    paths = [
        tmp_path / "0.bin",
        tmp_path / "1.bin",
        tmp_path / "dir/data.bin",
    ]

    for df, path in zip(dfs, paths):
        path.parent.mkdir(exist_ok=True)
        write_func(df, path)

    df = pl.concat(dfs)

    scan = scan_func

    if scan_func in [pl.scan_csv, pl.scan_ndjson]:
        scan = partial(scan, schema=df.schema)

    if scan_func is pl.scan_parquet:
        scan = partial(scan, glob=glob)

    out = scan(tmp_path).collect()
    assert_frame_equal(out, df)


@pytest.mark.write_disk()
def test_scan_glob_excludes_directories(tmp_path: Path) -> None:
    for dir in ["dir1", "dir2", "dir3"]:
        (tmp_path / dir).mkdir()

    df = pl.DataFrame({"a": [1, 2, 3]})

    df.write_parquet(tmp_path / "dir1/data.bin")
    df.write_parquet(tmp_path / "dir2/data.parquet")
    df.write_parquet(tmp_path / "data.parquet")

    assert_frame_equal(pl.scan_parquet(tmp_path / "**/*.bin").collect(), df)
    assert_frame_equal(pl.scan_parquet(tmp_path / "**/data*.bin").collect(), df)
    assert_frame_equal(
        pl.scan_parquet(tmp_path / "**/*").collect(), pl.concat(3 * [df])
    )
    assert_frame_equal(pl.scan_parquet(tmp_path / "*").collect(), df)


@pytest.mark.parametrize("file_name", ["a b", "a %25 b"])
@pytest.mark.write_disk()
def test_scan_async_whitespace_in_path(
    tmp_path: Path, monkeypatch: Any, file_name: str
) -> None:
    monkeypatch.setenv("POLARS_FORCE_ASYNC", "1")
    tmp_path.mkdir(exist_ok=True)

    path = tmp_path / f"{file_name}.parquet"
    df = pl.DataFrame({"x": 1})
    df.write_parquet(path)
    assert_frame_equal(pl.scan_parquet(path).collect(), df)
    assert_frame_equal(pl.scan_parquet(tmp_path).collect(), df)
    assert_frame_equal(pl.scan_parquet(tmp_path / "*").collect(), df)
    assert_frame_equal(pl.scan_parquet(tmp_path / "*.parquet").collect(), df)
    path.unlink()


@pytest.mark.write_disk()
def test_path_expansion_excludes_empty_files_17362(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)

    df = pl.DataFrame({"x": 1})
    df.write_parquet(tmp_path / "data.parquet")
    (tmp_path / "empty").touch()

    assert_frame_equal(pl.scan_parquet(tmp_path).collect(), df)
    assert_frame_equal(pl.scan_parquet(tmp_path / "*").collect(), df)


@pytest.mark.write_disk()
def test_scan_single_dir_differing_file_extensions_raises_17436(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)

    df = pl.DataFrame({"x": 1})
    df.write_parquet(tmp_path / "data.parquet")
    df.write_ipc(tmp_path / "data.ipc")

    with pytest.raises(
        pl.exceptions.InvalidOperationError, match="different file extensions"
    ):
        pl.scan_parquet(tmp_path).collect()

    for lf in [
        pl.scan_parquet(tmp_path / "*.parquet"),
        pl.scan_ipc(tmp_path / "*.ipc"),
    ]:
        assert_frame_equal(lf.collect(), df)

    # Ensure passing a glob doesn't trigger file extension checking
    with pytest.raises(
        pl.exceptions.ComputeError,
        match="parquet: File out of specification: The file must end with PAR1",
    ):
        pl.scan_parquet(tmp_path / "*").collect()


@pytest.mark.parametrize("format", ["parquet", "csv", "ndjson", "ipc"])
def test_scan_nonexistent_path(format: str) -> None:
    path_str = f"my-nonexistent-data.{format}"
    path = Path(path_str)
    assert not path.exists()

    scan_function = getattr(pl, f"scan_{format}")

    # Just calling the scan function should not raise any errors
    result = scan_function(path)
    assert isinstance(result, pl.LazyFrame)

    # Upon collection, it should fail
    with pytest.raises(FileNotFoundError):
        result.collect()


@pytest.mark.write_disk()
@pytest.mark.parametrize(
    ("scan_func", "write_func"),
    [
        (pl.scan_parquet, pl.DataFrame.write_parquet),
        (pl.scan_ipc, pl.DataFrame.write_ipc),
        (pl.scan_csv, pl.DataFrame.write_csv),
        (pl.scan_ndjson, pl.DataFrame.write_ndjson),
    ],
)
@pytest.mark.parametrize(
    "streaming",
    [True, False],
)
def test_scan_include_file_name(
    tmp_path: Path,
    scan_func: Callable[..., pl.LazyFrame],
    write_func: Callable[[pl.DataFrame, Path], None],
    streaming: bool,
) -> None:
    tmp_path.mkdir(exist_ok=True)
    dfs: list[pl.DataFrame] = []

    for x in ["1", "2"]:
        path = Path(f"{tmp_path}/{x}.bin").absolute()
        dfs.append(pl.DataFrame({"x": 10 * [x]}).with_columns(path=pl.lit(str(path))))
        write_func(dfs[-1].drop("path"), path)

    df = pl.concat(dfs)
    assert df.columns == ["x", "path"]

    with pytest.raises(
        pl.exceptions.DuplicateError,
        match=r'column name for file paths "x" conflicts with column name from file',
    ):
        scan_func(tmp_path, include_file_paths="x").collect(streaming=streaming)

    f = scan_func
    if scan_func in [pl.scan_csv, pl.scan_ndjson]:
        f = partial(f, schema=df.drop("path").schema)

    lf: pl.LazyFrame = f(tmp_path, include_file_paths="path")
    assert_frame_equal(lf.collect(streaming=streaming), df)

    # TODO: Support this with CSV
    if scan_func not in [pl.scan_csv, pl.scan_ndjson]:
        # Test projecting only the path column
        assert_frame_equal(
            lf.select("path").collect(streaming=streaming),
            df.select("path"),
        )

    # Test predicates
    for predicate in [pl.col("path") != pl.col("x"), pl.col("path") != ""]:
        assert_frame_equal(
            lf.filter(predicate).collect(streaming=streaming),
            df,
        )

    # Test codepaths that materialize empty DataFrames
    assert_frame_equal(lf.head(0).collect(streaming=streaming), df.head(0))


@pytest.mark.write_disk()
def test_async_path_expansion_bracket_17629(tmp_path: Path) -> None:
    path = tmp_path / "data.parquet"

    df = pl.DataFrame({"x": 1})
    df.write_parquet(path)

    assert_frame_equal(pl.scan_parquet(tmp_path / "[d]ata.parquet").collect(), df)
