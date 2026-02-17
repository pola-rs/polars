from __future__ import annotations

import io
import sys
import zlib
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from math import ceil
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

import polars as pl
from polars.exceptions import ComputeError
from polars.testing.asserts.frame import assert_frame_equal
from tests.unit.io.conftest import format_file_uri, normalize_path_separator_pl

if TYPE_CHECKING:
    from collections.abc import Callable

    from polars._typing import SchemaDict
    from tests.conftest import PlMonkeyPatch


@dataclass
class _RowIndex:
    name: str = "index"
    offset: int = 0


def _enable_force_async(plmonkeypatch: PlMonkeyPatch) -> None:
    """Modifies the provided plmonkeypatch context."""
    plmonkeypatch.setenv("POLARS_VERBOSE", "1")
    plmonkeypatch.setenv("POLARS_FORCE_ASYNC", "1")


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
            ".ipc": pl.scan_ipc,
            ".parquet": pl.scan_parquet,
            ".csv": pl.scan_csv,
            ".ndjson": pl.scan_ndjson,
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
            ".ipc": pl.DataFrame.write_ipc,
            ".parquet": pl.DataFrame.write_parquet,
            ".csv": pl.DataFrame.write_csv,
            ".ndjson": pl.DataFrame.write_ndjson,
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
def force_async(request: pytest.FixtureRequest, plmonkeypatch: PlMonkeyPatch) -> bool:
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
    # lexicographical ordering.
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


@pytest.mark.write_disk
def test_scan(
    capfd: Any, plmonkeypatch: PlMonkeyPatch, data_file: _DataFile, force_async: bool
) -> None:
    if force_async:
        _enable_force_async(plmonkeypatch)

    df = _scan(data_file.path, data_file.df.schema).collect()

    assert_frame_equal(df, data_file.df)


@pytest.mark.write_disk
def test_scan_with_limit(
    capfd: Any, plmonkeypatch: PlMonkeyPatch, data_file: _DataFile, force_async: bool
) -> None:
    if force_async:
        _enable_force_async(plmonkeypatch)

    df = _scan(data_file.path, data_file.df.schema).limit(4483).collect()

    assert_frame_equal(
        df,
        pl.DataFrame(
            {
                "sequence": range(4483),
            }
        ),
    )


@pytest.mark.write_disk
def test_scan_with_filter(
    capfd: Any, plmonkeypatch: PlMonkeyPatch, data_file: _DataFile, force_async: bool
) -> None:
    if force_async:
        _enable_force_async(plmonkeypatch)

    df = (
        _scan(data_file.path, data_file.df.schema)
        .filter(pl.col("sequence") % 2 == 0)
        .collect()
    )

    assert_frame_equal(
        df,
        pl.DataFrame(
            {
                "sequence": (2 * x for x in range(5000)),
            }
        ),
    )


@pytest.mark.write_disk
def test_scan_with_filter_and_limit(
    capfd: Any, plmonkeypatch: PlMonkeyPatch, data_file: _DataFile, force_async: bool
) -> None:
    if force_async:
        _enable_force_async(plmonkeypatch)

    df = (
        _scan(data_file.path, data_file.df.schema)
        .filter(pl.col("sequence") % 2 == 0)
        .limit(4483)
        .collect()
    )

    assert_frame_equal(
        df,
        pl.DataFrame(
            {
                "sequence": (2 * x for x in range(4483)),
            },
        ),
    )


@pytest.mark.write_disk
def test_scan_with_limit_and_filter(
    capfd: Any, plmonkeypatch: PlMonkeyPatch, data_file: _DataFile, force_async: bool
) -> None:
    if force_async:
        _enable_force_async(plmonkeypatch)

    df = (
        _scan(data_file.path, data_file.df.schema)
        .limit(4483)
        .filter(pl.col("sequence") % 2 == 0)
        .collect()
    )

    assert_frame_equal(
        df,
        pl.DataFrame(
            {
                "sequence": (2 * x for x in range(2242)),
            },
        ),
    )


@pytest.mark.write_disk
def test_scan_with_row_index_and_limit(
    capfd: Any, plmonkeypatch: PlMonkeyPatch, data_file: _DataFile, force_async: bool
) -> None:
    if force_async:
        _enable_force_async(plmonkeypatch)

    df = (
        _scan(data_file.path, data_file.df.schema, row_index=_RowIndex())
        .limit(4483)
        .collect()
    )

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


@pytest.mark.write_disk
def test_scan_with_row_index_and_filter(
    capfd: Any, plmonkeypatch: PlMonkeyPatch, data_file: _DataFile, force_async: bool
) -> None:
    if force_async:
        _enable_force_async(plmonkeypatch)

    df = (
        _scan(data_file.path, data_file.df.schema, row_index=_RowIndex())
        .filter(pl.col("sequence") % 2 == 0)
        .collect()
    )

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


@pytest.mark.write_disk
def test_scan_with_row_index_limit_and_filter(
    capfd: Any, plmonkeypatch: PlMonkeyPatch, data_file: _DataFile, force_async: bool
) -> None:
    if force_async:
        _enable_force_async(plmonkeypatch)

    df = (
        _scan(data_file.path, data_file.df.schema, row_index=_RowIndex())
        .limit(4483)
        .filter(pl.col("sequence") % 2 == 0)
        .collect()
    )

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


@pytest.mark.write_disk
def test_scan_with_row_index_projected_out(
    capfd: Any, plmonkeypatch: PlMonkeyPatch, data_file: _DataFile, force_async: bool
) -> None:
    if data_file.path.suffix == ".csv" and force_async:
        pytest.skip(reason="async reading of .csv not yet implemented")

    if force_async:
        _enable_force_async(plmonkeypatch)

    subset = next(iter(data_file.df.schema.keys()))
    df = (
        _scan(data_file.path, data_file.df.schema, row_index=_RowIndex())
        .select(subset)
        .collect()
    )

    assert_frame_equal(df, data_file.df.select(subset))


@pytest.mark.write_disk
def test_scan_with_row_index_filter_and_limit(
    capfd: Any, plmonkeypatch: PlMonkeyPatch, data_file: _DataFile, force_async: bool
) -> None:
    if data_file.path.suffix == ".csv" and force_async:
        pytest.skip(reason="async reading of .csv not yet implemented")

    if force_async:
        _enable_force_async(plmonkeypatch)

    df = (
        _scan(data_file.path, data_file.df.schema, row_index=_RowIndex())
        .filter(pl.col("sequence") % 2 == 0)
        .limit(4483)
        .collect()
    )

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


@pytest.mark.write_disk
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
    assert_frame_equal(
        scan_func(path)
        .head(0)
        .collect(engine="streaming" if streaming else "in-memory"),
        df.clear(),
    )


@pytest.mark.write_disk
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

    for df, path in zip(dfs, paths, strict=True):
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


@pytest.mark.write_disk
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
@pytest.mark.write_disk
def test_scan_async_whitespace_in_path(
    tmp_path: Path, plmonkeypatch: PlMonkeyPatch, file_name: str
) -> None:
    plmonkeypatch.setenv("POLARS_FORCE_ASYNC", "1")
    tmp_path.mkdir(exist_ok=True)

    path = tmp_path / f"{file_name}.parquet"
    df = pl.DataFrame({"x": 1})
    df.write_parquet(path)
    assert_frame_equal(pl.scan_parquet(path).collect(), df)
    assert_frame_equal(pl.scan_parquet(tmp_path).collect(), df)
    assert_frame_equal(pl.scan_parquet(tmp_path / "*").collect(), df)
    assert_frame_equal(pl.scan_parquet(tmp_path / "*.parquet").collect(), df)
    path.unlink()


@pytest.mark.write_disk
def test_path_expansion_excludes_empty_files_17362(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)

    df = pl.DataFrame({"x": 1})
    df.write_parquet(tmp_path / "data.parquet")
    (tmp_path / "empty").touch()

    assert_frame_equal(pl.scan_parquet(tmp_path).collect(), df)
    assert_frame_equal(pl.scan_parquet(tmp_path / "*").collect(), df)


@pytest.mark.write_disk
def test_path_expansion_empty_directory_does_not_panic(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)

    with pytest.raises(pl.exceptions.ComputeError):
        pl.scan_parquet(tmp_path).collect()

    with pytest.raises(pl.exceptions.ComputeError):
        pl.scan_parquet(tmp_path / "**/*").collect()


@pytest.mark.write_disk
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


@pytest.mark.write_disk
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
def test_scan_include_file_paths(
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

    df = pl.concat(dfs).with_columns(normalize_path_separator_pl(pl.col("path")))
    assert df.columns == ["x", "path"]

    with pytest.raises(
        pl.exceptions.DuplicateError,
        match=r'column name for file paths "x" conflicts with column name from file',
    ):
        scan_func(tmp_path, include_file_paths="x").collect(
            engine="streaming" if streaming else "in-memory"
        )

    f = scan_func
    if scan_func in [pl.scan_csv, pl.scan_ndjson]:
        f = partial(f, schema=df.drop("path").schema)

    lf: pl.LazyFrame = f(tmp_path, include_file_paths="path")
    assert_frame_equal(lf.collect(engine="streaming" if streaming else "in-memory"), df)

    # Test projecting only the path column
    q = lf.select("path")
    assert q.collect_schema() == {"path": pl.String}
    assert_frame_equal(
        q.collect(engine="streaming" if streaming else "in-memory"),
        df.select("path"),
    )

    q = q.select("path").head(3)
    assert q.collect_schema() == {"path": pl.String}
    assert_frame_equal(
        q.collect(engine="streaming" if streaming else "in-memory"),
        df.select("path").head(3),
    )

    # Test predicates
    for predicate in [pl.col("path") != pl.col("x"), pl.col("path") != ""]:
        assert_frame_equal(
            lf.filter(predicate).collect(
                engine="streaming" if streaming else "in-memory"
            ),
            df,
        )

    # Test codepaths that materialize empty DataFrames
    assert_frame_equal(
        lf.head(0).collect(engine="streaming" if streaming else "in-memory"),
        df.head(0),
    )


@pytest.mark.write_disk
def test_async_path_expansion_bracket_17629(tmp_path: Path) -> None:
    path = tmp_path / "data.parquet"

    df = pl.DataFrame({"x": 1})
    df.write_parquet(path)

    assert_frame_equal(pl.scan_parquet(tmp_path / "[d]ata.parquet").collect(), df)


@pytest.mark.parametrize(
    "method",
    ["parquet", "csv", "ipc", "ndjson"],
)
@pytest.mark.may_fail_auto_streaming  # unsupported negative slice offset -1 for CSV source
def test_scan_in_memory(method: str) -> None:
    f = io.BytesIO()
    df = pl.DataFrame(
        {
            "a": [1, 2, 3],
            "b": ["x", "y", "z"],
        }
    )

    (getattr(df, f"write_{method}"))(f)

    f.seek(0)
    result = (getattr(pl, f"scan_{method}"))(f).collect()
    assert_frame_equal(df, result)

    f.seek(0)
    result = (getattr(pl, f"scan_{method}"))(f).slice(1, 2).collect()
    assert_frame_equal(df.slice(1, 2), result)

    f.seek(0)
    result = (getattr(pl, f"scan_{method}"))(f).slice(-1, 1).collect()
    assert_frame_equal(df.slice(-1, 1), result)

    g = io.BytesIO()
    (getattr(df, f"write_{method}"))(g)

    f.seek(0)
    g.seek(0)
    result = (getattr(pl, f"scan_{method}"))([f, g]).collect()
    assert_frame_equal(df.vstack(df), result)

    f.seek(0)
    g.seek(0)
    result = (getattr(pl, f"scan_{method}"))([f, g]).slice(1, 2).collect()
    assert_frame_equal(df.vstack(df).slice(1, 2), result)

    f.seek(0)
    g.seek(0)
    result = (getattr(pl, f"scan_{method}"))([f, g]).slice(-1, 1).collect()
    assert_frame_equal(df.vstack(df).slice(-1, 1), result)


def test_scan_pyobject_zero_copy_buffer_mutate() -> None:
    f = io.BytesIO()

    df = pl.DataFrame({"x": [1, 2, 3, 4, 5]})
    df.write_ipc(f)
    f.seek(0)

    q = pl.scan_ipc(f)
    assert_frame_equal(q.collect(), df)

    f.write(b"AAA")
    assert_frame_equal(q.collect(), df)


@pytest.mark.parametrize(
    "method",
    ["csv", "ndjson"],
)
def test_scan_stringio(method: str) -> None:
    f = io.StringIO()
    df = pl.DataFrame(
        {
            "a": [1, 2, 3],
            "b": ["x", "y", "z"],
        }
    )

    (getattr(df, f"write_{method}"))(f)

    f.seek(0)
    result = (getattr(pl, f"scan_{method}"))(f).collect()
    assert_frame_equal(df, result)

    g = io.StringIO()
    (getattr(df, f"write_{method}"))(g)

    f.seek(0)
    g.seek(0)
    result = (getattr(pl, f"scan_{method}"))([f, g]).collect()
    assert_frame_equal(df.vstack(df), result)


def test_scan_double_collect_row_index_invalidates_cached_ir_18892() -> None:
    lf = pl.scan_csv(io.BytesIO(b"a\n1\n2\n3"))

    lf.collect()

    out = lf.with_row_index().collect()

    assert_frame_equal(
        out,
        pl.DataFrame(
            {"index": [0, 1, 2], "a": [1, 2, 3]},
            schema={"index": pl.get_index_type(), "a": pl.Int64},
        ),
    )


def test_scan_include_file_paths_respects_projection_pushdown() -> None:
    q = pl.scan_csv(b"a,b,c\na1,b1,c1", include_file_paths="path_name").select(
        ["a", "b"]
    )

    assert_frame_equal(q.collect(), pl.DataFrame({"a": "a1", "b": "b1"}))


def test_streaming_scan_csv_include_file_paths_18257(io_files_path: Path) -> None:
    lf = pl.scan_csv(
        io_files_path / "foods1.csv",
        include_file_paths="path",
    ).select("category", "path")

    assert lf.collect(engine="streaming").columns == ["category", "path"]


def test_streaming_scan_csv_with_row_index_19172(io_files_path: Path) -> None:
    lf = (
        pl.scan_csv(io_files_path / "foods1.csv", infer_schema=False)
        .with_row_index()
        .select("calories", "index")
        .head(1)
    )

    assert_frame_equal(
        lf.collect(engine="streaming"),
        pl.DataFrame(
            {"calories": "45", "index": 0},
            schema={"calories": pl.String, "index": pl.get_index_type()},
        ),
    )


@pytest.mark.write_disk
def test_predicate_hive_pruning_with_cast(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)

    df = pl.DataFrame({"x": 1})

    (p := (tmp_path / "date=2024-01-01")).mkdir()

    df.write_parquet(p / "1")

    (p := (tmp_path / "date=2024-01-02")).mkdir()

    # Write an invalid parquet file that will cause errors if polars attempts to
    # read it.
    # This works because `scan_parquet()` only looks at the first file during
    # schema inference.
    (p / "1").write_text("not a parquet file")

    expect = pl.DataFrame({"x": 1, "date": datetime(2024, 1, 1).date()})

    lf = pl.scan_parquet(tmp_path)

    q = lf.filter(pl.col("date") < datetime(2024, 1, 2).date())

    assert_frame_equal(q.collect(), expect)

    # This filter expr with stprtime is effectively what LazyFrame.sql()
    # generates
    q = lf.filter(
        pl.col("date")
        < pl.lit("2024-01-02").str.strptime(
            dtype=pl.Date, format="%Y-%m-%d", ambiguous="latest"
        )
    )

    assert_frame_equal(q.collect(), expect)

    q = lf.sql("select * from self where date < '2024-01-02'")
    print(q.explain())
    assert_frame_equal(q.collect(), expect)


def test_predicate_stats_eval_nested_binary() -> None:
    bufs: list[bytes] = []

    for i in range(10):
        b = io.BytesIO()
        pl.DataFrame({"x": i}).write_parquet(b)
        b.seek(0)
        bufs.append(b.read())

    assert_frame_equal(
        (
            pl.scan_parquet(bufs)
            .filter(pl.col("x") % 2 == 0)
            .collect(optimizations=pl.QueryOptFlags.none())
        ),
        pl.DataFrame({"x": [0, 2, 4, 6, 8]}),
    )

    assert_frame_equal(
        (
            pl.scan_parquet(bufs)
            # The literal eval depth limit is 4 -
            # * crates/polars-expr/src/expressions/mod.rs::PhysicalExpr::evaluate_inline
            .filter(pl.col("x") == pl.lit("222").str.slice(0, 1).cast(pl.Int64))
            .collect()
        ),
        pl.DataFrame({"x": [2]}),
    )


@pytest.mark.slow
@pytest.mark.parametrize("streaming", [True, False])
def test_scan_csv_bytesio_memory_usage(
    streaming: bool,
    # memory_usage_without_pyarrow: MemoryUsage,
) -> None:
    # memory_usage = memory_usage_without_pyarrow

    # Create CSV that is ~6-7 MB in size:
    f = io.BytesIO()
    df = pl.DataFrame({"mydata": pl.int_range(0, 1_000_000, eager=True)})
    df.write_csv(f)
    # assert 6_000_000 < f.tell() < 7_000_000
    f.seek(0, 0)

    # A lazy scan shouldn't make a full copy of the data:
    # starting_memory = memory_usage.get_current()
    assert (
        pl.scan_csv(f)
        .filter(pl.col("mydata") == 999_999)
        .collect(engine="streaming" if streaming else "in-memory")
        .item()
        == 999_999
    )
    # assert memory_usage.get_peak() - starting_memory < 1_000_000


@pytest.mark.parametrize(
    "scan_type",
    [
        (pl.DataFrame.write_parquet, pl.scan_parquet),
        (pl.DataFrame.write_ipc, pl.scan_ipc),
        (pl.DataFrame.write_csv, pl.scan_csv),
        (pl.DataFrame.write_ndjson, pl.scan_ndjson),
    ],
)
def test_only_project_row_index(scan_type: tuple[Any, Any]) -> None:
    write, scan = scan_type

    f = io.BytesIO()
    df = pl.DataFrame([pl.Series("a", [1, 2, 3], pl.UInt32)])
    write(df, f)

    f.seek(0)
    s = scan(f, row_index_name="row_index", row_index_offset=42)

    assert_frame_equal(
        s.select("row_index").collect(),
        pl.DataFrame({"row_index": [42, 43, 44]}),
        check_dtypes=False,
    )


@pytest.mark.parametrize(
    "scan_type",
    [
        (pl.DataFrame.write_parquet, pl.scan_parquet),
        (pl.DataFrame.write_ipc, pl.scan_ipc),
        (pl.DataFrame.write_csv, pl.scan_csv),
        (pl.DataFrame.write_ndjson, pl.scan_ndjson),
    ],
)
def test_only_project_include_file_paths(scan_type: tuple[Any, Any]) -> None:
    write, scan = scan_type

    f = io.BytesIO()
    df = pl.DataFrame([pl.Series("a", [1, 2, 3], pl.UInt32)])
    write(df, f)

    f.seek(0)
    s = scan(f, include_file_paths="file_path")

    # The exact value for in-memory buffers is undefined
    c = s.select("file_path").collect()
    assert c.height == 3
    assert c.columns == ["file_path"]


@pytest.mark.parametrize(
    "scan_type",
    [
        (pl.DataFrame.write_parquet, pl.scan_parquet),
        pytest.param(
            (pl.DataFrame.write_ipc, pl.scan_ipc),
            marks=pytest.mark.xfail(
                reason="has no allow_missing_columns parameter. https://github.com/pola-rs/polars/issues/21166"
            ),
        ),
        pytest.param(
            (pl.DataFrame.write_csv, pl.scan_csv),
            marks=pytest.mark.xfail(
                reason="has no allow_missing_columns parameter. https://github.com/pola-rs/polars/issues/21166"
            ),
        ),
        pytest.param(
            (pl.DataFrame.write_ndjson, pl.scan_ndjson),
            marks=pytest.mark.xfail(
                reason="has no allow_missing_columns parameter. https://github.com/pola-rs/polars/issues/21166"
            ),
        ),
    ],
)
def test_only_project_missing(scan_type: tuple[Any, Any]) -> None:
    write, scan = scan_type

    f = io.BytesIO()
    g = io.BytesIO()
    write(
        pl.DataFrame(
            [pl.Series("a", [], pl.UInt32), pl.Series("missing", [], pl.Int32)]
        ),
        f,
    )
    write(pl.DataFrame([pl.Series("a", [1, 2, 3], pl.UInt32)]), g)

    f.seek(0)
    g.seek(0)
    s = scan([f, g], missing_columns="insert")

    assert_frame_equal(
        s.select("missing").collect(),
        pl.DataFrame([pl.Series("missing", [None, None, None], pl.Int32)]),
    )


@pytest.mark.write_disk
@pytest.mark.parametrize(
    "scan_type",
    [
        (pl.DataFrame.write_parquet, pl.scan_parquet),
        (pl.DataFrame.write_ipc, pl.scan_ipc),
        (pl.DataFrame.write_csv, pl.scan_csv),
        (pl.DataFrame.write_ndjson, pl.scan_ndjson),
    ],
)
def test_async_read_21945(tmp_path: Path, scan_type: tuple[Any, Any]) -> None:
    f1 = tmp_path / "f1"
    f2 = tmp_path / "f2"

    pl.DataFrame({"value": [1, 2]}).write_parquet(f1)
    pl.DataFrame({"value": [3]}).write_parquet(f2)

    df = (
        pl.scan_parquet([format_file_uri(str(f1)), str(f2)], include_file_paths="foo")
        .filter(value=1)
        .collect()
    )

    assert_frame_equal(
        df,
        pl.DataFrame(
            {
                "value": [1],
                "foo": [format_file_uri(f1)],
            }
        ),
    )


@pytest.mark.write_disk
@pytest.mark.parametrize("with_str_contains", [False, True])
def test_hive_pruning_str_contains_21706(
    tmp_path: Path, capfd: Any, plmonkeypatch: PlMonkeyPatch, with_str_contains: bool
) -> None:
    df = pl.DataFrame(
        {
            "pdate": [20250301, 20250301, 20250302, 20250302, 20250303, 20250303],
            "prod_id": ["A1", "A2", "B1", "B2", "C1", "C2"],
            "price": [11, 22, 33, 44, 55, 66],
        }
    )

    df.write_parquet(tmp_path, partition_by=["pdate"])

    plmonkeypatch.setenv("POLARS_VERBOSE", "1")
    f = pl.col("pdate") == 20250303
    if with_str_contains:
        f = f & pl.col("prod_id").str.contains("1")

    df = pl.scan_parquet(tmp_path, hive_partitioning=True).filter(f).collect()

    captured = capfd.readouterr().err
    assert "allows skipping 2 / 3" in captured

    assert_frame_equal(
        df,
        pl.scan_parquet(tmp_path, hive_partitioning=True).collect().filter(f),
    )


@pytest.mark.skipif(
    sys.platform == "win32", reason="path characters not valid on Windows"
)
@pytest.mark.parametrize(
    ("scan", "write"),
    [
        (pl.scan_ipc, pl.DataFrame.write_ipc),
        (pl.scan_parquet, pl.DataFrame.write_parquet),
        (pl.scan_csv, pl.DataFrame.write_csv),
    ],
)
@pytest.mark.parametrize("file_name", ["%?", "[", "]"])
def test_scan_no_glob_special_chars_23292(
    tmp_path: Path, file_name: str, scan: Any, write: Any
) -> None:
    tmp_path.mkdir(exist_ok=True)

    path = tmp_path / file_name
    df = pl.DataFrame({"a": 1})
    write(df, path)

    assert_frame_equal(scan(path, glob=False).collect(), df)
    assert_frame_equal(scan(f"file://{path}", glob=False).collect(), df)


@pytest.mark.write_disk
@pytest.mark.parametrize(
    ("scan_function", "failed_message", "name_in_context"),
    [
        (
            pl.scan_parquet,
            "failed to retrieve first file schema (parquet)",
            "'parquet scan'",
        ),
        (pl.scan_ipc, "failed to retrieve first file schema (ipc)", "'ipc scan'"),
        (pl.scan_csv, "failed to retrieve file schemas (csv)", "'csv scan'"),
        (
            pl.scan_ndjson,
            "failed to retrieve first file schema (ndjson)",
            "'ndjson scan'",
        ),
    ],
)
def test_scan_empty_paths_friendly_error(
    tmp_path: Path,
    scan_function: Any,
    failed_message: str,
    name_in_context: str,
) -> None:
    q = scan_function(tmp_path)

    with pytest.raises(pl.exceptions.ComputeError) as exc:
        q.collect()

    exc_str = exc.exconly()

    assert (
        f"ComputeError: {failed_message}: expanded paths were empty "
        "(path expansion input: 'paths: "
    ) in exc_str

    assert "glob: true)." in exc_str
    assert exc_str.count(tmp_path.name) == 1

    if scan_function is pl.scan_parquet:
        assert (
            "Hint: passing a schema can allow this scan to succeed with an empty DataFrame."
            in exc_str
        )

    # Multiple input paths
    q = scan_function([tmp_path, tmp_path])

    with pytest.raises(pl.exceptions.ComputeError) as exc:
        q.collect()

    exc_str = exc.exconly()

    assert (
        f"ComputeError: {failed_message}: expanded paths were empty "
        "(path expansion input: 'paths: "
    ) in exc_str

    assert "glob: true)." in exc_str

    assert exc_str.count(tmp_path.name) == 2

    q = scan_function([])

    with pytest.raises(pl.exceptions.ComputeError) as exc:
        q.collect()

    exc_str = exc.exconly()

    # There is no "path expansion resulted in" for this error message as the
    # original input sources were empty.
    assert f"ComputeError: {failed_message}: empty input: paths: []" in exc_str

    if scan_function is pl.scan_parquet:
        assert (
            "Hint: passing a schema can allow this scan to succeed with an empty DataFrame."
            in exc_str
        )

    # TODO: glob parameter not supported in some scan types
    cx = (
        pytest.raises(pl.exceptions.ComputeError, match="glob: false")
        if (
            scan_function is pl.scan_csv
            or scan_function is pl.scan_parquet
            or scan_function is pl.scan_ipc
        )
        else pytest.raises(TypeError, match="unexpected keyword argument 'glob'")  # type: ignore[arg-type]
    )

    with cx:
        scan_function(tmp_path, glob=False).collect()


@pytest.mark.parametrize("paths", [[], ["file:///non-existent"]])
@pytest.mark.parametrize("scan_func", [pl.scan_parquet, pl.scan_csv, pl.scan_ndjson])
def test_scan_with_schema_skips_schema_inference(
    paths: list[str], scan_func: Any
) -> None:
    schema = {"A": pl.Int64}

    q = scan_func(paths, schema=schema).head(0)
    assert_frame_equal(q.collect(engine="streaming"), pl.DataFrame(schema=schema))


@pytest.mark.parametrize("format_name", ["csv", "ndjson", "lines"])
def test_scan_negative_slice_decompress(format_name: str) -> None:
    col_name = "x"

    # We could go through sink, but not for lines and this way we are testing
    # more narrowly.
    def format_line(val: int) -> str:
        if format_name == "csv":
            return f"{val}\n"
        elif format_name == "ndjson":
            return f'{{"{col_name}":{val}}}\n'
        elif format_name == "lines":
            return f"{val}\n"
        else:
            pytest.fail("unreachable")

    buf = bytearray()
    if format_name == "csv":
        buf.extend(f"{col_name}\n".encode())
    for val in range(47):
        buf.extend(format_line(val).encode())

    compressed_data = zlib.compress(buf, level=1)

    lf = getattr(pl, f"scan_{format_name}")(compressed_data).slice(-9, 5)
    if format_name == "lines":
        lf = lf.select(pl.col("lines").alias(col_name).str.to_integer())

    expected = [pl.Series("x", [38, 39, 40, 41, 42])]
    got = lf.collect(engine="streaming")
    assert_frame_equal(got, pl.DataFrame(expected))


def corrupt_compressed_impl(base_line: bytes, target_size: int) -> bytes:
    large_and_simple_data = base_line * round(target_size / len(base_line))
    compressed_data = zlib.compress(large_and_simple_data, level=0)

    corruption_start_pos = round(len(compressed_data) * 0.9)
    corruption_len = 500

    # The idea is to corrupt the input to make sure the scan never fully
    # decompresses the input.
    corrupted_data = bytearray(compressed_data)
    corrupted_data[corruption_start_pos : corruption_start_pos + corruption_len] = (
        b"\00"
    )
    # ~1.8MB of valid zlib compressed data to read before the corrupted data
    # appears.
    return bytes(corrupted_data)


@pytest.fixture(scope="session")
def corrupt_compressed_csv() -> bytes:
    return corrupt_compressed_impl(b"line_val\n", int(2e6))


@pytest.fixture(scope="session")
def corrupt_compressed_ndjson() -> bytes:
    # The decompressor is part of the line-batch provider which only stops once
    # processing noticed that it doesn't need anymore data. This can take a
    # bit. Tested with debug/release and calm/stressed machine.
    return corrupt_compressed_impl(b'{"line_val": 45}\n', int(100e6))


@pytest.mark.parametrize("schema", [{"line_val": pl.String}, None])
def test_scan_csv_streaming_decompression(
    corrupt_compressed_csv: bytes, schema: Any
) -> None:
    slice_count = 11

    df = (
        pl.scan_csv(io.BytesIO(corrupt_compressed_csv), schema=schema)
        .slice(0, slice_count)
        .collect(engine="streaming")
    )

    expected = [
        pl.Series("line_val", ["line_val"] * slice_count, dtype=pl.String),
    ]
    assert_frame_equal(df, pl.DataFrame(expected))


@pytest.mark.slow
@pytest.mark.parametrize("schema", [{"line_val": pl.Int64}, None])
def test_scan_ndjson_streaming_decompression(
    corrupt_compressed_ndjson: bytes, schema: Any
) -> None:
    slice_count = 11

    df = (
        pl.scan_ndjson(io.BytesIO(corrupt_compressed_ndjson), schema=schema)
        .slice(0, slice_count)
        .collect(engine="streaming")
    )

    expected = [
        pl.Series("line_val", [45] * slice_count, dtype=pl.Int64),
    ]
    assert_frame_equal(df, pl.DataFrame(expected))


def test_scan_file_uri_hostname_component() -> None:
    q = pl.scan_parquet("file://hostname:80/data.parquet")

    with pytest.raises(
        ComputeError,
        match="unsupported: non-empty hostname for 'file:' URI: 'hostname:80'",
    ):
        q.collect()


@pytest.mark.write_disk
@pytest.mark.parametrize("polars_force_async", ["0", "1"])
@pytest.mark.parametrize("n_repeats", [1, 2])
def test_scan_path_expansion_sorting_24528(
    tmp_path: Path,
    polars_force_async: str,
    n_repeats: int,
    plmonkeypatch: PlMonkeyPatch,
) -> None:
    plmonkeypatch.setenv("POLARS_FORCE_ASYNC", polars_force_async)

    relpaths = ["a.parquet", "a/a.parquet", "ab.parquet"]

    for p in relpaths:
        Path(tmp_path / p).parent.mkdir(exist_ok=True, parents=True)
        pl.DataFrame({"relpath": p}).write_parquet(tmp_path / p)

    assert_frame_equal(
        pl.scan_parquet(n_repeats * [tmp_path]).collect(),
        pl.DataFrame({"relpath": n_repeats * relpaths}),
    )

    assert_frame_equal(
        pl.scan_parquet(n_repeats * [f"{tmp_path}/**/*"]).collect(),
        pl.DataFrame({"relpath": n_repeats * relpaths}),
    )

    assert_frame_equal(
        pl.scan_parquet(n_repeats * [format_file_uri(tmp_path) + "/"]).collect(),
        pl.DataFrame({"relpath": n_repeats * relpaths}),
    )

    assert_frame_equal(
        pl.scan_parquet(n_repeats * [format_file_uri(f"{tmp_path}/**/*")]).collect(),
        pl.DataFrame({"relpath": n_repeats * relpaths}),
    )


def test_scan_sink_error_captures_path() -> None:
    storage_options = {
        "aws_endpoint_url": "http://localhost:333",
        "max_retries": 0,
    }

    q = pl.scan_parquet(
        "s3://.../...",
        storage_options=storage_options,
        credential_provider=None,
    )

    with pytest.raises(OSError, match=r"path: s3://.../..."):
        q.collect()

    with pytest.raises(OSError, match=r"path: s3://.../..."):
        pl.LazyFrame({"a": 1}).sink_parquet(
            "s3://.../...",
            storage_options=storage_options,
            credential_provider=None,
        )


# TODO: Uncomment file_format once properly instrumented
@pytest.mark.parametrize(
    "file_format",
    [
        "parquet",
        "ipc",
        # "csv",
        "ndjson",
    ],
)
@pytest.mark.parametrize("partitioned", [True, False])
@pytest.mark.write_disk
def test_scan_metrics(
    plmonkeypatch: PlMonkeyPatch,
    capfd: pytest.CaptureFixture[str],
    file_format: str,
    tmp_path: Path,
    partitioned: bool,
) -> None:
    path = tmp_path / "a"

    df = pl.DataFrame({"a": 1})

    getattr(pl.LazyFrame, f"sink_{file_format}")(
        df.lazy(),
        path
        if not partitioned
        else pl.PartitionBy("", file_path_provider=(lambda _: path), key="a"),
    )

    with plmonkeypatch.context() as cx:
        cx.setenv("POLARS_LOG_METRICS", "1")
        cx.setenv("POLARS_FORCE_ASYNC", "1")
        capfd.readouterr()
        out = getattr(pl, f"scan_{file_format}")(
            path,
        ).collect()
        capture = capfd.readouterr().err

    [line] = (x for x in capture.splitlines() if x.startswith("multi-scan"))

    logged_bytes_requested = int(
        pl.select(pl.lit(line).str.extract(r"total_bytes_requested=(\d+)")).item()
    )

    logged_bytes_received = int(
        pl.select(pl.lit(line).str.extract(r"total_bytes_received=(\d+)")).item()
    )

    # because of how metadata is accounted for, the bytes_requested may deviate
    # from the actual file_size
    file_size = path.stat().st_size
    # note, 131_072 is the maximum metadata size estimate for parquet, ipc
    upper_limit_bytes = min(2 * file_size, file_size + 131072)
    lower_limit_bytes = 2

    assert logged_bytes_requested <= upper_limit_bytes
    assert logged_bytes_requested >= lower_limit_bytes
    assert logged_bytes_received == logged_bytes_requested

    assert_frame_equal(out, df)
