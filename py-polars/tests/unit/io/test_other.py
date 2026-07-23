from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest

import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal
from tests.unit.utils.pathlike import HostilePathLike

if TYPE_CHECKING:
    from collections.abc import Callable

    from tests.conftest import PlMonkeyPatch


@pytest.mark.parametrize(
    "read_function",
    [
        pl.read_csv,
        pl.read_ipc,
        pl.read_json,
        pl.read_parquet,
        pl.read_avro,
        pl.scan_csv,
        pl.scan_ipc,
        pl.scan_parquet,
    ],
)
def test_read_missing_file(read_function: Callable[[Any], pl.DataFrame]) -> None:
    match = "\\(os error 2\\): fake_file_path"
    # The message associated with OS error 2 may differ per platform
    if sys.platform == "linux":
        match = "No such file or directory " + match

    if "scan" in read_function.__name__:
        with pytest.raises(FileNotFoundError, match=match):
            read_function("fake_file_path").collect()  # type: ignore[attr-defined]
    else:
        with pytest.raises(FileNotFoundError, match=match):
            read_function("fake_file_path")


@pytest.mark.parametrize(
    "write_method_name",
    [
        # "write_excel" not included
        # because it already raises a FileCreateError
        # from the underlying library dependency
        "write_csv",
        "write_ipc",
        "write_ipc_stream",
        "write_json",
        "write_ndjson",
        "write_parquet",
        "write_avro",
    ],
)
def test_write_missing_directory(write_method_name: str) -> None:
    df = pl.DataFrame({"a": [1]})
    non_existing_path = Path("non", "existing", "path")
    if non_existing_path.exists():
        pytest.fail(
            "Testing on a non existing path failed because the path does exist."
        )
    write_method = getattr(df, write_method_name)
    with pytest.raises(FileNotFoundError):
        write_method(non_existing_path)


@pytest.mark.parametrize(
    ("write_method", "read_function", "scan_function", "read_supports_list"),
    [
        # `read_supports_list` reflects existing behavior: the eager CSV/IPC
        # readers do not accept a list of paths (only their `scan_*` variants
        # do), while `read_parquet`/`read_ndjson` dispatch lists to a scan.
        ("write_parquet", pl.read_parquet, pl.scan_parquet, True),
        ("write_csv", pl.read_csv, pl.scan_csv, False),
        ("write_ipc", pl.read_ipc, pl.scan_ipc, False),
        ("write_ndjson", pl.read_ndjson, pl.scan_ndjson, True),
        ("write_avro", pl.read_avro, None, False),
        ("write_json", pl.read_json, None, False),
    ],
)
def test_read_scan_os_pathlike_17828(
    tmp_path: Path,
    write_method: str,
    read_function: Callable[[Any], pl.DataFrame],
    scan_function: Callable[[Any], pl.LazyFrame] | None,
    read_supports_list: bool,
) -> None:
    tmp_path.mkdir(exist_ok=True)
    df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    path = tmp_path / "data"
    getattr(df, write_method)(path)

    source = HostilePathLike(path)

    # A single `os.PathLike` must be treated as a path, not iterated over.
    assert_frame_equal(read_function(source), df)
    if scan_function is not None:
        assert_frame_equal(scan_function(source).collect(), df)

    # A list of `os.PathLike` must also be accepted where lists are supported.
    if read_supports_list:
        assert_frame_equal(read_function([source]), df)
    if scan_function is not None:
        assert_frame_equal(scan_function([source]).collect(), df)


@pytest.mark.parametrize(
    ("write_method", "read_function"),
    [
        ("write_parquet", pl.read_parquet),
        ("write_csv", pl.read_csv),
        ("write_ipc", pl.read_ipc),
        ("write_ipc_stream", pl.read_ipc_stream),
        ("write_ndjson", pl.read_ndjson),
        ("write_json", pl.read_json),
        ("write_avro", pl.read_avro),
    ],
)
def test_write_os_pathlike_17828(
    tmp_path: Path,
    write_method: str,
    read_function: Callable[[Any], pl.DataFrame],
) -> None:
    tmp_path.mkdir(exist_ok=True)
    df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    path = tmp_path / "data"

    # Writing must accept an `os.PathLike` target (not fall through to the
    # file-object branch).
    getattr(df, write_method)(HostilePathLike(path))

    assert_frame_equal(read_function(path), df)


def test_read_parquet_pyarrow_os_pathlike_17828(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    tmp_path.mkdir(exist_ok=True)
    df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    path = tmp_path / "data.parquet"
    df.write_parquet(path)

    # The PyArrow reader takes a separate code path from the native reader.
    assert_frame_equal(pl.read_parquet(HostilePathLike(path), use_pyarrow=True), df)


def test_read_schema_metadata_os_pathlike_17828(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)
    df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})

    parquet_path = tmp_path / "data.parquet"
    ipc_path = tmp_path / "data.ipc"
    ipc_stream_path = tmp_path / "data.ipcstream"
    df.write_parquet(parquet_path)
    df.write_ipc(ipc_path)
    df.write_ipc_stream(ipc_stream_path)

    assert set(pl.read_parquet_schema(HostilePathLike(parquet_path))) == {"a", "b"}
    assert set(pl.read_ipc_schema(HostilePathLike(ipc_path))) == {"a", "b"}
    assert isinstance(pl.read_parquet_metadata(HostilePathLike(parquet_path)), dict)

    # `read_ipc_stream` also accepts `os.PathLike`.
    assert_frame_equal(pl.read_ipc_stream(HostilePathLike(ipc_stream_path)), df)


def test_read_csv_batched_os_pathlike_17828(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)
    df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    csv_path = tmp_path / "data.csv"
    df.write_csv(csv_path)

    with pytest.deprecated_call():
        batched = pl.read_csv_batched(HostilePathLike(csv_path))
    batches = batched.next_batches(1)
    assert batches is not None
    assert_frame_equal(batches[0], df)


@pytest.mark.parametrize(
    ("write_method", "read_function"),
    [
        ("write_csv", pl.read_csv),
        ("write_ipc", pl.read_ipc),
    ],
)
def test_read_os_pathlike_force_async_17828(
    plmonkeypatch: PlMonkeyPatch,
    tmp_path: Path,
    write_method: str,
    read_function: Callable[[Any], pl.DataFrame],
) -> None:
    # The async dispatch checks `os.fspath(source).startswith("hf://")`; a
    # path-like with a non-path `__str__` would break that check if `str()`
    # were used instead of `os.fspath()`.
    plmonkeypatch.setenv("POLARS_FORCE_ASYNC", "1")

    tmp_path.mkdir(exist_ok=True)
    df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    path = tmp_path / "data"
    getattr(df, write_method)(path)

    assert_frame_equal(read_function(HostilePathLike(path)), df)


def test_read_missing_file_path_truncated() -> None:
    content = "lskdfj".join(str(i) for i in range(25))

    with pytest.raises(
        FileNotFoundError,
        match=r"\.\.\.lskdfj14lskdfj15lskdfj16lskdfj17lskdfj18lskdfj19lskdfj20lskdfj21lskdfj22lskdfj23lskdfj24 \(set POLARS_VERBOSE=1 to see full path\)",
    ):
        pl.read_csv(content)


def test_read_missing_file_path_expanded_when_polars_verbose_enabled(
    plmonkeypatch: PlMonkeyPatch,
) -> None:
    content = "lskdfj".join(str(i) for i in range(25))

    plmonkeypatch.setenv("POLARS_VERBOSE", "1")

    with pytest.raises(
        FileNotFoundError,
        match=content,
    ):
        pl.read_csv(content)


def test_copy() -> None:
    df = pl.DataFrame({"a": [1, 2], "b": ["a", None], "c": [True, False]})
    assert_frame_equal(copy.copy(df), df)
    assert_frame_equal(copy.deepcopy(df), df)

    a = pl.Series("a", [1, 2])
    assert_series_equal(copy.copy(a), a)
    assert_series_equal(copy.deepcopy(a), a)


def test_categorical_round_trip() -> None:
    df = pl.DataFrame({"ints": [1, 2, 3], "cat": ["a", "b", "c"]})
    df = df.with_columns(pl.col("cat").cast(pl.Categorical))

    tbl = df.to_arrow()
    assert "dictionary" in str(tbl["cat"].type)

    df2 = cast("pl.DataFrame", pl.from_arrow(tbl))
    assert df2.dtypes == [pl.Int64, pl.Categorical]


def test_from_different_chunks() -> None:
    s0 = pl.Series("a", [1, 2, 3, 4, None])
    s1 = pl.Series("b", [1, 2])
    s11 = pl.Series("b", [1, 2, 3])
    s1.append(s11)

    # check we don't panic
    df = pl.DataFrame([s0, s1])
    df.to_arrow()
    df = pl.DataFrame([s0, s1])
    out = df.to_pandas()
    assert list(out.columns) == ["a", "b"]
    assert out.shape == (5, 2)


def test_unit_io_subdir_has_no_init() -> None:
    # --------------------------------------------------------------------------------
    # If this test fails it means an '__init__.py' was added to 'tests/unit/io'.
    # See https://github.com/pola-rs/polars/pull/6889 for why this can cause issues.
    # --------------------------------------------------------------------------------
    # TLDR: it can mask the builtin 'io' module, causing a fatal python error.
    # --------------------------------------------------------------------------------
    io_dir = Path(__file__).parent
    assert io_dir.parts[-2:] == ("unit", "io")
    assert not (io_dir / "__init__.py").exists(), (
        "Found undesirable '__init__.py' in the 'unit.io' tests subdirectory"
    )


@pytest.mark.write_disk
@pytest.mark.parametrize(
    ("scan_funcs", "write_func"),
    [
        ([pl.scan_parquet, pl.read_parquet], pl.DataFrame.write_parquet),
        ([pl.scan_csv, pl.read_csv], pl.DataFrame.write_csv),
    ],
)
@pytest.mark.parametrize("char", ["[", "*"])
def test_no_glob(
    scan_funcs: list[Callable[[Any], pl.LazyFrame | pl.DataFrame]],
    write_func: Callable[[pl.DataFrame, Path], None],
    char: str,
    tmp_path: Path,
) -> None:
    if sys.platform == "win32" and char == "*":
        pytest.skip("unsupported glob char for windows")

    tmp_path.mkdir(exist_ok=True)

    df = pl.DataFrame({"x": 1})

    paths = [tmp_path / f"{char}", tmp_path / f"{char}1"]

    write_func(df, paths[0])
    write_func(df, paths[1])

    for func in scan_funcs:
        assert_frame_equal(func(paths[0], glob=False).lazy().collect(), df)  # type: ignore[call-arg]
