from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any, Callable, cast

import pytest

import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal


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


def test_read_missing_file_path_truncated() -> None:
    content = "lskdfj".join(str(i) for i in range(25))
    with pytest.raises(
        FileNotFoundError,
        match="\\.\\.\\.lskdfj14lskdfj15lskdfj16lskdfj17lskdfj18lskdfj19lskdfj20lskdfj21lskdfj22lskdfj23lskdfj24",
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

    df2 = cast(pl.DataFrame, pl.from_arrow(tbl))
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
    assert not (
        io_dir / "__init__.py"
    ).exists(), "Found undesirable '__init__.py' in the 'unit.io' tests subdirectory"


@pytest.mark.write_disk()
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
