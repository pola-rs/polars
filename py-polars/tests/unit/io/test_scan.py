from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

import polars as pl
from polars.testing.asserts.frame import assert_frame_equal

if TYPE_CHECKING:
    from pathlib import Path

    from polars.type_aliases import SchemaDict


@dataclass
class _RowIndex:
    name: str = "index"
    offset: int = 0


def _scan(
    file_path: Path,
    schema: SchemaDict | None = None,
    row_index: _RowIndex | None = None,
) -> pl.LazyFrame:
    suffix = file_path.suffix
    row_index_name = None if row_index is None else row_index.name
    row_index_offset = 0 if row_index is None else row_index.offset
    if suffix == ".ipc":
        return pl.scan_ipc(
            file_path,
            row_index_name=row_index_name,
            row_index_offset=row_index_offset,
        )
    if suffix == ".parquet":
        return pl.scan_parquet(
            file_path,
            row_index_name=row_index_name,
            row_index_offset=row_index_offset,
        )
    if suffix == ".csv":
        return pl.scan_csv(
            file_path,
            schema=schema,
            row_index_name=row_index_name,
            row_index_offset=row_index_offset,
        )
    msg = f"Unknown suffix {suffix}"
    raise NotImplementedError(msg)


def _write(df: pl.DataFrame, file_path: Path) -> None:
    suffix = file_path.suffix
    if suffix == ".ipc":
        return df.write_ipc(file_path)
    if suffix == ".parquet":
        return df.write_parquet(file_path)
    if suffix == ".csv":
        return df.write_csv(file_path)
    msg = f"Unknown suffix {suffix}"
    raise NotImplementedError(msg)


@pytest.fixture(
    scope="session",
    params=["csv", "ipc", "parquet"],
)
def data_file_extension(request: pytest.FixtureRequest) -> str:
    return f".{request.param}"


@pytest.fixture(scope="session")
def session_tmp_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("polars-test")


@dataclass
class _DataFile:
    path: Path
    df: pl.DataFrame


@pytest.fixture(scope="session")
def data_file_single(session_tmp_dir: Path, data_file_extension: str) -> _DataFile:
    file_path = (session_tmp_dir / "data").with_suffix(data_file_extension)
    df = pl.DataFrame(
        {
            "seq_int": range(10000),
            "seq_str": [f"{x}" for x in range(10000)],
        }
    )
    _write(df, file_path)
    return _DataFile(path=file_path, df=df)


@pytest.fixture(scope="session")
def data_file_glob(session_tmp_dir: Path, data_file_extension: str) -> _DataFile:
    row_counts = [
        100,
        186,
        95,
        185,
        90,
        84,
        115,
        81,
        87,
        217,
        126,
        85,
        98,
        122,
        129,
        122,
        1089,
        82,
        234,
        86,
        93,
        90,
        91,
        263,
        87,
        126,
        86,
        161,
        191,
        1368,
        403,
        192,
        102,
        98,
        115,
        81,
        111,
        305,
        92,
        534,
        431,
        150,
        90,
        128,
        152,
        118,
        127,
        124,
        229,
        368,
        81,
    ]
    assert sum(row_counts) == 10000
    assert (
        len(row_counts) < 100
    )  # need to make sure we pad file names with enough zeros, otherwise the lexographical ordering of the file names is not what we want.

    df = pl.DataFrame(
        {
            "seq_int": range(10000),
            "seq_str": [str(x) for x in range(10000)],
        }
    )

    row_offset = 0
    for index, row_count in enumerate(row_counts):
        file_path = (session_tmp_dir / f"data_{index:02}").with_suffix(
            data_file_extension
        )
        _write(df.slice(row_offset, row_count), file_path)
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
def test_scan(data_file: _DataFile) -> None:
    df = _scan(data_file.path, data_file.df.schema).collect()
    assert_frame_equal(df, data_file.df)


@pytest.mark.write_disk()
def test_scan_with_limit(data_file: _DataFile) -> None:
    df = _scan(data_file.path, data_file.df.schema).limit(100).collect()
    assert_frame_equal(df, data_file.df.limit(100))


@pytest.mark.write_disk()
def test_scan_with_row_index(data_file: _DataFile) -> None:
    df = _scan(data_file.path, data_file.df.schema, row_index=_RowIndex()).collect()
    assert_frame_equal(df, data_file.df.with_row_index())


@pytest.mark.write_disk()
def test_scan_with_row_index_and_predicate(data_file: _DataFile) -> None:
    df = (
        _scan(data_file.path, data_file.df.schema, row_index=_RowIndex())
        .filter(pl.col("seq_int") % 2 == 0)
        .collect()
    )
    assert_frame_equal(
        df, data_file.df.with_row_index().filter(pl.col("seq_int") % 2 == 0)
    )
