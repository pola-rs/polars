from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

import polars as pl
from polars.testing.asserts.frame import assert_frame_equal

if TYPE_CHECKING:
    from pathlib import Path

    from polars.type_aliases import SchemaDict


def _scan(file_path: Path, schema: SchemaDict | None = None) -> pl.LazyFrame:
    suffix = file_path.suffix
    if suffix == ".ipc":
        return pl.scan_ipc(file_path)
    if suffix == ".parquet":
        return pl.scan_parquet(file_path)
    if suffix == ".csv":
        return pl.scan_csv(file_path, schema=schema)
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
    params=[".csv", ".ipc", ".parquet"],
)
def data_file_extension(request: pytest.FixtureRequest) -> str:
    return request.param  # type: ignore[no-any-return]


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


@pytest.mark.write_disk()
def test_scan(data_file_single: _DataFile) -> None:
    df = _scan(data_file_single.path, data_file_single.df.schema).collect()
    assert_frame_equal(df, data_file_single.df)


@pytest.mark.write_disk()
def test_scan_with_limit(data_file_single: _DataFile) -> None:
    df = _scan(data_file_single.path, data_file_single.df.schema).limit(100).collect()
    assert_frame_equal(df, data_file_single.df.limit(100))
