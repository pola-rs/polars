from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import TYPE_CHECKING, Any

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


def _enable_force_async(monkeypatch: pytest.MonkeyPatch) -> None:
    """Modifies the provided monkeypatch context."""
    monkeypatch.setenv("POLARS_VERBOSE", "1")
    monkeypatch.setenv("POLARS_FORCE_ASYNC", "1")


def _assert_force_async(capfd: Any) -> None:
    """Calls `capfd.readouterr`, consuming the captured output so far."""
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

    if suffix == ".ipc":
        result = pl.scan_ipc(
            file_path,
            row_index_name=row_index_name,
            row_index_offset=row_index_offset,
        )
    elif suffix == ".parquet":
        result = pl.scan_parquet(
            file_path,
            row_index_name=row_index_name,
            row_index_offset=row_index_offset,
        )
    elif suffix == ".csv":
        result = pl.scan_csv(
            file_path,
            schema=schema,
            row_index_name=row_index_name,
            row_index_offset=row_index_offset,
        )
    else:
        msg = f"Unknown suffix {suffix}"
        raise NotImplementedError(msg)

    return result


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
    if data_file.path.suffix == ".csv" and force_async:
        pytest.skip(reason="async reading of .csv not yet implemented")

    if force_async:
        _enable_force_async(monkeypatch)

    df = _scan(data_file.path, data_file.df.schema).collect()

    if force_async:
        _assert_force_async(capfd)

    assert_frame_equal(df, data_file.df)


@pytest.mark.write_disk()
def test_scan_with_limit(
    capfd: Any, monkeypatch: pytest.MonkeyPatch, data_file: _DataFile, force_async: bool
) -> None:
    if data_file.path.suffix == ".csv" and force_async:
        pytest.skip(reason="async reading of .csv not yet implemented")

    if force_async:
        _enable_force_async(monkeypatch)

    df = _scan(data_file.path, data_file.df.schema).limit(4483).collect()

    if force_async:
        _assert_force_async(capfd)

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
    if data_file.path.suffix == ".csv" and force_async:
        pytest.skip(reason="async reading of .csv not yet implemented")

    if force_async:
        _enable_force_async(monkeypatch)

    df = (
        _scan(data_file.path, data_file.df.schema)
        .filter(pl.col("sequence") % 2 == 0)
        .collect()
    )

    if force_async:
        _assert_force_async(capfd)

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
    if data_file.path.suffix == ".csv" and force_async:
        pytest.skip(reason="async reading of .csv not yet implemented")

    if force_async:
        _enable_force_async(monkeypatch)

    df = (
        _scan(data_file.path, data_file.df.schema)
        .filter(pl.col("sequence") % 2 == 0)
        .limit(4483)
        .collect()
    )

    if force_async:
        _assert_force_async(capfd)

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
    if data_file.path.suffix == ".csv" and force_async:
        pytest.skip(reason="async reading of .csv not yet implemented")

    if force_async:
        _enable_force_async(monkeypatch)

    df = (
        _scan(data_file.path, data_file.df.schema)
        .limit(4483)
        .filter(pl.col("sequence") % 2 == 0)
        .collect()
    )

    if force_async:
        _assert_force_async(capfd)

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
    if data_file.path.suffix == ".csv" and force_async:
        pytest.skip(reason="async reading of .csv not yet implemented")

    if force_async:
        _enable_force_async(monkeypatch)

    df = (
        _scan(data_file.path, data_file.df.schema, row_index=_RowIndex())
        .limit(4483)
        .collect()
    )

    if force_async:
        _assert_force_async(capfd)

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
    if data_file.path.suffix == ".csv" and force_async:
        pytest.skip(reason="async reading of .csv not yet implemented")

    if force_async:
        _enable_force_async(monkeypatch)

    df = (
        _scan(data_file.path, data_file.df.schema, row_index=_RowIndex())
        .filter(pl.col("sequence") % 2 == 0)
        .collect()
    )

    if force_async:
        _assert_force_async(capfd)

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
    if data_file.path.suffix == ".csv" and force_async:
        pytest.skip(reason="async reading of .csv not yet implemented")

    if force_async:
        _enable_force_async(monkeypatch)

    df = (
        _scan(data_file.path, data_file.df.schema, row_index=_RowIndex())
        .limit(4483)
        .filter(pl.col("sequence") % 2 == 0)
        .collect()
    )

    if force_async:
        _assert_force_async(capfd)

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
        _assert_force_async(capfd)

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
