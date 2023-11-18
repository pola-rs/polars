from __future__ import annotations

import sys
from contextlib import suppress
from typing import TYPE_CHECKING, Any

import pytest

import polars as pl
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from pathlib import Path

    from polars.type_aliases import DbWriteEngine


def adbc_sqlite_driver_version(*args: Any, **kwargs: Any) -> str:
    with suppress(ModuleNotFoundError):  # not available on 3.8/windows
        import adbc_driver_sqlite

        return getattr(adbc_driver_sqlite, "__version__", "n/a")
    return "n/a"


@pytest.mark.skipif(
    sys.version_info > (3, 11),
    reason="connectorx cannot be installed on Python 3.12 yet.",
)
@pytest.mark.skipif(
    sys.version_info < (3, 9) or sys.platform == "win32",
    reason="adbc_driver_sqlite not available below Python 3.9 / on Windows",
)
@pytest.mark.write_disk()
@pytest.mark.parametrize("engine", ["adbc", "sqlalchemy"])
def test_write_database_create(engine: DbWriteEngine, tmp_path: Path) -> None:
    df = pl.DataFrame(
        {
            "id": [1234, 5678],
            "name": ["misc", "other"],
            "value": [1000.0, -9999.0],
        }
    )
    tmp_path.mkdir(exist_ok=True)
    test_db = str(tmp_path / f"test_{engine}.db")
    table_name = "test_create"

    df.write_database(
        table_name=table_name,
        connection=f"sqlite:///{test_db}",
        if_exists="replace",
        engine=engine,
    )
    result = pl.read_database_uri(f"SELECT * FROM {table_name}", f"sqlite:///{test_db}")
    assert_frame_equal(result, df)


@pytest.mark.skipif(
    sys.version_info > (3, 11),
    reason="connectorx cannot be installed on Python 3.12 yet.",
)
@pytest.mark.skipif(
    sys.version_info < (3, 9) or sys.platform == "win32",
    reason="adbc_driver_sqlite not available below Python 3.9 / on Windows",
)
@pytest.mark.write_disk()
@pytest.mark.parametrize("engine", ["adbc", "sqlalchemy"])
def test_write_database_append(engine: DbWriteEngine, tmp_path: Path) -> None:
    df = pl.DataFrame(
        {
            "key": ["xx", "yy", "zz"],
            "value": [123, None, 789],
            "other": [5.5, 7.0, None],
        }
    )

    tmp_path.mkdir(exist_ok=True)
    test_db = str(tmp_path / f"test_{engine}.db")
    table_name = "test_append"

    df.write_database(
        table_name=table_name,
        connection=f"sqlite:///{test_db}",
        if_exists="replace",
        engine=engine,
    )

    ExpectedError = NotImplementedError if engine == "adbc" else ValueError
    with pytest.raises(ExpectedError):
        df.write_database(
            table_name=table_name,
            connection=f"sqlite:///{test_db}",
            if_exists="fail",
            engine=engine,
        )

    df.write_database(
        table_name=table_name,
        connection=f"sqlite:///{test_db}",
        if_exists="append",
        engine=engine,
    )
    result = pl.read_database_uri(f"SELECT * FROM {table_name}", f"sqlite:///{test_db}")
    assert_frame_equal(result, pl.concat([df, df]))


@pytest.mark.skipif(
    sys.version_info < (3, 9) or sys.platform == "win32",
    reason="adbc_driver_sqlite not available below Python 3.9 / on Windows",
)
@pytest.mark.write_disk()
@pytest.mark.parametrize(
    "engine",
    [
        pytest.param(
            "adbc",
            marks=pytest.mark.xfail(  # see: https://github.com/apache/arrow-adbc/issues/1000
                reason="ADBC SQLite driver has a bug with quoted/qualified table names",
            ),
        ),
        pytest.param(
            "sqlalchemy",
            marks=pytest.mark.skipif(
                sys.version_info > (3, 11),
                reason="connectorx cannot be installed on Python 3.12 yet.",
            ),
        ),
    ],
)
def test_write_database_create_quoted_tablename(
    engine: DbWriteEngine, tmp_path: Path
) -> None:
    df = pl.DataFrame({"col x": [100, 200, 300], "col y": ["a", "b", "c"]})

    tmp_path.mkdir(exist_ok=True)
    test_db = str(tmp_path / f"test_{engine}.db")

    # table name requires quoting, and is qualified with the implicit 'main' schema
    table_name = 'main."test-append"'

    df.write_database(
        table_name=table_name,
        connection=f"sqlite:///{test_db}",
        if_exists="replace",
        engine=engine,
    )
    result = pl.read_database_uri(f"SELECT * FROM {table_name}", f"sqlite:///{test_db}")
    assert_frame_equal(result, df)


def test_write_database_errors() -> None:
    # confirm that invalid parameter values raise errors
    df = pl.DataFrame({"colx": [1, 2, 3]})

    with pytest.raises(
        ValueError, match="`table_name` appears to be invalid: 'w.x.y.z'"
    ):
        df.write_database(
            connection="sqlite:///:memory:", table_name="w.x.y.z", engine="sqlalchemy"
        )

    with pytest.raises(
        NotImplementedError, match="`if_exists = 'fail'` not supported for ADBC engine"
    ):
        df.write_database(
            connection="sqlite:///:memory:",
            table_name="test_errs",
            if_exists="fail",
            engine="adbc",
        )

    with pytest.raises(ValueError, match="'do_something' is not valid for if_exists"):
        df.write_database(
            connection="sqlite:///:memory:",
            table_name="main.test_errs",
            if_exists="do_something",  # type: ignore[arg-type]
            engine="sqlalchemy",
        )
