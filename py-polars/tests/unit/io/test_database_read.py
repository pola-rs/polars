from __future__ import annotations

import os
import sqlite3
import sys
from contextlib import suppress
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from sqlalchemy import create_engine

import polars as pl
from polars.exceptions import UnsuitableSQLError

if TYPE_CHECKING:
    from polars.type_aliases import DbReadEngine, SchemaDict


def adbc_sqlite_connect(*args: Any, **kwargs: Any) -> Any:
    with suppress(ModuleNotFoundError):  # not available on 3.8/windows
        from adbc_driver_sqlite.dbapi import connect

        return connect(*args, **kwargs)


@pytest.fixture()
def sample_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": [1, 2],
            "name": ["misc", "other"],
            "value": [100.0, -99.0],
            "date": ["2020-01-01", "2021-12-31"],
        }
    )


def create_temp_sqlite_db(test_db: str) -> None:
    Path(test_db).unlink(missing_ok=True)

    # NOTE: at the time of writing adcb/connectorx have weak SQLite support (poor or
    # no bool/date/datetime dtypes, for example) and there is a bug in connectorx that
    # causes float rounding < py 3.11, hence we are only testing/storing simple values
    # in this test db for now. as support improves, we can add/test additional dtypes).

    conn = sqlite3.connect(test_db)
    # ┌─────┬───────┬───────┬────────────┐
    # │ id  ┆ name  ┆ value ┆ date       │
    # │ --- ┆ ---   ┆ ---   ┆ ---        │
    # │ i64 ┆ str   ┆ f64   ┆ date       │
    # ╞═════╪═══════╪═══════╪════════════╡
    # │ 1   ┆ misc  ┆ 100.0 ┆ 2020-01-01 │
    # │ 2   ┆ other ┆ -99.0 ┆ 2021-12-31 │
    # └─────┴───────┴───────┴────────────┘
    conn.executescript(
        """
        CREATE TABLE test_data (
            id    INTEGER PRIMARY KEY,
            name  TEXT NOT NULL,
            value FLOAT,
            date  DATE
        );
        INSERT INTO test_data(name,value,date)
        VALUES ('misc',100.0,'2020-01-01'), ('other',-99.5,'2021-12-31');
        """
    )
    conn.close()


@pytest.mark.write_disk()
@pytest.mark.parametrize(
    (
        "read_method",
        "engine_or_connection_init",
        "expected_dtypes",
        "expected_dates",
        "schema_overrides",
    ),
    [
        pytest.param(
            "read_database_uri",
            "connectorx",
            {
                "id": pl.UInt8,
                "name": pl.Utf8,
                "value": pl.Float64,
                "date": pl.Date,
            },
            [date(2020, 1, 1), date(2021, 12, 31)],
            {"id": pl.UInt8},
            id="uri: connectorx",
        ),
        pytest.param(
            "read_database_uri",
            "adbc",
            {
                "id": pl.UInt8,
                "name": pl.Utf8,
                "value": pl.Float64,
                "date": pl.Utf8,
            },
            ["2020-01-01", "2021-12-31"],
            {"id": pl.UInt8},
            marks=pytest.mark.skipif(
                sys.version_info < (3, 9) or sys.platform == "win32",
                reason="adbc_driver_sqlite not available below Python 3.9 / on Windows",
            ),
            id="uri: adbc",
        ),
        pytest.param(
            "read_database",
            lambda path: sqlite3.connect(path, detect_types=True),
            {
                "id": pl.UInt8,
                "name": pl.Utf8,
                "value": pl.Float32,
                "date": pl.Date,
            },
            [date(2020, 1, 1), date(2021, 12, 31)],
            {"id": pl.UInt8, "value": pl.Float32},
            id="conn: sqlite3",
        ),
        pytest.param(
            "read_database",
            lambda path: create_engine(
                f"sqlite:///{path}",
                connect_args={"detect_types": sqlite3.PARSE_DECLTYPES},
            ).connect(),
            {
                "id": pl.Int64,
                "name": pl.Utf8,
                "value": pl.Float64,
                "date": pl.Date,
            },
            [date(2020, 1, 1), date(2021, 12, 31)],
            None,
            id="conn: sqlalchemy",
        ),
        pytest.param(
            "read_database",
            adbc_sqlite_connect,
            {
                "id": pl.Int64,
                "name": pl.Utf8,
                "value": pl.Float64,
                "date": pl.Utf8,
            },
            ["2020-01-01", "2021-12-31"],
            None,
            marks=pytest.mark.skipif(
                sys.version_info < (3, 9) or sys.platform == "win32",
                reason="adbc_driver_sqlite not available below Python 3.9 / on Windows",
            ),
            id="conn: adbc",
        ),
    ],
)
def test_read_database(
    read_method: str,
    engine_or_connection_init: Any,
    expected_dtypes: dict[str, pl.DataType],
    expected_dates: list[date | str],
    schema_overrides: SchemaDict | None,
    tmp_path: Path,
) -> None:
    tmp_path.mkdir(exist_ok=True)
    test_db = str(tmp_path / "test.db")
    create_temp_sqlite_db(test_db)

    if read_method == "read_database_uri":
        # instantiate the connection ourselves, using connectorx/adbc
        df = pl.read_database_uri(
            uri=f"sqlite:///{test_db}",
            query="SELECT * FROM test_data",
            engine=str(engine_or_connection_init),  # type: ignore[arg-type]
            schema_overrides=schema_overrides,
        )
    elif "adbc" in os.environ["PYTEST_CURRENT_TEST"]:
        # externally instantiated adbc connections
        with engine_or_connection_init(test_db) as conn, conn.cursor():
            df = pl.read_database(
                connection=conn,
                query="SELECT * FROM test_data",
                schema_overrides=schema_overrides,
            )
    else:
        # other user-supplied connections
        df = pl.read_database(
            connection=engine_or_connection_init(test_db),
            query="SELECT * FROM test_data WHERE name NOT LIKE '%polars%'",
            schema_overrides=schema_overrides,
        )

    assert df.schema == expected_dtypes
    assert df.shape == (2, 4)
    assert df["date"].to_list() == expected_dates


def test_read_database_mocked() -> None:
    class MockConnection:
        def __init__(self, driver: str) -> None:
            self.__class__.__module__ = driver
            self._cursor = MockCursor()

        def close(self) -> None:
            pass

        def cursor(self) -> Any:
            return self._cursor

    class MockCursor:
        def __init__(self) -> None:
            self.called: list[str] = []

        def __getattr__(self, item: str) -> Any:
            if "fetch" in item:
                self.called.append(item)
                return lambda *args, **kwargs: []
            super().__getattr__(item)  # type: ignore[misc]

        def close(self) -> Any:
            pass

        def execute(self, query: str) -> Any:
            return self

    # since we don't have access to snowflake/databricks/etc from CI we
    # mock them so we can check that we're calling the right methods
    for driver, batch_size, expected_call in (
        ("snowflake", None, "fetch_arrow_all"),
        ("snowflake", 10_000, "fetch_arrow_batches"),
        ("databricks", None, "fetchall_arrow"),
        ("databricks", 25_000, "fetchmany_arrow"),
        ("turbodbc", None, "fetchallarrow"),
        ("turbodbc", 50_000, "fetcharrowbatches"),
        ("adbc_driver_postgresql", None, "fetch_arrow_table"),
        ("adbc_driver_postgresql", 75_000, "fetch_arrow_table"),
    ):
        mc = MockConnection(driver)
        pl.read_database(
            connection=mc,
            query="SELECT * FROM test_data",
            batch_size=batch_size,
        )
        assert expected_call in mc.cursor().called


@pytest.mark.parametrize(
    ("read_method", "engine", "query", "database", "errclass", "err"),
    [
        pytest.param(
            "read_database_uri",
            "not_an_engine",
            "SELECT * FROM test_data",
            "sqlite",
            ValueError,
            "engine must be one of {'connectorx', 'adbc'}, got 'not_an_engine'",
            id="Not an available sql engine",
        ),
        pytest.param(
            "read_database_uri",
            "adbc",
            ["SELECT * FROM test_data", "SELECT * FROM test_data"],
            "sqlite",
            ValueError,
            "only a single SQL query string is accepted for adbc",
            id="Unavailable list of queries for adbc",
        ),
        pytest.param(
            "read_database_uri",
            "adbc",
            "SELECT * FROM test_data",
            "mysql",
            ImportError,
            "ADBC mysql driver not detected",
            id="Unavailable adbc driver",
        ),
        pytest.param(
            "read_database_uri",
            "adbc",
            "SELECT * FROM test_data",
            sqlite3.connect(":memory:"),
            TypeError,
            "expected connection to be a URI string",
            id="Invalid connection URI",
        ),
        pytest.param(
            "read_database",
            None,
            "SELECT * FROM imaginary_table",
            sqlite3.connect(":memory:"),
            sqlite3.OperationalError,
            "no such table: imaginary_table",
            id="Invalid read DB kwargs",
        ),
        pytest.param(
            "read_database",
            None,
            "SELECT * FROM imaginary_table",
            sys.getsizeof,  # not a connection
            TypeError,
            "Unrecognised connection .* unable to find 'execute' method",
            id="Invalid read DB kwargs",
        ),
        pytest.param(
            "read_database",
            None,
            "/* tag: misc */ INSERT INTO xyz VALUES ('polars')",
            sqlite3.connect(":memory:"),
            UnsuitableSQLError,
            "INSERT statements are not valid 'read' queries",
            id="Invalid statement type",
        ),
        pytest.param(
            "read_database",
            None,
            "DELETE FROM xyz WHERE id = 'polars'",
            sqlite3.connect(":memory:"),
            UnsuitableSQLError,
            "DELETE statements are not valid 'read' queries",
            id="Invalid statement type",
        ),
    ],
)
def test_read_database_exceptions(
    read_method: str,
    engine: DbReadEngine | None,
    query: str,
    database: Any,
    errclass: type,
    err: str,
    tmp_path: Path,
) -> None:
    if read_method == "read_database_uri":
        conn = f"{database}://test" if isinstance(database, str) else database
        params = {"uri": conn, "query": query, "engine": engine}
    else:
        params = {"connection": database, "query": query}

    read_database = getattr(pl, read_method)
    with pytest.raises(errclass, match=err):
        read_database(**params)
