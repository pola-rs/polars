from __future__ import annotations

import os
import sqlite3
import sys
from contextlib import suppress
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import pytest
from sqlalchemy import Integer, MetaData, Table, create_engine, func, select
from sqlalchemy.sql.expression import cast as alchemy_cast

import polars as pl
from polars.exceptions import UnsuitableSQLError
from polars.testing import assert_frame_equal

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
        "batch_size",
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
            None,
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
            None,
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
            None,
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
            None,
            marks=pytest.mark.skipif(
                sys.version_info < (3, 9) or sys.platform == "win32",
                reason="adbc_driver_sqlite not available below Python 3.9 / on Windows",
            ),
            id="conn: adbc (fetchall)",
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
            1,
            marks=pytest.mark.skipif(
                sys.version_info < (3, 9) or sys.platform == "win32",
                reason="adbc_driver_sqlite not available below Python 3.9 / on Windows",
            ),
            id="conn: adbc (batched)",
        ),
    ],
)
def test_read_database(
    read_method: str,
    engine_or_connection_init: Any,
    expected_dtypes: dict[str, pl.DataType],
    expected_dates: list[date | str],
    schema_overrides: SchemaDict | None,
    batch_size: int | None,
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
                batch_size=batch_size,
            )
    else:
        # other user-supplied connections
        df = pl.read_database(
            connection=engine_or_connection_init(test_db),
            query="SELECT * FROM test_data WHERE name NOT LIKE '%polars%'",
            schema_overrides=schema_overrides,
            batch_size=batch_size,
        )

    assert df.schema == expected_dtypes
    assert df.shape == (2, 4)
    assert df["date"].to_list() == expected_dates


def test_read_database_alchemy_selectable(tmp_path: Path) -> None:
    # setup underlying test data
    tmp_path.mkdir(exist_ok=True)
    create_temp_sqlite_db(test_db := str(tmp_path / "test.db"))
    conn = create_engine(f"sqlite:///{test_db}")
    t = Table("test_data", MetaData(), autoload_with=conn)

    # establish sqlalchemy "selectable" and validate usage
    selectable_query = select(
        alchemy_cast(func.strftime("%Y", t.c.date), Integer).label("year"),
        t.c.name,
        t.c.value,
    ).where(t.c.value < 0)

    assert_frame_equal(
        pl.read_database(selectable_query, connection=conn.connect()),
        pl.DataFrame({"year": [2021], "name": ["other"], "value": [-99.5]}),
    )


def test_read_database_parameterisd(tmp_path: Path) -> None:
    # setup underlying test data
    tmp_path.mkdir(exist_ok=True)
    create_temp_sqlite_db(test_db := str(tmp_path / "test.db"))
    conn = create_engine(f"sqlite:///{test_db}")

    # establish a parameterised query and validate usage
    parameterised_query = """
        SELECT CAST(STRFTIME('%Y',"date") AS INT) as "year", name, value
        FROM test_data WHERE value < :n
    """
    assert_frame_equal(
        pl.read_database(
            parameterised_query,
            connection=conn.connect(),
            execute_options={"parameters": {"n": 0}},
        ),
        pl.DataFrame({"year": [2021], "name": ["other"], "value": [-99.5]}),
    )


def test_read_database_mocked() -> None:
    arr = pl.DataFrame({"x": [1, 2, 3], "y": ["aa", "bb", "cc"]}).to_arrow()

    class MockConnection:
        def __init__(self, driver: str, batch_size: int | None = None) -> None:
            self.__class__.__module__ = driver
            self._cursor = MockCursor(batched=batch_size is not None)

        def close(self) -> None:
            pass

        def cursor(self) -> Any:
            return self._cursor

    class MockCursor:
        def __init__(self, batched: bool) -> None:
            self.called: list[str] = []
            self.batched = batched

        def __getattr__(self, item: str) -> Any:
            if "fetch" in item:
                res = (
                    (lambda *args, **kwargs: (arr for _ in range(1)))
                    if self.batched
                    else (lambda *args, **kwargs: arr)
                )
                self.called.append(item)
                return res
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
        mc = MockConnection(driver, batch_size)
        df = pl.read_database(
            connection=mc,
            query="SELECT * FROM test_data",
            batch_size=batch_size,
        )
        assert expected_call in mc.cursor().called
        assert df.rows() == [(1, "aa"), (2, "bb"), (3, "cc")]


class ExceptionTestParams(NamedTuple):
    """Clarify exception testing params."""

    read_method: str
    query: str | list[str]
    protocol: Any
    errclass: type[Exception]
    errmsg: str
    engine: str | None = None
    execute_options: dict[str, Any] | None = None
    kwargs: dict[str, Any] | None = None


@pytest.mark.parametrize(
    (
        "read_method",
        "query",
        "protocol",
        "errclass",
        "errmsg",
        "engine",
        "execute_options",
        "kwargs",
    ),
    [
        pytest.param(
            *ExceptionTestParams(
                read_method="read_database_uri",
                query="SELECT * FROM test_data",
                protocol="sqlite",
                errclass=ValueError,
                errmsg="engine must be one of {'connectorx', 'adbc'}, got 'not_an_engine'",
                engine="not_an_engine",
            ),
            id="Not an available sql engine",
        ),
        pytest.param(
            *ExceptionTestParams(
                read_method="read_database_uri",
                query=["SELECT * FROM test_data", "SELECT * FROM test_data"],
                protocol="sqlite",
                errclass=ValueError,
                errmsg="only a single SQL query string is accepted for adbc",
                engine="adbc",
            ),
            id="Unavailable list of queries for adbc",
        ),
        pytest.param(
            *ExceptionTestParams(
                read_method="read_database_uri",
                query="SELECT * FROM test_data",
                protocol="mysql",
                errclass=ImportError,
                errmsg="ADBC mysql driver not detected",
                engine="adbc",
            ),
            id="Unavailable adbc driver",
        ),
        pytest.param(
            *ExceptionTestParams(
                read_method="read_database_uri",
                query="SELECT * FROM test_data",
                protocol=sqlite3.connect(":memory:"),
                errclass=TypeError,
                errmsg="expected connection to be a URI string",
                engine="adbc",
            ),
            id="Invalid connection URI",
        ),
        pytest.param(
            *ExceptionTestParams(
                read_method="read_database",
                query="SELECT * FROM imaginary_table",
                protocol=sqlite3.connect(":memory:"),
                errclass=sqlite3.OperationalError,
                errmsg="no such table: imaginary_table",
            ),
            id="Invalid query (unrecognised table name)",
        ),
        pytest.param(
            *ExceptionTestParams(
                read_method="read_database",
                query="SELECT * FROM imaginary_table",
                protocol=sys.getsizeof,  # not a connection
                errclass=TypeError,
                errmsg="Unrecognised connection .* unable to find 'execute' method",
            ),
            id="Invalid read DB kwargs",
        ),
        pytest.param(
            *ExceptionTestParams(
                read_method="read_database",
                query="/* tag: misc */ INSERT INTO xyz VALUES ('polars')",
                protocol=sqlite3.connect(":memory:"),
                errclass=UnsuitableSQLError,
                errmsg="INSERT statements are not valid 'read' queries",
            ),
            id="Invalid statement type",
        ),
        pytest.param(
            *ExceptionTestParams(
                read_method="read_database",
                query="DELETE FROM xyz WHERE id = 'polars'",
                protocol=sqlite3.connect(":memory:"),
                errclass=UnsuitableSQLError,
                errmsg="DELETE statements are not valid 'read' queries",
            ),
            id="Invalid statement type",
        ),
        pytest.param(
            *ExceptionTestParams(
                read_method="read_database",
                engine="adbc",
                query="SELECT * FROM test_data",
                protocol=sqlite3.connect(":memory:"),
                errclass=TypeError,
                errmsg="takes no keyword arguments",
                execute_options={"parameters": {"n": 0}},
            ),
            id="Invalid execute_options",
        ),
        pytest.param(
            *ExceptionTestParams(
                read_method="read_database",
                engine="adbc",
                query="SELECT * FROM test_data",
                protocol=sqlite3.connect(":memory:"),
                errclass=ValueError,
                errmsg=r"`read_database` \*\*kwargs only exist for passthrough to `read_database_uri`",
                kwargs={"partition_on": "id"},
            ),
            id="Invalid kwargs",
        ),
    ],
)
def test_read_database_exceptions(
    read_method: str,
    query: str,
    protocol: Any,
    errclass: type[Exception],
    errmsg: str,
    engine: DbReadEngine | None,
    execute_options: dict[str, Any] | None,
    kwargs: dict[str, Any] | None,
    tmp_path: Path,
) -> None:
    if read_method == "read_database_uri":
        conn = f"{protocol}://test" if isinstance(protocol, str) else protocol
        params = {"uri": conn, "query": query, "engine": engine}
    else:
        params = {"connection": protocol, "query": query}
        if execute_options:
            params["execute_options"] = execute_options
        if kwargs is not None:
            params.update(kwargs)

    read_database = getattr(pl, read_method)
    with pytest.raises(errclass, match=errmsg):
        read_database(**params)
