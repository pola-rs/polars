from __future__ import annotations

import os
import sqlite3
import sys
from contextlib import suppress
from datetime import date
from pathlib import Path
from types import GeneratorType
from typing import TYPE_CHECKING, Any, NamedTuple

import pytest
from sqlalchemy import Integer, MetaData, Table, create_engine, func, select
from sqlalchemy.sql.expression import cast as alchemy_cast

import polars as pl
from polars.exceptions import UnsuitableSQLError
from polars.io.database import _ARROW_DRIVER_REGISTRY_
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    import pyarrow as pa

    from polars.type_aliases import DbReadEngine, SchemaDefinition, SchemaDict


def adbc_sqlite_connect(*args: Any, **kwargs: Any) -> Any:
    with suppress(ModuleNotFoundError):  # not available on 3.8/windows
        from adbc_driver_sqlite.dbapi import connect

        return connect(*args, **kwargs)


def create_temp_sqlite_db(test_db: str) -> None:
    Path(test_db).unlink(missing_ok=True)

    def convert_date(val: bytes) -> date:
        """Convert ISO 8601 date to datetime.date object."""
        return date.fromisoformat(val.decode())

    sqlite3.register_converter("date", convert_date)

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


class DatabaseReadTestParams(NamedTuple):
    """Clarify read test params."""

    read_method: str
    connect_using: Any
    expected_dtypes: SchemaDefinition
    expected_dates: list[date | str]
    schema_overrides: SchemaDict | None = None
    batch_size: int | None = None


class ExceptionTestParams(NamedTuple):
    """Clarify exception test params."""

    read_method: str
    query: str | list[str]
    protocol: Any
    errclass: type[Exception]
    errmsg: str
    engine: str | None = None
    execute_options: dict[str, Any] | None = None
    kwargs: dict[str, Any] | None = None


class MockConnection:
    """Mock connection class for databases we can't test in CI."""

    def __init__(
        self,
        driver: str,
        batch_size: int | None,
        test_data: pa.Table,
        repeat_batch_calls: bool,
    ) -> None:
        self.__class__.__module__ = driver
        self._cursor = MockCursor(
            repeat_batch_calls=repeat_batch_calls,
            batched=(batch_size is not None),
            test_data=test_data,
        )

    def close(self) -> None:  # noqa: D102
        pass

    def cursor(self) -> Any:  # noqa: D102
        return self._cursor


class MockCursor:
    """Mock cursor class for databases we can't test in CI."""

    def __init__(
        self,
        batched: bool,
        test_data: pa.Table,
        repeat_batch_calls: bool,
    ) -> None:
        self.resultset = MockResultSet(test_data, batched, repeat_batch_calls)
        self.called: list[str] = []
        self.batched = batched
        self.n_calls = 1

    def __getattr__(self, item: str) -> Any:
        if "fetch" in item:
            self.called.append(item)
            return self.resultset
        super().__getattr__(item)  # type: ignore[misc]

    def close(self) -> Any:  # noqa: D102
        pass

    def execute(self, query: str) -> Any:  # noqa: D102
        return self


class MockResultSet:
    """Mock resultset class for databases we can't test in CI."""

    def __init__(
        self, test_data: pa.Table, batched: bool, repeat_batch_calls: bool = False
    ):
        self.test_data = test_data
        self.repeat_batched_calls = repeat_batch_calls
        self.batched = batched
        self.n_calls = 1

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # noqa: D102
        if self.repeat_batched_calls:
            res = self.test_data[: None if self.n_calls else 0]
            self.n_calls -= 1
        else:
            res = iter((self.test_data,))
        return res


@pytest.mark.write_disk()
@pytest.mark.parametrize(
    (
        "read_method",
        "connect_using",
        "expected_dtypes",
        "expected_dates",
        "schema_overrides",
        "batch_size",
    ),
    [
        pytest.param(
            *DatabaseReadTestParams(
                read_method="read_database_uri",
                connect_using="connectorx",
                expected_dtypes={
                    "id": pl.UInt8,
                    "name": pl.Utf8,
                    "value": pl.Float64,
                    "date": pl.Date,
                },
                expected_dates=[date(2020, 1, 1), date(2021, 12, 31)],
                schema_overrides={"id": pl.UInt8},
            ),
            id="uri: connectorx",
            marks=pytest.mark.skipif(
                sys.version_info > (3, 11),
                reason="connectorx cannot be installed on Python 3.12 yet.",
            ),
        ),
        pytest.param(
            *DatabaseReadTestParams(
                read_method="read_database_uri",
                connect_using="adbc",
                expected_dtypes={
                    "id": pl.UInt8,
                    "name": pl.Utf8,
                    "value": pl.Float64,
                    "date": pl.Utf8,
                },
                expected_dates=["2020-01-01", "2021-12-31"],
                schema_overrides={"id": pl.UInt8},
            ),
            marks=pytest.mark.skipif(
                sys.version_info < (3, 9) or sys.platform == "win32",
                reason="adbc_driver_sqlite not available below Python 3.9 / on Windows",
            ),
            id="uri: adbc",
        ),
        pytest.param(
            *DatabaseReadTestParams(
                read_method="read_database",
                connect_using=lambda path: sqlite3.connect(path, detect_types=True),
                expected_dtypes={
                    "id": pl.UInt8,
                    "name": pl.Utf8,
                    "value": pl.Float32,
                    "date": pl.Date,
                },
                expected_dates=[date(2020, 1, 1), date(2021, 12, 31)],
                schema_overrides={"id": pl.UInt8, "value": pl.Float32},
            ),
            id="conn: sqlite3",
        ),
        pytest.param(
            *DatabaseReadTestParams(
                read_method="read_database",
                connect_using=lambda path: sqlite3.connect(path, detect_types=True),
                expected_dtypes={
                    "id": pl.Int32,
                    "name": pl.Utf8,
                    "value": pl.Float32,
                    "date": pl.Date,
                },
                expected_dates=[date(2020, 1, 1), date(2021, 12, 31)],
                schema_overrides={"id": pl.Int32, "value": pl.Float32},
                batch_size=1,
            ),
            id="conn: sqlite3",
        ),
        pytest.param(
            *DatabaseReadTestParams(
                read_method="read_database",
                connect_using=lambda path: create_engine(
                    f"sqlite:///{path}",
                    connect_args={"detect_types": sqlite3.PARSE_DECLTYPES},
                ).connect(),
                expected_dtypes={
                    "id": pl.Int64,
                    "name": pl.Utf8,
                    "value": pl.Float64,
                    "date": pl.Date,
                },
                expected_dates=[date(2020, 1, 1), date(2021, 12, 31)],
            ),
            id="conn: sqlalchemy",
        ),
        pytest.param(
            *DatabaseReadTestParams(
                read_method="read_database",
                connect_using=adbc_sqlite_connect,
                expected_dtypes={
                    "id": pl.Int64,
                    "name": pl.Utf8,
                    "value": pl.Float64,
                    "date": pl.Utf8,
                },
                expected_dates=["2020-01-01", "2021-12-31"],
            ),
            marks=pytest.mark.skipif(
                sys.version_info < (3, 9) or sys.platform == "win32",
                reason="adbc_driver_sqlite not available below Python 3.9 / on Windows",
            ),
            id="conn: adbc (fetchall)",
        ),
        pytest.param(
            *DatabaseReadTestParams(
                read_method="read_database",
                connect_using=adbc_sqlite_connect,
                expected_dtypes={
                    "id": pl.Int64,
                    "name": pl.Utf8,
                    "value": pl.Float64,
                    "date": pl.Utf8,
                },
                expected_dates=["2020-01-01", "2021-12-31"],
                batch_size=1,
            ),
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
    connect_using: Any,
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
            engine=str(connect_using),  # type: ignore[arg-type]
            schema_overrides=schema_overrides,
        )
    elif "adbc" in os.environ["PYTEST_CURRENT_TEST"]:
        # externally instantiated adbc connections
        with connect_using(test_db) as conn, conn.cursor():
            df = pl.read_database(
                connection=conn,
                query="SELECT * FROM test_data",
                schema_overrides=schema_overrides,
                batch_size=batch_size,
            )
    else:
        # other user-supplied connections
        df = pl.read_database(
            connection=connect_using(test_db),
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

    # establish parameterised queries and validate usage
    query = """
        SELECT CAST(STRFTIME('%Y',"date") AS INT) as "year", name, value
        FROM test_data
        WHERE value < {n}
    """
    for param, param_value in (
        (":n", {"n": 0}),
        ("?", (0,)),
        ("?", [0]),
    ):
        assert_frame_equal(
            pl.read_database(
                query.format(n=param),
                connection=conn.connect(),
                execute_options={"parameters": param_value},
            ),
            pl.DataFrame({"year": [2021], "name": ["other"], "value": [-99.5]}),
        )


@pytest.mark.parametrize(
    ("driver", "batch_size", "iter_batches", "expected_call"),
    [
        ("snowflake", None, False, "fetch_arrow_all"),
        ("snowflake", 10_000, False, "fetch_arrow_all"),
        ("snowflake", 10_000, True, "fetch_arrow_batches"),
        ("databricks", None, False, "fetchall_arrow"),
        ("databricks", 25_000, False, "fetchall_arrow"),
        ("databricks", 25_000, True, "fetchmany_arrow"),
        ("turbodbc", None, False, "fetchallarrow"),
        ("turbodbc", 50_000, False, "fetchallarrow"),
        ("turbodbc", 50_000, True, "fetcharrowbatches"),
        ("adbc_driver_postgresql", None, False, "fetch_arrow_table"),
        ("adbc_driver_postgresql", 75_000, False, "fetch_arrow_table"),
        ("adbc_driver_postgresql", 75_000, True, "fetch_arrow_table"),
    ],
)
def test_read_database_mocked(
    driver: str, batch_size: int | None, iter_batches: bool, expected_call: str
) -> None:
    # since we don't have access to snowflake/databricks/etc from CI we
    # mock them so we can check that we're calling the expected methods
    arrow = pl.DataFrame({"x": [1, 2, 3], "y": ["aa", "bb", "cc"]}).to_arrow()
    mc = MockConnection(
        driver,
        batch_size,
        test_data=arrow,
        repeat_batch_calls=_ARROW_DRIVER_REGISTRY_.get(driver, {}).get(  # type: ignore[call-overload]
            "repeat_batch_calls", False
        ),
    )
    res = pl.read_database(  # type: ignore[call-overload]
        query="SELECT * FROM test_data",
        connection=mc,
        iter_batches=iter_batches,
        batch_size=batch_size,
    )
    if iter_batches:
        assert isinstance(res, GeneratorType)
        res = pl.concat(res)

    assert expected_call in mc.cursor().called
    assert res.rows() == [(1, "aa"), (2, "bb"), (3, "cc")]


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
                query="SELECT * FROM test_data",
                protocol=sqlite3.connect(":memory:"),
                errclass=ValueError,
                errmsg=r"`read_database` \*\*kwargs only exist for passthrough to `read_database_uri`",
                kwargs={"partition_on": "id"},
            ),
            id="Invalid kwargs",
        ),
        pytest.param(
            *ExceptionTestParams(
                read_method="read_database",
                query="SELECT * FROM sqlite_master",
                protocol=sqlite3.connect(":memory:"),
                errclass=ValueError,
                kwargs={"iter_batches": True},
                errmsg="Cannot set `iter_batches` without also setting a non-zero `batch_size`",
            ),
            id="Invalid batch_size",
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


@pytest.mark.skipif(
    sys.version_info > (3, 11),
    reason="connectorx cannot be installed on Python 3.12 yet.",
)
@pytest.mark.parametrize(
    "uri",
    [
        "fakedb://123:456@account/database/schema?warehouse=warehouse&role=role",
        "fakedb://my#%us3r:p433w0rd@not_a_real_host:9999/database",
    ],
)
def test_read_database_cx_credentials(uri: str) -> None:
    # check that we masked the potential credentials leak; this isn't really
    # our responsibility (ideally would be handled by connectorx), but we
    # can reasonably mitigate the issue.
    with pytest.raises(BaseException, match=r"fakedb://\*\*\*:\*\*\*@\w+"):
        pl.read_database_uri("SELECT * FROM data", uri=uri)
