from __future__ import annotations

import re
import sys
from importlib import import_module
from typing import TYPE_CHECKING, Any, Iterable, Sequence, TypedDict

from polars.convert import from_arrow
from polars.exceptions import UnsuitableSQLError
from polars.utils.deprecation import (
    deprecate_renamed_parameter,
    issue_deprecation_warning,
)

if TYPE_CHECKING:
    from types import TracebackType

    if sys.version_info >= (3, 10):
        from typing import TypeAlias
    else:
        from typing_extensions import TypeAlias
    if sys.version_info >= (3, 11):
        from typing import Self
    else:
        from typing_extensions import Self

    from polars import DataFrame
    from polars.dependencies import pyarrow as pa
    from polars.type_aliases import ConnectionOrCursor, Cursor, DbReadEngine, SchemaDict

    try:
        from sqlalchemy.sql.expression import Selectable
    except ImportError:
        Selectable: TypeAlias = Any  # type: ignore[no-redef]


class _DriverProperties_(TypedDict):
    fetch_all: str
    fetch_batches: str | None
    exact_batch_size: bool | None


_ARROW_DRIVER_REGISTRY_: dict[str, _DriverProperties_] = {
    "adbc_.*": {
        "fetch_all": "fetch_arrow_table",
        "fetch_batches": None,
        "exact_batch_size": None,
    },
    "arrow_odbc_proxy": {
        "fetch_all": "fetchall",
        "fetch_batches": "fetchmany",
        "exact_batch_size": True,
    },
    "databricks": {
        "fetch_all": "fetchall_arrow",
        "fetch_batches": "fetchmany_arrow",
        "exact_batch_size": True,
    },
    "duckdb": {
        "fetch_all": "fetch_arrow_table",
        "fetch_batches": "fetch_record_batch",
        "exact_batch_size": True,
    },
    "snowflake": {
        "fetch_all": "fetch_arrow_all",
        "fetch_batches": "fetch_arrow_batches",
        "exact_batch_size": False,
    },
    "turbodbc": {
        "fetch_all": "fetchallarrow",
        "fetch_batches": "fetcharrowbatches",
        "exact_batch_size": False,
    },
}

_INVALID_QUERY_TYPES = {
    "ALTER",
    "ANALYZE",
    "CREATE",
    "DELETE",
    "DROP",
    "INSERT",
    "REPLACE",
    "UPDATE",
    "UPSERT",
    "USE",
    "VACUUM",
}


class ODBCCursorProxy:
    """Cursor proxy for ODBC connections (requires `arrow-odbc`)."""

    def __init__(self, connection_string: str) -> None:
        self.connection_string = connection_string
        self.execute_options: dict[str, Any] = {}
        self.query: str | None = None

    def close(self) -> None:
        """Close the cursor (n/a: nothing to close)."""

    def execute(self, query: str, **execute_options: Any) -> None:
        """Execute a query (n/a: just store query for the fetch* methods)."""
        self.execute_options = execute_options
        self.query = query

    def fetchmany(
        self, batch_size: int = 10_000
    ) -> Iterable[pa.RecordBatch | pa.Table]:
        """Fetch results in batches."""
        from arrow_odbc import read_arrow_batches_from_odbc

        yield from read_arrow_batches_from_odbc(
            query=self.query,
            batch_size=batch_size,
            connection_string=self.connection_string,
            **self.execute_options,
        )

    # internally arrow-odbc always reads batches
    fetchall = fetchmany


class ConnectionExecutor:
    """Abstraction for querying databases with user-supplied connection objects."""

    # indicate that we acquired a cursor (and are therefore responsible for closing
    # it on scope-exit). note that we should never close the underlying connection,
    # or a user-supplied cursor.
    acquired_cursor: bool = False

    def __init__(self, connection: ConnectionOrCursor) -> None:
        self.driver_name = (
            "arrow_odbc_proxy"
            if isinstance(connection, ODBCCursorProxy)
            else type(connection).__module__.split(".", 1)[0].lower()
        )
        self.cursor = self._normalise_cursor(connection)
        self.result: Any = None

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        # iif we created it, close the cursor (NOT the connection)
        if self.acquired_cursor:
            self.cursor.close()

    def __repr__(self) -> str:
        return f"<{type(self).__name__} module={self.driver_name!r}>"

    def _normalise_cursor(self, conn: ConnectionOrCursor) -> Cursor:
        """Normalise a connection object such that we have the query executor."""
        if self.driver_name == "sqlalchemy" and type(conn).__name__ == "Engine":
            # sqlalchemy engine; direct use is deprecated, so prefer the connection
            self.acquired_cursor = True
            return conn.connect()  # type: ignore[union-attr]
        elif hasattr(conn, "cursor"):
            # connection has a dedicated cursor; prefer over direct execute
            cursor = cursor() if callable(cursor := conn.cursor) else cursor
            self.acquired_cursor = True
            return cursor
        elif hasattr(conn, "execute"):
            # can execute directly (given cursor, sqlalchemy connection, etc)
            return conn  # type: ignore[return-value]

        raise TypeError(
            f"Unrecognised connection {conn!r}; unable to find 'execute' method"
        )

    @staticmethod
    def _fetch_arrow(
        result: Cursor, fetch_method: str, batch_size: int | None
    ) -> Iterable[pa.RecordBatch | pa.Table]:
        """Iterate over the result set, fetching arrow data in batches."""
        yield from getattr(result, fetch_method)(batch_size)

    @staticmethod
    def _fetchall_rows(result: Cursor) -> Iterable[Sequence[Any]]:
        """Fetch row data in a single call, returning the complete result set."""
        rows = result.fetchall()
        return (
            [tuple(row) for row in rows]
            if rows and not isinstance(rows[0], (list, tuple))
            else rows
        )

    def _fetchmany_rows(
        self, result: Cursor, batch_size: int | None
    ) -> Iterable[Sequence[Any]]:
        """Fetch row data incrementally, yielding over the complete result set."""
        while True:
            rows = result.fetchmany(batch_size)
            if not rows:
                break
            elif not isinstance(rows[0], (list, tuple)):
                for row in rows:
                    yield tuple(row)
            else:
                yield from rows

    def _from_arrow(
        self, batch_size: int | None, schema_overrides: SchemaDict | None
    ) -> DataFrame | None:
        """Return resultset data in Arrow format for frame init."""
        from polars import from_arrow

        try:
            for driver, driver_properties in _ARROW_DRIVER_REGISTRY_.items():
                if re.match(f"^{driver}$", self.driver_name):
                    size = batch_size if driver_properties["exact_batch_size"] else None
                    fetch_batches = driver_properties["fetch_batches"]
                    return from_arrow(  # type: ignore[return-value]
                        data=(
                            self._fetch_arrow(self.result, fetch_batches, size)
                            if batch_size and fetch_batches is not None
                            else getattr(self.result, driver_properties["fetch_all"])()
                        ),
                        schema_overrides=schema_overrides,
                    )
        except Exception as err:
            # eg: valid turbodbc/snowflake connection, but no arrow support
            # available in the underlying driver or this connection
            arrow_not_supported = (
                "does not support Apache Arrow",
                "Apache Arrow format is not supported",
            )
            if not any(e in str(err) for e in arrow_not_supported):
                raise

        return None

    def _from_rows(
        self, batch_size: int | None, schema_overrides: SchemaDict | None
    ) -> DataFrame | None:
        """Return resultset data row-wise for frame init."""
        from polars import DataFrame

        if hasattr(self.result, "fetchall"):
            description = (
                self.result.cursor.description
                if self.driver_name == "sqlalchemy"
                else self.result.description
            )
            column_names = [desc[0] for desc in description]
            return DataFrame(
                data=(
                    self._fetchall_rows(self.result)
                    if not batch_size
                    else self._fetchmany_rows(self.result, batch_size)
                ),
                schema=column_names,
                schema_overrides=schema_overrides,
                orient="row",
            )
        return None

    def execute(
        self,
        query: str | Selectable,
        *,
        options: dict[str, Any] | None = None,
        select_queries_only: bool = True,
    ) -> Self:
        """Execute a query and reference the result set."""
        if select_queries_only and isinstance(query, str):
            q = re.search(r"\w{3,}", re.sub(r"/\*(.|[\r\n])*?\*/", "", query))
            if (query_type := "" if not q else q.group(0)) in _INVALID_QUERY_TYPES:
                raise UnsuitableSQLError(
                    f"{query_type} statements are not valid 'read' queries"
                )

        if self.driver_name == "sqlalchemy" and isinstance(query, str):
            from sqlalchemy.sql import text

            query = text(query)  # type: ignore[assignment]

        if (result := self.cursor.execute(query, **(options or {}))) is None:
            result = self.cursor  # some cursors execute in-place

        self.result = result
        return self

    def to_frame(
        self, batch_size: int | None = None, schema_overrides: SchemaDict | None = None
    ) -> DataFrame:
        """
        Convert the result set to a DataFrame.

        Wherever possible we try to return arrow-native data directly; only
        fall back to initialising with row-level data if no other option.
        """
        if self.result is None:
            raise RuntimeError("Cannot return a frame before executing a query")

        for frame_init in (
            self._from_arrow,  # init from arrow-native data (most efficient option)
            self._from_rows,  # row-wise fallback covering sqlalchemy, dbapi2, pyodbc
        ):
            frame = frame_init(batch_size=batch_size, schema_overrides=schema_overrides)
            if frame is not None:
                return frame

        raise NotImplementedError(
            f"Currently no support for {self.driver_name!r} connection {self.cursor!r}"
        )


@deprecate_renamed_parameter("connection_uri", "connection", version="0.18.9")
def read_database(  # noqa D417
    query: str | Selectable,
    connection: ConnectionOrCursor | str,
    *,
    batch_size: int | None = None,
    schema_overrides: SchemaDict | None = None,
    execute_options: dict[str, Any] | None = None,
    **kwargs: Any,
) -> DataFrame:
    """
    Read the results of a SQL query into a DataFrame, given a connection object.

    Parameters
    ----------
    query
        SQL query to execute (if using a SQLAlchemy connection object this can
        be a suitable "Selectable", otherwise it is expected to be a string).
    connection
        An instantiated connection (or cursor/client object) that the query can be
        executed against. Can also pass a valid ODBC connection string, starting with
        "Driver=", in which case the ``arrow-odbc`` package will be used to establish
        the connection and return Arrow-native data to Polars.
    batch_size
        Enable batched data fetching (internally) instead of collecting all rows at
        once; this can be helpful for minimising the peak memory used for very large
        resultsets. Note that this parameter is *not* equivalent to a "limit"; you
        will always load all rows. If supported by the backend, this value is passed
        to the underlying query execution method (note that very low values will
        typically result in poor performance as it will result in many round-trips to
        the database as the data is returned). If the backend does not support changing
        the batch size, this parameter is ignored without error.
    schema_overrides
        A dictionary mapping column names to dtypes, used to override the schema
        inferred from the query cursor or given by the incoming Arrow data (depending
        on driver/backend). This can be useful if the given types can be more precisely
        defined (for example, if you know that a given column can be declared as `u32`
        instead of `i64`).
    execute_options
        These options will be passed through into the underlying query execution method
        as kwargs. In the case of connections made using an ODBC string (which use
        `arrow-odbc`) these options are passed to the ``read_arrow_batches_from_odbc``
        method.

    Notes
    -----
    * This function supports a wide range of native database drivers (ranging from local
      databases such as SQLite to large cloud databases such as Snowflake), as well as
      generic libraries such as ADBC, SQLAlchemy and various flavours of ODBC. If the
      backend supports returning Arrow data directly then this facility will be used to
      efficiently instantiate the DataFrame; otherwise, the DataFrame is initialised
      from row-wise data.

    * Support for Arrow Flight SQL data is available via the ``adbc-driver-flightsql``
      package; see https://arrow.apache.org/adbc/current/driver/flight_sql.html for
      more details about using this driver (notable databases implementing Flight SQL
      include Dremio and InfluxDB).

    * The ``read_database_uri`` function is likely to be noticeably faster than
      ``read_database`` if you are using a SQLAlchemy or DBAPI2 connection, as
      ``connectorx`` will optimise translation of the result set into Arrow format
      in Rust, whereas these libraries will return row-wise data to Python *before*
      we can load into Arrow. Note that you can easily determine the connection's
      URI from a SQLAlchemy engine object by calling ``str(conn.engine.url)``.

    * If polars has to create a cursor from your connection in order to execute the
      query then that cursor will be automatically closed when the query completes;
      however, polars will *never* close any other connection or cursor.

    See Also
    --------
    read_database_uri : Create a DataFrame from a SQL query using a URI string.

    Examples
    --------
    Instantiate a DataFrame from a SQL query against a user-supplied connection:

    >>> df = pl.read_database(
    ...     query="SELECT * FROM test_data",
    ...     connection=user_conn,
    ...     schema_overrides={"normalised_score": pl.UInt8},
    ... )  # doctest: +SKIP

    Use a parameterised SQLAlchemy query, passing values via ``execute_options``:

    >>> df = pl.read_database(
    ...     query="SELECT * FROM test_data WHERE metric > :value",
    ...     connection=alchemy_conn,
    ...     execute_options={"parameters": {"value": 0}},
    ... )  # doctest: +SKIP

    Instantiate a DataFrame using an ODBC connection string (requires ``arrow-odbc``)
    and set upper limits on the buffer size of variadic text/binary columns:

    >>> df = pl.read_database(
    ...     query="SELECT * FROM test_data",
    ...     connection="Driver={PostgreSQL};Server=localhost;Port=5432;Database=test;Uid=usr;Pwd=",
    ...     execute_options={"max_text_size": 512, "max_binary_size": 1024},
    ... )  # doctest: +SKIP

    """  # noqa: W505
    if isinstance(connection, str):
        # check for odbc connection string
        if re.sub(r"\s", "", connection[:20]).lower().startswith("driver="):
            try:
                import arrow_odbc  # noqa: F401
            except ModuleNotFoundError:
                raise ModuleNotFoundError(
                    "use of an ODBC connection string requires the `arrow-odbc` package"
                    "\n\nPlease run: pip install arrow-odbc"
                ) from None

            connection = ODBCCursorProxy(connection)
        else:
            # otherwise looks like a call to read_database_uri
            issue_deprecation_warning(
                message="Use of a string URI with 'read_database' is deprecated; use 'read_database_uri' instead",
                version="0.19.0",
            )
            if not isinstance(query, (list, str)):
                raise TypeError(
                    f"`read_database_uri` expects one or more string queries; found {type(query)}"
                )
            return read_database_uri(
                query,
                uri=connection,
                schema_overrides=schema_overrides,
                **kwargs,
            )

    # note: can remove this check (and **kwargs) once we drop the
    # pass-through deprecation support for read_database_uri
    if kwargs:
        raise ValueError(
            f"`read_database` **kwargs only exist for passthrough to `read_database_uri`: found {kwargs!r}"
        )

    # return frame from arbitrary connections using the executor abstraction
    with ConnectionExecutor(connection) as cx:
        return cx.execute(
            query=query,
            options=execute_options,
        ).to_frame(
            batch_size=batch_size,
            schema_overrides=schema_overrides,
        )


def read_database_uri(
    query: list[str] | str,
    uri: str,
    *,
    partition_on: str | None = None,
    partition_range: tuple[int, int] | None = None,
    partition_num: int | None = None,
    protocol: str | None = None,
    engine: DbReadEngine | None = None,
    schema_overrides: SchemaDict | None = None,
) -> DataFrame:
    """
    Read the results of a SQL query into a DataFrame, given a URI.

    Parameters
    ----------
    query
        Raw SQL query (or queries).
    uri
        A connectorx or ADBC connection URI string that starts with the backend's
        driver name, for example:

        * "postgresql://user:pass@server:port/database"
        * "snowflake://user:pass@account/database/schema?warehouse=warehouse&role=role"
    partition_on
        The column on which to partition the result (connectorx).
    partition_range
        The value range of the partition column (connectorx).
    partition_num
        How many partitions to generate (connectorx).
    protocol
        Backend-specific transfer protocol directive (connectorx); see connectorx
        documentation for more details.
    engine : {'connectorx', 'adbc'}
        Selects the engine used for reading the database (defaulting to connectorx):

        * ``'connectorx'``
          Supports a range of databases, such as PostgreSQL, Redshift, MySQL, MariaDB,
          Clickhouse, Oracle, BigQuery, SQL Server, and so on. For an up-to-date list
          please see the connectorx docs:

          * https://github.com/sfu-db/connector-x#supported-sources--destinations

        * ``'adbc'``
          Currently there is limited support for this engine, with a relatively small
          number of drivers available, most of which are still in development. For
          an up-to-date list of drivers please see the ADBC docs:

          * https://arrow.apache.org/adbc/
    schema_overrides
        A dictionary mapping column names to dtypes, used to override the schema
        given in the data returned by the query.

    Notes
    -----
    For ``connectorx``, ensure that you have ``connectorx>=0.3.2``. The documentation
    is available `here <https://sfu-db.github.io/connector-x/intro.html>`_.

    For ``adbc`` you will need to have installed ``pyarrow`` and the ADBC driver associated
    with the backend you are connecting to, eg: ``adbc-driver-postgresql``.

    See Also
    --------
    read_database : Create a DataFrame from a SQL query using a connection object.

    Examples
    --------
    Create a DataFrame from a SQL query using a single thread:

    >>> uri = "postgresql://username:password@server:port/database"
    >>> query = "SELECT * FROM lineitem"
    >>> pl.read_database_uri(query, uri)  # doctest: +SKIP

    Create a DataFrame in parallel using 10 threads by automatically partitioning
    the provided SQL on the partition column:

    >>> uri = "postgresql://username:password@server:port/database"
    >>> query = "SELECT * FROM lineitem"
    >>> pl.read_database_uri(
    ...     query,
    ...     uri,
    ...     partition_on="partition_col",
    ...     partition_num=10,
    ...     engine="connectorx",
    ... )  # doctest: +SKIP

    Create a DataFrame in parallel using 2 threads by explicitly providing two
    SQL queries:

    >>> uri = "postgresql://username:password@server:port/database"
    >>> queries = [
    ...     "SELECT * FROM lineitem WHERE partition_col <= 10",
    ...     "SELECT * FROM lineitem WHERE partition_col > 10",
    ... ]
    >>> pl.read_database_uri(queries, uri, engine="connectorx")  # doctest: +SKIP

    Read data from Snowflake using the ADBC driver:

    >>> df = pl.read_database_uri(
    ...     "SELECT * FROM test_table",
    ...     "snowflake://user:pass@company-org/testdb/public?warehouse=test&role=myrole",
    ...     engine="adbc",
    ... )  # doctest: +SKIP

    """  # noqa: W505
    if not isinstance(uri, str):
        raise TypeError(
            f"expected connection to be a URI string; found {type(uri).__name__!r}"
        )
    elif engine is None:
        engine = "connectorx"

    if engine == "connectorx":
        return _read_sql_connectorx(
            query,
            connection_uri=uri,
            partition_on=partition_on,
            partition_range=partition_range,
            partition_num=partition_num,
            protocol=protocol,
            schema_overrides=schema_overrides,
        )
    elif engine == "adbc":
        if not isinstance(query, str):
            raise ValueError("only a single SQL query string is accepted for adbc")
        return _read_sql_adbc(query, uri, schema_overrides)
    else:
        raise ValueError(
            f"engine must be one of {{'connectorx', 'adbc'}}, got {engine!r}"
        )


def _read_sql_connectorx(
    query: str | list[str],
    connection_uri: str,
    partition_on: str | None = None,
    partition_range: tuple[int, int] | None = None,
    partition_num: int | None = None,
    protocol: str | None = None,
    schema_overrides: SchemaDict | None = None,
) -> DataFrame:
    try:
        import connectorx as cx
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "connectorx is not installed"
            "\n\nPlease run: pip install connectorx>=0.3.2"
        ) from None

    tbl = cx.read_sql(
        conn=connection_uri,
        query=query,
        return_type="arrow2",
        partition_on=partition_on,
        partition_range=partition_range,
        partition_num=partition_num,
        protocol=protocol,
    )
    return from_arrow(tbl, schema_overrides=schema_overrides)  # type: ignore[return-value]


def _read_sql_adbc(
    query: str, connection_uri: str, schema_overrides: SchemaDict | None
) -> DataFrame:
    with _open_adbc_connection(connection_uri) as conn, conn.cursor() as cursor:
        cursor.execute(query)
        tbl = cursor.fetch_arrow_table()
    return from_arrow(tbl, schema_overrides=schema_overrides)  # type: ignore[return-value]


def _open_adbc_connection(connection_uri: str) -> Any:
    driver_name = connection_uri.split(":", 1)[0].lower()

    # map uri prefix to module when not 1:1
    module_suffix_map: dict[str, str] = {
        "postgres": "postgresql",
    }
    try:
        module_suffix = module_suffix_map.get(driver_name, driver_name)
        module_name = f"adbc_driver_{module_suffix}.dbapi"
        import_module(module_name)
        adbc_driver = sys.modules[module_name]
    except ImportError:
        raise ModuleNotFoundError(
            f"ADBC {driver_name} driver not detected"
            f"\n\nIf ADBC supports this database, please run: pip install adbc-driver-{driver_name} pyarrow"
        ) from None

    # some backends require the driver name to be stripped from the URI
    if driver_name in ("sqlite", "snowflake"):
        connection_uri = re.sub(f"^{driver_name}:/{{,3}}", "", connection_uri)

    return adbc_driver.connect(connection_uri)
