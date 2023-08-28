from __future__ import annotations

import re
import sys
from importlib import import_module
from typing import TYPE_CHECKING, Any, Iterable, Sequence, TypedDict

from polars.convert import from_arrow
from polars.utils.deprecation import (
    deprecate_renamed_parameter,
    issue_deprecation_warning,
)

if TYPE_CHECKING:
    from types import TracebackType

    if sys.version_info >= (3, 11):
        from typing import Self
    else:
        from typing_extensions import Self

    from polars import DataFrame
    from polars.dependencies import pyarrow as pa
    from polars.type_aliases import ConnectionOrCursor, Cursor, DbReadEngine


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
    "databricks": {
        "fetch_all": "fetchall_arrow",
        "fetch_batches": "fetchmany_arrow",
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


class ConnectionExecutor:
    """Abstraction for querying databases with user-supplied connection objects."""

    acquired_cursor = False

    def __init__(self, connection: ConnectionOrCursor) -> None:
        self.driver = type(connection).__module__.split(".", 1)[0].lower()
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
        return f"<{type(self).__name__} module={self.driver!r}>"

    def _normalise_cursor(self, conn: ConnectionOrCursor) -> Cursor:
        """Normalise a connection object such that we have the query executor."""
        if self.driver == "sqlalchemy" and type(conn).__name__ == "Engine":
            # sqlalchemy engine; direct use is deprecated, so get the connection
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
        size = (batch_size,) if batch_size else ()
        while result:  # type: ignore[truthy-bool]
            result = getattr(result, fetch_method)(*size)
            yield result

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

    def _from_arrow(self, batch_size: int | None) -> DataFrame | None:
        """Return resultset data in Arrow format for frame init."""
        from polars import DataFrame

        for driver, driver_properties in _ARROW_DRIVER_REGISTRY_.items():
            if re.match(f"^{driver}$", self.driver):
                size = batch_size if driver_properties["exact_batch_size"] else None
                fetch_batches = driver_properties["fetch_batches"]
                return DataFrame(
                    self._fetch_arrow(self.result, fetch_batches, size)
                    if batch_size and fetch_batches is not None
                    else getattr(self.result, driver_properties["fetch_all"])()
                )

        if self.driver == "duckdb":
            exec_kwargs = {"rows_per_batch": batch_size} if batch_size else {}
            return DataFrame(self.result.arrow(**exec_kwargs))

        return None

    def _from_rows(self, batch_size: int | None) -> DataFrame | None:
        """Return resultset data row-wise for frame init."""
        from polars import DataFrame

        if hasattr(self.result, "fetchall"):
            description = (
                self.result.cursor.description
                if self.driver == "sqlalchemy"
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
                orient="row",
            )
        return None

    def execute(self, query: str) -> Self:
        """Execute a query and reference the result set data."""
        if self.driver == "sqlalchemy":
            from sqlalchemy.sql import text

            query = text(query)  # type: ignore[assignment]

        if (result := self.cursor.execute(query)) is None:
            result = self.cursor  # some cursors execute in-place

        self.result = result
        return self

    def to_frame(self, batch_size: int | None = None) -> DataFrame:
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
            frame = frame_init(batch_size)
            if frame is not None:
                return frame

        raise NotImplementedError(
            f"Currently no support for {self.driver!r} connection {self.cursor!r}"
        )


@deprecate_renamed_parameter("connection_uri", "connection", version="0.18.9")
def read_database(  # noqa: D417
    query: str,
    connection: ConnectionOrCursor,
    batch_size: int | None = None,
    **kwargs: Any,
) -> DataFrame:
    """
    Read the results of a SQL query into a DataFrame, given a connection object.

    Parameters
    ----------
    query
        String SQL query to execute.
    connection
        An instantiated connection (or cursor/client object) that the query can be
        executed against.
    batch_size
        The number of rows to fetch each time as data is collected; if this option is
        supported by the backend it will be passed to the underlying query execution
        method (if the backend does not have such support it is ignored without error).

    Notes
    -----
    This function supports a wide range of native database drivers (ranging from SQLite
    to Snowflake), as well as libraries such as ADBC, SQLAlchemy and various flavours
    of ODBC. If the backend supports returning Arrow data directly then this facility
    will be used to efficiently instantiate the DataFrame; otherwise, the DataFrame
    is initialised from row-wise data.

    See Also
    --------
    read_database_uri : Create a DataFrame from a SQL query using a URI string.

    Examples
    --------
    Instantiate a DataFrame from a SQL query against a user-supplied connection:

    >>> df = pl.read_database(
    ...     query="SELECT * FROM test_data",
    ...     connection=conn,
    ... )  # doctest: +SKIP

    """
    if isinstance(connection, str):
        issue_deprecation_warning(
            message="Use of a string URI with 'read_database' is deprecated; use 'read_database_uri' instead",
            version="0.19.0",
        )
        return read_database_uri(query, uri=connection, **kwargs)
    elif kwargs:
        raise ValueError(
            f"'read_database' does not support arbitrary **kwargs: found {kwargs!r}"
        )

    with ConnectionExecutor(connection) as cx:
        return cx.execute(query).to_frame(batch_size)


def read_database_uri(
    query: list[str] | str,
    uri: str,
    *,
    partition_on: str | None = None,
    partition_range: tuple[int, int] | None = None,
    partition_num: int | None = None,
    protocol: str | None = None,
    engine: DbReadEngine | None = None,
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

    Notes
    -----
    For ``connectorx``, ensure that you have ``connectorx>=0.3.1``. The documentation
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
        )
    elif engine == "adbc":
        if not isinstance(query, str):
            raise ValueError("only a single SQL query string is accepted for adbc")
        return _read_sql_adbc(query, uri)
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
) -> DataFrame:
    try:
        import connectorx as cx
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "connectorx is not installed"
            "\n\nPlease run `pip install connectorx>=0.3.2`."
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
    return from_arrow(tbl)  # type: ignore[return-value]


def _read_sql_adbc(query: str, connection_uri: str) -> DataFrame:
    with _open_adbc_connection(connection_uri) as conn, conn.cursor() as cursor:
        cursor.execute(query)
        tbl = cursor.fetch_arrow_table()
    return from_arrow(tbl)  # type: ignore[return-value]


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
            "\n\nIf ADBC supports this database, please run:"
            " `pip install adbc-driver-{driver_name} pyarrow`"
        ) from None

    # some backends require the driver name to be stripped from the URI
    if driver_name in ("sqlite", "snowflake"):
        connection_uri = re.sub(f"^{driver_name}:/{{,3}}", "", connection_uri)

    return adbc_driver.connect(connection_uri)
