from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, cast

import polars.internals as pli
from polars.convert import from_arrow
from polars.utils import deprecate_nonkeyword_arguments

if TYPE_CHECKING:
    from polars.internals.type_aliases import DbReadEngine


@deprecate_nonkeyword_arguments()
def read_database(
    sql: list[str] | str,
    connection_uri: str,
    partition_on: str | None = None,
    partition_range: tuple[int, int] | None = None,
    partition_num: int | None = None,
    protocol: str | None = None,
    *,
    engine: DbReadEngine = "connectorx",
) -> pli.DataFrame:
    """
    Read a SQL query into a DataFrame.

    Parameters
    ----------
    sql
        Raw SQL query (or queries).
    connection_uri
        A connectorx compatible connection uri, for example

        * "postgresql://username:password@server:port/database"
    partition_on
        The column on which to partition the result.
    partition_range
        The value range of the partition column.
    partition_num
        How many partitions to generate.
    protocol
        Backend-specific transfer protocol directive; see connectorx documentation for
        details.
    engine : {'connectorx', 'adbc'}
        Select the engine used for reading the data.

        * ``'connectorx'``
          Supports a range of databases, such as PostgreSQL, Redshift, MySQL, MariaDB,
          Clickhouse, Oracle, BigQuery, SQL Server, and so on. For an up-to-date list
          please see the connectorx docs:

          * https://github.com/sfu-db/connector-x#supported-sources--destinations
        * ``'adbc'``
          Currently just PostgreSQL and SQLite are supported and these are both in
          development. When flight_sql is further in development and widely adopted
          this will make this significantly better. For an up-to-date list
          please see the adbc docs:

          * https://arrow.apache.org/adbc/0.1.0/driver/cpp/index.html

    Notes
    -----
    Make sure to install connectorx>=0.3.1. Read the documentation
    `here <https://sfu-db.github.io/connector-x/intro.html>`_.

    Examples
    --------
    Read a DataFrame from a SQL query using a single thread:

    >>> uri = "postgresql://username:password@server:port/database"
    >>> query = "SELECT * FROM lineitem"
    >>> pl.read_database(query, uri)  # doctest: +SKIP

    Read a DataFrame in parallel using 10 threads by automatically partitioning the
    provided SQL on the partition column:

    >>> uri = "postgresql://username:password@server:port/database"
    >>> query = "SELECT * FROM lineitem"
    >>> pl.read_database(
    ...     query, uri, partition_on="partition_col", partition_num=10
    ... )  # doctest: +SKIP

    Read a DataFrame in parallel using 2 threads by explicitly providing two SQL
    queries:

    >>> uri = "postgresql://username:password@server:port/database"
    >>> queries = [
    ...     "SELECT * FROM lineitem WHERE partition_col <= 10",
    ...     "SELECT * FROM lineitem WHERE partition_col > 10",
    ... ]
    >>> pl.read_database(queries, uri)  # doctest: +SKIP

    """
    if engine == "connectorx":
        return _read_sql_connectorx(
            connection_uri=connection_uri,
            sql=sql,
            partition_on=partition_on,
            partition_range=partition_range,
            partition_num=partition_num,
            protocol=protocol,
        )
    elif engine == "adbc":
        if isinstance(sql, str):
            return _read_sql_adbc(sql=sql, connection_uri=connection_uri)
        else:
            raise ValueError("Only a single SQL query string is accepted for adbc.")
    else:
        raise ValueError("Engine is not implemented, try either connectorx or adbc.")


@deprecate_nonkeyword_arguments()
def read_sql(
    sql: list[str] | str,
    connection_uri: str,
    partition_on: str | None = None,
    partition_range: tuple[int, int] | None = None,
    partition_num: int | None = None,
    protocol: str | None = None,
    *,
    engine: DbReadEngine = "connectorx",
) -> pli.DataFrame:
    """
    Read a SQL query into a DataFrame.

    Supports a range of databases, such as PostgreSQL, Redshift, MySQL, MariaDB,
    Clickhouse, Oracle, BigQuery, SQL Server, and so on. For an up-to-date list
    please see the connectorx docs:

    * https://github.com/sfu-db/connector-x#supported-sources--destinations

    Parameters
    ----------
    sql
        Raw SQL query (or queries).
    connection_uri
        A connectorx compatible connection uri, for example

        * "postgresql://username:password@server:port/database"
    partition_on
        The column on which to partition the result.
    partition_range
        The value range of the partition column.
    partition_num
        How many partitions to generate.
    protocol
        Backend-specific transfer protocol directive; see connectorx documentation for
        details.
    engine : {'connectorx', 'adbc'}
        Select the engine used for reading the data from sql.

    .. deprecated:: 0.16.10
        Use ``read_database`` instead.

    Notes
    -----
    Make sure to install connectorx>=0.3.1. Read the documentation
    `here <https://sfu-db.github.io/connector-x/intro.html>`_.

    Examples
    --------
    Read a DataFrame from a SQL query using a single thread:

    >>> uri = "postgresql://username:password@server:port/database"
    >>> query = "SELECT * FROM lineitem"
    >>> pl.read_sql(query, uri)  # doctest: +SKIP

    Read a DataFrame in parallel using 10 threads by automatically partitioning the
    provided SQL on the partition column:

    >>> uri = "postgresql://username:password@server:port/database"
    >>> query = "SELECT * FROM lineitem"
    >>> pl.read_sql(
    ...     query, uri, partition_on="partition_col", partition_num=10
    ... )  # doctest: +SKIP

    Read a DataFrame in parallel using 2 threads by explicitly providing two SQL
    queries:

    >>> uri = "postgresql://username:password@server:port/database"
    >>> queries = [
    ...     "SELECT * FROM lineitem WHERE partition_col <= 10",
    ...     "SELECT * FROM lineitem WHERE partition_col > 10",
    ... ]
    >>> pl.read_sql(queries, uri)  # doctest: +SKIP

    """
    warnings.warn(
        "`read_sql` has been renamed; this"
        " redirect is temporary, please use `read_database` instead",
        category=DeprecationWarning,
        stacklevel=2,
    )
    return read_database(
        sql=sql,
        connection_uri=connection_uri,
        partition_on=partition_on,
        partition_range=partition_range,
        partition_num=partition_num,
        protocol=protocol,
        engine=engine,
    )


def _read_sql_connectorx(
    sql: str | list[str],
    connection_uri: str,
    partition_on: str | None = None,
    partition_range: tuple[int, int] | None = None,
    partition_num: int | None = None,
    protocol: str | None = None,
) -> pli.DataFrame:
    try:
        import connectorx as cx
    except ImportError:
        raise ImportError(
            "connectorx is not installed. Please run `pip install connectorx>=0.3.1`."
        ) from None

    tbl = cx.read_sql(
        conn=connection_uri,
        query=sql,
        return_type="arrow2",
        partition_on=partition_on,
        partition_range=partition_range,
        partition_num=partition_num,
        protocol=protocol,
    )

    return cast(pli.DataFrame, from_arrow(tbl))


def _read_sql_adbc(sql: str, connection_uri: str) -> pli.DataFrame:
    with _open_adbc_connection(connection_uri) as conn:
        cursor = conn.cursor()
        cursor.execute(sql)
        tbl = cursor.fetch_arrow_table()
        cursor.close()
    return cast(pli.DataFrame, from_arrow(tbl))


def _open_adbc_connection(connection_uri: str) -> Any:
    if connection_uri.startswith("sqlite"):
        try:
            import adbc_driver_sqlite.dbapi as adbc  # type: ignore[import]
        except ImportError:
            raise ImportError(
                "ADBC sqlite driver not detected. Please run `pip install "
                "adbc_driver_sqlite`."
            ) from None
        connection_uri = connection_uri.replace(r"sqlite:///", "")
    elif connection_uri.startswith("postgres"):
        try:
            import adbc_driver_postgresql.dbapi as adbc  # type: ignore[import]
        except ImportError:
            raise ImportError(
                "ADBC postgresql driver not detected. Please run `pip install "
                "adbc_driver_postgresql`."
            ) from None
    else:
        raise ValueError("ADBC does not currently support this database.")
    return adbc.connect(connection_uri)
