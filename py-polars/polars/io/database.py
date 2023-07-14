from __future__ import annotations

import re
import sys
from importlib import import_module
from typing import TYPE_CHECKING, Any

from polars.convert import from_arrow

if TYPE_CHECKING:
    from polars import DataFrame
    from polars.type_aliases import DbReadEngine


def read_database(
    query: list[str] | str,
    connection_uri: str,
    *,
    partition_on: str | None = None,
    partition_range: tuple[int, int] | None = None,
    partition_num: int | None = None,
    protocol: str | None = None,
    engine: DbReadEngine = "connectorx",
) -> DataFrame:
    """
    Read a SQL query into a DataFrame.

    Parameters
    ----------
    query
        Raw SQL query (or queries).
    connection_uri
        A connectorx or ADBC connection URI that starts with the backend's
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
        Selects the engine used for reading the database:

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
    ...     query,
    ...     uri,
    ...     partition_on="partition_col",
    ...     partition_num=10,
    ...     engine="connectorx",
    ... )  # doctest: +SKIP

    Read a DataFrame in parallel using 2 threads by explicitly providing two SQL
    queries:

    >>> uri = "postgresql://username:password@server:port/database"
    >>> queries = [
    ...     "SELECT * FROM lineitem WHERE partition_col <= 10",
    ...     "SELECT * FROM lineitem WHERE partition_col > 10",
    ... ]
    >>> pl.read_database(queries, uri, engine="connectorx")  # doctest: +SKIP

    Read data from Snowflake using the ADBC driver:

    >>> df = pl.read_database(
    ...     "SELECT * FROM test_table",
    ...     "snowflake://user:pass@company-org/testdb/public?warehouse=test&role=myrole",
    ...     engine="adbc",
    ... )  # doctest: +SKIP

    """  # noqa: W505
    if engine == "connectorx":
        return _read_sql_connectorx(
            query,
            connection_uri,
            partition_on=partition_on,
            partition_range=partition_range,
            partition_num=partition_num,
            protocol=protocol,
        )
    elif engine == "adbc":
        if not isinstance(query, str):
            raise ValueError("Only a single SQL query string is accepted for adbc.")
        return _read_sql_adbc(query, connection_uri)
    else:
        raise ValueError(f"Engine {engine!r} not implemented; use connectorx or adbc.")


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
    except ImportError:
        raise ImportError(
            "connectorx is not installed. Please run `pip install connectorx>=0.3.1`."
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
    with _open_adbc_connection(connection_uri) as conn:
        cursor = conn.cursor()
        cursor.execute(query)
        tbl = cursor.fetch_arrow_table()
        cursor.close()
    return from_arrow(tbl)  # type: ignore[return-value]


def _open_adbc_connection(connection_uri: str) -> Any:
    driver_name = connection_uri.split(":", 1)[0].lower()

    # note: existing URI driver prefixes currently map 1:1 with
    # the adbc module suffix; update this map if that changes.
    module_suffix_map: dict[str, str] = {}
    try:
        module_suffix = module_suffix_map.get(driver_name, driver_name)
        module_name = f"adbc_driver_{module_suffix}.dbapi"
        import_module(module_name)
        adbc_driver = sys.modules[module_name]
    except ImportError:
        raise ImportError(
            f"ADBC {driver_name} driver not detected; if ADBC supports this database, "
            f"please run `pip install adbc-driver-{driver_name} pyarrow`"
        ) from None

    # some backends require the driver name to be stripped from the URI
    if driver_name in ("sqlite", "snowflake"):
        connection_uri = re.sub(f"^{driver_name}:/{{,3}}", "", connection_uri)

    return adbc_driver.connect(connection_uri)
