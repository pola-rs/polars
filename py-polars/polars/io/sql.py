from __future__ import annotations

from typing import TYPE_CHECKING

from polars.convert import from_arrow
from polars.utils import deprecate_nonkeyword_arguments

if TYPE_CHECKING:
    from polars.internals import DataFrame


@deprecate_nonkeyword_arguments()
def read_sql(
    sql: list[str] | str,
    connection_uri: str,
    partition_on: str | None = None,
    partition_range: tuple[int, int] | None = None,
    partition_num: int | None = None,
    protocol: str | None = None,
) -> DataFrame:
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

    return from_arrow(tbl)  # type: ignore[return-value]
