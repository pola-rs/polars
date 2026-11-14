# Databases

## Read from a database

Polars can read from a database using the `pl.read_database_uri` and `pl.read_database` functions,
or lazily with `pl.scan_database` (see [Scan a database lazily](#scan-a-database-lazily)).

### Difference between `read_database_uri` and `read_database`

Use `pl.read_database_uri` if you want to specify the database connection with a connection string
called a `uri`. For example, the following snippet shows a query to read all columns from the `foo`
table in a Postgres database where we use the `uri` to connect:

{{code_block('user-guide/io/database','read_uri',['read_database_uri'])}}

On the other hand, use `pl.read_database` if you want to connect via a connection engine created
with a library like SQLAlchemy.

{{code_block('user-guide/io/database','read_cursor',['read_database'])}}

Note that `pl.read_database_uri` is likely to be faster than `pl.read_database` if you are using a
SQLAlchemy or DBAPI2 connection as these connections may load the data row-wise into Python before
copying the data again to the column-wise Apache Arrow format.

### Engines

Polars doesn't manage connections and data transfer from databases by itself. Instead, external
libraries (known as _engines_) handle this.

When using `pl.read_database`, you specify the engine when you create the connection object. When
using `pl.read_database_uri`, you can specify one of two engines to read from the database:

- [ConnectorX](https://github.com/sfu-db/connector-x) and
- [ADBC](https://arrow.apache.org/docs/format/ADBC.html)

Both engines have native support for Apache Arrow and so can read data directly into a Polars
`DataFrame` without copying the data.

#### ConnectorX

ConnectorX is the default engine and
[supports numerous databases](https://github.com/sfu-db/connector-x#sources) including Postgres,
Mysql, SQL Server and Redshift. ConnectorX is written in Rust and stores data in Arrow format to
allow for zero-copy to Polars.

To read from one of the supported databases with `ConnectorX` you need to activate the additional
dependency `ConnectorX` when installing Polars or install it manually with

```shell
$ pip install connectorx
```

#### ADBC

ADBC (Arrow Database Connectivity) is an engine supported by the Apache Arrow project. ADBC aims to
be both an API standard for connecting to databases and libraries implementing this standard in a
range of languages.

It is still early days for ADBC so support for different databases is limited. At present, drivers
for ADBC are only available for [Postgres](https://pypi.org/project/adbc-driver-postgresql/),
[SQLite](https://pypi.org/project/adbc-driver-sqlite/) and
[Snowflake](https://pypi.org/project/adbc-driver-snowflake/). To install ADBC, you need to install
the driver for your database. For example, to install the driver for SQLite, you run:

```shell
$ pip install adbc-driver-sqlite
```

As ADBC is not the default engine, you must specify the engine as an argument to
`pl.read_database_uri`.

{{code_block('user-guide/io/database','adbc',['read_database_uri'])}}

## Scan a database lazily

!!! warning "Unstable"

    `pl.scan_database` is considered unstable. It may be changed at any point without it being
    considered a breaking change.

`pl.scan_database` reads a table or a query lazily, on the streaming engine. Projections, filters
and limits in the query plan are pushed into the SQL sent to the database, and a large table can be
read as several partitions over concurrent connections.

It is implemented by the `polars-database` plugin, which is installed with the `database` extra:

```shell
$ pip install 'polars[database]'
```

The examples on this page use a small SQLite database, built in this setup:

{{code_block('user-guide/io/database_scan','setup',[])}}

{{code_block('user-guide/io/database_scan','scan',['scan_database'])}}

The source is a table name (`"orders"`, `"public.orders"`, `'"My Schema"."Orders"'`) or a SQL query.
`explain()` shows the database, the driver, the partitions and the SQL of the first partition. A
filter the SQL applies exactly is not applied again by Polars:

{{code_block('user-guide/io/database_scan','scan_explain',['scan_database'])}}

```text
RESOLVER[name: polars_database: scan_database, cached_resolves: 1]
  ExternalReaderBuilder SCAN [0]
    database: sqlite via adbc:sqlite 1.12.0
    relation: "orders"
    partitions: 1
    pushed: projection 3/5, filters 1/1 (1 exact), limit: no
    renderer: sqlglot 30.24.0
    sql[0]: SELECT "id", "amount", "status" FROM "orders" WHERE "status" = 'paid'
  PROJECT 2/3 COLUMNS
```

The SQL is generated with [sqlglot](https://github.com/tobymao/sqlglot), a dependency of the plugin,
on the machine that plans the query.

### Filter pushdown

A filter is only pushed when the database cannot drop a row that Polars would keep. Each pushed
filter is either _exact_ (the database returns exactly the rows Polars keeps), or a _superset_ that
Polars applies again after the read. For example, under MySQL's default case-insensitive collation,
`pl.col("name") == "abc"` is pushed as a superset (the database also returns `'ABC'`, which Polars
then removes), while `pl.col("name") != "abc"` is not pushed at all.

Pushed today: comparisons of columns with literals (numbers, strings, booleans, dates, datetimes),
`is_null`, `is_in`, `is_between`, `eq_missing` / `ne_missing`, `str.starts_with` / `str.ends_with` /
`str.contains(literal=True)`, and `&`, `|`, `~` of those. Arithmetic, casts and other functions are
evaluated by Polars.

### Drivers

`scan_database` picks the best driver installed on the machine, per database: for example ADBC, then
ConnectorX, then psycopg, then SQLAlchemy for Postgres. `show_drivers` lists the candidates, which
of them are installed, and how to install the others:

{{code_block('user-guide/io/database_scan','show_drivers',[])}}

ADBC drivers are found as pip packages (`adbc-driver-postgresql`) and through
[ADBC driver manifests](https://arrow.apache.org/adbc/current/format/driver_manifests.html), which
includes drivers installed with [`dbc`](https://docs.columnar.tech/dbc/). Pin a driver with
`driver=` or with a SQLAlchemy-style URI scheme:

{{code_block('user-guide/io/database_scan','scan_driver',['scan_database'])}}

| Database           | Drivers, best first                                                               |
| ------------------ | --------------------------------------------------------------------------------- |
| PostgreSQL         | ADBC, ConnectorX, psycopg, psycopg2, pg8000, SQLAlchemy                           |
| MySQL / MariaDB    | ADBC, ConnectorX, MySQLdb, mysql-connector, PyMySQL, mariadb, SQLAlchemy          |
| SQL Server         | ADBC, mssql-python, ConnectorX, arrow-odbc, turbodbc, pyodbc, pymssql, SQLAlchemy |
| Oracle             | ADBC, oracledb, ConnectorX, SQLAlchemy                                            |
| SQLite             | ADBC, ConnectorX, sqlite3, SQLAlchemy                                             |
| DuckDB             | duckdb, ADBC, SQLAlchemy                                                          |
| Snowflake          | ADBC, snowflake-connector-python, SQLAlchemy                                      |
| BigQuery           | ADBC, google-cloud-bigquery, ConnectorX, SQLAlchemy                               |
| Databricks         | ADBC, databricks-sql-connector, SQLAlchemy                                        |
| ClickHouse         | ADBC, clickhouse-connect, ConnectorX, clickhouse-driver, SQLAlchemy               |
| Trino              | ADBC, ConnectorX, trino, SQLAlchemy                                               |
| Redshift           | ADBC, ConnectorX, redshift_connector, psycopg2, SQLAlchemy                        |
| Other (any DB-API) | a connection factory, or SQLAlchemy                                               |

The column types come from the database, not the driver: the same table has the same schema
whichever driver reads it.

### Partitions

By default (`partition_on="auto"`), a large table is split into ranges of its integer or date
primary key. `partition_on=<column>` partitions on another column, `partition_predicates=` gives one
partition per `WHERE` fragment, and `partition_on=None` reads with a single query. `max_connections`
limits the number of concurrent connections.

{{code_block('user-guide/io/database_scan','scan_partitions',['scan_database'])}}

Rows come back in no particular order unless `order_by=` is given. A `NUMERIC` column without a
precision is read as `Decimal(38, 9)`; a value with digits beyond the ninth decimal place is an
error, so pass `schema_overrides=` for such columns if you need more. Each partition is a separate
query, so concurrent writes to the table can make partitions see different states.

### Queries and connections

A query source is wrapped as a derived table (`SELECT … FROM (<query>) AS _pl WHERE …`), so filters
and projections are still pushed. A query that can't be wrapped runs as given, and Polars does the
rest. On MariaDB, which ignores an `ORDER BY` inside a derived table, a query ordered at its top
level (without `LIMIT`) isn't wrapped. A query whose rows depend on the time it runs (`now()`,
`CURRENT_TIMESTAMP`, `random()`, …) is read with a single query, never partitioned.

{{code_block('user-guide/io/database_scan','scan_query',['scan_database'])}}

Parameters of a query are bound by the driver, never written into the SQL. This needs a driver that
binds them itself (not ADBC or ConnectorX):

{{code_block('user-guide/io/database_scan','scan_params',['scan_database'])}}

Besides a URI, the connection can be an ODBC connection string, a SQLAlchemy `URL` or `Engine`, a
zero-argument function that returns a DB-API connection, or a live connection object. An `Engine`
opens the connections itself, with its pool, `creator` and `connect_args`. A function reaches any
database with a Python driver. A live connection is used for one partition at a time, and a plan
holding one can't be serialized. Connections are used from Polars' threads, so a `sqlite3`
connection needs `check_same_thread=False`. With a function or a live connection, `driver`,
`connect_options` and `credential_provider` don't apply: the connection object is already set up.

A serialized or pickled plan carries its URI as given, password included (`explain()` redacts it).
To keep the password out of a plan you ship elsewhere, leave it out of the URI and pass a
`credential_provider` that returns it when the scan connects.

{{code_block('user-guide/io/database_scan','scan_factory',['scan_database'])}}

## Write to a database

We can write to a database with Polars using the `pl.write_database` function.

### Engines

As with reading from a database above, Polars uses an _engine_ to write to a database. The currently
supported engines are:

- [SQLAlchemy](https://www.sqlalchemy.org/) and
- Arrow Database Connectivity (ADBC)

#### SQLAlchemy

With the default engine SQLAlchemy you can write to any database supported by SQLAlchemy. To use
this engine you need to install SQLAlchemy and Pandas

```shell
$ pip install SQLAlchemy pandas
```

In this example, we write the `DataFrame` to a table called `records` in the database

{{code_block('user-guide/io/database','write',['write_database'])}}

In the SQLAlchemy approach, Polars converts the `DataFrame` to a Pandas `DataFrame` backed by
PyArrow and then uses SQLAlchemy methods on a Pandas `DataFrame` to write to the database.

#### ADBC

ADBC can also be used to write to a database. Writing is supported for the same databases that
support reading with ADBC. As shown above, you need to install the appropriate ADBC driver for your
database.

{{code_block('user-guide/io/database','write_adbc',['write_database'])}}
