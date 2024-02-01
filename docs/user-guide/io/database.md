# Databases

## Read from a database

Polars can read from a database using the `pl.read_database_uri` and `pl.read_database` functions.

### Difference between `read_database_uri` and `read_database`

Use `pl.read_database_uri` if you want to specify the database connection with a connection string called a `uri`. For example, the following snippet shows a query to read all columns from the `foo` table in a Postgres database where we use the `uri` to connect:

{{code_block('user-guide/io/database','read_uri',['read_database_uri'])}}

On the other hand, use `pl.read_database` if you want to connect via a connection engine created with a library like SQLAlchemy.

{{code_block('user-guide/io/database','read_cursor',['read_database'])}}

Note that `pl.read_database_uri` is likely to be faster than `pl.read_database` if you are using a SQLAlchemy or DBAPI2 connection as these connections may load the data row-wise into Python before copying the data again to the column-wise Apache Arrow format.

### Engines

Polars doesn't manage connections and data transfer from databases by itself. Instead, external libraries (known as _engines_) handle this.

When using `pl.read_database`, you specify the engine when you create the connection object. When using `pl.read_database_uri`, you can specify one of two engines to read from the database:

- [ConnectorX](https://github.com/sfu-db/connector-x) and
- [ADBC](https://arrow.apache.org/docs/format/ADBC.html)

Both engines have native support for Apache Arrow and so can read data directly into a Polars `DataFrame` without copying the data.

#### ConnectorX

ConnectorX is the default engine and [supports numerous databases](https://github.com/sfu-db/connector-x#sources) including Postgres, Mysql, SQL Server and Redshift. ConnectorX is written in Rust and stores data in Arrow format to allow for zero-copy to Polars.

To read from one of the supported databases with `ConnectorX` you need to activate the additional dependency `ConnectorX` when installing Polars or install it manually with

```shell
$ pip install connectorx
```

Note: connectorx cannot be installed on ARM architectures from pip. See [this thread](https://github.com/sfu-db/connector-x/issues/186) for manual build instructions.

#### ADBC

ADBC (Arrow Database Connectivity) is an engine supported by the Apache Arrow project. ADBC aims to be both an API standard for connecting to databases and libraries implementing this standard in a range of languages.

It is still early days for ADBC so support for different databases is still limited. At present drivers for ADBC are only available for [Postgres and SQLite](https://arrow.apache.org/adbc/0.1.0/driver/cpp/index.html). To install ADBC you need to install the driver for your database. For example to install the driver for SQLite you run

```shell
$ pip install adbc-driver-sqlite
```

As ADBC is not the default engine you must specify the engine as an argument to `pl.read_database_uri`

{{code_block('user-guide/io/database','adbc',['read_database_uri'])}}

## Write to a database

We can write to a database with Polars using the `pl.write_database` function.

### Engines

As with reading from a database above Polars uses an _engine_ to write to a database. The currently supported engines are:

- [SQLAlchemy](https://www.sqlalchemy.org/) and
- Arrow Database Connectivity (ADBC)

#### SQLAlchemy

With the default engine SQLAlchemy you can write to any database supported by SQLAlchemy. To use this engine you need to install SQLAlchemy and Pandas

```shell
$ pip install SQLAlchemy pandas
```

In this example, we write the `DataFrame` to a table called `records` in the database

{{code_block('user-guide/io/database','write',['write_database'])}}

In the SQLAlchemy approach, Polars converts the `DataFrame` to a Pandas `DataFrame` backed by PyArrow and then uses SQLAlchemy methods on a Pandas `DataFrame` to write to the database.

#### ADBC

As with reading from a database, you can also use ADBC to write to a SQLite or Posgres database. As shown above, you need to install the appropriate ADBC driver for your database.

{{code_block('user-guide/io/database','write_adbc',['write_database'])}}
