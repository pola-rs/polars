# Introduction

While Polars supports interaction with SQL, it's recommended that users familiarize themselves with
the [expression syntax](../concepts/expressions-and-contexts.md#expressions) to produce more
readable and expressive code. As the DataFrame interface is primary, new features are typically
added to the expression API first. However, if you already have an existing SQL codebase or prefer
the use of SQL, Polars does offers support for this.

!!! note Execution

    There is no separate SQL engine because Polars translates SQL queries into [expressions](../concepts/expressions-and-contexts.md#expressions), which are then executed using its own engine. This approach ensures that Polars maintains its performance and scalability advantages as a native DataFrame library, while still providing users with the ability to work with SQL.

## Context

Polars uses the `SQLContext` object to manage SQL queries. The context contains a mapping of
`DataFrame` and `LazyFrame` identifier names to their corresponding datasets[^1]. The example below
starts a `SQLContext`:

{{code_block('user-guide/sql/intro','context',['SQLContext'])}}

```python exec="on" session="user-guide/sql"
--8<-- "python/user-guide/sql/intro.py:setup"
--8<-- "python/user-guide/sql/intro.py:context"
```

## Register Dataframes

There are several ways to register DataFrames during `SQLContext` initialization.

- register all `LazyFrame` and `DataFrame` objects in the global namespace.
- register explicitly via a dictionary mapping, or kwargs.

{{code_block('user-guide/sql/intro','register_context',['SQLContext'])}}

```python exec="on" session="user-guide/sql"
--8<-- "python/user-guide/sql/intro.py:register_context"
```

We can also register Pandas DataFrames by converting them to Polars first.

{{code_block('user-guide/sql/intro','register_pandas',['SQLContext'])}}

```python exec="on" session="user-guide/sql"
--8<-- "python/user-guide/sql/intro.py:register_pandas"
```

!!! note Pandas

    Converting a Pandas DataFrame backed by Numpy will trigger a potentially expensive conversion; however, if the Pandas DataFrame is already backed by Arrow then the conversion will be significantly cheaper (and in some cases close to free).

Once the `SQLContext` is initialized, we can register additional Dataframes or unregister existing
Dataframes with:

- `register`
- `register_globals`
- `register_many`
- `unregister`

### Attach an Iceberg catalog

Instead of registering each table, you can attach an Iceberg catalog with `register_catalog`.
Queries then read its tables by name:

```python
from pyiceberg.catalog import load_catalog

ctx = pl.SQLContext()
ctx.register_catalog("lake", load_catalog("lake"), default_namespace="tpch", default=True)

ctx.execute("SELECT * FROM lineitem")            # lake: tpch.lineitem
ctx.execute("SELECT * FROM tpch.lineitem")       # the default catalog: tpch.lineitem
ctx.execute("SELECT * FROM lake.tpch.lineitem")  # catalog lake: tpch.lineitem
```

A table name is resolved in this order:

1. a CTE, or a table registered in the context;
2. a table of the attached catalog that the name starts with;
3. a table of the default catalog (with `default=True`), where a name without a namespace gets
   `default_namespace`.

Catalog tables are scanned lazily, like `scan_iceberg`: a table loads when the query's schema is
first needed. The name after the catalog is passed to the catalog as written, so catalog identifiers
are case-sensitive. A column can be qualified by any trailing part of the table's full name:
`lineitem.k`, `tpch.lineitem.k` or `lake.tpch.lineitem.k`.

With a default catalog, any name that is neither a CTE nor a registered table is looked up in the
catalog, so a misspelled name is too: a missing table raises `NoSuchTableError` when it loads, which
can be as late as `collect()`. Writing the full name, such as `lake.tpch.lineitem`, makes the
catalog explicit.

Catalogs attached inside a `with SQLContext()` block are detached on exit, like the tables
registered there.

#### Create and drop catalog tables

`CREATE TABLE` and `DROP TABLE` resolve the table name as a query does, and act on the catalog:

```python
ctx.execute("CREATE TABLE orders (id BIGINT NOT NULL, amount DOUBLE)")  # empty lake.tpch.orders
ctx.execute("CREATE TABLE big_orders AS SELECT * FROM orders WHERE amount > 1000")
ctx.execute("CREATE TEMP TABLE staging AS SELECT * FROM orders")  # registered in the context
ctx.execute("DROP TABLE big_orders")  # dropped from the catalog
```

- The statements run at `execute()`. `CREATE TABLE ... AS` commits the rows together with the
  create, so a failed query creates no table. A catalog that cannot stage a create gets an empty
  table first, dropped again if filling it fails, unless another writer changed it.
- `NOT NULL` makes a column a required field. `LOCATION '...'` and
  `WITH ('format-version' = 2, ...)` set the table's location and properties.
- `CREATE TEMP TABLE` always creates a registered table instead.
- `DROP TABLE` drops a registered table of that name first, else the catalog table, permanently.
  `DROP TABLE ... PURGE` also deletes the table's data files.
- `TRUNCATE` and `DELETE` are not supported on catalog tables.

See `SQLContext.register_catalog` for the details.

!!! warning "Serialized plans"

    A serialized plan over catalog tables can contain the catalog's properties and storage
    options, which may include credentials. Treat serialized plans as secret.

## Execute queries and collect results

SQL queries are always executed in lazy mode to take advantage of the full set of query planning
optimizations, so we have two options to collect the result:

- Set the parameter `eager` to True in `SQLContext`; this ensures that Polars automatically collects
  the LazyFrame results from `execute` calls.
- Set the parameter `eager` to True when executing a query with `execute`, or explicitly collect the
  result using `collect`.

We execute SQL queries by calling `execute` on a `SQLContext`.

{{code_block('user-guide/sql/intro','execute',['SQLregister','SQLexecute'])}}

```python exec="on" result="text" session="user-guide/sql"
--8<-- "python/user-guide/sql/intro.py:execute"
```

## Execute queries from multiple sources

SQL queries can be executed just as easily from multiple sources. In the example below, we register:

- a CSV file (loaded lazily)
- a NDJSON file (loaded lazily)
- a Pandas DataFrame

And join them together using SQL. Lazy reading lets you only load the necessary rows and columns
from the files.

In the same way, it's possible to register cloud datalakes (S3, Azure Data Lake). A PyArrow dataset
can point to the datalake, then Polars can read it with `scan_pyarrow_dataset`.

{{code_block('user-guide/sql/intro','execute_multiple_sources',['SQLregister','SQLexecute'])}}

```python exec="on" result="text" session="user-guide/sql"
--8<-- "python/user-guide/sql/intro.py:prepare_multiple_sources"
--8<-- "python/user-guide/sql/intro.py:execute_multiple_sources"
--8<-- "python/user-guide/sql/intro.py:clean_multiple_sources"
```

[^1]: Additionally it also tracks the [common table expressions](./cte.md) as well.

## Compatibility

Polars does not support the complete SQL specification, but it does support a subset of the most
common statement types.

!!! note Dialect

    Where possible, Polars aims to follow PostgreSQL syntax definitions and function behaviour.

For example, here is a non-exhaustive list of some of the supported functionality:

- Write a `CREATE` statements: `CREATE TABLE xxx AS ...`
- Write a `SELECT` statements containing:`WHERE`,`ORDER`,`LIMIT`,`GROUP BY`,`UNION` and `JOIN`
  clauses ...
- Write Common Table Expressions (CTE's) such as: `WITH tablename AS`
- Explain a query: `EXPLAIN SELECT ...`
- List registered tables: `SHOW TABLES`
- Drop a table: `DROP TABLE [IF EXISTS] tablename`
- Truncate a table: `TRUNCATE TABLE [IF EXISTS] tablename`

The following are some features that are not yet supported:

- `INSERT`, `UPDATE` or `DELETE` statements
- Meta queries such as `ANALYZE`

In the upcoming sections we will cover each of the statements in more detail.
