# Introduction

While Polars supports interaction with SQL, it's recommended that users familiarize themselves with
the [expression syntax](../concepts/expressions.md) to produce more readable and expressive code. As the DataFrame
interface is primary, new features are typically added to the expression API first. However, if you already have an
existing SQL codebase or prefer the use of SQL, Polars does offers support for this.

!!! note Execution

    There is no separate SQL engine because Polars translates SQL queries into [expressions](../concepts/expressions.md), which are then executed using its own engine. This approach ensures that Polars maintains its performance and scalability advantages as a native DataFrame library, while still providing users with the ability to work with SQL.

## Context

Polars uses the `SQLContext` object to manage SQL queries. The context contains a mapping of `DataFrame` and `LazyFrame`
identifier names to their corresponding datasets[^1]. The example below starts a `SQLContext`:

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

Once the `SQLContext` is initialized, we can register additional Dataframes or unregister existing Dataframes with:

- `register`
- `register_globals`
- `register_many`
- `unregister`

## Execute queries and collect results

SQL queries are always executed in lazy mode to take advantage of the full set of query planning optimizations, so we
have two options to collect the result:

- Set the parameter `eager_execution` to True in `SQLContext`; this ensures that Polars automatically collects the
  LazyFrame results from `execute` calls.
- Set the parameter `eager` to True when executing a query with `execute`, or explicitly collect the result
  using `collect`.

We execute SQL queries by calling `execute` on a `SQLContext`.

{{code_block('user-guide/sql/intro','execute',['SQLregister','SQLexecute'])}}

```python exec="on" result="text" session="user-guide/sql"
--8<-- "python/user-guide/sql/intro.py:execute"
```

## Execute queries from multiple sources

SQL queries can be executed just as easily from multiple sources.
In the example below, we register:

- a CSV file (loaded lazily)
- a NDJSON file (loaded lazily)
- a Pandas DataFrame

And join them together using SQL.
Lazy reading allows to only load the necessary rows and columns from the files.

In the same way, it's possible to register cloud datalakes (S3, Azure Data Lake). A PyArrow dataset can point to the
datalake, then Polars can read it with `scan_pyarrow_dataset`.

{{code_block('user-guide/sql/intro','execute_multiple_sources',['SQLregister','SQLexecute'])}}

```python exec="on" result="text" session="user-guide/sql"
--8<-- "python/user-guide/sql/intro.py:prepare_multiple_sources"
--8<-- "python/user-guide/sql/intro.py:execute_multiple_sources"
--8<-- "python/user-guide/sql/intro.py:clean_multiple_sources"
```

[^1]: Additionally it also tracks the [common table expressions](./cte.md) as well.

## Compatibility

Polars does not support the complete SQL specification, but it does support a subset of the most common statement types.

!!! note Dialect

    Where possible, Polars aims to follow PostgreSQL syntax definitions and function behaviour.

For example, here is a non-exhaustive list of some of the supported functionality:

- Write a `CREATE` statements: `CREATE TABLE xxx AS ...`
- Write a `SELECT` statements containing:`WHERE`,`ORDER`,`LIMIT`,`GROUP BY`,`UNION` and `JOIN` clauses ...
- Write Common Table Expressions (CTE's) such as: `WITH tablename AS`
- Explain a query: `EXPLAIN SELECT ...`
- List registered tables: `SHOW TABLES`
- Drop a table: `DROP TABLE tablename`
- Truncate a table: `TRUNCATE TABLE tablename`

The following are some features that are not yet supported:

- `INSERT`, `UPDATE` or `DELETE` statements
- Meta queries such as `ANALYZE`

In the upcoming sections we will cover each of the statements in more detail.
