# Introduction

While Polars does support writing queries in SQL, it's recommended that users familiarize themselves with the [expression syntax](../concepts/expressions.md) for more readable and expressive code. As a primarily DataFrame library, new features will typically be added to the expression API first. However, if you already have an existing SQL codebase or prefer to use SQL, Polars also offers support for SQL queries.

!!! note Execution

    In Polars, there is no separate SQL engine because Polars translates SQL queries into [expressions](../concepts/expressions.md), which are then executed using its built-in execution engine. This approach ensures that Polars maintains its performance and scalability advantages as a native DataFrame library while still providing users with the ability to work with SQL queries.

## Context

Polars uses the `SQLContext` to manage SQL queries . The context contains a dictionary mapping `DataFrames` and `LazyFrames` names to their corresponding datasets[^1]. The example below starts a `SQLContext`:

{{code_block('user-guide/sql/intro','context',['SQLContext'])}}

```python exec="on" session="user-guide/sql"
--8<-- "python/user-guide/sql/intro.py:setup"
--8<-- "python/user-guide/sql/intro.py:context"
```

## Register Dataframes

There are 2 ways to register DataFrames in the `SQLContext`:

- register all `LazyFrames` and `DataFrames` in the global namespace
- register them one by one

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

    Converting a Pandas DataFrame backed by Numpy to Polars triggers a conversion to the Arrow format. This conversion has a computation cost. Converting a Pandas DataFrame backed by Arrow on the other hand will be free or almost free.

Once the `SQLContext` is initialized, we can register additional Dataframes or unregister existing Dataframes with:

- `register`
- `register_globals`
- `register_many`
- `unregister`

## Execute queries and collect results

SQL queries are always executed in lazy mode to benefit from lazy optimizations, so we have 2 options to collect the result:

- Set the parameter `eager_execution` to True in `SQLContext`. With this parameter, Polars will automatically collect SQL results
- Set the parameter `eager` to True when executing a query with `execute`, or collect the result with `collect`.

We execute SQL queries by calling `execute` on a `SQLContext`.

{{code_block('user-guide/sql/intro','execute',['SQLregister','SQLexecute'])}}

```python exec="on" result="text" session="user-guide/sql"
--8<-- "python/user-guide/sql/intro.py:execute"
```

## Execute queries from multiple sources

SQL queries can be executed just as easily from multiple sources.
In the example below, we register :

- a CSV file loaded lazily
- a NDJSON file loaded lazily
- a Pandas DataFrame

And we join them together with SQL.
Lazy reading allows to only load the necessary rows and columns from the files.

In the same way, it's possible to register cloud datalakes (S3, Azure Data Lake). A PyArrow dataset can point to the datalake, then Polars can read it with `scan_pyarrow_dataset`.

{{code_block('user-guide/sql/intro','execute_multiple_sources',['SQLregister','SQLexecute'])}}

```python exec="on" result="text" session="user-guide/sql"
--8<-- "python/user-guide/sql/intro.py:prepare_multiple_sources"
--8<-- "python/user-guide/sql/intro.py:execute_multiple_sources"
--8<-- "python/user-guide/sql/intro.py:clean_multiple_sources"
```

[^1]: Additionally it also tracks the [common table expressions](./cte.md) as well.

## Compatibility

Polars does not support the full SQL language, in Polars you are allowed to:

- Write a `CREATE` statements `CREATE TABLE xxx AS ...`
- Write a `SELECT` statements with all generic elements (`GROUP BY`, `WHERE`,`ORDER`,`LIMIT`,`JOIN`, ...)
- Write Common Table Expressions (CTE's) (`WITH tablename AS`)
- Show an overview of all tables `SHOW TABLES`

The following is not yet supported:

- `INSERT`, `UPDATE` or `DELETE` statements
- Table aliasing (e.g. `SELECT p.Name from pokemon AS p`)
- Meta queries such as `ANALYZE`, `EXPLAIN`

In the upcoming sections we will cover each of the statements in more details.
