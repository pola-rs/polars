# SHOW TABLES

In Polars, the `SHOW TABLES` statement is used to list all the tables that have been registered in the current `SQLContext`. When you register a DataFrame with the `SQLContext`, you give it a name that can be used to refer to the DataFrame in subsequent SQL statements. The `SHOW TABLES` statement allows you to see a list of all the registered tables, along with their names.

The syntax for the `SHOW TABLES` statement in Polars is as follows:

```
SHOW TABLES
```

Here's an example of how to use the `SHOW TABLES` statement in Polars:

{{code_block('user-guide/sql/show','show',['SQLregister','SQLexecute'])}}

```python exec="on" result="text" session="user-guide/sql/show"
--8<-- "python/user-guide/sql/show.py:setup"
--8<-- "python/user-guide/sql/show.py:show"
```

In this example, we create two DataFrames and register them with the `SQLContext` using different names. We then execute a `SHOW TABLES` statement using the `execute()` method of the `SQLContext` object, which returns a DataFrame containing a list of all the registered tables and their names. The resulting DataFrame is then printed using the `print()` function.

Note that the `SHOW TABLES` statement only lists tables that have been registered with the current `SQLContext`. If you register a DataFrame with a different `SQLContext` or in a different Python session, it will not appear in the list of tables returned by `SHOW TABLES`.
