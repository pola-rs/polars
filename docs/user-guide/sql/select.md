# SELECT

In Polars SQL, the `SELECT` statement is used to retrieve data from a table into a `DataFrame`. The basic syntax of a `SELECT` statement in Polars SQL is as follows:

```sql
SELECT column1, column2, ...
FROM table_name;
```

Here, `column1`, `column2`, etc. are the columns that you want to select from the table. You can also use the wildcard `*` to select all columns. `table_name` is the name of the table or that you want to retrieve data from. In the sections below we will cover some of the more common SELECT variants

{{code_block('user-guide/sql/select','df',['SQLregister','SQLexecute'])}}

```python exec="on" result="text" session="user-guide/sql/select"
--8<-- "python/user-guide/sql/select.py:setup"
--8<-- "python/user-guide/sql/select.py:df"
```

### GROUP BY

The `GROUP BY` statement is used to group rows in a table by one or more columns and compute aggregate functions on each group.

{{code_block('user-guide/sql/select','group_by',['SQLexecute'])}}

```python exec="on" result="text" session="user-guide/sql/select"
--8<-- "python/user-guide/sql/select.py:group_by"
```

### ORDER BY

The `ORDER BY` statement is used to sort the result set of a query by one or more columns in ascending or descending order.

{{code_block('user-guide/sql/select','orderby',['SQLexecute'])}}

```python exec="on" result="text" session="user-guide/sql/select"
--8<-- "python/user-guide/sql/select.py:orderby"
```

### JOIN

{{code_block('user-guide/sql/select','join',['SQLregister_many','SQLexecute'])}}

```python exec="on" result="text" session="user-guide/sql/select"
--8<-- "python/user-guide/sql/select.py:join"
```

### Functions

Polars provides a wide range of SQL functions, including:

- Mathematical functions: `ABS`, `EXP`, `LOG`, `ASIN`, `ACOS`, `ATAN`, etc.
- String functions: `LOWER`, `UPPER`, `LTRIM`, `RTRIM`, `STARTS_WITH`,`ENDS_WITH`.
- Aggregation functions: `SUM`, `AVG`, `MIN`, `MAX`, `COUNT`, `STDDEV`, `FIRST` etc.
- Array functions: `EXPLODE`, `UNNEST`,`ARRAY_SUM`,`ARRAY_REVERSE`, etc.

For a full list of supported functions go the [API documentation](https://docs.rs/polars-sql/latest/src/polars_sql/keywords.rs.html). The example below demonstrates how to use a function in a query

{{code_block('user-guide/sql/select','functions',['SQLquery'])}}

```python exec="on" result="text" session="user-guide/sql/select"
--8<-- "python/user-guide/sql/select.py:functions"
```

### Table Functions

In the examples earlier we first generated a DataFrame which we registered in the `SQLContext`. Polars also support directly reading from CSV, Parquet, JSON and IPC in your SQL query using table functions `read_xxx`.

{{code_block('user-guide/sql/select','tablefunctions',['SQLexecute'])}}

```python exec="on" result="text" session="user-guide/sql/select"
--8<-- "python/user-guide/sql/select.py:tablefunctions"
```
