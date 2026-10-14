# SELECT

In Polars SQL, the `SELECT` statement is used to retrieve data from a table into a `DataFrame`. The
basic syntax of a `SELECT` statement in Polars SQL is as follows:

```sql
SELECT column1, column2, ...
FROM table_name;
```

Here, `column1`, `column2`, etc. are the columns that you want to select from the table. You can
also use the wildcard `*` to select all columns. `table_name` is the name of the table or that you
want to retrieve data from. In the sections below we will cover some of the more common SELECT
variants

{{code_block('user-guide/sql/select','df',['SQLregister','SQLexecute'])}}

```python exec="on" result="text" session="user-guide/sql/select"
--8<-- "python/user-guide/sql/select.py:setup"
--8<-- "python/user-guide/sql/select.py:df"
```

### GROUP BY

The `GROUP BY` statement is used to group rows in a table by one or more columns and compute
aggregate functions on each group.

{{code_block('user-guide/sql/select','group_by',['SQLexecute'])}}

```python exec="on" result="text" session="user-guide/sql/select"
--8<-- "python/user-guide/sql/select.py:group_by"
```

### ORDER BY

The `ORDER BY` statement is used to sort the result set of a query by one or more columns in
ascending or descending order.

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

For a full list of supported functions go the
[API documentation](https://docs.rs/polars-sql/latest/src/polars_sql/keywords.rs.html). The example
below demonstrates how to use a function in a query

{{code_block('user-guide/sql/select','functions',['SQLexecute'])}}

```python exec="on" result="text" session="user-guide/sql/select"
--8<-- "python/user-guide/sql/select.py:functions"
```

### Table Functions

In the examples earlier we first generated a DataFrame which we registered in the `SQLContext`.
Polars also support directly reading from CSV, Parquet, JSON and IPC in your SQL query using table
functions `read_xxx`.

{{code_block('user-guide/sql/select','tablefunctions',['SQLexecute'])}}

```python exec="on" result="text" session="user-guide/sql/select"
--8<-- "python/user-guide/sql/select.py:tablefunctions"
```

`read_csv` also accepts named options after the path, with the same names and meaning as the options
of `scan_csv`. Write them as `name => value` (or `name := value`):

{{code_block('user-guide/sql/select','tablefunctionoptions',['SQLexecute'])}}

```python exec="on" result="text" session="user-guide/sql/select"
--8<-- "python/user-guide/sql/select.py:tablefunctionoptions"
```

The supported options are:

- Layout: `has_header`, `separator`, `quote_char`, `comment_prefix`, `skip_rows`,
  `skip_rows_after_header`, `skip_lines`, `encoding`.
- Names and types: `new_columns`, `schema`, `schema_overrides`, `infer_schema`,
  `infer_schema_length`, `infer_schema_files`, `try_parse_dates`, `decimal_comma`.
- Values and errors: `null_values`, `ignore_errors`.
- Ragged rows and multiple files: `truncate_ragged_lines`, `missing_columns`, `extra_columns`,
  `include_file_paths`.

Values are SQL literals:

- `NULL` stands for `None`, as in `infer_schema_length => NULL` to infer the schema from the whole
  file, or `quote_char => NULL` to turn quoting off.
- Lists use brackets: `new_columns => ['a', 'b']`.
- `schema`, `schema_overrides` and per-column `null_values` take a dictionary, with types written as
  SQL type names: `schema_overrides => {'id': 'INT', 'name': 'VARCHAR'}`. `schema_overrides` also
  accepts a list of type names, by position.
- Escape sequences need an `E''` string, as in `separator => E'\t'` for tab-separated files.

Unlike `scan_csv`, `read_csv` tries to parse dates by default (`try_parse_dates => true`).

Some options are unstable: those marked as such in the
[`scan_csv`](https://docs.pola.rs/api/python/stable/reference/api/polars.scan_csv.html) API
reference are unstable in `read_csv` too, and may be changed at any point without it being
considered a breaking change.

`read_ndjson` reads newline-delimited JSON (one object per line) and accepts the options of
`scan_ndjson` in the same way: `schema`, `schema_overrides`, `infer_schema_length`, `ignore_errors`
and `include_file_paths`.

{{code_block('user-guide/sql/select','tablefunctionndjson',['SQLexecute'])}}

```python exec="on" result="text" session="user-guide/sql/select"
--8<-- "python/user-guide/sql/select.py:tablefunctionndjson"
```

- The schema is inferred from the first 100 rows, so a key that first appears later is dropped. Use
  `infer_schema_length => NULL` to infer it from the whole file.
- A nested object read as `'VARCHAR'` keeps its JSON text.
- `read_json` reads the same files, but takes only the path. For a JSON array or a pretty-printed
  JSON document, use `pl.read_json` instead.
