# Universally unique identifiers (UUIDs)

UUIDs are 128-bit identifiers commonly used as database keys, event identifiers, and distributed
system identifiers. The native `UUID` data type stores each non-null value in 16 bytes, instead of
keeping its 36-character text representation. This makes the intended schema explicit and lets
Polars validate values once, then sort, filter, join, group, and compare them without repeatedly
parsing strings.

## Creating UUID columns

Python [`uuid.UUID`](https://docs.python.org/3/library/uuid.html#uuid.UUID) objects are inferred as
the Polars `UUID` data type. You can also specify the data type explicitly.

{{code_block('user-guide/concepts/uuids','construction',[],['DataFrame','UUID'],[])}}

```python exec="on" result="text" session="user-guide/uuids"
--8<-- "python/user-guide/concepts/uuids.py:construction"
```

Use a cast to parse existing string or binary columns. A strict cast reports the first invalid
value. Set `strict=False` to replace invalid values with `null`.

{{code_block('user-guide/concepts/uuids','parsing',[],['cast','UUID'],[])}}

```python exec="on" result="text" session="user-guide/uuids"
--8<-- "python/user-guide/concepts/uuids.py:parsing"
```

UUID values use their unsigned 128-bit order for sorting and comparisons. They work as keys in the
same expressions and operations as other scalar data types, including filters, joins, `group_by`,
`unique`, and `is_in`. Python `uuid.UUID` objects can be used directly as scalar values in these
expressions.

{{code_block('user-guide/concepts/uuids','filtering',[],['filter'],[])}}

```python exec="on" result="text" session="user-guide/uuids"
--8<-- "python/user-guide/concepts/uuids.py:filtering"
```

## Generating and inspecting UUIDs

Use `uuid4` to generate random UUIDs. Use `uuid7` when identifiers should retain creation-time
order. Both functions take an explicit row count and can return either an expression or an eager
series.

{{code_block('user-guide/concepts/uuids','generation',[],['uuid4','uuid7'],[])}}

```python exec="on" result="text" session="user-guide/uuids"
--8<-- "python/user-guide/concepts/uuids.py:generation"
```

The `uuid` namespace exposes the version field for every UUID. Version 7 UUIDs also encode a UTC
millisecond timestamp, available with `uuid.timestamp`. By default, timestamp extraction raises for
a value that is not version 7; use `strict=False` to return `null` for those values instead.

{{code_block('user-guide/concepts/uuids','inspection',[],[],[])}}

```python exec="on" result="text" session="user-guide/uuids"
--8<-- "python/user-guide/concepts/uuids.py:inspection"
```

## Interchange

Polars preserves the logical type when exchanging UUID columns with systems that understand it:

- Python values round-trip as `uuid.UUID` objects.
- Arrow uses the canonical `arrow.uuid` extension type backed by fixed-size 16-byte binary values.
- Parquet uses the standard UUID logical annotation.
- CSV, JSON, and NDJSON use lowercase, hyphenated UUID text.

Canonical text, compact text without hyphens, braced text, and `urn:uuid:` text can all be parsed.
The resulting UUID values compare consistently with Python, PostgreSQL, and DuckDB UUID values.
