# IO Plugins

Besides [expression plugins](./expr_plugins.md), Polars supports IO plugins for data sources that it
cannot read by itself. Queries over such sources still benefit from the query optimizer and the
streaming engine. There are two interfaces:

- [File readers](#file-readers) let Polars drive the reading of each source, as it does for its
  built-in formats. Together with a [`LazyFrameResolver`](#tables-with-lazyframeresolver), they also
  cover table formats.
- [Generator-based IO sources](#generator-based-io-sources) provide one generator for the whole
  source.

## File readers

!!! warning "Unstable"

    This functionality is considered **unstable**. It may be changed at any point without it being
    considered a breaking change.

A file reader reads one source of a custom format, and Polars drives the scan across all sources. A
`FileReaderBuilder` creates one `FileReader` per source and declares which operations the readers
apply themselves. `pl.scan_external_reader` turns a builder and a list of sources into a
`LazyFrame`.

Sources are `str` or `bytes` values that reach the builder unchanged, so they need not be file
paths. Directories and glob patterns are only expanded with `expand_paths=True`.

### Example format

The example reads a made-up format. Its first line is a JSON header with the row count, which stands
in for the metadata of a real format. One JSON object per row follows.

```
{"num_rows": 3}
{"city": "Oslo", "temperature": 3.5}
{"city": "Lima", "temperature": 18.0}
{"city": "Pune", "temperature": 27.1}
```

### The reader

```python
import json
from collections.abc import Iterator, Sequence
from itertools import islice
from typing import Any

import polars as pl
from polars.io.external_reader import (
    FileReader,
    FileReaderBuilder,
    FilterExpr,
    ReaderCapabilities,
)


class CountedJsonReader(FileReader):
    def __init__(self, path: str, schema: pl.Schema) -> None:
        self.path = path
        self.file_schema = schema
        self.num_rows = 0

    def fetch_metadata(self) -> None:
        # Only the header line is read here.
        with open(self.path) as f:
            self.num_rows = json.loads(f.readline())["num_rows"]

    def schema(self) -> pl.Schema:
        return self.file_schema

    def n_rows_in_file(self, limit: int | None = None) -> int:
        return self.num_rows

    def collect_batches(
        self,
        *,
        columns: Sequence[str],
        row_index: tuple[str, int] | None,
        pre_slice: tuple[int, int] | None,
        filters: Sequence[FilterExpr],
        num_pipelines: int,
    ) -> Iterator[pl.DataFrame]:
        offset, length = pre_slice if pre_slice is not None else (0, self.num_rows)
        if offset >= self.num_rows:
            return  # The slice starts after this file.
        batch_size = 2

        with open(self.path) as f:
            f.readline()  # Skip the header.
            lines = islice(f, offset, offset + length)
            position = offset

            while batch := list(islice(lines, batch_size)):
                rows = [json.loads(line) for line in batch]
                # With no columns requested, the height must still be correct.
                df = (
                    pl.from_dicts(rows, schema=self.file_schema).select(columns)
                    if columns
                    else pl.DataFrame(height=len(rows))
                )

                if row_index is not None:
                    name, start = row_index
                    df = df.with_row_index(name, offset=start + position)

                position += len(batch)
                yield df
```

`fetch_metadata` is called first and loads the metadata of the source, here the header line.
`schema` and `n_rows_in_file` describe the source. Polars asks for the row count when it needs to
know where the next source starts, and counting may stop at the optional `limit`.

`collect_batches` yields the data as DataFrames, and Polars stops consuming it once it has enough
rows. Its arguments are:

- `columns` lists the columns to read. The list is empty when a query only needs the number of rows,
  as with `select(pl.len())`. The DataFrames must have the correct height in that case too.
- `row_index` holds the name of a row index column and the index of the first row in this source. It
  is only passed when the builder declares support for it.
- `pre_slice` holds the offset and length of a slice, counted from the start of this source. The
  slice can extend past the end of the source. It is also only passed when the builder declares
  support for it.
- `filters` holds predicates that the reader may use. They are described [further below](#filters).
- `num_pipelines` is a hint for how much parallelism the reader may use.

The operations apply in a fixed order: row index, then slice, then filters. The row index therefore
numbers rows from the start of the source, not of the slice. The batch size of two only shows that a
reader can yield its data in several pieces.

### The builder

```python
class CountedJsonReaderBuilder(FileReaderBuilder):
    def __init__(self, schema: pl.Schema) -> None:
        self.schema = schema

    def reader_capabilities(self) -> ReaderCapabilities:
        return ReaderCapabilities(row_index=True, pre_slice=True)

    def build_file_reader(
        self, source: str | bytes, multi_scan_context: dict[Any, Any]
    ) -> FileReader:
        assert isinstance(source, str)
        return CountedJsonReader(source, self.schema)

    def explain_properties(self) -> dict[str, str]:
        return {"format": "counted JSON lines"}


def scan_counted_json(paths: list[str], schema: pl.Schema) -> pl.LazyFrame:
    return pl.scan_external_reader(
        CountedJsonReaderBuilder(schema), sources=paths, schema=schema
    )
```

`ReaderCapabilities` declares which operations the reader applies itself. Polars applies the others
to the DataFrames that the reader yields.

`build_file_reader` may be called from several threads at once. The `multi_scan_context` dictionary
is shared by all readers of a scan, for example to hold a thread pool. `explain_properties` adds
lines to the output of `LazyFrame.explain()`. The optional hooks `initialize_serial` and `on_drop`
are described in the [API reference](https://docs.pola.rs/api/python/stable/reference/plugins.html).

### Using the reader

The helper below writes two files in the example format. The scan then reads both as one table.

```python
def write_counted_json(path: str, df: pl.DataFrame) -> None:
    with open(path, "w") as f:
        f.write(json.dumps({"num_rows": df.height}) + "\n")
        for row in df.iter_rows(named=True):
            f.write(json.dumps(row) + "\n")


schema = pl.Schema({"city": pl.String, "temperature": pl.Float64})

write_counted_json(
    "part-0.cjson",
    pl.DataFrame({"city": ["Oslo", "Lima", "Pune"], "temperature": [3.5, 18.0, 27.1]}),
)
write_counted_json(
    "part-1.cjson",
    pl.DataFrame({"city": ["Kyiv", "Nuuk"], "temperature": [6.2, -4.0]}),
)

lf = scan_counted_json(["part-0.cjson", "part-1.cjson"], schema)
print(lf.collect())
```

```
shape: (5, 2)
┌──────┬─────────────┐
│ city ┆ temperature │
│ ---  ┆ ---         │
│ str  ┆ f64         │
╞══════╪═════════════╡
│ Oslo ┆ 3.5         │
│ Lima ┆ 18.0        │
│ Pune ┆ 27.1        │
│ Kyiv ┆ 6.2         │
│ Nuuk ┆ -4.0        │
└──────┴─────────────┘
```

The row index and the slice are pushed into the scan, and the slice spans both files.

```python
q = lf.with_row_index().slice(2, 2)
print(q.explain())
print(q.collect())
```

```
ExternalReaderBuilder SCAN [part-0.cjson, part-1.cjson]
  format: counted JSON lines
PROJECT */2 COLUMNS
SLICE: Positive { offset: 2, len: 2 }
ROW_INDEX: index
shape: (2, 3)
┌───────┬──────┬─────────────┐
│ index ┆ city ┆ temperature │
│ ---   ┆ ---  ┆ ---         │
│ u32   ┆ str  ┆ f64         │
╞═══════╪══════╪═════════════╡
│ 2     ┆ Pune ┆ 27.1        │
│ 3     ┆ Kyiv ┆ 6.2         │
└───────┴──────┴─────────────┘
```

### Filters

A reader that declares `supported_filter` receives the predicates of the query in `filters`. Each is
a `FilterExpr`, whose `expr` holds the Polars expression and whose `pyarrow_expr` holds a PyArrow
equivalent, or `None` if there is none. Filters are only passed when the reader also applies every
row index and slice of the scan.

With `supported_filter="partial"`, the reader may use the filters to skip data, and Polars filters
the yielded rows again. With `supported_filter="full"`, the reader must apply them exactly.

## Tables with `LazyFrameResolver`

!!! warning "Unstable"

    This functionality is considered **unstable**. It may be changed at any point without it being
    considered a breaking change.

A `LazyFrameResolver` defers resolving the paths to scan until the query runs. The resolver sees the
columns, row limit and filters that the query requests, and returns the `LazyFrame` to scan. Table
formats and catalogs use this to scan the current list of files and to skip files that the filters
rule out.

### Example catalog

The example catalog is a JSON file with a version number and the data files of the table. Each file
holds the measurements of one day, which the catalog records next to its path.

```python
def add_file(catalog_path: str, path: str, day: str, df: pl.DataFrame) -> None:
    write_counted_json(path, df.with_columns(day=pl.lit(day)))

    try:
        with open(catalog_path) as f:
            catalog = json.load(f)
    except FileNotFoundError:
        catalog = {"version": 0, "files": []}

    catalog["version"] += 1
    catalog["files"].append({"path": path, "day": day})

    with open(catalog_path, "w") as f:
        json.dump(catalog, f)
```

### The resolver

```python
from polars.lazyframe.resolver import (
    LazyFrameResolver,
    ResolvedLazyFrameProps,
)


class CatalogTable(LazyFrameResolver):
    def __init__(self, catalog_path: str, schema: pl.Schema) -> None:
        self.catalog_path = catalog_path
        self.table_schema = schema

    def schema(self) -> pl.Schema:
        return self.table_schema

    def resolve_lazyframe(
        self,
        *,
        projection: list[str] | None,
        limit: int | None,
        filters: list[FilterExpr],
        filter_columns: list[str],
        filter_drop_columns_idx: int | None,
        existing_resolved_version_key: str | None,
    ) -> pl.LazyFrame | tuple[pl.LazyFrame | None, ResolvedLazyFrameProps]:
        with open(self.catalog_path) as f:
            catalog = json.load(f)

        version_key = str(catalog["version"])
        if version_key == existing_resolved_version_key:
            # The table has not changed since the previous resolve.
            return None, ResolvedLazyFrameProps(version_key=version_key)

        files = pl.DataFrame(catalog["files"])

        # Skip the files whose day cannot match. Polars still applies every
        # filter afterwards, because none is reported as applied.
        for f in filters:
            if set(f.expr.meta.root_names()) <= {"day"}:
                files = files.filter(f.expr)

        lf = scan_counted_json(files["path"].to_list(), self.table_schema)

        if limit is not None:
            lf = lf.head(limit)
        if projection is not None:
            lf = lf.select(projection) if projection else lf.drop("*")

        return lf, ResolvedLazyFrameProps(version_key=version_key)
```

`schema` returns the schema of the table without resolving it, as Polars needs it to build the query
plan. `resolve_lazyframe` receives:

- `projection`: the columns to return, or `None` for all. An empty list means that only the row
  count is needed, so the result must keep its height, as `lf.drop("*")` does.
- `limit`: the maximum number of rows needed, or `None`.
- `filters`: the predicates of the query as `FilterExpr` objects. `filter_columns` lists the columns
  they refer to, and `filter_drop_columns_idx` marks the columns at the end of `projection` that
  only the filters need.
- `existing_resolved_version_key`: the version key of the previous resolve, described below.

The example resolver evaluates the filters on `day` against the catalog, so a query for one day only
opens the files of that day. Polars applies all filters again on the scanned rows.

### Versions and applied filters

A resolver returns a `LazyFrame`, optionally paired with `ResolvedLazyFrameProps`.

`version_key` identifies the version of the table. Without it, Polars caches the result and reuses
it for the same arguments. With it, Polars calls the resolver at every `collect()` and passes the
key of the previous resolve with the same arguments as `existing_resolved_version_key`. If the table
is unchanged, the resolver returns `None` with that key to reuse the previous result. Returning
`None` with any other key raises an error. The example takes the key from the catalog, so new files
show up in the next query.

`applied_filters` lists the filters that the returned `LazyFrame` applies fully, and Polars skips
them. The example leaves it empty, because it only skips whole files.

`cse_eq` tells whether two resolvers represent the same table, so that common subplan elimination
can read it once. It returns `False` by default.

### Using the resolver

`lazy()` creates a `LazyFrame` from the resolver. `explain()` resolves the table and shows that only
one of the two files is scanned.

```python
add_file(
    "catalog.json",
    "day-1.cjson",
    "2026-11-01",
    pl.DataFrame({"city": ["Oslo", "Lima"], "temperature": [3.5, 18.0]}),
)
add_file(
    "catalog.json",
    "day-2.cjson",
    "2026-11-02",
    pl.DataFrame({"city": ["Oslo", "Lima"], "temperature": [1.0, 19.5]}),
)

table_schema = pl.Schema(
    {"city": pl.String, "temperature": pl.Float64, "day": pl.String}
)
table = CatalogTable("catalog.json", table_schema).lazy()

q = table.filter(pl.col("day") == "2026-11-02", pl.col("temperature") > 5).select(
    "city", "temperature"
)
print(q.explain())
print(q.collect())
```

```
RESOLVER[name: CatalogTable, cached_resolves: 1]
  simple π 2/2 ["city", "temperature"]
    ExternalReaderBuilder SCAN [day-2.cjson]
      format: counted JSON lines
    PROJECT 3/3 COLUMNS
    SELECTION: (col("day") == "2026-11-02") & (col("temperature") > 5.0)
shape: (1, 2)
┌──────┬─────────────┐
│ city ┆ temperature │
│ ---  ┆ ---         │
│ str  ┆ f64         │
╞══════╪═════════════╡
│ Lima ┆ 19.5        │
└──────┴─────────────┘
```

A file added to the catalog shows up in the next run of the same query.

```python
add_file(
    "catalog.json",
    "day-2-late.cjson",
    "2026-11-02",
    pl.DataFrame({"city": ["Pune"], "temperature": [27.0]}),
)
print(q.collect())
```

```
shape: (2, 2)
┌──────┬─────────────┐
│ city ┆ temperature │
│ ---  ┆ ---         │
│ str  ┆ f64         │
╞══════╪═════════════╡
│ Lima ┆ 19.5        │
│ Pune ┆ 27.0        │
└──────┴─────────────┘
```

## Generator-based IO sources

Generator-based IO sources are registered with `register_io_source`. Because sources can move data
zero copy via Arrow FFI and sources can produce large chunks of data before returning, we've decided
to interface to IO plugins via Python for now, as we don't think the short time the GIL is needed
should lead to any contention.

E.g. an IO source can read their dataframe's in rust and only at the rendez-vous move the data
zero-copy having only a short time the GIL is needed.

### Use case

You want IO plugins if you have a source file not supported by Polars and you want to benefit from
optimizations like projection pushdown, predicate pushdown, early stopping and support of our
streaming engine.

### Example

So let's write a simple, very bad, custom CSV source and register that as an IO plugin. I want to
stress that this is a very bad example and is only given for learning purposes.

First we define some imports we need:

```python
# Use python for csv parsing.
import csv
import polars as pl
# Used to register a new generator on every instantiation.
from polars.io.plugins import register_io_source
from typing import Iterator
import io
```

#### Parsing the schema

Every `scan` function in Polars has to be able to provide the schema of the data it reads. For this
simple csv parser we will always read the data as `pl.String`. The only thing that differs are the
field names and the number of fields.

```python
def parse_schema(csv_str: str) -> pl.Schema:
    first_line = csv_str.split("\n")[0]

    return pl.Schema({k: pl.String for k in first_line.split(",")})
```

If we run this with small csv file `"a,b,c\n1,2,3"` we get the schema:
`Schema([('a', String), ('b', String), ('c', String)])`.

```python
>>> print(parse_schema("a,b,c\n1,2,3"))
Schema([('a', String), ('b', String), ('c', String)])
```

#### Writing the source

Next up is the actual source. For this we create an outer and an inner function. The outer function
`my_scan_csv` is the user facing function. This function will accept the file name and other
potential arguments you would need for reading the source. For csv files, these arguments could be
"delimiter", "quote_char" and such.

This outer function calls `register_io_source` which accepts a `callable` and a `schema`. The schema
is the Polars schema of the complete source file (independent of projection pushdown).

The callable is a function that will return a generator that produces `pl.DataFrame` objects.

The arguments of this function are predefined and this function must accept:

- `with_columns`

  Columns that are projected. The reader must project these columns if applied

- `predicate`

  Polars expression. The reader must filter their rows accordingly.

- `n_rows`

  Materialize only n rows from the source. The reader can stop when `n_rows` are read.

- `batch_size`

  A hint of the ideal batch size the reader's generator must produce.

The inner function is the actual implementation of the IO source and can also call into Rust/C++ or
wherever the IO plugin is written. If you want to see an IO source implemented in Rust, take a look
at our [plugins repository](https://github.com/pola-rs/pyo3-polars/tree/main/example/io_plugin).

```python
def my_scan_csv(csv_str: str) -> pl.LazyFrame:
    schema = parse_schema(csv_str)

    def source_generator(
        with_columns: list[str] | None,
        predicate: pl.Expr | None,
        n_rows: int | None,
        batch_size: int | None,
    ) -> Iterator[pl.DataFrame]:
        """
        Generator function that creates the source.
        This function will be registered as IO source.
        """
        if batch_size is None:
            batch_size = 100

        # Initialize the reader.
        reader = csv.reader(io.StringIO(csv_str), delimiter=',')
        # Skip the header.
        _ = next(reader)

        # Ensure we don't read more rows than requested from the engine
        while n_rows is None or n_rows > 0:
            if n_rows is not None:
                batch_size = min(batch_size, n_rows)

            rows = []

            for _ in range(batch_size):
                try:
                    row = next(reader)
                except StopIteration:
                    n_rows = 0
                    break
                rows.append(row)

            df = pl.from_records(rows, schema=schema, orient="row")
            n_rows -= df.height

            # If we would make a performant reader, we would not read these
            # columns at all.
            if with_columns is not None:
                df = df.select(with_columns)

            # If the source supports predicate pushdown, the expression can be parsed
            # to skip rows/groups.
            if predicate is not None:
                df = df.filter(predicate)

            yield df

    return register_io_source(io_source=source_generator, schema=schema)
```

#### Taking it for a (very slow) spin

Finally we can test our source:

```python
csv_str1 = """a,b,c,d
1,2,3,4
9,10,11,2
1,2,3,4
1,122,3,4"""

print(my_scan_csv(csv_str1).collect())


csv_str2 = """a,b
1,2
9,10
1,2
1,122"""

print(my_scan_csv(csv_str2).head(2).collect())
```

Running the script above would print the following output to the console:

```
shape: (4, 4)
┌─────┬─────┬─────┬─────┐
│ a   ┆ b   ┆ c   ┆ d   │
│ --- ┆ --- ┆ --- ┆ --- │
│ str ┆ str ┆ str ┆ str │
╞═════╪═════╪═════╪═════╡
│ 1   ┆ 2   ┆ 3   ┆ 4   │
│ 9   ┆ 10  ┆ 11  ┆ 2   │
│ 1   ┆ 2   ┆ 3   ┆ 4   │
│ 1   ┆ 122 ┆ 3   ┆ 4   │
└─────┴─────┴─────┴─────┘
shape: (2, 2)
┌─────┬─────┐
│ a   ┆ b   │
│ --- ┆ --- │
│ str ┆ str │
╞═════╪═════╡
│ 1   ┆ 2   │
│ 9   ┆ 10  │
└─────┴─────┘
```

### For engine authors: identifying a source

This section is for people writing a Polars execution engine. If you are writing an IO source, you
can skip it.

Polars does not call your source directly. It calls a wrapper that deserializes the predicate and
forwards the call. Engines may need to identify sources they know about, for example to provide
execution context or call additional methods.

To make that possible, the wrapper satisfies the `IOSourceScanFunction` protocol, which exposes the
registered source:

```python
from polars.io.plugins import IOSourceScanFunction

# `scan_fn` is the scan function of the `PythonScan` node being executed.
if isinstance(scan_fn, IOSourceScanFunction):
    source = scan_fn.io_source  # exactly the object passed to register_io_source
    if isinstance(source, MyEngineSource):
        ...
```

This is as unstable as the rest of the IO plugin API.

## Further reading

- [Rust example (distribution source)](https://github.com/pola-rs/pyo3-polars/tree/main/example/io_plugin)
- [Lance reader in Polars (`FileReader` and `LazyFrameResolver`)](https://github.com/pola-rs/polars/tree/main/py-polars/src/polars/io/lance)
