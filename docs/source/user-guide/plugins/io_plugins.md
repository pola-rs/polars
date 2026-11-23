# IO Plugins

An IO plugin interface is exposed that can be used to implement custom file reading logic. This is
useful for file formats that polars does not have a native reader for.

- [File readers](#file-readers), which get called by polars to read a user-supplied path. This can
  be used together with a [`LazyFrameResolver`](#tables-with-lazyframeresolver), which allows custom
  resolution logic for the list of paths to scan.
- [Generator-based IO sources](#generator-based-io-sources) provide one generator for the whole
  source.

## File readers

!!! warning "Unstable"

    This functionality is considered **unstable**. It may be changed at any point without it being
    considered a breaking change.

The file reader implementation interface is split into 2 parts:

- `FileReaderBuilder`: Gets called by polars to build `FileReader`s for each path that is being
  scanned.
- `FileReader`: Responsible for reading the data of a single path.

`pl.scan_external_reader` takes a builder, a list of sources and a schema, and returns a
`LazyFrame`.

### Example format

Given this example NDJSON data, where the first row is a JSON-formatted header line:

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

<!-- dprint-ignore-start -->

- `fetch_metadata` is called by polars first to allow the file reader to perform metadata fetching
  before the actual read.
- `schema` and `n_rows_in_file` describe the source. A `limit` is provided to allow readers
  (optionally) to stop counting rows early.
- `collect_batches` yields the data as `DataFrame`s:
    - `columns` lists the columns to read. **If this is given as an empty list, the reader should
      still produce 0-column DataFrames with the correct height.**
    - `row_index` holds the name of a row index column and the index of the first row in this source
      that the reader should apply. It is not passed unless the `FileReaderBuilder` declares that the
      reader supports applying the row index.
    - `pre_slice` holds the offset and length of a slice, counted from the start of this source. Note,
      the slice can extend past the end of the source. It is also only passed when the builder
      declares support for it.
    - `filters` holds predicates that the reader may use. They are described
      [further below](#filters).
    - `num_pipelines` is a hint for how much parallelism the reader should use.

<!-- dprint-ignore-end -->

The order which operations should be applied is: row index, then slice, then filters.

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

- `ReaderCapabilities` declares which operations the reader applies itself. Polars applies the
  others to the DataFrames that the reader yields.
- `build_file_reader` may be called from several threads at once. A `multi_scan_context` dictionary
  is provided as a shared context accessible by all file readers during a scan, which can be used
  for e.g. synchronization, or holding a shared threadpool.
- `explain_properties` adds lines to the output of `LazyFrame.explain()`.

The optional hooks `initialize_serial` and `on_drop` are described in the
[API reference](https://docs.pola.rs/api/python/stable/reference/plugins.html).

### Using the reader

This helper writes 2 files in the example format, which are then scanned as a single table:

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

Row index and slice get pushed down into the scan. Note the slice spans across both files:

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

Readers that declare `supported_filter` in their `ReaderCapabilities` receive the predicates of the
query in `filters`. Each one is a `FilterExpr`:

- `expr` holds the polars expression.
- `pyarrow_expr` holds an equivalent PyArrow expression, or `None` if the expression cannot be
  converted.

Filters are not passed unless the reader also supports applying the row index and slice of the scan.

- `supported_filter="partial"`: The reader may use the filters to skip data. Polars will filter the
  yielded rows again afterwards.
- `supported_filter="full"`: **The reader must apply the filters exactly.**

## Tables with `LazyFrameResolver`

!!! warning "Unstable"

    This functionality is considered **unstable**. It may be changed at any point without it being
    considered a breaking change.

A `LazyFrameResolver` allows the paths to scan to be decided when the query runs instead of when it
is built. It gets called by polars with the columns, row limit and filters of the query, and returns
the `LazyFrame` to scan. This is useful for e.g. table formats and catalogs, which can use it to
scan the latest list of files, and skip files that cannot match the filters.

### Example catalog

Given this example catalog, which is a JSON file holding a version number and the list of data files
in the table. Each data file holds the measurements of a single day, which is also recorded in the
catalog:

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

<!-- dprint-ignore-start -->

- `schema` returns the schema of the table. It is called by polars when building the query plan, so
  it should not resolve the table.
- `resolve_lazyframe` returns the `LazyFrame` to scan:
    - `projection` lists the columns to return, or `None` for all columns. **If this is given as an
      empty list, the result should still have the correct height** (e.g. using `lf.drop("*")`).
    - `limit` is the maximum number of rows needed, or `None`.
    - `filters` holds the predicates of the query as `FilterExpr`s. `filter_columns` lists the columns
      they refer to, and `filter_drop_columns_idx` marks the columns at the end of `projection` that
      are only needed by the filters.
    - `existing_resolved_version_key` is the version key of the previous resolve, described
      [below](#versions-and-applied-filters).

<!-- dprint-ignore-end -->

The example resolver applies the filters on `day` to the list of files in the catalog, so that a
query for a single day only opens the files of that day. Polars will still apply all filters again
afterwards.

### Versions and applied filters

A resolver can optionally return a `ResolvedLazyFrameProps` alongside the `LazyFrame`:

<!-- dprint-ignore-start -->

- `version_key` identifies the version of the table:
    - If it is not given, polars caches the resolved `LazyFrame` and reuses it for calls with the same
      arguments.
    - If it is given, polars calls the resolver on every `collect()`, passing the key of the previous
      resolve with the same arguments as `existing_resolved_version_key`. If the table is unchanged,
      the resolver can return `None` with that same key to reuse the previous result. Note, returning
      `None` with any other key raises an error.
- `applied_filters` lists the filters that are fully applied by the returned `LazyFrame`. Polars
  will not apply these again. The example leaves it empty, as it only skips whole files.

<!-- dprint-ignore-end -->

The example takes the version key from the catalog, so that new files show up in the next query.

A resolver can also implement `cse_eq` to tell whether two resolvers represent the same table. This
allows common subplan elimination to read the table only once. It returns `False` by default.

### Using the resolver

`lazy()` creates a `LazyFrame` from the resolver. Note `explain()` resolves the table, and shows
that only one of the two files gets scanned:

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

A file added to the catalog shows up the next time the same query is run:

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
zero copy via Arrow FFI and sources can produce large chunks of data before returning, IO plugins
are interfaced via Python for now, as the short time the GIL is needed is not expected to lead to
any contention.

E.g. an IO source can read their dataframe's in rust and only at the rendez-vous move the data
zero-copy having only a short time the GIL is needed.

### Use case

IO plugins are useful for source files not supported by Polars that should still benefit from
optimizations like projection pushdown, predicate pushdown, early stopping and support of the
streaming engine.

### Example

This example writes a simple, very bad, custom CSV source and registers it as an IO plugin. Note, it
is only given for learning purposes.

First, the imports that are needed:

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
simple csv parser the data is always read as `pl.String`. The only thing that differs are the field
names and the number of fields.

```python
def parse_schema(csv_str: str) -> pl.Schema:
    first_line = csv_str.split("\n")[0]

    return pl.Schema({k: pl.String for k in first_line.split(",")})
```

Running this with the small csv file `"a,b,c\n1,2,3"` gives the schema:
`Schema([('a', String), ('b', String), ('c', String)])`.

```python
>>> print(parse_schema("a,b,c\n1,2,3"))
Schema([('a', String), ('b', String), ('c', String)])
```

#### Writing the source

Next up is the actual source, which consists of an outer and an inner function. The outer function
`my_scan_csv` is the user facing function. This function will accept the file name and other
potential arguments needed for reading the source. For csv files, these arguments could be
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
wherever the IO plugin is written. For an IO source implemented in Rust, see the
[plugins repository](https://github.com/pola-rs/pyo3-polars/tree/main/example/io_plugin).

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

        # Don't read more rows than requested by the engine.
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

            # A performant reader would not read these
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

Finally, the source can be tested:

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

This section is for people writing a Polars execution engine, and can be skipped when writing an IO
source.

Polars does not call the source directly. It calls a wrapper that deserializes the predicate and
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
