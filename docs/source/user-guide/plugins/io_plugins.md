# IO Plugins

Besides [expression plugins](./expr_plugins.md), we also support IO plugins. These allow you to
register different file formats as sources to the Polars engines. Because sources can move data zero
copy via Arrow FFI and sources can produce large chunks of data before returning, we've decided to
interface to IO plugins via Python for now, as we don't think the short time the GIL is needed
should lead to any contention.

E.g. an IO source can read their dataframe's in rust and only at the rendez-vous move the data
zero-copy having only a short time the GIL is needed.

## Use case

You want IO plugins if you have a source file not supported by Polars and you want to benefit from
optimizations like projection pushdown, predicate pushdown, early stopping and support of our
streaming engine.

## Example

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

### Parsing the schema

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

### Writing the source

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

            df = pl.from_records(rows, schema=schema)
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

    return register_io_source(callable=source_generator, schema=schema)
```

### Taking it for a (very slow) spin

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
│ 1   ┆ 9   ┆ 1   ┆ 1   │
│ 2   ┆ 10  ┆ 2   ┆ 122 │
│ 3   ┆ 11  ┆ 3   ┆ 3   │
│ 4   ┆ 2   ┆ 4   ┆ 4   │
└─────┴─────┴─────┴─────┘
shape: (2, 2)
┌─────┬─────┐
│ a   ┆ b   │
│ --- ┆ --- │
│ str ┆ str │
╞═════╪═════╡
│ 1   ┆ 9   │
│ 2   ┆ 10  │
└─────┴─────┘
```

## Further reading

- [Rust example (distribution source)](https://github.com/pola-rs/pyo3-polars/tree/main/example/io_plugin)
