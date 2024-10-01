# Installation

Polars is a library and installation is as simple as invoking the package manager of the corresponding programming language.

=== ":fontawesome-brands-python: Python"

    ``` bash
    pip install polars

    # Or for legacy CPUs without AVX2 support
    pip install polars-lts-cpu
    ```

=== ":fontawesome-brands-rust: Rust"

    ``` shell
    cargo add polars -F lazy

    # Or Cargo.toml
    [dependencies]
    polars = { version = "x", features = ["lazy", ...]}
    ```

## Big Index

By default, Polars dataframes are limited to 2<sup>32</sup> rows (~4.3 billion).
Increase this limit to 2<sup>64</sup> (~18 quintillion) by enabling the big index extension:

=== ":fontawesome-brands-python: Python"

    ``` bash
    pip install polars-u64-idx
    ```

=== ":fontawesome-brands-rust: Rust"

    ``` shell
    cargo add polars -F bigidx

    # Or Cargo.toml
    [dependencies]
    polars = { version = "x", features = ["bigidx", ...] }
    ```

## Legacy CPU

To install Polars for Python on an old CPU without [AVX](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions) support, run:

=== ":fontawesome-brands-python: Python"

    ``` bash
    pip install polars-lts-cpu
    ```

## Importing

To use the library, simply import it into your project:

=== ":fontawesome-brands-python: Python"

    ``` python
    import polars as pl
    ```

=== ":fontawesome-brands-rust: Rust"

    ``` rust
    use polars::prelude::*;
    ```

## Feature flags

By using the above command you install the core of Polars onto your system.
However, depending on your use case, you might want to install the optional dependencies as well.
These are made optional to minimize the footprint.
The flags are different depending on the programming language.
Throughout the user guide we will mention when a functionality used requires an additional dependency.

### Python

```text
# For example
pip install 'polars[numpy,fsspec]'
```

#### All

| Tag | Description                        |
| --- | ---------------------------------- |
| all | Install all optional dependencies. |

#### GPU

| Tag | Description                 |
| --- | --------------------------- |
| gpu | Run queries on NVIDIA GPUs. |

!!! note

    To install the GPU engine, you need to pass
    `--extra-index-url=https://pypi.nvidia.com` to `pip`. See [GPU
    support](gpu-support.md) for more detailed instructions and
    prerequisites.

#### Interoperability

| Tag      | Description                                        |
| -------- | -------------------------------------------------- |
| pandas   | Convert data to and from pandas dataframes/series. |
| numpy    | Convert data to and from NumPy arrays.             |
| pyarrow  | Convert data to and from PyArrow tables/arrays.    |
| pydantic | Convert data from Pydantic models to Polars.       |

#### Excel

| Tag        | Description                                      |
| ---------- | ------------------------------------------------ |
| calamine   | Read from Excel files with the calamine engine.  |
| openpyxl   | Read from Excel files with the openpyxl engine.  |
| xlsx2csv   | Read from Excel files with the xlsx2csv engine.  |
| xlsxwriter | Write to Excel files with the XlsxWriter engine. |
| excel      | Install all supported Excel engines.             |

#### Database

| Tag        | Description                                                                          |
| ---------- | ------------------------------------------------------------------------------------ |
| adbc       | Read from and write to databases with the Arrow Database Connectivity (ADBC) engine. |
| connectorx | Read from databases with the ConnectorX engine.                                      |
| sqlalchemy | Write to databases with the SQLAlchemy engine.                                       |
| database   | Install all supported database engines.                                              |

#### Cloud

| Tag    | Description                                 |
| ------ | ------------------------------------------- |
| fsspec | Read from and write to remote file systems. |

#### Other I/O

| Tag       | Description                          |
| --------- | ------------------------------------ |
| deltalake | Read from and write to Delta tables. |
| iceberg   | Read from Apache Iceberg tables.     |

#### Other

| Tag         | Description                                     |
| ----------- | ----------------------------------------------- |
| async       | Collect LazyFrames asynchronously.              |
| cloudpickle | Serialize user-defined functions.               |
| graph       | Visualize LazyFrames as a graph.                |
| plot        | Plot dataframes through the `plot` namespace.   |
| style       | Style dataframes through the `style` namespace. |
| timezone    | Timezone support[^note].                        |

[^note]: Only needed if you are on Python < 3.9 or you are on Windows.

### Rust

```toml
# Cargo.toml
[dependencies]
polars = { version = "0.26.1", features = ["lazy", "temporal", "describe", "json", "parquet", "dtype-datetime"] }
```

The opt-in features are:

<!-- dprint-ignore-start -->

- Additional data types:
    - `dtype-date`
    - `dtype-datetime`
    - `dtype-time`
    - `dtype-duration`
    - `dtype-i8`
    - `dtype-i16`
    - `dtype-u8`
    - `dtype-u16`
    - `dtype-categorical`
    - `dtype-struct`
- `lazy` - Lazy API:
    - `regex` - Use regexes in [column selection](crate::lazy::dsl::col).
    - `dot_diagram` - Create dot diagrams from lazy logical plans.
- `sql` - Pass SQL queries to Polars.
- `streaming` - Be able to process datasets that are larger than RAM.
- `random` - Generate arrays with randomly sampled values
- `ndarray`- Convert from `DataFrame` to `ndarray`
- `temporal` - Conversions between [Chrono](https://docs.rs/chrono/) and Polars for temporal data types
- `timezones` - Activate timezone support.
- `strings` - Extra string utilities for `StringChunked`:
    - `string_pad` - for `pad_start`, `pad_end`, `zfill`.
    - `string_to_integer` - for `parse_int`.
- `object` - Support for generic ChunkedArrays called `ObjectChunked<T>` (generic over `T`).
  These are downcastable from Series through the [Any](https://doc.rust-lang.org/std/any/index.html) trait.
- Performance related:
    - `nightly` - Several nightly only features such as SIMD and specialization.
    - `performant` - more fast paths, slower compile times.
    - `bigidx` - Activate this feature if you expect >> 2<sup>32</sup> rows.
    This allows polars to scale up way beyond that by using `u64` as an index.
    Polars will be a bit slower with this feature activated as many data structures
    are less cache efficient.
    - `cse` - Activate common subplan elimination optimization.
- IO related:
    - `serde` - Support for [serde](https://crates.io/crates/serde) serialization and deserialization.
    Can be used for JSON and more serde supported serialization formats.
    - `serde-lazy` - Support for [serde](https://crates.io/crates/serde) serialization and deserialization.
    Can be used for JSON and more serde supported serialization formats.
    - `parquet` - Read Apache Parquet format.
    - `json` - JSON serialization.
    - `ipc` - Arrow's IPC format serialization.
    - `decompress` - Automatically infer compression of csvs and decompress them.
    Supported compressions:
      - zip
      - gzip
- Dataframe operations:
    - `dynamic_group_by` - Group by based on a time window instead of predefined keys.
    Also activates rolling window group by operations.
    - `sort_multiple` - Allow sorting a dataframe on multiple columns.
    - `rows` - Create dataframe from rows and extract rows from `dataframes`.
    Also activates `pivot` and `transpose` operations.
    - `join_asof` - Join ASOF, to join on nearest keys instead of exact equality match.
    - `cross_join` - Create the Cartesian product of two dataframes.
    - `semi_anti_join` - SEMI and ANTI joins.
    - `row_hash` - Utility to hash dataframe rows to `UInt64Chunked`.
    - `diagonal_concat` - Diagonal concatenation thereby combining different schemas.
    - `dataframe_arithmetic` - Arithmetic between dataframes and other dataframes or series.
    - `partition_by` - Split into multiple dataframes partitioned by groups.
- Series/expression operations:
    - `is_in` - [Check for membership in series](crate::chunked_array::ops::IsIn)
    - `zip_with` - [Zip two `Series` / `ChunkedArray`s](crate::chunked_array::ops::ChunkZip)
    - `round_series` - round underlying float types of series.
    - `repeat_by` - Repeat element in an array a number of times specified by another array.
    - `is_first_distinct` - Check if element is first unique value.
    - `is_last_distinct` - Check if element is last unique value.
    - `checked_arithmetic` - checked arithmetic returning `None` on invalid operations.
    - `dot_product` - Dot/inner product on series and expressions.
    - `concat_str` - Concatenate string data in linear time.
    - `reinterpret` - Utility to reinterpret bits to signed/unsigned.
    - `take_opt_iter` - Take from a series with `Iterator<Item=Option<usize>>`.
    - `mode` - [Return the most frequently occurring value(s)](crate::chunked_array::ops::ChunkUnique::mode).
    - `cum_agg` - `cum_sum`, `cum_min`, and `cum_max`, aggregations.
    - `rolling_window` - rolling window functions, like `rolling_mean`.
    - `interpolate` - [interpolate `None` values](crate::chunked_array::ops::Interpolate).
    - `extract_jsonpath` - [Run `jsonpath` queries on `StringChunked`](https://goessner.net/articles/JsonPath/).
    - `list` - List utils:
      - `list_gather` - take sublist by multiple indices.
    - `rank` - Ranking algorithms.
    - `moment` - Kurtosis and skew statistics.
    - `ewma` - Exponential moving average windows.
    - `abs` - Get absolute values of series.
    - `arange` - Range operation on series.
    - `product` - Compute the product of a series.
    - `diff` - `diff` operation.
    - `pct_change` - Compute change percentages.
    - `unique_counts` - Count unique values in expressions.
    - `log` - Logarithms for series.
    - `list_to_struct` - Convert `List` to `Struct` data types.
    - `list_count` - Count elements in lists.
    - `list_eval` - Apply expressions over list elements.
    - `cumulative_eval` - Apply expressions over cumulatively increasing windows.
    - `arg_where` - Get indices where condition holds.
    - `search_sorted` - Find indices where elements should be inserted to maintain order.
    - `offset_by` - Add an offset to dates that take months and leap years into account.
    - `trigonometry` - Trigonometric functions.
    - `sign` - Compute the element-wise sign of a series.
    - `propagate_nans` - `NaN`-propagating min/max aggregations.
- Dataframe pretty printing:
    - `fmt` - Activate dataframe formatting.

<!-- dprint-ignore-end -->
