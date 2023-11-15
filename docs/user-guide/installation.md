# Installation

Polars is a library and installation is as simple as invoking the package manager of the corresponding programming language.

=== ":fontawesome-brands-python: Python"

    ``` bash
    pip install polars
    ```

=== ":fontawesome-brands-rust: Rust"

    ``` shell
    cargo add polars -F lazy

    # Or Cargo.toml
    [dependencies]
    polars = { version = "x", features = ["lazy", ...]}
    ```

## Importing

To use the library import it into your project

=== ":fontawesome-brands-python: Python"

    ``` python
    import polars as pl
    ```

=== ":fontawesome-brands-rust: Rust"

    ``` rust
    use polars::prelude::*;
    ```

## Feature Flags

By using the above command you install the core of Polars onto your system. However depending on your use case you might want to install the optional dependencies as well. These are made optional to minimize the footprint. The flags are different depending on the programming language. Throughout the user guide we will mention when a functionality is used that requires an additional dependency.

### Python

```text
# For example
pip install polars[numpy, fsspec]
```

| Tag        | Description                                                                                                                           |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| all        | Install all optional dependencies (all of the following)                                                                              |
| pandas     | Install with Pandas for converting data to and from Pandas Dataframes/Series                                                          |
| numpy      | Install with numpy for converting data to and from numpy arrays                                                                       |
| pyarrow    | Reading data formats using PyArrow                                                                                                    |
| fsspec     | Support for reading from remote file systems                                                                                          |
| connectorx | Support for reading from SQL databases                                                                                                |
| xlsx2csv   | Support for reading from Excel files                                                                                                  |
| deltalake  | Support for reading from Delta Lake Tables                                                                                            |
| timezone   | Timezone support, only needed if 1. you are on Python < 3.9 and/or 2. you are on Windows, otherwise no dependencies will be installed |

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
- `lazy` - Lazy API
    - `lazy_regex` - Use regexes in [column selection](crate::lazy::dsl::col)
    - `dot_diagram` - Create dot diagrams from lazy logical plans.
- `sql` - Pass SQL queries to polars.
- `streaming` - Be able to process datasets that are larger than RAM.
- `random` - Generate arrays with randomly sampled values
- `ndarray`- Convert from `DataFrame` to `ndarray`
- `temporal` - Conversions between [Chrono](https://docs.rs/chrono/) and Polars for temporal data types
- `timezones` - Activate timezone support.
- `strings` - Extra string utilities for `Utf8Chunked`
    - `string_pad` - `pad_start`, `pad_end`, `zfill`
    - `string_to_integer` - `parse_int`
- `object` - Support for generic ChunkedArrays called `ObjectChunked<T>` (generic over `T`).
  These are downcastable from Series through the [Any](https://doc.rust-lang.org/std/any/index.html) trait.
- Performance related:
    - `nightly` - Several nightly only features such as SIMD and specialization.
    - `performant` - more fast paths, slower compile times.
    - `bigidx` - Activate this feature if you expect >> 2^32 rows. This has not been needed by anyone.
    This allows polars to scale up way beyond that by using `u64` as an index.
    Polars will be a bit slower with this feature activated as many data structures
    are less cache efficient.
    - `cse` - Activate common subplan elimination optimization
- IO related:
    - `serde` - Support for [serde](https://crates.io/crates/serde) serialization and deserialization.
    Can be used for JSON and more serde supported serialization formats.
    - `serde-lazy` - Support for [serde](https://crates.io/crates/serde) serialization and deserialization.
    Can be used for JSON and more serde supported serialization formats.
    - `parquet` - Read Apache Parquet format
    - `json` - JSON serialization
    - `ipc` - Arrow's IPC format serialization
    - `decompress` - Automatically infer compression of csvs and decompress them.
    Supported compressions:
      - zip
      - gzip

- `DataFrame` operations:
    - `dynamic_group_by` - Group by based on a time window instead of predefined keys.
    Also activates rolling window group by operations.
    - `sort_multiple` - Allow sorting a `DataFrame` on multiple columns
    - `rows` - Create `DataFrame` from rows and extract rows from `DataFrames`.
    And activates `pivot` and `transpose` operations
    - `join_asof` - Join ASOF, to join on nearest keys instead of exact equality match.
    - `cross_join` - Create the cartesian product of two DataFrames.
    - `semi_anti_join` - SEMI and ANTI joins.
    - `group_by_list` - Allow group by operation on keys of type List.
    - `row_hash` - Utility to hash DataFrame rows to UInt64Chunked
    - `diagonal_concat` - Concat diagonally thereby combining different schemas.
    - `horizontal_concat` - Concat horizontally and extend with null values if lengths don't match
    - `dataframe_arithmetic` - Arithmetic on (Dataframe and DataFrames) and (DataFrame on Series)
    - `partition_by` - Split into multiple DataFrames partitioned by groups.
- `Series`/`Expression` operations:
    - `is_in` - [Check for membership in `Series`](crate::chunked_array::ops::IsIn)
    - `zip_with` - [Zip two Series/ ChunkedArrays](crate::chunked_array::ops::ChunkZip)
    - `round_series` - round underlying float types of `Series`.
    - `repeat_by` - [Repeat element in an Array N times, where N is given by another array.
    - `is_first_distinct` - Check if element is first unique value.
    - `is_last_distinct` - Check if element is last unique value.
    - `checked_arithmetic` - checked arithmetic/ returning `None` on invalid operations.
    - `dot_product` - Dot/inner product on Series and Expressions.
    - `concat_str` - Concat string data in linear time.
    - `reinterpret` - Utility to reinterpret bits to signed/unsigned
    - `take_opt_iter` - Take from a Series with `Iterator<Item=Option<usize>>`
    - `mode` - [Return the most occurring value(s)](crate::chunked_array::ops::ChunkUnique::mode)
    - `cum_agg` - cumsum, cummin, cummax aggregation.
    - `rolling_window` - rolling window functions, like rolling_mean
    - `interpolate` [interpolate None values](crate::chunked_array::ops::Interpolate)
    - `extract_jsonpath` - [Run jsonpath queries on Utf8Chunked](https://goessner.net/articles/JsonPath/)
    - `list` - List utils.
      - `list_take` take sublist by multiple indices
    - `rank` - Ranking algorithms.
    - `moment` - kurtosis and skew statistics
    - `ewma` - Exponential moving average windows
    - `abs` - Get absolute values of Series
    - `arange` - Range operation on Series
    - `product` - Compute the product of a Series.
    - `diff` - `diff` operation.
    - `pct_change` - Compute change percentages.
    - `unique_counts` - Count unique values in expressions.
    - `log` - Logarithms for `Series`.
    - `list_to_struct` - Convert `List` to `Struct` dtypes.
    - `list_count` - Count elements in lists.
    - `list_eval` - Apply expressions over list elements.
    - `cumulative_eval` - Apply expressions over cumulatively increasing windows.
    - `arg_where` - Get indices where condition holds.
    - `search_sorted` - Find indices where elements should be inserted to maintain order.
    - `date_offset` Add an offset to dates that take months and leap years into account.
    - `trigonometry` Trigonometric functions.
    - `sign` Compute the element-wise sign of a Series.
    - `propagate_nans` NaN propagating min/max aggregations.
- `DataFrame` pretty printing
    - `fmt` - Activate DataFrame formatting

<!-- dprint-ignore-end -->
