# Polars Core

This crate is a submodule of Polars. It contains the core functionality for the Polars DataFrame library.

# Features

| Feature              | Description                                                                                                   |
| -------------------- | ------------------------------------------------------------------------------------------------------------- |
| simd                 | Enables the use of SIMD instructions to accelerate computations.                                              |
| nightly              | Enables the use of unstable and experimental features that are only available on the nightly version of Rust. |
| avx512               | Enables the use of AVX-512 instructions to accelerate computations.                                           |
| docs                 | Includes documentation-related features.                                                                      |
| temporal             | temporal data types: `Datetime`, `Date`, `Duration` and `Time`.                                               |
| random               | the `Random` RankMethod                                                                                       |
| default              | Includes default features (docs, temporal, and private).                                                      |
| lazy                 | lazy evaluation.                                                                                              |
| performant           | Provides faster operations at the expense of compilation time.                                                |
| strings              | Provides extra utilities for Utf8Chunked.                                                                     |
| object               | ObjectChunked<T> (downcastable Series of any type).                                                           |
| fmt                  | Pretty formatting of DataFrames                                                                               |
| fmt_no_tty           | Pretty formatting of DataFrames without tty.                                                                  |
| sort_multiple        | sorting by multiple columns.                                                                                  |
| rows                 | Enables the creation of a DataFrame from row values and includes `pivot` operation.                           |
| is_in                | `is_in` operation.                                                                                            |
| zip_with             | Enables the use of `hmin, hmax, zip_with, zip_with_same_type` operations.                                     |
| round_series         | Enables the use of rounding operations: (`round, floor, ceil, clip, clip_max, clip_min`)                      |
| checked_arithmetic   | Enables checked arithmetic (addition, subtraction, multiplication, and division) for integer data types.      |
| repeat_by            | `repeat_by` operation.                                                                                        |
| is_first             | `is_first` operation.                                                                                         |
| is_last              | `is_last` operation.                                                                                          |
| asof_join            | Enables the `AsOf` Join Type.                                                                                 |
| cross_join           | Enables the `Cross` Join Type.                                                                                |
| dot_product          | `dot` operation.                                                                                              |
| concat_str           | `concat_str` operation.                                                                                       |
| row_hash             | `row_hash` operation.                                                                                         |
| mode                 | `mode` operation.                                                                                             |
| cum_agg              | Enables cumulative aggregation (cumsum, cummin, etc.).                                                        |
| rolling_window       | Enables the use of rolling window functions.                                                                  |
| diff                 | `diff` operation.                                                                                             |
| moment               | `moment` operation.                                                                                           |
| diagonal_concat      | `diagonal_concat` operation.                                                                                  |
| horizontal_concat    | `horizontal_concat` operation.                                                                                |
| abs                  | `abs` operation.                                                                                              |
| ewma                 | Enables exponential weighted moving average (EWMA) support.                                                   |
| dataframe_arithmetic | Enables arithmetic operations between DataFrames.                                                             |
| product              | `product` operation.                                                                                          |
| unique_counts        | `unique_counts` operation.                                                                                    |
| partition_by         | `partition_by` operation.                                                                                     |
| semi_anti_join       | `semi_anti_join` operation.                                                                                   |
| chunked_ids          | `chunked_ids` operation.                                                                                      |
| describe             | `describe` operation.                                                                                         |
| timezones            | Enables timezone parsing via chrono-tz                                                                        |
| dynamic_groupby      | `dynamic_groupby` operation                                                                                   |
| dtype-date           | `Date` DataType                                                                                               |
| dtype-datetime       | `Datetime` DataType                                                                                           |
| dtype-time           | `Time` DataType.                                                                                              |
| dtype-i8             | `Int8` DataType                                                                                               |
| dtype-i16            | `Int16` DataType                                                                                              |
| dtype-u8             | `UInt8` DataType                                                                                              |
| dtype-u16            | `UInt16` DataType                                                                                             |
| dtype-decimal        | `Decimal` DataType                                                                                            |
| dtype-categorical    | `Categorical` DataType                                                                                        |
| dtype-struct         | `Struct` DataType                                                                                             |
| parquet              | Enables reading & writing for Parquet files.                                                                  |
| bigidx               | large arrays and indexes in the code.                                                                         |
| python               | Python integration.                                                                                           |
| serde                | (de)serialization of Polars types                                                                             |
| serde-lazy           | (de)serialization LazyFrames & Exprs                                                                          |
| async                | async support.                                                                                                |
| aws                  | reading from AWS S3 object store.                                                                             |
| azure                | reading from Azure Blob Storage.                                                                              |
| gcp                  | reading from Google Cloud Storage.                                                                            |
| private              | Enables private APIs. <sup> [1](#footnote1)</sup>                                                             |

<sup><a name="footnote1">1</a></sup> Private APIs in `polars` are not intended for public use and may change without notice. Use at your own risk.
