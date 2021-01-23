# Polars
[![rust docs](https://docs.rs/polars/badge.svg)](https://docs.rs/polars/latest/polars/)
![Build and test](https://github.com/ritchie46/polars/workflows/Build%20and%20test/badge.svg)
[![](http://meritbadge.herokuapp.com/polars)](https://crates.io/crates/polars)
[![Gitter](https://badges.gitter.im/polars-rs/community.svg)](https://gitter.im/polars-rs/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

## Blazingly fast DataFrames in Rust & Python

Polars is a blazingly fast DataFrames library implemented in Rust. Its memory model uses Apache Arrow as backend. 

It currently consists of an eager API similar to pandas and a lazy API that is somewhat similar to spark. 
Amongst more, Polars has the following functionalities.

To learn more about the inner workings of Polars read the [WIP book](https://ritchie46.github.io/polars-book/).


| Functionality                                     | Eager | Lazy (DataFrame) | Lazy (Series) |
|---------------------------------------------------|-------|------------------|---------------|
| Filters                                           | ✔     | ✔                | ✔             |
| Shifts                                            | ✔     | ✔                | ✔             |
| Joins                                             | ✔     | ✔                |               |
| GroupBys + aggregations                           | ✔     | ✔                |               |
| Comparisons                                       | ✔     | ✔                | ✔             |
| Arithmetic                                        | ✔     |                  | ✔             |
| Sorting                                           | ✔     | ✔                | ✔             |
| Reversing                                         | ✔     | ✔                | ✔             |
| Closure application (User Defined Functions)      | ✔     |                  | ✔             |
| SIMD                                              | ✔     |                  | ✔             |
| Pivots                                            | ✔     | ✗                |               |
| Melts                                             | ✔     | ✗                |               |
| Filling nulls + fill strategies                   | ✔     | ✗                | ✔             |
| Aggregations                                      | ✔     | ✔                | ✔             |
| Moving Window aggregates                          | ✔     | ✗                | ✗             |
| Find unique values                                | ✔     |                  | ✗             |
| Rust iterators                                    | ✔     |                  | ✔             |
| IO (csv, json, parquet, Arrow IPC                 | ✔     | ✗                |               |
| Query optimization: (predicate pushdown)          | ✗     | ✔                |               |
| Query optimization: (projection pushdown)         | ✗     | ✔                |               |
| Query optimization: (type coercion)               | ✗     | ✔                |               |
| Query optimization: (simplify expressions)        | ✗     | ✔                |               |
| Query optimization: (aggregate pushdown)          | ✗     | ✔                |               |

**Note that almost all eager operations supported by Eager on `Series`/`ChunkedArrays` can be used in Lazy via UDF's**


## Documentation
Want to know about all the features Polars support? Read the docs!

#### Rust
* [Documentation (stable)](https://docs.rs/polars/latest/polars/). 
* [Documentation (master branch)](https://ritchie46.github.io/polars). 
    * [DataFrame](https://ritchie46.github.io/polars/polars/frame/struct.DataFrame.html) 
    * [Series](https://ritchie46.github.io/polars/polars/series/enum.Series.html)
    * [ChunkedArray](https://ritchie46.github.io/polars/polars/chunked_array/struct.ChunkedArray.html)
    * [Traits for ChunkedArray](https://ritchie46.github.io/polars/polars/chunked_array/ops/index.html)
    * [Time/ DateTime utilities](https://ritchie46.github.io/polars/polars/doc/time/index.html)
    * [Groupby, aggregations and pivots](https://ritchie46.github.io/polars/polars/frame/group_by/struct.GroupBy.html)
    * [Lazy DataFrame](https://ritchie46.github.io/polars/polars/lazy/frame/struct.LazyFrame.html)
* [the book](https://ritchie46.github.io/polars-book/)
* [10 minutes to Polars notebook](examples/10_minutes_to_polars.ipynb)
    
#### Python
* installation guide: `pip install py-polars`
* [the book](https://ritchie46.github.io/polars-book/)
* [Reference guide](https://ritchie46.github.io/polars/pypolars/index.html)

## Performance
Polars is written to be performant, and it is! But don't take my word for it, take a look at the results in 
[h2oai's db-benchmark](https://h2oai.github.io/db-benchmark/).

## Cargo Features

Additional cargo features:

* `temporal (default)`
    - Conversions between Chrono and Polars for temporal data
* `simd (default)`
    - SIMD operations
* `parquet`
    - Read Apache Parquet format
* `json`
    - Json serialization
* `ipc`
    - Arrow's IPC format serialization
* `random`
    - Generate array's with randomly sampled values
* `ndarray`
    - Convert from `DataFrame` to `ndarray`
* `lazy`
    - Lazy api
* `strings`
    - String utilities for `Utf8Chunked`
* `object`
    - Support for generic ChunkedArray's called `ObjectChunked<T>` (generic over `T`). 
      These will downcastable from Series through the [Any](https://doc.rust-lang.org/std/any/index.html) trait.

## Contribution
Want to contribute? Read our [contribution guideline](./CONTRIBUTING.md).


## Env vars
* `POLARS_PAR_SORT_BOUND` -> Sets the lower bound of rows at which Polars will use a parallel sorting algorithm.
                             Default is 1M rows.
* `POLARS_FMT_MAX_COLS` -> maximum number of columns shown when formatting DataFrames.
* `POLARS_FMT_MAX_ROWS` -> maximum number of rows shown when formatting DataFrames.
* `POLARS_TABLE_WIDTH` -> width of the tables used during DataFrame formatting.
* `POLARS_MAX_THREADS` -> maximum number of threads used in join algorithm. Default is unbounded.
