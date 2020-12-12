# Polars
[![rust docs](https://docs.rs/polars/badge.svg)](https://docs.rs/polars/latest/polars/)
![Build and test](https://github.com/ritchie46/polars/workflows/Build%20and%20test/badge.svg)
[![](http://meritbadge.herokuapp.com/polars)](https://crates.io/crates/polars)
[![Gitter](https://badges.gitter.im/polars-rs/community.svg)](https://gitter.im/polars-rs/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

## Blazingly fast DataFrames in Rust & Python

Polars is a blazingly fast DataFrames library implemented in Rust. Its memory model uses Apache Arrow as backend. 

It currently consists of an eager API similar to pandas and a lazy API that is somewhat similar to spark. 
Amongst more, Polars has the following functionalities.

To learn more about the inner workings of Polars read the [WIP book](https://ritchie46.github.io/polars-book/book/book/index.html).


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
* [the book](https://ritchie46.github.io/polars-book/book/book/index.html)
* [10 minutes to Polars notebook](examples/10_minutes_to_polars.ipynb)
    
#### Python
* installation guide: `pip install py-polars`
* [the book](https://ritchie46.github.io/polars-book/book/book/index.html)
* [Reference guide](https://ritchie46.github.io/polars/pypolars/index.html)
* [10 minutes to py-polars notebook](examples/10_minutes_to_pypolars.ipynb)
* [lazy py-polars notebook](examples/lazy_py-polars.ipynb)

## Performance
Polars is written to be performant. Below are some comparisons with pandas and pydatatable DataFrame library **(lower is better)**.

#### GroupBy
![](pandas_cmp/img/groupby10_.png)
![](pandas_cmp/img/groupby10_mem.png)

#### Joins
![](pandas_cmp/img/join_80_000.png)

## Cargo Features

Additional cargo features:

* `temporal (default)`
    - Conversions between Chrono and Polars for temporal data
* `simd (default)`
    - SIMD operations
* `parquet`
    - Read Apache Parquet format
* `random`
    - Generate array's with randomly sampled values
* `ndarray`
    - Convert from `DataFrame` to `ndarray`
* `lazy`
    - Lazy api
* `strings`
    - String utilities for `Utf8Chunked`

## Contribution
Want to contribute? Read our [contribution guideline](./CONTRIBUTING.md).


## Env vars
* `POLARS_PAR_COLUMN_BP` -> breakpoint for (some) parallel operations on columns. 
    If the number of columns exceeds this it will run in parallel
* `POLARS_FMT_MAX_COLS` -> maximum number of columns shown when formatting DataFrames
* `POLARS_FMT_MAX_ROW` -> maximum number of rows shown when formatting DataFrames
* `POLARS_TABLE_WIDTH` -> width of the tables used during DataFrame formatting
