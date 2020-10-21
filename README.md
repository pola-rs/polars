# Polars
[![rust docs](https://docs.rs/polars/badge.svg)](https://docs.rs/polars/latest/polars/)
![Build, test and docs](https://github.com/ritchie46/polars/workflows/Build,%20test%20and%20docs/badge.svg)
[![](http://meritbadge.herokuapp.com/polars)](https://crates.io/crates/polars)
[![Gitter](https://badges.gitter.im/polars-rs/community.svg)](https://gitter.im/polars-rs/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

## Blazingly fast  DataFrames in Rust

Polars is a blazingly fast DataFrames library implemented in Rust. Its memory model uses Apache Arrow as backend. 

It currently consists of an eager API similar to pandas and a lazy API that is somewhat similar to spark. 
Amongst more, Polars has the following functionalities.


| Functionality                                     | Eager | Lazy (DataFrame) | Lazy (Series) |
|---------------------------------------------------|-------|------------------|---------------|
| Filters                                           | ✔     | ✔                | ✔             |
| Shifts                                            | ✔     | ✗                | ✔             |
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
| Aggregations                                      | ✔     | ✗                | ✔             |
| Find unique values                                | ✔     |                  | ✗             |
| Rust iterators                                    | ✔     |                  | ✗             |
| IO (csv, json, parquet, Arrow IPC                 | ✔     | ✗                |               |
| Query optimization: (predicate pushdown)          | ✗     | ✔                |               |
| Query optimization: (projection pushdown)         | ✗     | ✔                |               |
| Query optimization: (type coercion)               | ✗     | ✔                |               |

**Note that almost all eager operations supported by Eager on `Series`/`ChunkedArrays` can be used in Lazy via UDF's**


## Documentation
Want to know about all the features Polars supports? [Check the current master docs](https://ritchie46.github.io/polars). 

Most features are described on the [DataFrame](https://ritchie46.github.io/polars/polars/frame/struct.DataFrame.html), 
[Series](https://ritchie46.github.io/polars/polars/series/enum.Series.html), and [ChunkedArray](https://ritchie46.github.io/polars/polars/chunked_array/struct.ChunkedArray.html)
structs in that order. For `ChunkedArray` a lot of functionality is also defined by `Traits` in the 
[ops module](https://ritchie46.github.io/polars/polars/chunked_array/ops/index.html).
Other useful parts of the documentation are:
* [Time/ DateTime utilities](https://ritchie46.github.io/polars/polars/doc/time/index.html)
* [Groupby, aggregations and pivots](https://ritchie46.github.io/polars/polars/frame/group_by/struct.GroupBy.html)
* [Lazy DataFrame](https://ritchie46.github.io/polars/polars/lazy/frame/struct.LazyFrame.html)


## Performance
Polars is written to be performant. Below are some comparisons with the (also very fast) Pandas DataFrame library.

#### GroupBy
![](pandas_cmp/img/groupby10_.png)

#### Joins
![](pandas_cmp/img/join_80_000.png)


## First run in Rust
Take a look at the [10 minutes to Polars notebook](examples/10_minutes_to_polars.ipynb) to get you started.
Want to run the notebook yourself? Clone the repo and run `$ cargo c && docker-compose up`. This will spin up a jupyter
notebook on `http://localhost:8891`. The notebooks are in the `/examples` directory.
 
Oh yeah.. and get a cup of coffee because compilation will take a while during the first run.

## First run in Python
A subset of the Polars functionality is also exposed through Python bindings. You can install them with:

`$ pip install py-polars`

Next you can check the [10 minutes to py-polars notebook](examples/10_minutes_to_pypolars.ipynb) or take a look 
at the [reference](https://py-polars.readthedocs.io/en/latest/).



## Features

Additional cargo features:

* `pretty` (default)
    - pretty printing of DataFrames
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
