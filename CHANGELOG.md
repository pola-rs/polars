# Changelog Polars (Rust crate)

The Python bindings `polars` have their own changelog.

## \[future\] Polars v0.13

* performance
  - Vast reduction of compile times by making compilation dtypes of Series opt-in.
  - Fast multi-threaded csv parser. Fixes multiple gripes of old parser.

* features
  - Series / ChunkedArray implementations
    * Series::week
    * Series::weekday
    * Series::arg_min
    * Series::arg_max

* breaking
  - ChunkedArray::arg_unique return UInt32Chunked instead of Vec<u32>

* bug fixes
  - various

## Polars v0.12
* Lot's of bug fixes

## Polars v0.10 / v0.11

* CSV Read IO
    - Parallel csv reader
* Sample DataFrames/ Series
* Performance increase in take kernel
* Performance increase in ChunkedArray builders
* Join operation on multiple columns.
* ~3.5 x performance increase in groupby operations (measured on db-benchmark),
  due to embarrassingly parallel grouping and better branch prediction (tight loops).
* Performance increase on join operation due to better branch prediction.
* Categorical datatype and global string cache (BETA).

* Lazy
    - Lot's of bug fixes in optimizer.
    - Parallel execution of Physical plan
    - Partition window function
    - More simplify expression optimizations.
    - Caching
    - Alpha release of Aggregate pushdown optimization.
* Start of general Object type in ChunkedArray/DataFrames/Series
