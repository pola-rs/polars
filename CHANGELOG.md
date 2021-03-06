# Changelog polars/ py-polars

## \[future\] Polars v0.13

### \[future\] py-polars 0.7.0
* feature
  - \[python\] add parquet compression
  - \[python\] implicitly download raw files from the web in `read_parquet`, `read_csv`.
  
* bug fix
  - \[python\] support file buffers for reading and writing csv and parquet
  - \[python | rust\] fix csv-parser: allow new-line character in a string field
  - \[python | rust\] don't let predicate-pushdown pass shift operation to maintain correctness.

### py-polars 0.6.7
* performance
  - \[python | rust\] use mimalloc global allocator
  - \[python | rust\] undo performance regression on large number of threads
* bug fix
  - \[python | rust\] fix accidental over-allocation in csv-parser
  - \[python\] support agg (dictionary aggregation) for downsample

### py-polars 0.6.6
* performance
  - \[python | rust\] categorical type groupby keys (use size hint)
  - \[python | rust\] remove indirection layer in vector hasher
  - \[python | rust\] improve performance of null array creation
* bug fix
  - \[python\] implement set_with_mask for Boolean type
  - \[python | rust\] don't panic (instead return null) in dataframe aggregation `std` and `var`
* other
  - \[rust\] internal refactors


### py-polars 0.6.5
* bug fix
  - \[python\] fix various pyarrow related bugs
  
### py-polars 0.6.4
* feature
  - \[python\] render html tables
* performance
  - \[python\] default to pyarrow for parquet reading
  - \[python | rust\] use u32 instead of usize in groupby and join to increase cache coherence and reduce memory pressure.

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
