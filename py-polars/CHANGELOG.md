# Changelog polars (Python bindings)

The Rust crate `polars` has its own changelog.

### \[future\] polars 0.7.0
* name change: Python bindings module renamed from pypolars to polars
* name change: Python bindings package renamed from py-polars to polars

* feature
  - \[python\] eager: DataFrame fold for horizontal aggregation.
  - \[python\] add parquet compression
  - \[python\] shift_and_fill expression
  - \[python\] implicitly download raw files from the web in `read_parquet`, `read_csv`.
  - \[python | rust\] methods for local peak finding in numerical series
  - \[python | rust\] faster query optimization due to local memory arena's.
  - \[rust\] reduce default compile time by making less features default.
  - \[python | rust\] Series zip_with implicitly cast to supertype.
  
* bug fix
  - \[python\] support file buffers for reading and writing csv and parquet
  - \[python | rust\] fix csv-parser: allow new-line character in a string field
  - \[python | rust\] don't let predicate-pushdown pass shift | sort operation to maintain correctness.

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
