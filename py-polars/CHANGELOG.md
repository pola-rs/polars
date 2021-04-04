# Changelog polars (Python bindings)

The Rust crate `polars` has its own changelog.

### polars 0.7.4
* performance
  - \[python | rust\] multi-threaded outer join
  - \[python | rust\] better performance in groupby on multiple keys (faster hashmap comparisons)
  - \[python | rust\] better performance in multi column joins

* bug fix
  - \[python\] make horizontal aggregations null aware

* feature
  - \[python | rust\] Downsample by week
  - \[python | rust\] join by unlimited columns
  - \[python\] Create a list Series directly.
  - \[python\] Create DataFrame from np.ndarray
  

### polars 0.7.3
* bug fix
  - \[python\] pandas to polars date64, maintain time information
  - \[python\] fix bug in Date64 Series.year
  - \[python\] fix bug Series.mean (did not correct for null values) #484
  - \[python | rust \] fix bug in rolling windows #484
  - \[python | rust \] fix bug lazy csv parser #459

* feature
  - \[python | rust\] Series methods
    * Series.week
    * Series.weekday
    * Series.arg_min
    * Series.arg_max
    * Series.shape

### polars 0.7.2
* bug fix
  - \[python\] More pyarrow -> polars conversions.
    
* feature
  - \[python\] DataFrame methods: \[ shift_and_fill\].
  - \[python\] eager: sum, min, max, mean horizontal aggregation.

### polars 0.7.1
* performance
  - \[python | rust\] arrow arrays have a layer of indirection less; 10/20% performance improvement

### polars 0.7.0
* name change: Python bindings module renamed from pypolars to polars
* name change: Python bindings package renamed from py-polars to polars

* feature
  - \[python\] lazy: DataFrame methods: \[ tail, first, last \].
  - \[python\] eager: DataFrame fold for horizontal aggregation.
  - \[python\] eager: Series methods: \[median, quantile, is_in, to_frame\]
  - \[python\] eager: iterate over groupby and yield groups' DataFrames
  - \[python\] eager: groupby.get_group('value')
  - \[python\] add parquet compression
  - \[python\] shift_and_fill expression
  - \[python\] implicitly download raw files from the web in `read_parquet`, `read_csv`.
  - \[python | rust\] methods for local peak finding in numerical series
  - \[python | rust\] faster query optimization due to local memory arena's.
  - \[rust\] reduce default compile time by making less features default.
  - \[python | rust\] Series zip_with implicitly cast to supertype.
  - \[python | rust\] window functions have a `min_periods` argument to control when to compute a result
  
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
