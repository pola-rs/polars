# Changelog polars (Python bindings)

The Rust crate `polars` has its own changelog.

### Polars 0.8.10
* feature
  - is_first expr/method
  - asof join added
  - eager io can open multiple sources with ffspec
  - resolve `~` to homedir
  - python arange add step and run eager
* performance
  - use fast csv-parser for more python memory buffers/streams
* bug fix
  - kleene or and and operations
  - maybe fix rayon deadlock
  - concat is a pure function
  - string addition lhs broadcast

### Polars 0.8.9
* feature
  - correct type hints for python 3.6
  - csv-parser option to ignore comment lines
* performance
  - improve take on DataFrame
  - remove bound checks in buffer creation
  - improve performance of sorting by multiple columns
  - improve argsort performance
* bug fix
  - fix backward/forward fill
  - window groupby context
  - fix is_duplicated dispatch

### Polars 0.8.8
* bug fix
  - fix UB due to slice in take kernel
  - fix join for dates
  
### Polars 0.8.7
* feature
  - from_pandas accept series and date range #875
  - expr: forward_fill, backward_fill #874
  - gzipped file support in csv parser
* performance
  - reduce memory usage of multi-key groupby
  - improve variance and std-dev aggregation
* bug fix
  - cast to large-utf8 before collecting chunks #870
  - various

### Polars 0.8.6
* performance
  - improve hashing performance for grouping on two keys for 64 bit and 32 and 64 bit data.
  - improve cache coherence take operation of multiple chunks
* bug fix
  - fix replaxing string with None #802

### Polars 0.8.5
* feature
  - improve compatibility with pyarrow csv parser
* performance
  - improve hashing performance for grouping on two keys for 64 bit and 32 and 64 bit data.
  - improve cache coherence take operation of multiple chunks
  - fast path for categorical unique
  - decrease memory fragmentation and usage of csv-parser
* bug fix
  - split utf8 data only at valid char boundaries #789
  - fix bug in outer join due to new partitioning algorithm

### Polars 0.8.4
* feature
  - Series.round
  - head/ limit aliases
* performance
  - partitioned hashing
  
### Polars 0.8.0
* breaking change
  - `str` namespace Series.str_* methods to Series.str.<method>
  - `dt` namespace Series datetime related methods to Series.dt.<method>
    
* feature
  - DataFrame.rows method
  - apply on object types
  - `Series.dt.to_python_datetime`
  - `Series.dt.timestamp`
  
* bug fix
  - preserve date64 in round trip to parquet #723
  - during arrow conversion coerce categorical to utf8 (this preserves string data) #725
  - fix bug in csv skip rows

* performance
  - improve hashing of string data in groupby and join
  - improve numeric hashing in join
  - fast path for filtering no data and all date (upstream)

### polars 0.7.19
* feature
  - window function by multiple group columns

* bug fix
  - fix bug in argsort multiple
  - fix bug in filter with nulls (upstream)

* performance
  - improve numeric hashing in groupby
  - fast paths for filters (upstream)

### polars 0.7.18
* feature
  - argsort multiple columns

### polars 0.7.17
* feature
  - support more indexing
  - scan_csv low memory argument
  - Series.filter accept list of expressions
  - object type:
      - zip
      - take -> join / groupby agg
      - agg first/ last

* performance
  - change memory usage of csv-parser
  - binary aggregation in parallel
  - determine groupby keys in threadpool

### polars 0.7.16
* feature
  - Series literal may have any length
  - change globaly string cache behavior
  - Add Expr.arg_sort
  - Make literals typed

* bug fix
  - Fix Expr.fill_none
  - set offset in null buffers (fixes aggregation with null values)

* performance
  - sample cardinality in groupby and choose algorithm

### polars 0.7.15
* feature
  - join allows expression syntax
  - use pyarrow as default ipc backend
  
* bug fix
  - fix deadlock in window expressions

### polars 0.7.13 / 0.7.14 (patch) 2021-05-08

* bug fix
  - fix bug in cum_sum #604
  
* feature
  - DataFrame.describe method #606
  - Multi-level sorting of a DataFrame #607
  - Expand functionality of Expr.is_in #614
  - Csv-parser low_memory option #615
  - Allow expressions in `pl.arange` #611
    
* performance
  - sort().reverse() optimization #605

### polars 0.7.12
* bug fix
  - null handling in mean, std, var, and cov aggregations. #595
  - rev-mapping of categorical stored duplicates. #595
  - fix memory surge after csv-parsing #593

### polars 0.7.11 
* bug fix
  - Throw error on join from different string cache #584
  - fix covariance of array with null values #585

* feature
  - Series describe method #569
  - dsl: take, arg_unique, unique
  - allow lazy expressions in Eager API # 588
  - describe Series

* performance
  - fix accidental expensive appends #592
  - remove chunk_id from ChunkedArray #593


### polars 0.7.8 -> 0.7.9 (patched)
* bug fix
  - ensure column name persist after pyarrow cast #563
  - make sure that `agg_list` maintains dtype #567
  - fix panic in physical dispatch of Date dtypes

* feature
  - Implicitly Cast dtypes to temporal types in csv parser #560
  - Series describe method #569

* performance
  - Cache and improve window functions performance #570

### polars 0.7.7
* bug fix
  - fix bug with pyarrow chunkedarray: #545

* feature
  - DataFrame.apply method
  - Make a Series a Literal
  - Make None a Literal
    
* performance
  - Update arrow
    * faster iterators
    * faster kernels

### polars 0.7.6
* bug fix
  - fix bug in downsample: #537

* feature
  - cast categorical in csv parser: #533
  - add many groupby-context aware operations: #534
  - dowcast by month: #537

* performance
  - improve iterator in no null case: #538
  - remove indirection: #536

### polars 0.7.5
* bug fix
  - fix bug in vectorized hashing algorithm that affected groupbys with null values: #523
  - fix bug in downsample: 528
  - change median algorithm: #527

* feature
  - use lazy groupby API/DSL in eager API: #522
  - make sort groupby-context aware: #522

* performance
  - improve sort algorithms for sort and argsort: #526

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
  - \[python\] ~Create a list Series directly.~
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
