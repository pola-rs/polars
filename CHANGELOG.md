# Changelog Polars

The Python bindings of `polars` have their own changelog.

### Polars 0.16.0
* features
  - read compression streams
  - improve window exprs
  - interpolate series
  - add flattened window exprs
  - Series::fill_nan
  - improve consistency of agg exprs
  - prefix/suffix exprs
  - DataFrame::transpose
  - rolling window functions in lazy
  - add .arr namespace
  - argsort expr
  - rank: (min, max, avg, dense, ordinal)
  - add null behvavior of horizontal sum/mean
  - skew aggregation
  - kurtosis aggregation
  - and a lot more
* bug fix
  - Categorical type append
  - fix serde
  - csv decompression 33f69fe33e70cbeffda04ea218df525bcf6fce8a
  - fix O^2 pivot.count
  - improve type-casting optimizer
  - fix agg_quantile
  - fix explode/flatten on slice lists
  - fix bug in utf8/categorical cast
  - fix float join on multiple columns
  - csv parser handle files without EOF
  - fix df explode on df's with single column
  - change asof join behavior
  - fix bugs in predicate pushdown
  - and a lot more
* performance
  - lazy drop nulls optimization
  - improve fill_null performance
  - improve primitive sorting
  - improve hashing performance
  - improve csv parser
  - unstable sort in stable groupby
  - improve performance of list iteration

### Polars 0.15.0
* feature
  - extract jsonpath
  - more object support
* performance
  - improve list take performance
* bug fix
  - don't panic in out of bounds take, but error
  - update offsets in case of utf8lossy
* breaking
  - take returns error
  - parsers take `W: Write` instead of `&mut W`

### Polars 0.14.8
* feature
  - concat_str function
  - more object support
  - hash and row_hash function/ expr
  - reinterpret function/ expr
  - Series.mode expr/function
  - csv file decompression
  - read_sql support
* performance
  - divide and conquer binary expressions

### Polars 0.14.7
* feature
  - cross join added
  - dot-product
* performance
  - improve csv-parser performance by ~25%
* bug fix
  - fix ub of AlignedVec dealloc
  - various minor

### Polars 0.14.6
* feature
  - is_first expr/method
  - asof join added
  - resolve `~` to homedir
* performance
  - use fast csv-parser for more input types
* bug fix
  - kleene or and and operations
  - maybe fix rayon deadlock
  - string addition lhs broadcast

### Polars 0.14.5
* feature
  - csv-parser option to ignore comment lines
* performance
  - improve take on DataFrame
  - remove bound checks in buffer creation
  - improve performance of sorting by multiple columns
  - improve argsort performance
* bug fix
  - fix backward/forward fill
  - window groupby context

## Polars v0.13

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
