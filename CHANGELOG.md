# Changelog Polars (Rust crate)

The Python bindings of `polars` have their own changelog.

### Polars 0.16.0
* features
  - [read compression streams](05c672965c4f05d78c3d649fa200562d8a78f222)
  - [improve window exprs](49a52f71215b428ee362843c1620983126ed19a4)
  - [interpolate series](28a858da832a12c44f3e352080c06fa06f6798f8)
  - [add flattened window exprs](eaf8bacb5356756d893b041cd06a71771e527597)
  - [Series::fill_nan](38f7acb49e84a605c6c6318447c2dfd9ad41b20f)
  - [improve consistency of agg exprs](93e82d5b6abe4e35389f6c282d01c8a3234f1192)
  - [prefix/suffix exprs](prefix/suffix)
  - [DataFrame::transpose](c3310fc5c68b55b3a5ff11480f36203937fba958)
  - [rolling window functions in lazy](1429dbc6e443cd1dbeada0aa9146dfedf9f7eaaf)
  - [add .arr namespace](42f54e30cc3c40fdbd3271a5a4e10666135a8fa6)
  - [argsort expr](c9c41316ef701e058e65b1b114292b2baf3c032e)
  - [rank: (min, max, avg, dense, ordinal)][a42a78142125b84e2acaba51f60bc37e5c92cda6]
  - [add null behvavior of horizontal sum/mean](e86417a10c4908e5d77d2ec39e8a5b5061101301)
  - [skew aggregation](5b69beebae78da6bf1eb9360394c457e7899c7b1)
  - [kurtosis aggregation](5a21ddcd01c676536dfbf44342b98ba28d6b7831)
  - and a lot more
* bug fix
  - [Categorical type append](b301358246453337d3a391a97616cfb5e68d1e4d)
  - [fix serde](1447a570f604e3a920972bc87dbf3b4f7fd19ac5)
  - [csv decompression][33f69fe33e70cbeffda04ea218df525bcf6fce8a]
  - [fix O^2 pivot.count](21f77c7349da80d937bd7cb6aaeed4af266157d0)
  - [improve type-casting optimizer](ca69f6cab7b095b91e07f3d7899f89a6525a9754)
  - [fix agg_quantile](1f61a1c89fd9ae9f037a8d07bb9dd7bc4dc1a2ef)
  - [fix explode/flatten on slice lists](938cd9ab033e5dac9cf83fc9d915c825a9007bdb)
  - [fix bug in utf8/categorical cast](ecaad33855d9a32f4e822f4b1b548806da724791)
  - [fix float join on multiple columns](e597e93ec52206a6b4a64c8c388e1bab3a405660)
  - [csv parser handle files without EOF](81a79b5be80890fa34ccc011354898ea77a45267)
  - [fix df explode on df's with single column](ef12a1204bc715434433531cf449b0ccbeca9660)
  - [change asof join behavior](bd17e69ff9b1157443fc9381a89cd7a3d2346471)
  - fix bugs in predicate pushdown
  - and a lot more
* performance
  - [lazy drop nulls optimization](4ab0d890e4f07f52cebbd5c2c4accef11f103019)
  - [improve fill_null performance](ae5e6af290086193c35c288f4af9d13edd5795a0)
  - [improve primitive sorting](f0253094bdacda9886e5521d5d380c40ee3b1639)
  - [improve hashing performance](64847767f6af488b8c72ce7e06facec8230fbf86)
  - improve csv parser (faster float parsing/ less utf8 checks)
  - [unstable sort in stable groupby](97139effbc109fdfb7c0cbe1922649b1965c3649)
  - improve performance of list iteration (this is used in most groupby + apply exprs)



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
  - fix ub of alignedvec dealloc
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
