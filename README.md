# Polars
[![rust docs](https://docs.rs/polars/badge.svg)](https://docs.rs/polars/latest/polars/)
![Build, test and docs](https://github.com/ritchie46/polars/workflows/Build,%20test%20and%20docs/badge.svg)
[![](http://meritbadge.herokuapp.com/polars)](https://crates.io/crates/polars)

## Blazingly fast in memory DataFrames in Rust

Polars is a DataFrames library implemented in Rust, using Apache Arrow as backend. 
Its focus is being a fast in memory DataFrame library. 

Polars is in rapid development, but it already supports most features needed for a useful DataFrame library. Do you
miss something, please make an issue and/or sent a PR.

## First run
Take a look at the [10 minutes to Polars notebook](examples/10_minutes_to_polars.ipynb) to get you started.
Want to run the notebook yourself? Clone the repo and run `$ cargo c && docker-compose up`. This will spin up a jupyter
notebook on `http://localhost:8891`. The notebooks are in the `/examples` directory.
 
Oh yeah.. and get a cup of coffee because compilation will take while during the first run.

## Python
A subset of the Polars functionality is also exposed through Python bindings. You can install them for linux with:

`$ pip install py-polars`

Next you can check the [10 minutes to py-polars notebook](examples/10_minutes_to_pypolars.ipynb) or take a look 
at the [reference](https://py-polars.readthedocs.io/en/latest/).


## Documentation
Want to know what features Polars support? [Check the current master docs](https://ritchie46.github.io/polars). 

Most features are described on the [DataFrame](https://ritchie46.github.io/polars/polars/frame/struct.DataFrame.html), 
[Series](https://ritchie46.github.io/polars/polars/series/enum.Series.html), and [ChunkedArray](https://ritchie46.github.io/polars/polars/chunked_array/struct.ChunkedArray.html)
structs in that order. For `ChunkedArray` a lot of functionality is also defined by `Traits` in the 
[ops module](https://ritchie46.github.io/polars/polars/chunked_array/ops/index.html).

## Performance
Polars is written to be performant. Below are some comparisons with the (also very fast) Pandas DataFrame library.

#### GroupBy
![](pandas_cmp/img/groupby10_.png)

#### Joins
![](pandas_cmp/img/join_80_000.png)

## Functionality

### Read and write CSV | JSON | IPC | Parquet

```rust
 use polars::prelude::*;
 use std::fs::File;
 
 fn example() -> Result<DataFrame> {
     let file = File::open("iris.csv")
                     .expect("could not open file");
 
     CsvReader::new(file)
             .infer_schema(None)
             .has_header(true)
             .finish()
 }
```

### Joins

```rust
 use polars::prelude::*;

 fn join() -> Result<DataFrame> {
     // Create first df.
     let temp = df!("days" => &[0, 1, 2, 3, 4],
                    "temp" => &[22.1, 19.9, 7., 2., 3.])?;

     // Create second df.
     let rain = df!("days" => &[1, 2],
                    "rain" => &[0.1, 0.2])?;

     // Left join on days column.
     temp.left_join(&rain, "days", "days")
 }

 println!("{:?}", join().unwrap());
```

```text
 +------+------+------+
 | days | temp | rain |
 | ---  | ---  | ---  |
 | i32  | f64  | f64  |
 +======+======+======+
 | 0    | 22.1 | null |
 +------+------+------+
 | 1    | 19.9 | 0.1  |
 +------+------+------+
 | 2    | 7    | 0.2  |
 +------+------+------+
 | 3    | 2    | null |
 +------+------+------+
 | 4    | 3    | null |
 +------+------+------+
```

### Groupby's | aggregations | pivots

```rust
 use polars::prelude::*;
 fn groupby_sum(df: &DataFrame) -> Result<DataFrame> {
     df.groupby("column_name")?
     .select("agg_column_name")
     .sum()
 }
```

### Arithmetic
```rust
 use polars::prelude::*;
 let s: Series = [1, 2, 3].iter().collect();
 let s_squared = &s * &s;
```

### Rust iterators

```rust
 use polars::prelude::*;
 
 let s: Series = [1, 2, 3].iter().collect();
 let s_squared: Series = s.i32()
      .expect("datatype mismatch")
      .into_iter()
      .map(|optional_v| {
              match optional_v {
                  Some(v) => Some(v * v),
                  None => None, // null value
              }
          }).collect();
```

### Apply custom closures
```rust
 use polars::prelude::*;
 
 let s: Series = Series::new("values", [Some(1.0), None, Some(3.0)]);
 // null values are ignored automatically
 let squared = s.f64()
     .unwrap()
     .apply(|value| value.powf(2.0))
     .into_series();
 
 assert_eq!(Vec::from(squared.f64().unwrap()), &[Some(1.0), None, Some(9.0)])
```

### Comparisons

```rust
 use polars::prelude::*;
 use itertools::Itertools;
 let s = Series::new("dollars", &[1, 2, 3]);
 let mask = s.eq(1);
 let valid = [true, false, false].iter();
 
 assert_eq!(Vec::from(mask), &[Some(true), Some(false), Some(false)]);
```

## Temporal data types

```rust
 let dates = &[
 "2020-08-21",
 "2020-08-21",
 "2020-08-22",
 "2020-08-23",
 "2020-08-22",
 ];
 // date format
 let fmt = "%Y-%m-%d";
 // create date series
 let s0 = Date32Chunked::parse_from_str_slice("date", dates, fmt)
         .into_series();
```

## And more...

* [DataFrame](https://ritchie46.github.io/polars/polars/frame/struct.DataFrame.html)
* [Series](https://ritchie46.github.io/polars/polars/series/enum.Series.html)
* [ChunkedArray](https://ritchie46.github.io/polars/polars/chunked_array/struct.ChunkedArray.html)
     - [Operations implemented by Traits](https://ritchie46.github.io/polars/polars/chunked_array/ops/index.html)
* [Time/ DateTime utilities](https://ritchie46.github.io/polars/polars/doc/time/index.html)
* [Groupby, aggregations and pivots](https://ritchie46.github.io/polars/polars/frame/group_by/struct.GroupBy.html)

## Features

Additional cargo features:

* `pretty` (default)
    - pretty printing of DataFrames
* `temporal (default)`
    - Conversions between Chrono and Polars for temporal data
* `simd`
    - SIMD operations
* `paquet`
    - Read Apache Parquet format
* `random`
    - Generate array's with randomly sampled values
* `ndarray`
    - Convert from `DataFrame` to `ndarray`
