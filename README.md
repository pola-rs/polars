# Polars
[![rust docs](https://docs.rs/polars/badge.svg)](https://docs.rs/polars/latest/polars/)
[![Build Status](https://travis-ci.com/ritchie46/polars.svg?branch=master)](https://travis-ci.com/ritchie46/polars)
[![](http://meritbadge.herokuapp.com/polars)](https://crates.io/crates/polars)

## Blazingly fast in memory DataFrames in Rust

Polars is a DataFrames library implemented in Rust, using Apache Arrow as backend. It's focus is being a fast in memory
DataFrame library that only supports core functionality.

## Documentation
[Read more in the docs. (master branch)](https://ritchie46.github.io/polars)

### Performance

#### GroupBy
![](pandas_cmp/img/groupby10_.png)

#### Joins
![](pandas_cmp/img/join_80_000.png)

## Run the python bindings!
`$ docker run --rm -p 8890:8890 ritchie46/py-polars` and open `http://localhost:8890` in your browser.

## Functionality

### Read and write CSV/ JSON

```rust
use polars::prelude::*;
use std::fs::File;

fn example() -> Result<DataFrame> {
    let file = File::open("iris.csv").expect("could not open file");

    CsvReader::new(file)
            .infer_schema(None)
            .has_header(true)
            .finish()
}
```

### Joins

```rust
use polars::prelude::*;

// Create first df.
let s0 = Series::new("days", &[0, 1, 2, 3, 4]);
let s1 = Series::new("temp", &[22.1, 19.9, 7., 2., 3.]);
let temp = DataFrame::new(vec![s0, s1]).unwrap();

// Create second df.
let s0 = Series::new("days", &[1, 2]);
let s1 = Series::new("rain", &[0.1, 0.2]);
let rain = DataFrame::new(vec![s0, s1]).unwrap();

// Left join on days column.
let joined = temp.left_join(&rain, "days", "days");
println!("{}", joined.unwrap())
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

### GroupBys

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

assert_eq!(Vec::from(mask.bool().unwrap()), &[Some(true), Some(false), Some(false)]);
```

## And more...

* [DataFrame](https://ritchie46.github.io/polars/polars/frame/struct.DataFrame.html)
* [Series](https://ritchie46.github.io/polars/polars/series/enum.Series.html)
* [ChunkedArray](https://ritchie46.github.io/polars/polars/chunked_array/struct.ChunkedArray.html)
* [Time/ DateTime utilities](https://ritchie46.github.io/polars/polars/doc/time/index.html)

## Features

Additional cargo features:

* `pretty` (default)
    - pretty printing of DataFrames
* `simd`
    - SIMD operations
* `temporal`
    - Conversions between [Chrono](https://docs.rs/chrono/latest/chrono/) and Polars for temporal data
