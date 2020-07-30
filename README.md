# Polars
[![rust docs](https://docs.rs/polars/badge.svg)](https://docs.rs/polars/latest/polars/)
[![Build Status](https://travis-ci.com/ritchie46/polars.svg?branch=master)](https://travis-ci.com/ritchie46/polars)

## Blazingly fast in memory DataFrames in Rust

Polars is a DataFrames library implemented in Rust, using Apache Arrow as backend. It's focus is being a fast in memory
DataFrame library that only supports core functionality.

## Documentation
[Read more in the docs.](https://ritchie46.github.io/polars)

### Performance

#### GroupBy
![](pandas_cmp/img/groupby10_.png)

#### Joins
![](pandas_cmp/img/join_80_000.png)

## Run the python bindings!
`$ docker run --rm -p 8890:8890 ritchie46/py-polars` and open `http://localhost:8890` in your browser.

## Functionality

### Series
- [x] cast
- [x] take by index/ boolean mask
- [x] limit
- [x] Rust iterators! (So any function you can think of)
- [x] append
- [x] aggregation: min, max, sum
- [x] arithmetic
- [x] comparison
- [x] sorting

### DataFrame
- [x] take by index/ boolean mask
- [x] limit
- [x] join: inner, left, and outer
- [x] column ops: drop, select, rename
- [x] group by: min, max, sum, mean, count
- [x] concat (horizontal)
- [x] read csv
- [x] write csv
- [x] write json
- [x] read json
- [ ] write parquet
- [ ] read parquet
- [x] sorting

### Data types
- [x] null
- [x] boolean
- [x] u32
- [x] i32
- [x] i64
- [x] f32
- [x] f64
- [x] utf-8
- [x] date
- [x] time


## Examples

```rust
use polars::prelude::*;

// Create first df.
let s0 = Series::new("days", &[0, 1, 2, 3, 4]);
let s1 = Series::new("temp", &[22.1, 19.9, 7., 2., 3.]);
let temp = DataFrame::new_from_columns(vec![s0, s1]).unwrap();

// Create second df.
let s0 = Series::new("days", &[1, 2]);
let s1 = Series::new("rain", &[0.1, 0.2]);
let rain = DataFrame::new_from_columns(vec![s0, s1]).unwrap();

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

### Arithmetic
```rust
use polars::prelude::*;

let s: Series = [1, 2, 3].iter().collect(); 
let s_squared = &s * &s;
```

### Custom functions
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


