//!
//! # Polars Eager cookbook
//!
//! This page should serve a cookbook to quickly get you started with most fundamental operations
//! executed on a `ChunkedArray`, `Series` or `DataFrame`.
//!
//! ## Tree Of Contents
//!
//! * [Creation of data structures](#creation-of-data-structures)
//!     - [ChunkedArray](#chunkedarray)
//!     - [Series](#series)
//!     - [DataFrame](#dataframe)
//! * [Arithmetic](#arithmetic)
//! * [Comparisons](#comparisons)
//! * [Apply functions/ closures](#apply-functions-closures)
//!     - [Series / ChunkedArrays](#dataframe-1)
//!     - [DataFrame](#dataframe-1)
//! * [Filter](#filter)
//! * [Sort](#sort)
//! * [Joins](#joins)
//! * [GroupBy](#groupby)
//!     - [pivot](#pivot)
//! * [Melt](#melt)
//! * [Explode](#explode)
//! * [IO](#IO)
//!     - [Read CSV](#read-csv)
//!     - [Write CSV](#write-csv)
//!     - [Read IPC](#read-ipc)
//!     - [Write IPC](#write-ipc)
//!     - [Read Parquet](#read-parquet)
//!     - [Write Parquet](#write-parquet)
//! * [Various](#various)
//!     - [Replace NaN](#replace-nan)
//!     * [Extracting data](#extracting-data)
//!
//! ## Creation of Data structures
//!
//! ### ChunkedArray
//!
//! ```
//! use polars::prelude::*;
//!
//! // use iterators
//! let ca: UInt32Chunked = (0..10).map(Some).collect();
//!
//! // from slices
//! let ca = UInt32Chunked::new("foo", &[1, 2, 3]);
//!
//! // use builders
//! let mut builder = PrimitiveChunkedBuilder::<UInt32Type>::new("foo", 10);
//! for value in 0..10 {
//!     builder.append_value(value);
//! }
//! let ca = builder.finish();
//! ```
//!
//! ### Series
//!
//! ```
//! use polars::prelude::*;
//!
//! // use iterators
//! let s: Series = (0..10).map(Some).collect();
//!
//! // from slices
//! let s = Series::new("foo", &[1, 2, 3]);
//!
//! // from a chunked-array
//! let ca = UInt32Chunked::new("foo", &[Some(1), None, Some(3)]);
//! let s = ca.into_series();
//! ```
//!
//! ### DataFrame
//!
//! ```
//! use polars::prelude::*;
//! use polars::df;
//! # fn example() -> PolarsResult<()> {
//!
//! // use macro
//! let df = df! [
//!     "names" => ["a", "b", "c"],
//!     "values" => [1, 2, 3],
//!     "values_nulls" => [Some(1), None, Some(3)]
//! ]?;
//!
//! // from a Vec<Series>
//! let s1 = Series::new("names", &["a", "b", "c"]);
//! let s2 = Series::new("values", &[Some(1), None, Some(3)]);
//! let df = DataFrame::new(vec![s1, s2])?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Arithmetic
//! Arithmetic can be done on both `Series` and `ChunkedArray`s. The most notable difference is that
//! a `Series` coerces the data to match the underlying data types.
//!
//! ```
//! use polars::prelude::*;
//! # fn example() -> PolarsResult<()> {
//! let s_int = Series::new("a", &[1, 2, 3]);
//! let s_flt = Series::new("b", &[1.0, 2.0, 3.0]);
//!
//! let added = &s_int + &s_flt;
//! let subtracted = &s_int - &s_flt;
//! let multiplied = &s_int * &s_flt;
//! let divided = &s_int / &s_flt;
//! let moduloed = &s_int % &s_flt;
//!
//!
//! // on chunked-arrays we first need to cast to same types
//! let ca_int = s_int.i32()?;
//! let ca_flt = s_flt.f32()?;
//!
//! ca_int.cast(&DataType::Float32)?.f32()? * ca_flt;
//! ca_flt.cast(&DataType::Int32)?.i32()? * ca_int;
//!
//! // we can also do arithmetic with numeric values
//! let multiplied = ca_int * 2.0;
//! let multiplied = s_flt * 2.0;
//!
//! // or broadcast Series to match the operands type
//! let added = &s_int * &Series::new("broadcast_me", &[10]);
//!
//! # Ok(())
//! # }
//! ```
//!
//! Because Rusts Orphan Rule doesn't allow use to implement left side operations, we need to call
//! such operation directly.
//!
//! ```rust
//! # use polars::prelude::*;
//! let series = Series::new("foo", [1, 2, 3]);
//!
//! // 1 / s
//! let divide_one_by_s = 1.div(&series);
//!
//! // 1 - s
//! let subtract_one_by_s = 1.sub(&series);
//! ```
//!
//! For `ChunkedArray`s this left hand side operations can be done with the `apply` method.
//!
//! ```rust
//! # use polars::prelude::*;
//! let ca = UInt32Chunked::new("foo", &[1, 2, 3]);
//!
//! // 1 / ca
//! let divide_one_by_ca = ca.apply(|rhs| 1 / rhs);
//! ```
//!
//! ## Comparisons
//!
//! `Series` and `ChunkedArray`s can be used in comparison operations to create `boolean` masks/predicates.
//!
//! ```
//! use polars::prelude::*;
//! # fn example() -> PolarsResult<()> {
//!
//! let s = Series::new("a", &[1, 2, 3]);
//! let ca = UInt32Chunked::new("b", &[Some(3), None, Some(1)]);
//!
//! // compare Series with numeric values
//! // ==
//! s.equal(2);
//! // !=
//! s.not_equal(2);
//! // >
//! s.gt(2);
//! // >=
//! s.gt_eq(2);
//! // <
//! s.lt(2);
//! // <=
//! s.lt_eq(2);
//!
//!
//! // compare Series with Series
//! // ==
//! s.equal(&s);
//! // !=
//! s.not_equal(&s);
//! // >
//! s.gt(&s);
//! // >=
//! s.gt_eq(&s);
//! // <
//! s.lt(&s);
//! // <=
//! s.lt_eq(&s);
//!
//!
//! // compare chunked-array with numeric values
//! // ==
//! ca.equal(2);
//! // !=
//! ca.not_equal(2);
//! // >
//! ca.gt(2);
//! // >=
//! ca.gt_eq(2);
//! // <
//! ca.lt(2);
//! // <=
//! ca.lt_eq(2);
//!
//! // compare chunked-array with chunked-array
//! // ==
//! ca.equal(&ca);
//! // !=
//! ca.not_equal(&ca);
//! // >
//! ca.gt(&ca);
//! // >=
//! ca.gt_eq(&ca);
//! // <
//! ca.lt(&ca);
//! // <=
//! ca.lt_eq(&ca);
//!
//! // use iterators
//! let a: BooleanChunked = ca.into_iter()
//!     .map(|opt_value| {
//!          match opt_value {
//!          Some(value) => value < 10,
//!          None => false
//! }}).collect();
//!
//! # Ok(())
//! # }
//! ```
//!
//!
//! ## Apply functions/ closures
//!
//! See all possible [apply methods here](crate::chunked_array::ops::ChunkApply).
//!
//! ### Series / ChunkedArrays
//!
//! ```
//! use polars::prelude::*;
//! # fn example() -> PolarsResult<()> {
//!
//! // apply a closure over all values
//! let s = Series::new("foo", &[Some(1), Some(2), None]);
//! s.i32()?.apply(|value| value * 20);
//!
//! // count string lengths
//! let s = Series::new("foo", &["foo", "bar", "foobar"]);
//! s.utf8()?.apply_cast_numeric::<_, UInt64Type>(|str_val| str_val.len() as u64);
//!
//! # Ok(())
//! # }
//! ```
//!
//!
//! ### Multiple columns
//!
//! ```
//! use polars::prelude::*;
//! fn my_black_box_function(a: f32, b: f32) -> f32 {
//!     // do something
//!     a
//! }
//!
//! fn apply_multiples(col_a: &Series, col_b: &Series) -> Float32Chunked {
//!     match (col_a.dtype(), col_b.dtype()) {
//!         (DataType::Float32, DataType::Float32) => {
//!             // downcast to `ChunkedArray`
//!             let a = col_a.f32().unwrap();
//!             let b = col_b.f32().unwrap();
//!
//!             a.into_iter()
//!                 .zip(b.into_iter())
//!                 .map(|(opt_a, opt_b)| match (opt_a, opt_b) {
//!                     (Some(a), Some(b)) => Some(my_black_box_function(a, b)),
//!                     // if any of the two value is `None` we propagate that null
//!                     _ => None,
//!                 })
//!                 .collect()
//!         }
//!         _ => panic!("unexpected dtypes"),
//!     }
//! }
//! ```
//!
//! ### DataFrame
//!
//! ```
//! use polars::prelude::*;
//! use polars::df;
//! # fn example() -> PolarsResult<()> {
//!
//! let mut df = df![
//!     "letters" => ["a", "b", "c", "d"],
//!     "numbers" => [1, 2, 3, 4]
//! ]?;
//!
//!
//! // coerce numbers to floats
//! df.try_apply("number", |s: &Series| s.cast(&DataType::Float64))?;
//!
//! // transform letters to uppercase letters
//! df.try_apply("letters", |s: &Series| {
//!     Ok(s.utf8()?.to_uppercase())
//! });
//!
//! # Ok(())
//! # }
//! ```
//!
//! ## Filter
//! ```
//! use polars::prelude::*;
//!
//! # fn example(df: &DataFrame) -> PolarsResult<()> {
//! // create a mask to filter out null values
//! let mask = df.column("sepal.width")?.is_not_null();
//!
//! // select column
//! let s = df.column("sepal.length")?;
//!
//! // apply filter on a Series
//! let filtered_series = s.filter(&mask);
//!
//! // apply the filter on a DataFrame
//! let filtered_df = df.filter(&mask)?;
//!
//! # Ok(())
//! # }
//! ```
//!
//! ## Sort
//! ```
//! use polars::prelude::*;
//! use polars::df;
//!
//! # fn example() -> PolarsResult<()> {
//! let df = df![
//!     "a" => [1, 2, 3],
//!     "b" => ["a", "a", "b"]
//! ]?;
//! // sort this DataFrame by multiple columns
//!
//! // ordering of the columns
//! let reverse = vec![true, false];
//! // columns to sort by
//! let by = &["b", "a"];
//! // do the sort operation
//! let sorted = df.sort(by, reverse)?;
//!
//! // sorted:
//!
//! // ╭─────┬─────╮
//! // │ a   ┆ b   │
//! // │ --- ┆ --- │
//! // │ i64 ┆ str │
//! // ╞═════╪═════╡
//! // │ 1   ┆ "a" │
//! // │ 2   ┆ "a" │
//! // │ 3   ┆ "b" │
//! // ╰─────┴─────╯
//!
//! # Ok(())
//! # }
//! ```
//!
//! ## Joins
//!
//! ```
//! use polars::prelude::*;
//! use polars::df;
//!
//! # fn example() -> PolarsResult<()> {
//! // Create first df.
//! let temp = df!("days" => &[0, 1, 2, 3, 4],
//!                "temp" => &[22.1, 19.9, 7., 2., 3.],
//!                "other" => &[1, 2, 3, 4, 5]
//! )?;
//!
//! // Create second df.
//! let rain = df!("days" => &[1, 2],
//!                "rain" => &[0.1, 0.2],
//!                "other" => &[1, 2, 3, 4, 5]
//! )?;
//!
//! // join on a single column
//! temp.left_join(&rain, ["days"], ["days"]);
//! temp.inner_join(&rain, ["days"], ["days"]);
//! temp.outer_join(&rain, ["days"], ["days"]);
//!
//! // join on multiple columns
//! temp.join(&rain, vec!["days", "other"], vec!["days", "other"], JoinType::Left, None);
//!
//! # Ok(())
//! # }
//! ```
//!
//! ## Groupby
//!
//! Note that Polars lazy is a lot more powerful in and more performant in groupby operations.
//! In lazy a myriad of aggregations can be combined from expressions.
//!
//! See more in:
//!
//! * [Groupby](crate::frame::groupby::GroupBy)
//!
//! ### GroupBy
//! ```
//! use polars::prelude::*;
//!
//! # fn example(df: &DataFrame) -> PolarsResult<()> {
//!  // groupby "groups" | sum "foo"
//!  let out = df.groupby(["groups"])?
//!     .select(["foo"])
//!     .sum();
//!
//! # Ok(())
//! # }
//!
//! ```
//!
//! ### Pivot
//!
//! ```
//! use polars::prelude::*;
//! use polars::df;
//!
//! # fn example(df: &DataFrame) -> PolarsResult<()> {
//!  let df = df!("foo" => ["A", "A", "B", "B", "C"],
//!      "N" => [1, 2, 2, 4, 2],
//!      "bar" => ["k", "l", "m", "n", "0"]
//!      )?;
//!
//! // groupby "foo" | pivot "bar" column | aggregate "N"
//!  let pivoted = df.groupby(["foo"])?
//!     .pivot(["bar"], ["N"])
//!     .first();
//!
//! // pivoted:
//! // +-----+------+------+------+------+------+
//! // | foo | o    | n    | m    | l    | k    |
//! // | --- | ---  | ---  | ---  | ---  | ---  |
//! // | str | i32  | i32  | i32  | i32  | i32  |
//! // +=====+======+======+======+======+======+
//! // | "A" | null | null | null | 2    | 1    |
//! // +-----+------+------+------+------+------+
//! // | "B" | null | 4    | 2    | null | null |
//! // +-----+------+------+------+------+------+
//! // | "C" | 2    | null | null | null | null |
//! // +-----+------+------+------+------+------+!
//!
//! # Ok(())
//! # }
//! ```
//!
//! ## Melt
//!
//! ```
//! use polars::prelude::*;
//! use polars::df;
//!
//! # fn example(df: &DataFrame) -> PolarsResult<()> {
//! let df = df!["A" => &["a", "b", "a"],
//!              "B" => &[1, 3, 5],
//!              "C" => &[10, 11, 12],
//!              "D" => &[2, 4, 6]
//!     ]?;
//!
//! let melted = df.melt(&["A", "B"], &["C", "D"]).unwrap();
//! // melted:
//!
//! // +-----+-----+----------+-------+
//! // | A   | B   | variable | value |
//! // | --- | --- | ---      | ---   |
//! // | str | i32 | str      | i32   |
//! // +=====+=====+==========+=======+
//! // | "a" | 1   | "C"      | 10    |
//! // +-----+-----+----------+-------+
//! // | "b" | 3   | "C"      | 11    |
//! // +-----+-----+----------+-------+
//! // | "a" | 5   | "C"      | 12    |
//! // +-----+-----+----------+-------+
//! // | "a" | 1   | "D"      | 2     |
//! // +-----+-----+----------+-------+
//! // | "b" | 3   | "D"      | 4     |
//! // +-----+-----+----------+-------+
//! // | "a" | 5   | "D"      | 6     |
//! // +-----+-----+----------+-------+
//!
//! # Ok(())
//! # }
//! ```
//!
//! ## Explode
//!
//! ```
//! use polars::prelude::*;
//! use polars::df;
//!
//! # fn example(df: &DataFrame) -> PolarsResult<()> {
//! let s0 = Series::new("a", &[1i64, 2, 3]);
//! let s1 = Series::new("b", &[1i64, 1, 1]);
//! let s2 = Series::new("c", &[2i64, 2, 2]);
//! // construct a new ListChunked for a slice of Series.
//! let list = Series::new("foo", &[s0, s1, s2]);
//!
//! // construct a few more Series.
//! let s0 = Series::new("B", [1, 2, 3]);
//! let s1 = Series::new("C", [1, 1, 1]);
//! let df = DataFrame::new(vec![list, s0, s1])?;
//!
//! let exploded = df.explode(["foo"])?;
//! // exploded:
//!
//! // +-----+-----+-----+
//! // | foo | B   | C   |
//! // | --- | --- | --- |
//! // | i64 | i32 | i32 |
//! // +=====+=====+=====+
//! // | 1   | 1   | 1   |
//! // +-----+-----+-----+
//! // | 2   | 1   | 1   |
//! // +-----+-----+-----+
//! // | 3   | 1   | 1   |
//! // +-----+-----+-----+
//! // | 1   | 2   | 1   |
//! // +-----+-----+-----+
//! // | 1   | 2   | 1   |
//! // +-----+-----+-----+
//! // | 1   | 2   | 1   |
//! // +-----+-----+-----+
//! // | 2   | 3   | 1   |
//! // +-----+-----+-----+
//! // | 2   | 3   | 1   |
//! // +-----+-----+-----+
//! // | 2   | 3   | 1   |
//! // +-----+-----+-----+
//!
//! # Ok(())
//! # }
//! ```
//!
//! ## IO
//!
//! ### Read CSV
//!
//! ```
//! use polars::prelude::*;
//!
//! # fn example(df: &DataFrame) -> PolarsResult<()> {
//! // read from path
//! let df = CsvReader::from_path("iris_csv")?
//!             .infer_schema(None)
//!             .has_header(true)
//!             .finish()?;
//! # Ok(())
//! # }
//! ```
//!
//! ### Write CSV
//!
//! ```
//! use polars::prelude::*;
//! use std::fs::File;
//!
//! # fn example(df: &mut DataFrame) -> PolarsResult<()> {
//! // create a file
//! let mut file = File::create("example.csv").expect("could not create file");
//!
//! // write DataFrame to file
//! CsvWriter::new(&mut file)
//!     .has_header(true)
//!     .with_delimiter(b',')
//!     .finish(df);
//! # Ok(())
//! # }
//! ```
//!
//! ### Read IPC
//! ```
//! use polars::prelude::*;
//! use std::fs::File;
//!
//! # fn example(df: &DataFrame) -> PolarsResult<()> {
//! // open file
//! let file = File::open("file.ipc").expect("file not found");
//!
//! // read to DataFrame
//! let df = IpcReader::new(file)
//!    .finish()?;
//! # Ok(())
//! # }
//! ```
//!
//! ### Write IPC
//! ```
//! use polars::prelude::*;
//! use std::fs::File;
//!
//! # fn example(df: &mut DataFrame) -> PolarsResult<()> {
//! // create a file
//! let mut file = File::create("file.ipc").expect("could not create file");
//!
//! // write DataFrame to file
//! IpcWriter::new(&mut file)
//!     .finish(df)
//! # }
//! ```
//!
//! ### Read Parquet
//!
//! ```
//! use polars::prelude::*;
//! use std::fs::File;
//!
//! # fn example(df: &DataFrame) -> PolarsResult<()> {
//! // open file
//! let file = File::open("some_file.parquet").unwrap();
//!
//! // read to DataFrame
//! let df = ParquetReader::new(file).finish()?;
//! # Ok(())
//! # }
//! ```
//!
//! ### Write Parquet
//! ```
//! use polars::prelude::*;
//! use std::fs::File;
//!
//! # fn example(df: &mut DataFrame) -> PolarsResult<()> {
//! // create a file
//! let file = File::create("example.parquet").expect("could not create file");
//!
//! ParquetWriter::new(file)
//!     .finish(df)
//! # }
//! ```
//!
//! # Various
//!
//! ## Replace NaN with Missing.
//! The floating point [Not a Number: NaN](https://en.wikipedia.org/wiki/NaN) is conceptually different
//! than missing data in Polars. In the snippet below we show how we can replace `NaN` values with
//! missing values, by setting them to `None`.
//! ```
//! use polars::prelude::*;
//! use polars::df;
//!
//! /// Replaces NaN with missing values.
//! fn fill_nan_with_nulls() -> PolarsResult<DataFrame> {
//!     let nan = f64::NAN;
//!
//!     let mut df = df! {
//!        "a" => [nan, 1.0, 2.0],
//!        "b" => [nan, 1.0, 2.0]
//!     }
//!     .unwrap();
//!
//!     for idx in 0..df.width() {
//!         df.try_apply_at_idx(idx, |series| {
//!             let mask = series.is_nan()?;
//!             let ca = series.f64()?;
//!             ca.set(&mask, None)
//!         })?;
//!     }
//!     Ok(df)
//! }
//! ```
//!
//! ## Extracting data
//!
//! To be able to extract data out of `Series`, either by iterating over them or converting them
//! to other datatypes like a `Vec<T>`, we first need to downcast them to a `ChunkedArray<T>`. This
//! is needed because we don't know the data type that is hold by the `Series`.
//!
//! ```
//! use polars::prelude::*;
//! use polars::df;
//!
//! fn extract_data() -> PolarsResult<()> {
//!     let df = df! [
//!        "a" => [None, Some(1.0f32), Some(2.0)],
//!        "str" => ["foo", "bar", "ham"]
//!     ]?;
//!
//!     // first extract ChunkedArray to get the inner type.
//!     let ca = df.column("a")?.f32()?;
//!
//!     // Then convert to vec
//!     let to_vec: Vec<Option<f32>> = Vec::from(ca);
//!
//!     // We can also do this with iterators
//!     let ca = df.column("str")?.utf8()?;
//!     let to_vec: Vec<Option<&str>> = ca.into_iter().collect();
//!     let to_vec_no_options: Vec<&str> = ca.into_no_null_iter().collect();
//!
//!     Ok(())
//! }
//! ```
//!
//!
