//! DataFrame module.
#[cfg(feature = "zip_with")]
use std::borrow::Cow;
use std::{mem, ops};

use polars_utils::itertools::Itertools;
use rayon::prelude::*;

#[cfg(feature = "algorithm_group_by")]
use crate::chunked_array::ops::unique::is_unique_helper;
use crate::prelude::*;
#[cfg(feature = "row_hash")]
use crate::utils::split_df;
use crate::utils::{slice_offsets, try_get_supertype, NoNull};

#[cfg(feature = "dataframe_arithmetic")]
mod arithmetic;
mod chunks;
pub mod explode;
mod from;
#[cfg(feature = "algorithm_group_by")]
pub mod group_by;
pub(crate) mod horizontal;
#[cfg(any(feature = "rows", feature = "object"))]
pub mod row;
mod top_k;
mod upstream_traits;

use arrow::record_batch::RecordBatch;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use smartstring::alias::String as SmartString;

use crate::chunked_array::cast::CastOptions;
#[cfg(feature = "row_hash")]
use crate::hashing::_df_rows_to_hashes_threaded_vertical;
#[cfg(feature = "zip_with")]
use crate::prelude::min_max_binary::min_max_binary_series;
use crate::prelude::sort::{argsort_multiple_row_fmt, prepare_arg_sort};
use crate::series::IsSorted;
use crate::POOL;

#[derive(Copy, Clone, Debug)]
pub enum NullStrategy {
    Ignore,
    Propagate,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Default, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum UniqueKeepStrategy {
    /// Keep the first unique row.
    First,
    /// Keep the last unique row.
    Last,
    /// Keep None of the unique rows.
    None,
    /// Keep any of the unique rows
    /// This allows more optimizations
    #[default]
    Any,
}

fn ensure_names_unique<T, F>(items: &[T], mut get_name: F) -> PolarsResult<()>
where
    F: FnMut(&T) -> &str,
{
    // Always unique.
    if items.len() <= 1 {
        return Ok(());
    }

    if items.len() <= 4 {
        // Too small to be worth spawning a hashmap for, this is at most 6 comparisons.
        for i in 0..items.len() - 1 {
            let name = get_name(&items[i]);
            for other in items.iter().skip(i + 1) {
                if name == get_name(other) {
                    polars_bail!(duplicate = name);
                }
            }
        }
    } else {
        let mut names = PlHashSet::with_capacity(items.len());
        for item in items {
            let name = get_name(item);
            if !names.insert(name) {
                polars_bail!(duplicate = name);
            }
        }
    }
    Ok(())
}

/// A contiguous growable collection of `Series` that have the same length.
///
/// ## Use declarations
///
/// All the common tools can be found in [`crate::prelude`] (or in `polars::prelude`).
///
/// ```rust
/// use polars_core::prelude::*; // if the crate polars-core is used directly
/// // use polars::prelude::*;      if the crate polars is used
/// ```
///
/// # Initialization
/// ## Default
///
/// A `DataFrame` can be initialized empty:
///
/// ```rust
/// # use polars_core::prelude::*;
/// let df = DataFrame::default();
/// assert!(df.is_empty());
/// ```
///
/// ## Wrapping a `Vec<Series>`
///
/// A `DataFrame` is built upon a `Vec<Series>` where the `Series` have the same length.
///
/// ```rust
/// # use polars_core::prelude::*;
/// let s1 = Series::new("Fruit", &["Apple", "Apple", "Pear"]);
/// let s2 = Series::new("Color", &["Red", "Yellow", "Green"]);
///
/// let df: PolarsResult<DataFrame> = DataFrame::new(vec![s1, s2]);
/// ```
///
/// ## Using a macro
///
/// The [`df!`] macro is a convenient method:
///
/// ```rust
/// # use polars_core::prelude::*;
/// let df: PolarsResult<DataFrame> = df!("Fruit" => &["Apple", "Apple", "Pear"],
///                                       "Color" => &["Red", "Yellow", "Green"]);
/// ```
///
/// ## Using a CSV file
///
/// See the `polars_io::csv::CsvReader`.
///
/// # Indexing
/// ## By a number
///
/// The `Index<usize>` is implemented for the `DataFrame`.
///
/// ```rust
/// # use polars_core::prelude::*;
/// let df = df!("Fruit" => &["Apple", "Apple", "Pear"],
///              "Color" => &["Red", "Yellow", "Green"])?;
///
/// assert_eq!(df[0], Series::new("Fruit", &["Apple", "Apple", "Pear"]));
/// assert_eq!(df[1], Series::new("Color", &["Red", "Yellow", "Green"]));
/// # Ok::<(), PolarsError>(())
/// ```
///
/// ## By a `Series` name
///
/// ```rust
/// # use polars_core::prelude::*;
/// let df = df!("Fruit" => &["Apple", "Apple", "Pear"],
///              "Color" => &["Red", "Yellow", "Green"])?;
///
/// assert_eq!(df["Fruit"], Series::new("Fruit", &["Apple", "Apple", "Pear"]));
/// assert_eq!(df["Color"], Series::new("Color", &["Red", "Yellow", "Green"]));
/// # Ok::<(), PolarsError>(())
/// ```
#[derive(Clone)]
pub struct DataFrame {
    pub(crate) columns: Vec<Series>,
}

impl DataFrame {
    /// Returns an estimation of the total (heap) allocated size of the `DataFrame` in bytes.
    ///
    /// # Implementation
    /// This estimation is the sum of the size of its buffers, validity, including nested arrays.
    /// Multiple arrays may share buffers and bitmaps. Therefore, the size of 2 arrays is not the
    /// sum of the sizes computed from this function. In particular, [`StructArray`]'s size is an upper bound.
    ///
    /// When an array is sliced, its allocated size remains constant because the buffer unchanged.
    /// However, this function will yield a smaller number. This is because this function returns
    /// the visible size of the buffer, not its total capacity.
    ///
    /// FFI buffers are included in this estimation.
    pub fn estimated_size(&self) -> usize {
        self.columns.iter().map(|s| s.estimated_size()).sum()
    }

    // Reduce monomorphization.
    pub fn _apply_columns(&self, func: &(dyn Fn(&Series) -> Series)) -> Vec<Series> {
        self.columns.iter().map(func).collect()
    }

    // Reduce monomorphization.
    pub fn _apply_columns_par(
        &self,
        func: &(dyn Fn(&Series) -> Series + Send + Sync),
    ) -> Vec<Series> {
        POOL.install(|| self.columns.par_iter().map(func).collect())
    }

    // Reduce monomorphization.
    fn try_apply_columns_par(
        &self,
        func: &(dyn Fn(&Series) -> PolarsResult<Series> + Send + Sync),
    ) -> PolarsResult<Vec<Series>> {
        POOL.install(|| self.columns.par_iter().map(func).collect())
    }

    // Reduce monomorphization.
    fn try_apply_columns(
        &self,
        func: &(dyn Fn(&Series) -> PolarsResult<Series> + Send + Sync),
    ) -> PolarsResult<Vec<Series>> {
        self.columns.iter().map(func).collect()
    }

    /// Get the index of the column.
    fn check_name_to_idx(&self, name: &str) -> PolarsResult<usize> {
        self.get_column_index(name)
            .ok_or_else(|| polars_err!(col_not_found = name))
    }

    fn check_already_present(&self, name: &str) -> PolarsResult<()> {
        polars_ensure!(
            self.columns.iter().all(|s| s.name() != name),
            Duplicate: "column with name {:?} is already present in the DataFrame", name
        );
        Ok(())
    }

    /// Reserve additional slots into the chunks of the series.
    pub(crate) fn reserve_chunks(&mut self, additional: usize) {
        for s in &mut self.columns {
            // SAFETY:
            // do not modify the data, simply resize.
            unsafe { s.chunks_mut().reserve(additional) }
        }
    }

    /// Create a DataFrame from a Vector of Series.
    ///
    /// # Example
    ///
    /// ```
    /// # use polars_core::prelude::*;
    /// let s0 = Series::new("days", [0, 1, 2].as_ref());
    /// let s1 = Series::new("temp", [22.1, 19.9, 7.].as_ref());
    ///
    /// let df = DataFrame::new(vec![s0, s1])?;
    /// # Ok::<(), PolarsError>(())
    /// ```
    pub fn new(columns: Vec<Series>) -> PolarsResult<Self> {
        ensure_names_unique(&columns, |s| s.name())?;

        if columns.len() > 1 {
            let first_len = columns[0].len();
            for col in &columns {
                polars_ensure!(
                    col.len() == first_len,
                    ShapeMismatch: "could not create a new DataFrame: series {:?} has length {} while series {:?} has length {}",
                    columns[0].len(), first_len, col.name(), col.len()
                );
            }
        }

        Ok(DataFrame { columns })
    }

    /// Converts a sequence of columns into a DataFrame, broadcasting length-1
    /// columns to match the other columns.
    pub fn new_with_broadcast(columns: Vec<Series>) -> PolarsResult<Self> {
        ensure_names_unique(&columns, |s| s.name())?;
        unsafe { Self::new_with_broadcast_no_checks(columns) }
    }

    /// Converts a sequence of columns into a DataFrame, broadcasting length-1
    /// columns to match the other columns.
    ///  
    /// # Safety
    /// Does not check that the column names are unique (which they must be).
    pub unsafe fn new_with_broadcast_no_checks(mut columns: Vec<Series>) -> PolarsResult<Self> {
        // The length of the longest non-unit length column determines the
        // broadcast length. If all columns are unit-length the broadcast length
        // is one.
        let broadcast_len = columns
            .iter()
            .map(|s| s.len())
            .filter(|l| *l != 1)
            .max()
            .unwrap_or(1);

        for col in &mut columns {
            // Length not equal to the broadcast len, needs broadcast or is an error.
            let len = col.len();
            if len != broadcast_len {
                if len != 1 {
                    let name = col.name().to_owned();
                    let longest_column = columns.iter().max_by_key(|c| c.len()).unwrap().name();
                    polars_bail!(
                        ShapeMismatch: "could not create a new DataFrame: series {:?} has length {} while series {:?} has length {}",
                        name, len, longest_column, broadcast_len
                    );
                }
                *col = col.new_from_index(0, broadcast_len);
            }
        }
        Ok(unsafe { DataFrame::new_no_checks(columns) })
    }

    /// Creates an empty `DataFrame` usable in a compile time context (such as static initializers).
    ///
    /// # Example
    ///
    /// ```rust
    /// use polars_core::prelude::DataFrame;
    /// static EMPTY: DataFrame = DataFrame::empty();
    /// ```
    pub const fn empty() -> Self {
        // SAFETY: An empty dataframe cannot have length mismatches or duplicate names
        unsafe { DataFrame::new_no_checks(Vec::new()) }
    }

    /// Create an empty `DataFrame` with empty columns as per the `schema`.
    pub fn empty_with_schema(schema: &Schema) -> Self {
        let cols = schema
            .iter()
            .map(|(name, dtype)| Series::new_empty(name, dtype))
            .collect();
        unsafe { DataFrame::new_no_checks(cols) }
    }

    /// Create an empty `DataFrame` with empty columns as per the `schema`.
    pub fn empty_with_arrow_schema(schema: &ArrowSchema) -> Self {
        let cols = schema
            .fields
            .iter()
            .map(|fld| Series::new_empty(fld.name.as_str(), &(fld.data_type().into())))
            .collect();
        unsafe { DataFrame::new_no_checks(cols) }
    }

    /// Removes the last `Series` from the `DataFrame` and returns it, or [`None`] if it is empty.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let s1 = Series::new("Ocean", &["Atlantic", "Indian"]);
    /// let s2 = Series::new("Area (km²)", &[106_460_000, 70_560_000]);
    /// let mut df = DataFrame::new(vec![s1.clone(), s2.clone()])?;
    ///
    /// assert_eq!(df.pop(), Some(s2));
    /// assert_eq!(df.pop(), Some(s1));
    /// assert_eq!(df.pop(), None);
    /// assert!(df.is_empty());
    /// # Ok::<(), PolarsError>(())
    /// ```
    pub fn pop(&mut self) -> Option<Series> {
        self.columns.pop()
    }

    /// Add a new column at index 0 that counts the rows.
    ///
    /// # Example
    ///
    /// ```
    /// # use polars_core::prelude::*;
    /// let df1: DataFrame = df!("Name" => &["James", "Mary", "John", "Patricia"])?;
    /// assert_eq!(df1.shape(), (4, 1));
    ///
    /// let df2: DataFrame = df1.with_row_index("Id", None)?;
    /// assert_eq!(df2.shape(), (4, 2));
    /// println!("{}", df2);
    ///
    /// # Ok::<(), PolarsError>(())
    /// ```
    ///
    /// Output:
    ///
    /// ```text
    ///  shape: (4, 2)
    ///  +-----+----------+
    ///  | Id  | Name     |
    ///  | --- | ---      |
    ///  | u32 | str      |
    ///  +=====+==========+
    ///  | 0   | James    |
    ///  +-----+----------+
    ///  | 1   | Mary     |
    ///  +-----+----------+
    ///  | 2   | John     |
    ///  +-----+----------+
    ///  | 3   | Patricia |
    ///  +-----+----------+
    /// ```
    pub fn with_row_index(&self, name: &str, offset: Option<IdxSize>) -> PolarsResult<Self> {
        let mut columns = Vec::with_capacity(self.columns.len() + 1);
        let offset = offset.unwrap_or(0);

        let mut ca = IdxCa::from_vec(
            name,
            (offset..(self.height() as IdxSize) + offset).collect(),
        );
        ca.set_sorted_flag(IsSorted::Ascending);
        columns.push(ca.into_series());

        columns.extend_from_slice(&self.columns);
        DataFrame::new(columns)
    }

    /// Add a row index column in place.
    pub fn with_row_index_mut(&mut self, name: &str, offset: Option<IdxSize>) -> &mut Self {
        let offset = offset.unwrap_or(0);
        let mut ca = IdxCa::from_vec(
            name,
            (offset..(self.height() as IdxSize) + offset).collect(),
        );
        ca.set_sorted_flag(IsSorted::Ascending);

        self.columns.insert(0, ca.into_series());
        self
    }

    /// Create a new `DataFrame` but does not check the length or duplicate occurrence of the `Series`.
    ///
    /// It is advised to use [DataFrame::new] in favor of this method.
    ///
    /// # Safety
    ///
    /// It is the callers responsibility to uphold the contract of all `Series`
    /// having an equal length and a unique name, if not this may panic down the line.
    pub const unsafe fn new_no_checks(columns: Vec<Series>) -> DataFrame {
        DataFrame { columns }
    }

    /// Create a new `DataFrame` but does not check the length of the `Series`,
    /// only check for duplicates.
    ///
    /// It is advised to use [DataFrame::new] in favor of this method.
    ///
    /// # Safety
    ///
    /// It is the callers responsibility to uphold the contract of all `Series`
    /// having an equal length, if not this may panic down the line.
    pub unsafe fn new_no_length_checks(columns: Vec<Series>) -> PolarsResult<DataFrame> {
        ensure_names_unique(&columns, |s| s.name())?;
        Ok(DataFrame { columns })
    }

    /// Shrink the capacity of this DataFrame to fit its length.
    pub fn shrink_to_fit(&mut self) {
        // Don't parallelize this. Memory overhead
        for s in &mut self.columns {
            s.shrink_to_fit();
        }
    }

    /// Aggregate all the chunks in the DataFrame to a single chunk.
    pub fn as_single_chunk(&mut self) -> &mut Self {
        // Don't parallelize this. Memory overhead
        for s in &mut self.columns {
            *s = s.rechunk();
        }
        self
    }

    /// Aggregate all the chunks in the DataFrame to a single chunk in parallel.
    /// This may lead to more peak memory consumption.
    pub fn as_single_chunk_par(&mut self) -> &mut Self {
        self.as_single_chunk();
        // if self.columns.iter().any(|s| s.n_chunks() > 1) {
        //     self.columns = self._apply_columns_par(&|s| s.rechunk());
        // }
        self
    }

    /// Returns true if the chunks of the columns do not align and re-chunking should be done
    pub fn should_rechunk(&self) -> bool {
        // Fast check. It is also needed for correctness, as code below doesn't check if the number
        // of chunks is equal.
        if !self.get_columns().iter().map(|s| s.n_chunks()).all_equal() {
            return true;
        }

        // From here we check chunk lengths.
        let mut chunk_lengths = self.columns.iter().map(|s| s.chunk_lengths());
        match chunk_lengths.next() {
            None => false,
            Some(first_column_chunk_lengths) => {
                // Fast Path for single Chunk Series
                if first_column_chunk_lengths.len() == 1 {
                    return chunk_lengths.any(|cl| cl.len() != 1);
                }
                // Always rechunk if we have more chunks than rows.
                // except when we have an empty df containing a single chunk
                let height = self.height();
                let n_chunks = first_column_chunk_lengths.len();
                if n_chunks > height && !(height == 0 && n_chunks == 1) {
                    return true;
                }
                // Slow Path for multi Chunk series
                let v: Vec<_> = first_column_chunk_lengths.collect();
                for cl in chunk_lengths {
                    if cl.enumerate().any(|(idx, el)| Some(&el) != v.get(idx)) {
                        return true;
                    }
                }
                false
            },
        }
    }

    /// Ensure all the chunks in the [`DataFrame`] are aligned.
    pub fn align_chunks(&mut self) -> &mut Self {
        if self.should_rechunk() {
            self.as_single_chunk_par()
        } else {
            self
        }
    }

    /// Get the [`DataFrame`] schema.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let df: DataFrame = df!("Thing" => &["Observable universe", "Human stupidity"],
    ///                         "Diameter (m)" => &[8.8e26, f64::INFINITY])?;
    ///
    /// let f1: Field = Field::new("Thing", DataType::String);
    /// let f2: Field = Field::new("Diameter (m)", DataType::Float64);
    /// let sc: Schema = Schema::from_iter(vec![f1, f2]);
    ///
    /// assert_eq!(df.schema(), sc);
    /// # Ok::<(), PolarsError>(())
    /// ```
    pub fn schema(&self) -> Schema {
        self.columns.as_slice().into()
    }

    /// Get a reference to the [`DataFrame`] columns.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let df: DataFrame = df!("Name" => &["Adenine", "Cytosine", "Guanine", "Thymine"],
    ///                         "Symbol" => &["A", "C", "G", "T"])?;
    /// let columns: &[Series] = df.get_columns();
    ///
    /// assert_eq!(columns[0].name(), "Name");
    /// assert_eq!(columns[1].name(), "Symbol");
    /// # Ok::<(), PolarsError>(())
    /// ```
    #[inline]
    pub fn get_columns(&self) -> &[Series] {
        &self.columns
    }

    #[inline]
    /// Get mutable access to the underlying columns.
    ///
    /// # Safety
    /// The caller must ensure the length of all [`Series`] remains equal.
    pub unsafe fn get_columns_mut(&mut self) -> &mut Vec<Series> {
        &mut self.columns
    }

    /// Take ownership of the underlying columns vec.
    pub fn take_columns(self) -> Vec<Series> {
        self.columns
    }

    /// Iterator over the columns as [`Series`].
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let s1: Series = Series::new("Name", &["Pythagoras' theorem", "Shannon entropy"]);
    /// let s2: Series = Series::new("Formula", &["a²+b²=c²", "H=-Σ[P(x)log|P(x)|]"]);
    /// let df: DataFrame = DataFrame::new(vec![s1.clone(), s2.clone()])?;
    ///
    /// let mut iterator = df.iter();
    ///
    /// assert_eq!(iterator.next(), Some(&s1));
    /// assert_eq!(iterator.next(), Some(&s2));
    /// assert_eq!(iterator.next(), None);
    /// # Ok::<(), PolarsError>(())
    /// ```
    pub fn iter(&self) -> std::slice::Iter<'_, Series> {
        self.columns.iter()
    }

    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let df: DataFrame = df!("Language" => &["Rust", "Python"],
    ///                         "Designer" => &["Graydon Hoare", "Guido van Rossum"])?;
    ///
    /// assert_eq!(df.get_column_names(), &["Language", "Designer"]);
    /// # Ok::<(), PolarsError>(())
    /// ```
    pub fn get_column_names(&self) -> Vec<&str> {
        self.columns.iter().map(|s| s.name()).collect()
    }

    /// Get the [`Vec<String>`] representing the column names.
    pub fn get_column_names_owned(&self) -> Vec<SmartString> {
        self.columns.iter().map(|s| s.name().into()).collect()
    }

    /// Set the column names.
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let mut df: DataFrame = df!("Mathematical set" => &["ℕ", "ℤ", "𝔻", "ℚ", "ℝ", "ℂ"])?;
    /// df.set_column_names(&["Set"])?;
    ///
    /// assert_eq!(df.get_column_names(), &["Set"]);
    /// # Ok::<(), PolarsError>(())
    /// ```
    pub fn set_column_names<S: AsRef<str>>(&mut self, names: &[S]) -> PolarsResult<()> {
        polars_ensure!(
            names.len() == self.width(),
            ShapeMismatch: "{} column names provided for a DataFrame of width {}",
            names.len(), self.width()
        );
        ensure_names_unique(names, |s| s.as_ref())?;

        let columns = mem::take(&mut self.columns);
        self.columns = columns
            .into_iter()
            .zip(names)
            .map(|(s, name)| {
                let mut s = s;
                s.rename(name.as_ref());
                s
            })
            .collect();
        Ok(())
    }

    /// Get the data types of the columns in the [`DataFrame`].
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let venus_air: DataFrame = df!("Element" => &["Carbon dioxide", "Nitrogen"],
    ///                                "Fraction" => &[0.965, 0.035])?;
    ///
    /// assert_eq!(venus_air.dtypes(), &[DataType::String, DataType::Float64]);
    /// # Ok::<(), PolarsError>(())
    /// ```
    pub fn dtypes(&self) -> Vec<DataType> {
        self.columns.iter().map(|s| s.dtype().clone()).collect()
    }

    /// The number of chunks per column
    pub fn n_chunks(&self) -> usize {
        match self.columns.first() {
            None => 0,
            Some(s) => s.n_chunks(),
        }
    }

    /// Get a reference to the schema fields of the [`DataFrame`].
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let earth: DataFrame = df!("Surface type" => &["Water", "Land"],
    ///                            "Fraction" => &[0.708, 0.292])?;
    ///
    /// let f1: Field = Field::new("Surface type", DataType::String);
    /// let f2: Field = Field::new("Fraction", DataType::Float64);
    ///
    /// assert_eq!(earth.fields(), &[f1, f2]);
    /// # Ok::<(), PolarsError>(())
    /// ```
    pub fn fields(&self) -> Vec<Field> {
        self.columns
            .iter()
            .map(|s| s.field().into_owned())
            .collect()
    }

    /// Get (height, width) of the [`DataFrame`].
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let df0: DataFrame = DataFrame::default();
    /// let df1: DataFrame = df!("1" => &[1, 2, 3, 4, 5])?;
    /// let df2: DataFrame = df!("1" => &[1, 2, 3, 4, 5],
    ///                          "2" => &[1, 2, 3, 4, 5])?;
    ///
    /// assert_eq!(df0.shape(), (0 ,0));
    /// assert_eq!(df1.shape(), (5, 1));
    /// assert_eq!(df2.shape(), (5, 2));
    /// # Ok::<(), PolarsError>(())
    /// ```
    pub fn shape(&self) -> (usize, usize) {
        match self.columns.as_slice() {
            &[] => (0, 0),
            v => (v[0].len(), v.len()),
        }
    }

    /// Get the width of the [`DataFrame`] which is the number of columns.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let df0: DataFrame = DataFrame::default();
    /// let df1: DataFrame = df!("Series 1" => &[0; 0])?;
    /// let df2: DataFrame = df!("Series 1" => &[0; 0],
    ///                          "Series 2" => &[0; 0])?;
    ///
    /// assert_eq!(df0.width(), 0);
    /// assert_eq!(df1.width(), 1);
    /// assert_eq!(df2.width(), 2);
    /// # Ok::<(), PolarsError>(())
    /// ```
    pub fn width(&self) -> usize {
        self.columns.len()
    }

    /// Get the height of the [`DataFrame`] which is the number of rows.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let df0: DataFrame = DataFrame::default();
    /// let df1: DataFrame = df!("Currency" => &["€", "$"])?;
    /// let df2: DataFrame = df!("Currency" => &["€", "$", "¥", "£", "₿"])?;
    ///
    /// assert_eq!(df0.height(), 0);
    /// assert_eq!(df1.height(), 2);
    /// assert_eq!(df2.height(), 5);
    /// # Ok::<(), PolarsError>(())
    /// ```
    pub fn height(&self) -> usize {
        self.shape().0
    }

    /// Returns `true` if the [`DataFrame`] contains no rows.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let df1: DataFrame = DataFrame::default();
    /// assert!(df1.is_empty());
    ///
    /// let df2: DataFrame = df!("First name" => &["Forever"],
    ///                          "Last name" => &["Alone"])?;
    /// assert!(!df2.is_empty());
    /// # Ok::<(), PolarsError>(())
    /// ```
    pub fn is_empty(&self) -> bool {
        self.height() == 0
    }

    /// Add multiple [`Series`] to a [`DataFrame`].
    /// The added `Series` are required to have the same length.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let df1: DataFrame = df!("Element" => &["Copper", "Silver", "Gold"])?;
    /// let s1: Series = Series::new("Proton", &[29, 47, 79]);
    /// let s2: Series = Series::new("Electron", &[29, 47, 79]);
    ///
    /// let df2: DataFrame = df1.hstack(&[s1, s2])?;
    /// assert_eq!(df2.shape(), (3, 3));
    /// println!("{}", df2);
    /// # Ok::<(), PolarsError>(())
    /// ```
    ///
    /// Output:
    ///
    /// ```text
    /// shape: (3, 3)
    /// +---------+--------+----------+
    /// | Element | Proton | Electron |
    /// | ---     | ---    | ---      |
    /// | str     | i32    | i32      |
    /// +=========+========+==========+
    /// | Copper  | 29     | 29       |
    /// +---------+--------+----------+
    /// | Silver  | 47     | 47       |
    /// +---------+--------+----------+
    /// | Gold    | 79     | 79       |
    /// +---------+--------+----------+
    /// ```
    pub fn hstack(&self, columns: &[Series]) -> PolarsResult<Self> {
        let mut new_cols = self.columns.clone();
        new_cols.extend_from_slice(columns);
        DataFrame::new(new_cols)
    }

    /// Concatenate a [`DataFrame`] to this [`DataFrame`] and return as newly allocated [`DataFrame`].
    ///
    /// If many `vstack` operations are done, it is recommended to call [`DataFrame::align_chunks`].
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let df1: DataFrame = df!("Element" => &["Copper", "Silver", "Gold"],
    ///                          "Melting Point (K)" => &[1357.77, 1234.93, 1337.33])?;
    /// let df2: DataFrame = df!("Element" => &["Platinum", "Palladium"],
    ///                          "Melting Point (K)" => &[2041.4, 1828.05])?;
    ///
    /// let df3: DataFrame = df1.vstack(&df2)?;
    ///
    /// assert_eq!(df3.shape(), (5, 2));
    /// println!("{}", df3);
    /// # Ok::<(), PolarsError>(())
    /// ```
    ///
    /// Output:
    ///
    /// ```text
    /// shape: (5, 2)
    /// +-----------+-------------------+
    /// | Element   | Melting Point (K) |
    /// | ---       | ---               |
    /// | str       | f64               |
    /// +===========+===================+
    /// | Copper    | 1357.77           |
    /// +-----------+-------------------+
    /// | Silver    | 1234.93           |
    /// +-----------+-------------------+
    /// | Gold      | 1337.33           |
    /// +-----------+-------------------+
    /// | Platinum  | 2041.4            |
    /// +-----------+-------------------+
    /// | Palladium | 1828.05           |
    /// +-----------+-------------------+
    /// ```
    pub fn vstack(&self, other: &DataFrame) -> PolarsResult<Self> {
        let mut df = self.clone();
        df.vstack_mut(other)?;
        Ok(df)
    }

    /// Concatenate a [`DataFrame`] to this [`DataFrame`]
    ///
    /// If many `vstack` operations are done, it is recommended to call [`DataFrame::align_chunks`].
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let mut df1: DataFrame = df!("Element" => &["Copper", "Silver", "Gold"],
    ///                          "Melting Point (K)" => &[1357.77, 1234.93, 1337.33])?;
    /// let df2: DataFrame = df!("Element" => &["Platinum", "Palladium"],
    ///                          "Melting Point (K)" => &[2041.4, 1828.05])?;
    ///
    /// df1.vstack_mut(&df2)?;
    ///
    /// assert_eq!(df1.shape(), (5, 2));
    /// println!("{}", df1);
    /// # Ok::<(), PolarsError>(())
    /// ```
    ///
    /// Output:
    ///
    /// ```text
    /// shape: (5, 2)
    /// +-----------+-------------------+
    /// | Element   | Melting Point (K) |
    /// | ---       | ---               |
    /// | str       | f64               |
    /// +===========+===================+
    /// | Copper    | 1357.77           |
    /// +-----------+-------------------+
    /// | Silver    | 1234.93           |
    /// +-----------+-------------------+
    /// | Gold      | 1337.33           |
    /// +-----------+-------------------+
    /// | Platinum  | 2041.4            |
    /// +-----------+-------------------+
    /// | Palladium | 1828.05           |
    /// +-----------+-------------------+
    /// ```
    pub fn vstack_mut(&mut self, other: &DataFrame) -> PolarsResult<&mut Self> {
        if self.width() != other.width() {
            polars_ensure!(
                self.width() == 0,
                ShapeMismatch:
                "unable to append to a DataFrame of width {} with a DataFrame of width {}",
                self.width(), other.width(),
            );
            self.columns.clone_from(&other.columns);
            return Ok(self);
        }

        self.columns
            .iter_mut()
            .zip(other.columns.iter())
            .try_for_each::<_, PolarsResult<_>>(|(left, right)| {
                ensure_can_extend(left, right)?;
                left.append(right)?;
                Ok(())
            })?;
        Ok(self)
    }

    /// Concatenate a [`DataFrame`] to this [`DataFrame`]
    ///
    /// If many `vstack` operations are done, it is recommended to call [`DataFrame::align_chunks`].
    ///
    /// # Panics
    /// Panics if the schema's don't match.
    pub fn vstack_mut_unchecked(&mut self, other: &DataFrame) {
        self.columns
            .iter_mut()
            .zip(other.columns.iter())
            .for_each(|(left, right)| {
                left.append(right).expect("should not fail");
            });
    }

    /// Extend the memory backed by this [`DataFrame`] with the values from `other`.
    ///
    /// Different from [`vstack`](Self::vstack) which adds the chunks from `other` to the chunks of this [`DataFrame`]
    /// `extend` appends the data from `other` to the underlying memory locations and thus may cause a reallocation.
    ///
    /// If this does not cause a reallocation, the resulting data structure will not have any extra chunks
    /// and thus will yield faster queries.
    ///
    /// Prefer `extend` over `vstack` when you want to do a query after a single append. For instance during
    /// online operations where you add `n` rows and rerun a query.
    ///
    /// Prefer `vstack` over `extend` when you want to append many times before doing a query. For instance
    /// when you read in multiple files and when to store them in a single `DataFrame`. In the latter case, finish the sequence
    /// of `append` operations with a [`rechunk`](Self::align_chunks).
    pub fn extend(&mut self, other: &DataFrame) -> PolarsResult<()> {
        polars_ensure!(
            self.width() == other.width(),
            ShapeMismatch:
            "unable to extend a DataFrame of width {} with a DataFrame of width {}",
            self.width(), other.width(),
        );
        self.columns
            .iter_mut()
            .zip(other.columns.iter())
            .try_for_each::<_, PolarsResult<_>>(|(left, right)| {
                ensure_can_extend(left, right)?;
                left.extend(right)?;
                Ok(())
            })
    }

    /// Remove a column by name and return the column removed.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let mut df: DataFrame = df!("Animal" => &["Tiger", "Lion", "Great auk"],
    ///                             "IUCN" => &["Endangered", "Vulnerable", "Extinct"])?;
    ///
    /// let s1: PolarsResult<Series> = df.drop_in_place("Average weight");
    /// assert!(s1.is_err());
    ///
    /// let s2: Series = df.drop_in_place("Animal")?;
    /// assert_eq!(s2, Series::new("Animal", &["Tiger", "Lion", "Great auk"]));
    /// # Ok::<(), PolarsError>(())
    /// ```
    pub fn drop_in_place(&mut self, name: &str) -> PolarsResult<Series> {
        let idx = self.check_name_to_idx(name)?;
        Ok(self.columns.remove(idx))
    }

    /// Return a new [`DataFrame`] where all null values are dropped.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use polars_core::prelude::*;
    /// let df1: DataFrame = df!("Country" => ["Malta", "Liechtenstein", "North Korea"],
    ///                         "Tax revenue (% GDP)" => [Some(32.7), None, None])?;
    /// assert_eq!(df1.shape(), (3, 2));
    ///
    /// let df2: DataFrame = df1.drop_nulls::<String>(None)?;
    /// assert_eq!(df2.shape(), (1, 2));
    /// println!("{}", df2);
    /// # Ok::<(), PolarsError>(())
    /// ```
    ///
    /// Output:
    ///
    /// ```text
    /// shape: (1, 2)
    /// +---------+---------------------+
    /// | Country | Tax revenue (% GDP) |
    /// | ---     | ---                 |
    /// | str     | f64                 |
    /// +=========+=====================+
    /// | Malta   | 32.7                |
    /// +---------+---------------------+
    /// ```
    pub fn drop_nulls<S: AsRef<str>>(&self, subset: Option<&[S]>) -> PolarsResult<Self> {
        let selected_series;

        let mut iter = match subset {
            Some(cols) => {
                selected_series = self.select_series(cols)?;
                selected_series.iter()
            },
            None => self.columns.iter(),
        };

        // fast path for no nulls in df
        if iter.clone().all(|s| !s.has_nulls()) {
            return Ok(self.clone());
        }

        let mask = iter
            .next()
            .ok_or_else(|| polars_err!(NoData: "no data to drop nulls from"))?;
        let mut mask = mask.is_not_null();

        for s in iter {
            mask = mask & s.is_not_null();
        }
        self.filter(&mask)
    }

    /// Drop a column by name.
    /// This is a pure method and will return a new [`DataFrame`] instead of modifying
    /// the current one in place.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let df1: DataFrame = df!("Ray type" => &["α", "β", "X", "γ"])?;
    /// let df2: DataFrame = df1.drop("Ray type")?;
    ///
    /// assert!(df2.is_empty());
    /// # Ok::<(), PolarsError>(())
    /// ```
    pub fn drop(&self, name: &str) -> PolarsResult<Self> {
        let idx = self.check_name_to_idx(name)?;
        let mut new_cols = Vec::with_capacity(self.columns.len() - 1);

        self.columns.iter().enumerate().for_each(|(i, s)| {
            if i != idx {
                new_cols.push(s.clone())
            }
        });

        Ok(unsafe { DataFrame::new_no_checks(new_cols) })
    }

    /// Drop columns that are in `names`.
    pub fn drop_many<S: AsRef<str>>(&self, names: &[S]) -> Self {
        let names: PlHashSet<_> = names.iter().map(|s| s.as_ref()).collect();
        self.drop_many_amortized(&names)
    }

    /// Drop columns that are in `names` without allocating a [`HashSet`](std::collections::HashSet).
    pub fn drop_many_amortized(&self, names: &PlHashSet<&str>) -> DataFrame {
        if names.is_empty() {
            return self.clone();
        }
        let mut new_cols = Vec::with_capacity(self.columns.len().saturating_sub(names.len()));
        self.columns.iter().for_each(|s| {
            if !names.contains(&s.name()) {
                new_cols.push(s.clone())
            }
        });

        unsafe { DataFrame::new_no_checks(new_cols) }
    }

    /// Insert a new column at a given index without checking for duplicates.
    /// This can leave the [`DataFrame`] at an invalid state
    fn insert_column_no_name_check(
        &mut self,
        index: usize,
        series: Series,
    ) -> PolarsResult<&mut Self> {
        polars_ensure!(
            self.width() == 0 || series.len() == self.height(),
            ShapeMismatch: "unable to add a column of length {} to a DataFrame of height {}",
            series.len(), self.height(),
        );
        self.columns.insert(index, series);
        Ok(self)
    }

    /// Insert a new column at a given index.
    pub fn insert_column<S: IntoSeries>(
        &mut self,
        index: usize,
        column: S,
    ) -> PolarsResult<&mut Self> {
        let series = column.into_series();
        self.check_already_present(series.name())?;
        self.insert_column_no_name_check(index, series)
    }

    fn add_column_by_search(&mut self, series: Series) -> PolarsResult<()> {
        if let Some(idx) = self.get_column_index(series.name()) {
            self.replace_column(idx, series)?;
        } else {
            self.columns.push(series);
        }
        Ok(())
    }

    /// Add a new column to this [`DataFrame`] or replace an existing one.
    pub fn with_column<S: IntoSeries>(&mut self, column: S) -> PolarsResult<&mut Self> {
        fn inner(df: &mut DataFrame, mut series: Series) -> PolarsResult<&mut DataFrame> {
            let height = df.height();
            if series.len() == 1 && height > 1 {
                series = series.new_from_index(0, height);
            }

            if series.len() == height || df.get_columns().is_empty() {
                df.add_column_by_search(series)?;
                Ok(df)
            }
            // special case for literals
            else if height == 0 && series.len() == 1 {
                let s = series.clear();
                df.add_column_by_search(s)?;
                Ok(df)
            } else {
                polars_bail!(
                    ShapeMismatch: "unable to add a column of length {} to a DataFrame of height {}",
                    series.len(), height,
                );
            }
        }
        let series = column.into_series();
        inner(self, series)
    }

    /// Adds a column to the [`DataFrame`] without doing any checks
    /// on length or duplicates.
    ///
    /// # Safety
    /// The caller must ensure `column.len() == self.height()` .
    pub unsafe fn with_column_unchecked(&mut self, column: Series) -> &mut Self {
        #[cfg(debug_assertions)]
        {
            return self.with_column(column).unwrap();
        }
        #[cfg(not(debug_assertions))]
        {
            self.get_columns_mut().push(column);
            self
        }
    }

    fn add_column_by_schema(&mut self, s: Series, schema: &Schema) -> PolarsResult<()> {
        let name = s.name();
        if let Some((idx, _, _)) = schema.get_full(name) {
            // schema is incorrect fallback to search
            if self.columns.get(idx).map(|s| s.name()) != Some(name) {
                self.add_column_by_search(s)?;
            } else {
                self.replace_column(idx, s)?;
            }
        } else {
            self.columns.push(s);
        }
        Ok(())
    }

    pub fn _add_columns(&mut self, columns: Vec<Series>, schema: &Schema) -> PolarsResult<()> {
        for (i, s) in columns.into_iter().enumerate() {
            // we need to branch here
            // because users can add multiple columns with the same name
            if i == 0 || schema.get(s.name()).is_some() {
                self.with_column_and_schema(s, schema)?;
            } else {
                self.with_column(s.clone())?;
            }
        }
        Ok(())
    }

    /// Add a new column to this [`DataFrame`] or replace an existing one.
    /// Uses an existing schema to amortize lookups.
    /// If the schema is incorrect, we will fallback to linear search.
    pub fn with_column_and_schema<S: IntoSeries>(
        &mut self,
        column: S,
        schema: &Schema,
    ) -> PolarsResult<&mut Self> {
        let mut series = column.into_series();

        let height = self.height();
        if series.len() == 1 && height > 1 {
            series = series.new_from_index(0, height);
        }

        if series.len() == height || self.columns.is_empty() {
            self.add_column_by_schema(series, schema)?;
            Ok(self)
        }
        // special case for literals
        else if height == 0 && series.len() == 1 {
            let s = series.clear();
            self.add_column_by_schema(s, schema)?;
            Ok(self)
        } else {
            polars_bail!(
                ShapeMismatch: "unable to add a column of length {} to a DataFrame of height {}",
                series.len(), height,
            );
        }
    }

    /// Get a row in the [`DataFrame`]. Beware this is slow.
    ///
    /// # Example
    ///
    /// ```
    /// # use polars_core::prelude::*;
    /// fn example(df: &mut DataFrame, idx: usize) -> Option<Vec<AnyValue>> {
    ///     df.get(idx)
    /// }
    /// ```
    pub fn get(&self, idx: usize) -> Option<Vec<AnyValue>> {
        match self.columns.first() {
            Some(s) => {
                if s.len() <= idx {
                    return None;
                }
            },
            None => return None,
        }
        // SAFETY: we just checked bounds
        unsafe { Some(self.columns.iter().map(|s| s.get_unchecked(idx)).collect()) }
    }

    /// Select a [`Series`] by index.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let df: DataFrame = df!("Star" => &["Sun", "Betelgeuse", "Sirius A", "Sirius B"],
    ///                         "Absolute magnitude" => &[4.83, -5.85, 1.42, 11.18])?;
    ///
    /// let s1: Option<&Series> = df.select_at_idx(0);
    /// let s2: Series = Series::new("Star", &["Sun", "Betelgeuse", "Sirius A", "Sirius B"]);
    ///
    /// assert_eq!(s1, Some(&s2));
    /// # Ok::<(), PolarsError>(())
    /// ```
    pub fn select_at_idx(&self, idx: usize) -> Option<&Series> {
        self.columns.get(idx)
    }

    /// Select a mutable series by index.
    ///
    /// *Note: the length of the Series should remain the same otherwise the DataFrame is invalid.*
    /// For this reason the method is not public
    fn select_at_idx_mut(&mut self, idx: usize) -> Option<&mut Series> {
        self.columns.get_mut(idx)
    }

    /// Select column(s) from this [`DataFrame`] by range and return a new [`DataFrame`]
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let df = df! {
    ///     "0" => &[0, 0, 0],
    ///     "1" => &[1, 1, 1],
    ///     "2" => &[2, 2, 2]
    /// }?;
    ///
    /// assert!(df.select(&["0", "1"])?.equals(&df.select_by_range(0..=1)?));
    /// assert!(df.equals(&df.select_by_range(..)?));
    /// # Ok::<(), PolarsError>(())
    /// ```
    pub fn select_by_range<R>(&self, range: R) -> PolarsResult<Self>
    where
        R: ops::RangeBounds<usize>,
    {
        // This function is copied from std::slice::range (https://doc.rust-lang.org/std/slice/fn.range.html)
        // because it is the nightly feature. We should change here if this function were stable.
        fn get_range<R>(range: R, bounds: ops::RangeTo<usize>) -> ops::Range<usize>
        where
            R: ops::RangeBounds<usize>,
        {
            let len = bounds.end;

            let start: ops::Bound<&usize> = range.start_bound();
            let start = match start {
                ops::Bound::Included(&start) => start,
                ops::Bound::Excluded(start) => start.checked_add(1).unwrap_or_else(|| {
                    panic!("attempted to index slice from after maximum usize");
                }),
                ops::Bound::Unbounded => 0,
            };

            let end: ops::Bound<&usize> = range.end_bound();
            let end = match end {
                ops::Bound::Included(end) => end.checked_add(1).unwrap_or_else(|| {
                    panic!("attempted to index slice up to maximum usize");
                }),
                ops::Bound::Excluded(&end) => end,
                ops::Bound::Unbounded => len,
            };

            if start > end {
                panic!("slice index starts at {start} but ends at {end}");
            }
            if end > len {
                panic!("range end index {end} out of range for slice of length {len}",);
            }

            ops::Range { start, end }
        }

        let colnames = self.get_column_names_owned();
        let range = get_range(range, ..colnames.len());

        self._select_impl(&colnames[range])
    }

    /// Get column index of a [`Series`] by name.
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let df: DataFrame = df!("Name" => &["Player 1", "Player 2", "Player 3"],
    ///                         "Health" => &[100, 200, 500],
    ///                         "Mana" => &[250, 100, 0],
    ///                         "Strength" => &[30, 150, 300])?;
    ///
    /// assert_eq!(df.get_column_index("Name"), Some(0));
    /// assert_eq!(df.get_column_index("Health"), Some(1));
    /// assert_eq!(df.get_column_index("Mana"), Some(2));
    /// assert_eq!(df.get_column_index("Strength"), Some(3));
    /// assert_eq!(df.get_column_index("Haste"), None);
    /// # Ok::<(), PolarsError>(())
    /// ```
    pub fn get_column_index(&self, name: &str) -> Option<usize> {
        self.columns.iter().position(|s| s.name() == name)
    }

    /// Get column index of a [`Series`] by name.
    pub fn try_get_column_index(&self, name: &str) -> PolarsResult<usize> {
        self.get_column_index(name)
            .ok_or_else(|| polars_err!(col_not_found = name))
    }

    /// Select a single column by name.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let s1: Series = Series::new("Password", &["123456", "[]B$u$g$s$B#u#n#n#y[]{}"]);
    /// let s2: Series = Series::new("Robustness", &["Weak", "Strong"]);
    /// let df: DataFrame = DataFrame::new(vec![s1.clone(), s2])?;
    ///
    /// assert_eq!(df.column("Password")?, &s1);
    /// # Ok::<(), PolarsError>(())
    /// ```
    pub fn column(&self, name: &str) -> PolarsResult<&Series> {
        let idx = self.try_get_column_index(name)?;
        Ok(self.select_at_idx(idx).unwrap())
    }

    /// Selected multiple columns by name.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let df: DataFrame = df!("Latin name" => &["Oncorhynchus kisutch", "Salmo salar"],
    ///                         "Max weight (kg)" => &[16.0, 35.89])?;
    /// let sv: Vec<&Series> = df.columns(&["Latin name", "Max weight (kg)"])?;
    ///
    /// assert_eq!(&df[0], sv[0]);
    /// assert_eq!(&df[1], sv[1]);
    /// # Ok::<(), PolarsError>(())
    /// ```
    pub fn columns<I, S>(&self, names: I) -> PolarsResult<Vec<&Series>>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        names
            .into_iter()
            .map(|name| self.column(name.as_ref()))
            .collect()
    }

    /// Select column(s) from this [`DataFrame`] and return a new [`DataFrame`].
    ///
    /// # Examples
    ///
    /// ```
    /// # use polars_core::prelude::*;
    /// fn example(df: &DataFrame) -> PolarsResult<DataFrame> {
    ///     df.select(["foo", "bar"])
    /// }
    /// ```
    pub fn select<I, S>(&self, selection: I) -> PolarsResult<Self>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let cols = selection
            .into_iter()
            .map(|s| SmartString::from(s.as_ref()))
            .collect::<Vec<_>>();
        self._select_impl(&cols)
    }

    pub fn _select_impl(&self, cols: &[SmartString]) -> PolarsResult<Self> {
        ensure_names_unique(cols, |s| s.as_str())?;
        self._select_impl_unchecked(cols)
    }

    pub fn _select_impl_unchecked(&self, cols: &[SmartString]) -> PolarsResult<Self> {
        let selected = self.select_series_impl(cols)?;
        Ok(unsafe { DataFrame::new_no_checks(selected) })
    }

    /// Select with a known schema.
    pub fn select_with_schema<I, S>(&self, selection: I, schema: &SchemaRef) -> PolarsResult<Self>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let cols = selection
            .into_iter()
            .map(|s| SmartString::from(s.as_ref()))
            .collect::<Vec<_>>();
        self.select_with_schema_impl(&cols, schema, true)
    }

    /// Select with a known schema. This doesn't check for duplicates.
    pub fn select_with_schema_unchecked<I, S>(
        &self,
        selection: I,
        schema: &Schema,
    ) -> PolarsResult<Self>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let cols = selection
            .into_iter()
            .map(|s| SmartString::from(s.as_ref()))
            .collect::<Vec<_>>();
        self.select_with_schema_impl(&cols, schema, false)
    }

    fn select_with_schema_impl(
        &self,
        cols: &[SmartString],
        schema: &Schema,
        check_duplicates: bool,
    ) -> PolarsResult<Self> {
        if check_duplicates {
            ensure_names_unique(cols, |s| s.as_str())?;
        }
        let selected = self.select_series_impl_with_schema(cols, schema)?;
        Ok(unsafe { DataFrame::new_no_checks(selected) })
    }

    /// A non generic implementation to reduce compiler bloat.
    fn select_series_impl_with_schema(
        &self,
        cols: &[SmartString],
        schema: &Schema,
    ) -> PolarsResult<Vec<Series>> {
        cols.iter()
            .map(|name| {
                let index = schema.try_get_full(name)?.0;
                Ok(self.columns[index].clone())
            })
            .collect()
    }

    pub fn select_physical<I, S>(&self, selection: I) -> PolarsResult<Self>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let cols = selection
            .into_iter()
            .map(|s| SmartString::from(s.as_ref()))
            .collect::<Vec<_>>();
        self.select_physical_impl(&cols)
    }

    fn select_physical_impl(&self, cols: &[SmartString]) -> PolarsResult<Self> {
        ensure_names_unique(cols, |s| s.as_str())?;
        let selected = self.select_series_physical_impl(cols)?;
        Ok(unsafe { DataFrame::new_no_checks(selected) })
    }

    /// Select column(s) from this [`DataFrame`] and return them into a [`Vec`].
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let df: DataFrame = df!("Name" => &["Methane", "Ethane", "Propane"],
    ///                         "Carbon" => &[1, 2, 3],
    ///                         "Hydrogen" => &[4, 6, 8])?;
    /// let sv: Vec<Series> = df.select_series(&["Carbon", "Hydrogen"])?;
    ///
    /// assert_eq!(df["Carbon"], sv[0]);
    /// assert_eq!(df["Hydrogen"], sv[1]);
    /// # Ok::<(), PolarsError>(())
    /// ```
    pub fn select_series(&self, selection: impl IntoVec<SmartString>) -> PolarsResult<Vec<Series>> {
        let cols = selection.into_vec();
        self.select_series_impl(&cols)
    }

    fn _names_to_idx_map(&self) -> PlHashMap<&str, usize> {
        self.columns
            .iter()
            .enumerate()
            .map(|(i, s)| (s.name(), i))
            .collect()
    }

    /// A non generic implementation to reduce compiler bloat.
    fn select_series_physical_impl(&self, cols: &[SmartString]) -> PolarsResult<Vec<Series>> {
        let selected = if cols.len() > 1 && self.columns.len() > 10 {
            let name_to_idx = self._names_to_idx_map();
            cols.iter()
                .map(|name| {
                    let idx = *name_to_idx
                        .get(name.as_str())
                        .ok_or_else(|| polars_err!(col_not_found = name))?;
                    Ok(self
                        .select_at_idx(idx)
                        .unwrap()
                        .to_physical_repr()
                        .into_owned())
                })
                .collect::<PolarsResult<Vec<_>>>()?
        } else {
            cols.iter()
                .map(|c| self.column(c).map(|s| s.to_physical_repr().into_owned()))
                .collect::<PolarsResult<Vec<_>>>()?
        };

        Ok(selected)
    }

    /// A non generic implementation to reduce compiler bloat.
    fn select_series_impl(&self, cols: &[SmartString]) -> PolarsResult<Vec<Series>> {
        let selected = if cols.len() > 1 && self.columns.len() > 10 {
            // we hash, because there are user that having millions of columns.
            // # https://github.com/pola-rs/polars/issues/1023
            let name_to_idx = self._names_to_idx_map();

            cols.iter()
                .map(|name| {
                    let idx = *name_to_idx
                        .get(name.as_str())
                        .ok_or_else(|| polars_err!(col_not_found = name))?;
                    Ok(self.select_at_idx(idx).unwrap().clone())
                })
                .collect::<PolarsResult<Vec<_>>>()?
        } else {
            cols.iter()
                .map(|c| self.column(c).cloned())
                .collect::<PolarsResult<Vec<_>>>()?
        };

        Ok(selected)
    }

    /// Select a mutable series by name.
    /// *Note: the length of the Series should remain the same otherwise the DataFrame is invalid.*
    /// For this reason the method is not public
    fn select_mut(&mut self, name: &str) -> Option<&mut Series> {
        let opt_idx = self.get_column_index(name);

        opt_idx.and_then(|idx| self.select_at_idx_mut(idx))
    }

    /// Take the [`DataFrame`] rows by a boolean mask.
    ///
    /// # Example
    ///
    /// ```
    /// # use polars_core::prelude::*;
    /// fn example(df: &DataFrame) -> PolarsResult<DataFrame> {
    ///     let mask = df.column("sepal_width")?.is_not_null();
    ///     df.filter(&mask)
    /// }
    /// ```
    pub fn filter(&self, mask: &BooleanChunked) -> PolarsResult<Self> {
        let new_col = self.try_apply_columns_par(&|s| s.filter(mask))?;
        Ok(unsafe { DataFrame::new_no_checks(new_col) })
    }

    /// Same as `filter` but does not parallelize.
    pub fn _filter_seq(&self, mask: &BooleanChunked) -> PolarsResult<Self> {
        let new_col = self.try_apply_columns(&|s| s.filter(mask))?;
        Ok(unsafe { DataFrame::new_no_checks(new_col) })
    }

    /// Take [`DataFrame`] rows by index values.
    ///
    /// # Example
    ///
    /// ```
    /// # use polars_core::prelude::*;
    /// fn example(df: &DataFrame) -> PolarsResult<DataFrame> {
    ///     let idx = IdxCa::new("idx", &[0, 1, 9]);
    ///     df.take(&idx)
    /// }
    /// ```
    pub fn take(&self, indices: &IdxCa) -> PolarsResult<Self> {
        let new_col = POOL.install(|| self.try_apply_columns_par(&|s| s.take(indices)))?;

        Ok(unsafe { DataFrame::new_no_checks(new_col) })
    }

    /// # Safety
    /// The indices must be in-bounds.
    pub unsafe fn take_unchecked(&self, idx: &IdxCa) -> Self {
        self.take_unchecked_impl(idx, true)
    }

    /// # Safety
    /// The indices must be in-bounds.
    pub unsafe fn take_unchecked_impl(&self, idx: &IdxCa, allow_threads: bool) -> Self {
        let cols = if allow_threads {
            POOL.install(|| self._apply_columns_par(&|s| s.take_unchecked(idx)))
        } else {
            self.columns.iter().map(|s| s.take_unchecked(idx)).collect()
        };
        unsafe { DataFrame::new_no_checks(cols) }
    }

    pub(crate) unsafe fn take_slice_unchecked(&self, idx: &[IdxSize]) -> Self {
        self.take_slice_unchecked_impl(idx, true)
    }

    unsafe fn take_slice_unchecked_impl(&self, idx: &[IdxSize], allow_threads: bool) -> Self {
        let cols = if allow_threads {
            POOL.install(|| self._apply_columns_par(&|s| s.take_slice_unchecked(idx)))
        } else {
            self.columns
                .iter()
                .map(|s| s.take_slice_unchecked(idx))
                .collect()
        };
        unsafe { DataFrame::new_no_checks(cols) }
    }

    /// Rename a column in the [`DataFrame`].
    ///
    /// # Example
    ///
    /// ```
    /// # use polars_core::prelude::*;
    /// fn example(df: &mut DataFrame) -> PolarsResult<&mut DataFrame> {
    ///     let original_name = "foo";
    ///     let new_name = "bar";
    ///     df.rename(original_name, new_name)
    /// }
    /// ```
    pub fn rename(&mut self, column: &str, name: &str) -> PolarsResult<&mut Self> {
        if column == name {
            return Ok(self);
        }
        polars_ensure!(
            self.columns.iter().all(|c| c.name() != name),
            Duplicate: "column rename attempted with already existing name \"{name}\""
        );
        self.select_mut(column)
            .ok_or_else(|| polars_err!(col_not_found = column))
            .map(|s| s.rename(name))?;
        Ok(self)
    }

    /// Sort [`DataFrame`] in place.
    ///
    /// See [`DataFrame::sort`] for more instruction.
    pub fn sort_in_place(
        &mut self,
        by: impl IntoVec<SmartString>,
        sort_options: SortMultipleOptions,
    ) -> PolarsResult<&mut Self> {
        let by_column = self.select_series(by)?;
        self.columns = self.sort_impl(by_column, sort_options, None)?.columns;
        Ok(self)
    }

    #[doc(hidden)]
    /// This is the dispatch of Self::sort, and exists to reduce compile bloat by monomorphization.
    pub fn sort_impl(
        &self,
        by_column: Vec<Series>,
        mut sort_options: SortMultipleOptions,
        slice: Option<(i64, usize)>,
    ) -> PolarsResult<Self> {
        if by_column.is_empty() {
            polars_bail!(ComputeError: "No columns selected for sorting");
        }
        // note that the by_column argument also contains evaluated expression from
        // polars-lazy that may not even be present in this dataframe. therefore
        // when we try to set the first columns as sorted, we ignore the error as
        // expressions are not present (they are renamed to _POLARS_SORT_COLUMN_i.
        let first_descending = sort_options.descending[0];
        let first_by_column = by_column[0].name().to_string();

        let set_sorted = |df: &mut DataFrame| {
            // Mark the first sort column as sorted; if the column does not exist it
            // is ok, because we sorted by an expression not present in the dataframe
            let _ = df.apply(&first_by_column, |s| {
                let mut s = s.clone();
                if first_descending {
                    s.set_sorted_flag(IsSorted::Descending)
                } else {
                    s.set_sorted_flag(IsSorted::Ascending)
                }
                s
            });
        };
        if self.is_empty() {
            let mut out = self.clone();
            set_sorted(&mut out);
            return Ok(out);
        }
        if let Some((0, k)) = slice {
            return self.bottom_k_impl(k, by_column, sort_options);
        }

        #[cfg(feature = "dtype-struct")]
        let has_struct = by_column
            .iter()
            .any(|s| matches!(s.dtype(), DataType::Struct(_)));

        #[cfg(not(feature = "dtype-struct"))]
        #[allow(non_upper_case_globals)]
        const has_struct: bool = false;

        // a lot of indirection in both sorting and take
        let mut df = self.clone();
        let df = df.as_single_chunk_par();
        let mut take = match (by_column.len(), has_struct) {
            (1, false) => {
                let s = &by_column[0];
                let options = SortOptions {
                    descending: sort_options.descending[0],
                    nulls_last: sort_options.nulls_last[0],
                    multithreaded: sort_options.multithreaded,
                    maintain_order: sort_options.maintain_order,
                };
                // fast path for a frame with a single series
                // no need to compute the sort indices and then take by these indices
                // simply sort and return as frame
                if df.width() == 1 && df.check_name_to_idx(s.name()).is_ok() {
                    let mut out = s.sort_with(options)?;
                    if let Some((offset, len)) = slice {
                        out = out.slice(offset, len);
                    }
                    return Ok(out.into_frame());
                }
                s.arg_sort(options)
            },
            _ => {
                if sort_options.nulls_last.iter().all(|&x| x)
                    || has_struct
                    || std::env::var("POLARS_ROW_FMT_SORT").is_ok()
                {
                    argsort_multiple_row_fmt(
                        &by_column,
                        sort_options.descending,
                        sort_options.nulls_last,
                        sort_options.multithreaded,
                    )?
                } else {
                    let (first, other) = prepare_arg_sort(by_column, &mut sort_options)?;
                    first.arg_sort_multiple(&other, &sort_options)?
                }
            },
        };

        if let Some((offset, len)) = slice {
            take = take.slice(offset, len);
        }

        // SAFETY:
        // the created indices are in bounds
        let mut df = unsafe { df.take_unchecked_impl(&take, sort_options.multithreaded) };
        set_sorted(&mut df);
        Ok(df)
    }

    /// Return a sorted clone of this [`DataFrame`].
    ///
    /// # Example
    ///
    /// Sort by a single column with default options:
    /// ```
    /// # use polars_core::prelude::*;
    /// fn sort_by_sepal_width(df: &DataFrame) -> PolarsResult<DataFrame> {
    ///     df.sort(["sepal_width"], Default::default())
    /// }
    /// ```
    /// Sort by a single column with specific order:
    /// ```
    /// # use polars_core::prelude::*;
    /// fn sort_with_specific_order(df: &DataFrame, descending: bool) -> PolarsResult<DataFrame> {
    ///     df.sort(
    ///         ["sepal_width"],
    ///         SortMultipleOptions::new()
    ///             .with_order_descending(descending)
    ///     )
    /// }
    /// ```
    /// Sort by multiple columns with specifying order for each column:
    /// ```
    /// # use polars_core::prelude::*;
    /// fn sort_by_multiple_columns_with_specific_order(df: &DataFrame) -> PolarsResult<DataFrame> {
    ///     df.sort(
    ///         &["sepal_width", "sepal_length"],
    ///         SortMultipleOptions::new()
    ///             .with_order_descending_multi([false, true])
    ///     )
    /// }
    /// ```
    /// See [`SortMultipleOptions`] for more options.
    ///
    /// Also see [`DataFrame::sort_in_place`].
    pub fn sort(
        &self,
        by: impl IntoVec<SmartString>,
        sort_options: SortMultipleOptions,
    ) -> PolarsResult<Self> {
        let mut df = self.clone();
        df.sort_in_place(by, sort_options)?;
        Ok(df)
    }

    /// Replace a column with a [`Series`].
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let mut df: DataFrame = df!("Country" => &["United States", "China"],
    ///                         "Area (km²)" => &[9_833_520, 9_596_961])?;
    /// let s: Series = Series::new("Country", &["USA", "PRC"]);
    ///
    /// assert!(df.replace("Nation", s.clone()).is_err());
    /// assert!(df.replace("Country", s).is_ok());
    /// # Ok::<(), PolarsError>(())
    /// ```
    pub fn replace<S: IntoSeries>(&mut self, column: &str, new_col: S) -> PolarsResult<&mut Self> {
        self.apply(column, |_| new_col.into_series())
    }

    /// Replace or update a column. The difference between this method and [DataFrame::with_column]
    /// is that now the value of `column: &str` determines the name of the column and not the name
    /// of the `Series` passed to this method.
    pub fn replace_or_add<S: IntoSeries>(
        &mut self,
        column: &str,
        new_col: S,
    ) -> PolarsResult<&mut Self> {
        let mut new_col = new_col.into_series();
        new_col.rename(column);
        self.with_column(new_col)
    }

    /// Replace column at index `idx` with a [`Series`].
    ///
    /// # Example
    ///
    /// ```ignored
    /// # use polars_core::prelude::*;
    /// let s0 = Series::new("foo", &["ham", "spam", "egg"]);
    /// let s1 = Series::new("ascii", &[70, 79, 79]);
    /// let mut df = DataFrame::new(vec![s0, s1])?;
    ///
    /// // Add 32 to get lowercase ascii values
    /// df.replace_column(1, df.select_at_idx(1).unwrap() + 32);
    /// # Ok::<(), PolarsError>(())
    /// ```
    pub fn replace_column<S: IntoSeries>(
        &mut self,
        index: usize,
        new_column: S,
    ) -> PolarsResult<&mut Self> {
        polars_ensure!(
            index < self.width(),
            ShapeMismatch:
            "unable to replace at index {}, the DataFrame has only {} columns",
            index, self.width(),
        );
        let mut new_column = new_column.into_series();
        polars_ensure!(
            new_column.len() == self.height(),
            ShapeMismatch:
            "unable to replace a column, series length {} doesn't match the DataFrame height {}",
            new_column.len(), self.height(),
        );
        let old_col = &mut self.columns[index];
        mem::swap(old_col, &mut new_column);
        Ok(self)
    }

    /// Apply a closure to a column. This is the recommended way to do in place modification.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let s0 = Series::new("foo", &["ham", "spam", "egg"]);
    /// let s1 = Series::new("names", &["Jean", "Claude", "van"]);
    /// let mut df = DataFrame::new(vec![s0, s1])?;
    ///
    /// fn str_to_len(str_val: &Series) -> Series {
    ///     str_val.str()
    ///         .unwrap()
    ///         .into_iter()
    ///         .map(|opt_name: Option<&str>| {
    ///             opt_name.map(|name: &str| name.len() as u32)
    ///          })
    ///         .collect::<UInt32Chunked>()
    ///         .into_series()
    /// }
    ///
    /// // Replace the names column by the length of the names.
    /// df.apply("names", str_to_len);
    /// # Ok::<(), PolarsError>(())
    /// ```
    /// Results in:
    ///
    /// ```text
    /// +--------+-------+
    /// | foo    |       |
    /// | ---    | names |
    /// | str    | u32   |
    /// +========+=======+
    /// | "ham"  | 4     |
    /// +--------+-------+
    /// | "spam" | 6     |
    /// +--------+-------+
    /// | "egg"  | 3     |
    /// +--------+-------+
    /// ```
    pub fn apply<F, S>(&mut self, name: &str, f: F) -> PolarsResult<&mut Self>
    where
        F: FnOnce(&Series) -> S,
        S: IntoSeries,
    {
        let idx = self.check_name_to_idx(name)?;
        self.apply_at_idx(idx, f)
    }

    /// Apply a closure to a column at index `idx`. This is the recommended way to do in place
    /// modification.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let s0 = Series::new("foo", &["ham", "spam", "egg"]);
    /// let s1 = Series::new("ascii", &[70, 79, 79]);
    /// let mut df = DataFrame::new(vec![s0, s1])?;
    ///
    /// // Add 32 to get lowercase ascii values
    /// df.apply_at_idx(1, |s| s + 32);
    /// # Ok::<(), PolarsError>(())
    /// ```
    /// Results in:
    ///
    /// ```text
    /// +--------+-------+
    /// | foo    | ascii |
    /// | ---    | ---   |
    /// | str    | i32   |
    /// +========+=======+
    /// | "ham"  | 102   |
    /// +--------+-------+
    /// | "spam" | 111   |
    /// +--------+-------+
    /// | "egg"  | 111   |
    /// +--------+-------+
    /// ```
    pub fn apply_at_idx<F, S>(&mut self, idx: usize, f: F) -> PolarsResult<&mut Self>
    where
        F: FnOnce(&Series) -> S,
        S: IntoSeries,
    {
        let df_height = self.height();
        let width = self.width();
        let col = self.columns.get_mut(idx).ok_or_else(|| {
            polars_err!(
                ComputeError: "invalid column index: {} for a DataFrame with {} columns",
                idx, width
            )
        })?;
        let name = col.name().to_string();
        let new_col = f(col).into_series();
        match new_col.len() {
            1 => {
                let new_col = new_col.new_from_index(0, df_height);
                let _ = mem::replace(col, new_col);
            },
            len if (len == df_height) => {
                let _ = mem::replace(col, new_col);
            },
            len => polars_bail!(
                ShapeMismatch:
                "resulting Series has length {} while the DataFrame has height {}",
                len, df_height
            ),
        }

        // make sure the name remains the same after applying the closure
        unsafe {
            let col = self.columns.get_unchecked_mut(idx);
            col.rename(&name);
        }
        Ok(self)
    }

    /// Apply a closure that may fail to a column at index `idx`. This is the recommended way to do in place
    /// modification.
    ///
    /// # Example
    ///
    /// This is the idiomatic way to replace some values a column of a `DataFrame` given range of indexes.
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let s0 = Series::new("foo", &["ham", "spam", "egg", "bacon", "quack"]);
    /// let s1 = Series::new("values", &[1, 2, 3, 4, 5]);
    /// let mut df = DataFrame::new(vec![s0, s1])?;
    ///
    /// let idx = vec![0, 1, 4];
    ///
    /// df.try_apply("foo", |s| {
    ///     s.str()?
    ///     .scatter_with(idx, |opt_val| opt_val.map(|string| format!("{}-is-modified", string)))
    /// });
    /// # Ok::<(), PolarsError>(())
    /// ```
    /// Results in:
    ///
    /// ```text
    /// +---------------------+--------+
    /// | foo                 | values |
    /// | ---                 | ---    |
    /// | str                 | i32    |
    /// +=====================+========+
    /// | "ham-is-modified"   | 1      |
    /// +---------------------+--------+
    /// | "spam-is-modified"  | 2      |
    /// +---------------------+--------+
    /// | "egg"               | 3      |
    /// +---------------------+--------+
    /// | "bacon"             | 4      |
    /// +---------------------+--------+
    /// | "quack-is-modified" | 5      |
    /// +---------------------+--------+
    /// ```
    pub fn try_apply_at_idx<F, S>(&mut self, idx: usize, f: F) -> PolarsResult<&mut Self>
    where
        F: FnOnce(&Series) -> PolarsResult<S>,
        S: IntoSeries,
    {
        let width = self.width();
        let col = self.columns.get_mut(idx).ok_or_else(|| {
            polars_err!(
                ComputeError: "invalid column index: {} for a DataFrame with {} columns",
                idx, width
            )
        })?;
        let name = col.name().to_string();

        let _ = mem::replace(col, f(col).map(|s| s.into_series())?);

        // make sure the name remains the same after applying the closure
        unsafe {
            let col = self.columns.get_unchecked_mut(idx);
            col.rename(&name);
        }
        Ok(self)
    }

    /// Apply a closure that may fail to a column. This is the recommended way to do in place
    /// modification.
    ///
    /// # Example
    ///
    /// This is the idiomatic way to replace some values a column of a `DataFrame` given a boolean mask.
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let s0 = Series::new("foo", &["ham", "spam", "egg", "bacon", "quack"]);
    /// let s1 = Series::new("values", &[1, 2, 3, 4, 5]);
    /// let mut df = DataFrame::new(vec![s0, s1])?;
    ///
    /// // create a mask
    /// let values = df.column("values")?;
    /// let mask = values.lt_eq(1)? | values.gt_eq(5_i32)?;
    ///
    /// df.try_apply("foo", |s| {
    ///     s.str()?
    ///     .set(&mask, Some("not_within_bounds"))
    /// });
    /// # Ok::<(), PolarsError>(())
    /// ```
    /// Results in:
    ///
    /// ```text
    /// +---------------------+--------+
    /// | foo                 | values |
    /// | ---                 | ---    |
    /// | str                 | i32    |
    /// +=====================+========+
    /// | "not_within_bounds" | 1      |
    /// +---------------------+--------+
    /// | "spam"              | 2      |
    /// +---------------------+--------+
    /// | "egg"               | 3      |
    /// +---------------------+--------+
    /// | "bacon"             | 4      |
    /// +---------------------+--------+
    /// | "not_within_bounds" | 5      |
    /// +---------------------+--------+
    /// ```
    pub fn try_apply<F, S>(&mut self, column: &str, f: F) -> PolarsResult<&mut Self>
    where
        F: FnOnce(&Series) -> PolarsResult<S>,
        S: IntoSeries,
    {
        let idx = self.try_get_column_index(column)?;
        self.try_apply_at_idx(idx, f)
    }

    /// Slice the [`DataFrame`] along the rows.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let df: DataFrame = df!("Fruit" => &["Apple", "Grape", "Grape", "Fig", "Fig"],
    ///                         "Color" => &["Green", "Red", "White", "White", "Red"])?;
    /// let sl: DataFrame = df.slice(2, 3);
    ///
    /// assert_eq!(sl.shape(), (3, 2));
    /// println!("{}", sl);
    /// # Ok::<(), PolarsError>(())
    /// ```
    /// Output:
    /// ```text
    /// shape: (3, 2)
    /// +-------+-------+
    /// | Fruit | Color |
    /// | ---   | ---   |
    /// | str   | str   |
    /// +=======+=======+
    /// | Grape | White |
    /// +-------+-------+
    /// | Fig   | White |
    /// +-------+-------+
    /// | Fig   | Red   |
    /// +-------+-------+
    /// ```
    #[must_use]
    pub fn slice(&self, offset: i64, length: usize) -> Self {
        if offset == 0 && length == self.height() {
            return self.clone();
        }
        if length == 0 {
            return self.clear();
        }
        let col = self
            .columns
            .iter()
            .map(|s| s.slice(offset, length))
            .collect::<Vec<_>>();
        unsafe { DataFrame::new_no_checks(col) }
    }

    /// Split [`DataFrame`] at the given `offset`.
    pub fn split_at(&self, offset: i64) -> (Self, Self) {
        let (a, b) = self.columns.iter().map(|s| s.split_at(offset)).unzip();
        let a = unsafe { DataFrame::new_no_checks(a) };
        let b = unsafe { DataFrame::new_no_checks(b) };
        (a, b)
    }

    pub fn clear(&self) -> Self {
        let col = self.columns.iter().map(|s| s.clear()).collect::<Vec<_>>();
        unsafe { DataFrame::new_no_checks(col) }
    }

    #[must_use]
    pub fn slice_par(&self, offset: i64, length: usize) -> Self {
        if offset == 0 && length == self.height() {
            return self.clone();
        }
        let columns = self._apply_columns_par(&|s| s.slice(offset, length));
        unsafe { DataFrame::new_no_checks(columns) }
    }

    #[must_use]
    pub fn _slice_and_realloc(&self, offset: i64, length: usize) -> Self {
        if offset == 0 && length == self.height() {
            return self.clone();
        }
        let columns = self._apply_columns(&|s| {
            let mut out = s.slice(offset, length);
            out.shrink_to_fit();
            out
        });
        unsafe { DataFrame::new_no_checks(columns) }
    }

    /// Get the head of the [`DataFrame`].
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let countries: DataFrame =
    ///     df!("Rank by GDP (2021)" => &[1, 2, 3, 4, 5],
    ///         "Continent" => &["North America", "Asia", "Asia", "Europe", "Europe"],
    ///         "Country" => &["United States", "China", "Japan", "Germany", "United Kingdom"],
    ///         "Capital" => &["Washington", "Beijing", "Tokyo", "Berlin", "London"])?;
    /// assert_eq!(countries.shape(), (5, 4));
    ///
    /// println!("{}", countries.head(Some(3)));
    /// # Ok::<(), PolarsError>(())
    /// ```
    ///
    /// Output:
    ///
    /// ```text
    /// shape: (3, 4)
    /// +--------------------+---------------+---------------+------------+
    /// | Rank by GDP (2021) | Continent     | Country       | Capital    |
    /// | ---                | ---           | ---           | ---        |
    /// | i32                | str           | str           | str        |
    /// +====================+===============+===============+============+
    /// | 1                  | North America | United States | Washington |
    /// +--------------------+---------------+---------------+------------+
    /// | 2                  | Asia          | China         | Beijing    |
    /// +--------------------+---------------+---------------+------------+
    /// | 3                  | Asia          | Japan         | Tokyo      |
    /// +--------------------+---------------+---------------+------------+
    /// ```
    #[must_use]
    pub fn head(&self, length: Option<usize>) -> Self {
        let col = self
            .columns
            .iter()
            .map(|s| s.head(length))
            .collect::<Vec<_>>();
        unsafe { DataFrame::new_no_checks(col) }
    }

    /// Get the tail of the [`DataFrame`].
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let countries: DataFrame =
    ///     df!("Rank (2021)" => &[105, 106, 107, 108, 109],
    ///         "Apple Price (€/kg)" => &[0.75, 0.70, 0.70, 0.65, 0.52],
    ///         "Country" => &["Kosovo", "Moldova", "North Macedonia", "Syria", "Turkey"])?;
    /// assert_eq!(countries.shape(), (5, 3));
    ///
    /// println!("{}", countries.tail(Some(2)));
    /// # Ok::<(), PolarsError>(())
    /// ```
    ///
    /// Output:
    ///
    /// ```text
    /// shape: (2, 3)
    /// +-------------+--------------------+---------+
    /// | Rank (2021) | Apple Price (€/kg) | Country |
    /// | ---         | ---                | ---     |
    /// | i32         | f64                | str     |
    /// +=============+====================+=========+
    /// | 108         | 0.63               | Syria   |
    /// +-------------+--------------------+---------+
    /// | 109         | 0.63               | Turkey  |
    /// +-------------+--------------------+---------+
    /// ```
    #[must_use]
    pub fn tail(&self, length: Option<usize>) -> Self {
        let col = self
            .columns
            .iter()
            .map(|s| s.tail(length))
            .collect::<Vec<_>>();
        unsafe { DataFrame::new_no_checks(col) }
    }

    /// Iterator over the rows in this [`DataFrame`] as Arrow RecordBatches.
    ///
    /// # Panics
    ///
    /// Panics if the [`DataFrame`] that is passed is not rechunked.
    ///
    /// This responsibility is left to the caller as we don't want to take mutable references here,
    /// but we also don't want to rechunk here, as this operation is costly and would benefit the caller
    /// as well.
    pub fn iter_chunks(&self, compat_level: CompatLevel, parallel: bool) -> RecordBatchIter {
        // If any of the columns is binview and we don't convert `compat_level` we allow parallelism
        // as we must allocate arrow strings/binaries.
        let parallel = if parallel && compat_level.0 >= 1 {
            self.columns.len() > 1
                && self
                    .columns
                    .iter()
                    .any(|s| matches!(s.dtype(), DataType::String | DataType::Binary))
        } else {
            false
        };

        RecordBatchIter {
            columns: &self.columns,
            idx: 0,
            n_chunks: self.n_chunks(),
            compat_level,
            parallel,
        }
    }

    /// Iterator over the rows in this [`DataFrame`] as Arrow RecordBatches as physical values.
    ///
    /// # Panics
    ///
    /// Panics if the [`DataFrame`] that is passed is not rechunked.
    ///
    /// This responsibility is left to the caller as we don't want to take mutable references here,
    /// but we also don't want to rechunk here, as this operation is costly and would benefit the caller
    /// as well.
    pub fn iter_chunks_physical(&self) -> PhysRecordBatchIter<'_> {
        PhysRecordBatchIter {
            iters: self.columns.iter().map(|s| s.chunks().iter()).collect(),
        }
    }

    /// Get a [`DataFrame`] with all the columns in reversed order.
    #[must_use]
    pub fn reverse(&self) -> Self {
        let col = self.columns.iter().map(|s| s.reverse()).collect::<Vec<_>>();
        unsafe { DataFrame::new_no_checks(col) }
    }

    /// Shift the values by a given period and fill the parts that will be empty due to this operation
    /// with `Nones`.
    ///
    /// See the method on [Series](crate::series::SeriesTrait::shift) for more info on the `shift` operation.
    #[must_use]
    pub fn shift(&self, periods: i64) -> Self {
        let col = self._apply_columns_par(&|s| s.shift(periods));
        unsafe { DataFrame::new_no_checks(col) }
    }

    /// Replace None values with one of the following strategies:
    /// * Forward fill (replace None with the previous value)
    /// * Backward fill (replace None with the next value)
    /// * Mean fill (replace None with the mean of the whole array)
    /// * Min fill (replace None with the minimum of the whole array)
    /// * Max fill (replace None with the maximum of the whole array)
    ///
    /// See the method on [Series](crate::series::Series::fill_null) for more info on the `fill_null` operation.
    pub fn fill_null(&self, strategy: FillNullStrategy) -> PolarsResult<Self> {
        let col = self.try_apply_columns_par(&|s| s.fill_null(strategy))?;

        Ok(unsafe { DataFrame::new_no_checks(col) })
    }

    /// Aggregate the column horizontally to their min values.
    #[cfg(feature = "zip_with")]
    pub fn min_horizontal(&self) -> PolarsResult<Option<Series>> {
        let min_fn = |acc: &Series, s: &Series| min_max_binary_series(acc, s, true);

        match self.columns.len() {
            0 => Ok(None),
            1 => Ok(Some(self.columns[0].clone())),
            2 => min_fn(&self.columns[0], &self.columns[1]).map(Some),
            _ => {
                // the try_reduce_with is a bit slower in parallelism,
                // but I don't think it matters here as we parallelize over columns, not over elements
                POOL.install(|| {
                    self.columns
                        .par_iter()
                        .map(|s| Ok(Cow::Borrowed(s)))
                        .try_reduce_with(|l, r| min_fn(&l, &r).map(Cow::Owned))
                        // we can unwrap the option, because we are certain there is a column
                        // we started this operation on 3 columns
                        .unwrap()
                        .map(|cow| Some(cow.into_owned()))
                })
            },
        }
    }

    /// Aggregate the column horizontally to their max values.
    #[cfg(feature = "zip_with")]
    pub fn max_horizontal(&self) -> PolarsResult<Option<Series>> {
        let max_fn = |acc: &Series, s: &Series| min_max_binary_series(acc, s, false);

        match self.columns.len() {
            0 => Ok(None),
            1 => Ok(Some(self.columns[0].clone())),
            2 => max_fn(&self.columns[0], &self.columns[1]).map(Some),
            _ => {
                // the try_reduce_with is a bit slower in parallelism,
                // but I don't think it matters here as we parallelize over columns, not over elements
                POOL.install(|| {
                    self.columns
                        .par_iter()
                        .map(|s| Ok(Cow::Borrowed(s)))
                        .try_reduce_with(|l, r| max_fn(&l, &r).map(Cow::Owned))
                        // we can unwrap the option, because we are certain there is a column
                        // we started this operation on 3 columns
                        .unwrap()
                        .map(|cow| Some(cow.into_owned()))
                })
            },
        }
    }

    /// Sum all values horizontally across columns.
    pub fn sum_horizontal(&self, null_strategy: NullStrategy) -> PolarsResult<Option<Series>> {
        let apply_null_strategy =
            |s: Series, null_strategy: NullStrategy| -> PolarsResult<Series> {
                if let NullStrategy::Ignore = null_strategy {
                    // if has nulls
                    if s.null_count() > 0 {
                        return s.fill_null(FillNullStrategy::Zero);
                    }
                }
                Ok(s)
            };

        let sum_fn =
            |acc: Series, s: Series, null_strategy: NullStrategy| -> PolarsResult<Series> {
                let acc: Series = apply_null_strategy(acc, null_strategy)?;
                let s = apply_null_strategy(s, null_strategy)?;
                // This will do owned arithmetic and can be mutable
                std::ops::Add::add(acc, s)
            };

        let non_null_cols = self
            .columns
            .iter()
            .filter(|x| x.dtype() != &DataType::Null)
            .collect::<Vec<_>>();

        match non_null_cols.len() {
            0 => {
                if self.columns.is_empty() {
                    Ok(None)
                } else {
                    // all columns are null dtype, so result is null dtype
                    Ok(Some(self.columns[0].clone()))
                }
            },
            1 => Ok(Some(apply_null_strategy(
                if non_null_cols[0].dtype() == &DataType::Boolean {
                    non_null_cols[0].cast(&DataType::UInt32)?
                } else {
                    non_null_cols[0].clone()
                },
                null_strategy,
            )?)),
            2 => sum_fn(
                non_null_cols[0].clone(),
                non_null_cols[1].clone(),
                null_strategy,
            )
            .map(Some),
            _ => {
                // the try_reduce_with is a bit slower in parallelism,
                // but I don't think it matters here as we parallelize over columns, not over elements
                let out = POOL.install(|| {
                    non_null_cols
                        .into_par_iter()
                        .cloned()
                        .map(Ok)
                        .try_reduce_with(|l, r| sum_fn(l, r, null_strategy))
                        // We can unwrap because we started with at least 3 columns, so we always get a Some
                        .unwrap()
                });
                out.map(Some)
            },
        }
    }

    /// Compute the mean of all values horizontally across columns.
    pub fn mean_horizontal(&self, null_strategy: NullStrategy) -> PolarsResult<Option<Series>> {
        match self.columns.len() {
            0 => Ok(None),
            1 => Ok(Some(match self.columns[0].dtype() {
                dt if dt != &DataType::Float32 && (dt.is_numeric() || dt == &DataType::Boolean) => {
                    self.columns[0].cast(&DataType::Float64)?
                },
                _ => self.columns[0].clone(),
            })),
            _ => {
                let columns = self
                    .columns
                    .iter()
                    .filter(|s| {
                        let dtype = s.dtype();
                        dtype.is_numeric() || matches!(dtype, DataType::Boolean)
                    })
                    .cloned()
                    .collect();
                let numeric_df = unsafe { DataFrame::new_no_checks(columns) };

                let sum = || numeric_df.sum_horizontal(null_strategy);

                let null_count = || {
                    numeric_df
                        .columns
                        .par_iter()
                        .map(|s| {
                            s.is_null()
                                .cast_with_options(&DataType::UInt32, CastOptions::NonStrict)
                        })
                        .reduce_with(|l, r| {
                            let l = l?;
                            let r = r?;
                            let result = std::ops::Add::add(&l, &r)?;
                            PolarsResult::Ok(result)
                        })
                        // we can unwrap the option, because we are certain there is a column
                        // we started this operation on 2 columns
                        .unwrap()
                };

                let (sum, null_count) = POOL.install(|| rayon::join(sum, null_count));
                let sum = sum?;
                let null_count = null_count?;

                // value lengths: len - null_count
                let value_length: UInt32Chunked =
                    (numeric_df.width().sub(&null_count)).u32().unwrap().clone();

                // make sure that we do not divide by zero
                // by replacing with None
                let value_length = value_length
                    .set(&value_length.equal(0), None)?
                    .into_series()
                    .cast(&DataType::Float64)?;

                sum.map(|sum| std::ops::Div::div(&sum, &value_length))
                    .transpose()
            },
        }
    }

    /// Pipe different functions/ closure operations that work on a DataFrame together.
    pub fn pipe<F, B>(self, f: F) -> PolarsResult<B>
    where
        F: Fn(DataFrame) -> PolarsResult<B>,
    {
        f(self)
    }

    /// Pipe different functions/ closure operations that work on a DataFrame together.
    pub fn pipe_mut<F, B>(&mut self, f: F) -> PolarsResult<B>
    where
        F: Fn(&mut DataFrame) -> PolarsResult<B>,
    {
        f(self)
    }

    /// Pipe different functions/ closure operations that work on a DataFrame together.
    pub fn pipe_with_args<F, B, Args>(self, f: F, args: Args) -> PolarsResult<B>
    where
        F: Fn(DataFrame, Args) -> PolarsResult<B>,
    {
        f(self, args)
    }

    /// Drop duplicate rows from a [`DataFrame`].
    /// *This fails when there is a column of type List in DataFrame*
    ///
    /// Stable means that the order is maintained. This has a higher cost than an unstable distinct.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use polars_core::prelude::*;
    /// let df = df! {
    ///               "flt" => [1., 1., 2., 2., 3., 3.],
    ///               "int" => [1, 1, 2, 2, 3, 3, ],
    ///               "str" => ["a", "a", "b", "b", "c", "c"]
    ///           }?;
    ///
    /// println!("{}", df.unique_stable(None, UniqueKeepStrategy::First, None)?);
    /// # Ok::<(), PolarsError>(())
    /// ```
    /// Returns
    ///
    /// ```text
    /// +-----+-----+-----+
    /// | flt | int | str |
    /// | --- | --- | --- |
    /// | f64 | i32 | str |
    /// +=====+=====+=====+
    /// | 1   | 1   | "a" |
    /// +-----+-----+-----+
    /// | 2   | 2   | "b" |
    /// +-----+-----+-----+
    /// | 3   | 3   | "c" |
    /// +-----+-----+-----+
    /// ```
    #[cfg(feature = "algorithm_group_by")]
    pub fn unique_stable(
        &self,
        subset: Option<&[String]>,
        keep: UniqueKeepStrategy,
        slice: Option<(i64, usize)>,
    ) -> PolarsResult<DataFrame> {
        self.unique_impl(true, subset, keep, slice)
    }

    /// Unstable distinct. See [`DataFrame::unique_stable`].
    #[cfg(feature = "algorithm_group_by")]
    pub fn unique(
        &self,
        subset: Option<&[String]>,
        keep: UniqueKeepStrategy,
        slice: Option<(i64, usize)>,
    ) -> PolarsResult<DataFrame> {
        self.unique_impl(false, subset, keep, slice)
    }

    #[cfg(feature = "algorithm_group_by")]
    pub fn unique_impl(
        &self,
        maintain_order: bool,
        subset: Option<&[String]>,
        keep: UniqueKeepStrategy,
        slice: Option<(i64, usize)>,
    ) -> PolarsResult<Self> {
        let names = match &subset {
            Some(s) => s.iter().map(|s| &**s).collect(),
            None => self.get_column_names(),
        };
        let mut df = self.clone();
        // take on multiple chunks is terrible
        df.as_single_chunk_par();

        let columns = match (keep, maintain_order) {
            (UniqueKeepStrategy::First | UniqueKeepStrategy::Any, true) => {
                let gb = df.group_by_stable(names)?;
                let groups = gb.get_groups();
                let (offset, len) = slice.unwrap_or((0, groups.len()));
                let groups = groups.slice(offset, len);
                df._apply_columns_par(&|s| unsafe { s.agg_first(&groups) })
            },
            (UniqueKeepStrategy::Last, true) => {
                // maintain order by last values, so the sorted groups are not correct as they
                // are sorted by the first value
                let gb = df.group_by(names)?;
                let groups = gb.get_groups();

                let func = |g: GroupsIndicator| match g {
                    GroupsIndicator::Idx((_first, idx)) => idx[idx.len() - 1],
                    GroupsIndicator::Slice([first, len]) => first + len - 1,
                };

                let last_idx: NoNull<IdxCa> = match slice {
                    None => groups.iter().map(func).collect(),
                    Some((offset, len)) => {
                        let (offset, len) = slice_offsets(offset, len, groups.len());
                        groups.iter().skip(offset).take(len).map(func).collect()
                    },
                };

                let last_idx = last_idx.sort(false);
                return Ok(unsafe { df.take_unchecked(&last_idx) });
            },
            (UniqueKeepStrategy::First | UniqueKeepStrategy::Any, false) => {
                let gb = df.group_by(names)?;
                let groups = gb.get_groups();
                let (offset, len) = slice.unwrap_or((0, groups.len()));
                let groups = groups.slice(offset, len);
                df._apply_columns_par(&|s| unsafe { s.agg_first(&groups) })
            },
            (UniqueKeepStrategy::Last, false) => {
                let gb = df.group_by(names)?;
                let groups = gb.get_groups();
                let (offset, len) = slice.unwrap_or((0, groups.len()));
                let groups = groups.slice(offset, len);
                df._apply_columns_par(&|s| unsafe { s.agg_last(&groups) })
            },
            (UniqueKeepStrategy::None, _) => {
                let df_part = df.select(names)?;
                let mask = df_part.is_unique()?;
                let mask = match slice {
                    None => mask,
                    Some((offset, len)) => mask.slice(offset, len),
                };
                return df.filter(&mask);
            },
        };
        Ok(unsafe { DataFrame::new_no_checks(columns) })
    }

    /// Get a mask of all the unique rows in the [`DataFrame`].
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use polars_core::prelude::*;
    /// let df: DataFrame = df!("Company" => &["Apple", "Microsoft"],
    ///                         "ISIN" => &["US0378331005", "US5949181045"])?;
    /// let ca: ChunkedArray<BooleanType> = df.is_unique()?;
    ///
    /// assert!(ca.all());
    /// # Ok::<(), PolarsError>(())
    /// ```
    #[cfg(feature = "algorithm_group_by")]
    pub fn is_unique(&self) -> PolarsResult<BooleanChunked> {
        let gb = self.group_by(self.get_column_names())?;
        let groups = gb.take_groups();
        Ok(is_unique_helper(
            groups,
            self.height() as IdxSize,
            true,
            false,
        ))
    }

    /// Get a mask of all the duplicated rows in the [`DataFrame`].
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use polars_core::prelude::*;
    /// let df: DataFrame = df!("Company" => &["Alphabet", "Alphabet"],
    ///                         "ISIN" => &["US02079K3059", "US02079K1079"])?;
    /// let ca: ChunkedArray<BooleanType> = df.is_duplicated()?;
    ///
    /// assert!(!ca.all());
    /// # Ok::<(), PolarsError>(())
    /// ```
    #[cfg(feature = "algorithm_group_by")]
    pub fn is_duplicated(&self) -> PolarsResult<BooleanChunked> {
        let gb = self.group_by(self.get_column_names())?;
        let groups = gb.take_groups();
        Ok(is_unique_helper(
            groups,
            self.height() as IdxSize,
            false,
            true,
        ))
    }

    /// Create a new [`DataFrame`] that shows the null counts per column.
    #[must_use]
    pub fn null_count(&self) -> Self {
        let cols = self
            .columns
            .iter()
            .map(|s| Series::new(s.name(), &[s.null_count() as IdxSize]))
            .collect();
        unsafe { Self::new_no_checks(cols) }
    }

    /// Hash and combine the row values
    #[cfg(feature = "row_hash")]
    pub fn hash_rows(
        &mut self,
        hasher_builder: Option<PlRandomState>,
    ) -> PolarsResult<UInt64Chunked> {
        let dfs = split_df(self, POOL.current_num_threads(), false);
        let (cas, _) = _df_rows_to_hashes_threaded_vertical(&dfs, hasher_builder)?;

        let mut iter = cas.into_iter();
        let mut acc_ca = iter.next().unwrap();
        for ca in iter {
            acc_ca.append(&ca)?;
        }
        Ok(acc_ca.rechunk())
    }

    /// Get the supertype of the columns in this DataFrame
    pub fn get_supertype(&self) -> Option<PolarsResult<DataType>> {
        self.columns
            .iter()
            .map(|s| Ok(s.dtype().clone()))
            .reduce(|acc, b| try_get_supertype(&acc?, &b.unwrap()))
    }

    /// Take by index values given by the slice `idx`.
    /// # Warning
    /// Be careful with allowing threads when calling this in a large hot loop
    /// every thread split may be on rayon stack and lead to SO
    #[doc(hidden)]
    pub unsafe fn _take_unchecked_slice(&self, idx: &[IdxSize], allow_threads: bool) -> Self {
        self._take_unchecked_slice_sorted(idx, allow_threads, IsSorted::Not)
    }

    /// Take by index values given by the slice `idx`. Use this over `_take_unchecked_slice`
    /// if the index value in `idx` are sorted. This will maintain sorted flags.
    ///
    /// # Warning
    /// Be careful with allowing threads when calling this in a large hot loop
    /// every thread split may be on rayon stack and lead to SO
    #[doc(hidden)]
    pub unsafe fn _take_unchecked_slice_sorted(
        &self,
        idx: &[IdxSize],
        allow_threads: bool,
        sorted: IsSorted,
    ) -> Self {
        #[cfg(debug_assertions)]
        {
            if idx.len() > 2 {
                match sorted {
                    IsSorted::Ascending => {
                        assert!(idx[0] <= idx[idx.len() - 1]);
                    },
                    IsSorted::Descending => {
                        assert!(idx[0] >= idx[idx.len() - 1]);
                    },
                    _ => {},
                }
            }
        }
        let mut ca = IdxCa::mmap_slice("", idx);
        ca.set_sorted_flag(sorted);
        self.take_unchecked_impl(&ca, allow_threads)
    }

    #[cfg(all(feature = "partition_by", feature = "algorithm_group_by"))]
    #[doc(hidden)]
    pub fn _partition_by_impl(
        &self,
        cols: &[String],
        stable: bool,
        include_key: bool,
    ) -> PolarsResult<Vec<DataFrame>> {
        let groups = if stable {
            self.group_by_stable(cols)?.take_groups()
        } else {
            self.group_by(cols)?.take_groups()
        };

        // drop key columns prior to calculation if requested
        let df = if include_key {
            self.clone()
        } else {
            self.drop_many(cols)
        };

        // don't parallelize this
        // there is a lot of parallelization in take and this may easily SO
        POOL.install(|| {
            match groups {
                GroupsProxy::Idx(idx) => {
                    // Rechunk as the gather may rechunk for every group #17562.
                    let mut df = df.clone();
                    df.as_single_chunk_par();
                    Ok(idx
                        .into_par_iter()
                        .map(|(_, group)| {
                            // groups are in bounds
                            unsafe {
                                df._take_unchecked_slice_sorted(&group, false, IsSorted::Ascending)
                            }
                        })
                        .collect())
                },
                GroupsProxy::Slice { groups, .. } => Ok(groups
                    .into_par_iter()
                    .map(|[first, len]| df.slice(first as i64, len as usize))
                    .collect()),
            }
        })
    }

    /// Split into multiple DataFrames partitioned by groups
    #[cfg(feature = "partition_by")]
    pub fn partition_by(
        &self,
        cols: impl IntoVec<String>,
        include_key: bool,
    ) -> PolarsResult<Vec<DataFrame>> {
        let cols = cols.into_vec();
        self._partition_by_impl(&cols, false, include_key)
    }

    /// Split into multiple DataFrames partitioned by groups
    /// Order of the groups are maintained.
    #[cfg(feature = "partition_by")]
    pub fn partition_by_stable(
        &self,
        cols: impl IntoVec<String>,
        include_key: bool,
    ) -> PolarsResult<Vec<DataFrame>> {
        let cols = cols.into_vec();
        self._partition_by_impl(&cols, true, include_key)
    }

    /// Unnest the given `Struct` columns. This means that the fields of the `Struct` type will be
    /// inserted as columns.
    #[cfg(feature = "dtype-struct")]
    pub fn unnest<I: IntoVec<String>>(&self, cols: I) -> PolarsResult<DataFrame> {
        let cols = cols.into_vec();
        self.unnest_impl(cols.into_iter().collect())
    }

    #[cfg(feature = "dtype-struct")]
    fn unnest_impl(&self, cols: PlHashSet<String>) -> PolarsResult<DataFrame> {
        let mut new_cols = Vec::with_capacity(std::cmp::min(self.width() * 2, self.width() + 128));
        let mut count = 0;
        for s in &self.columns {
            if cols.contains(s.name()) {
                let ca = s.struct_()?.clone();
                new_cols.extend_from_slice(&ca.fields_as_series());
                count += 1;
            } else {
                new_cols.push(s.clone())
            }
        }
        if count != cols.len() {
            // one or more columns not found
            // the code below will return an error with the missing name
            let schema = self.schema();
            for col in cols {
                let _ = schema
                    .get(&col)
                    .ok_or_else(|| polars_err!(col_not_found = col))?;
            }
        }
        DataFrame::new(new_cols)
    }
}

pub struct RecordBatchIter<'a> {
    columns: &'a Vec<Series>,
    idx: usize,
    n_chunks: usize,
    compat_level: CompatLevel,
    parallel: bool,
}

impl<'a> Iterator for RecordBatchIter<'a> {
    type Item = RecordBatch;

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx >= self.n_chunks {
            None
        } else {
            // Create a batch of the columns with the same chunk no.
            let batch_cols = if self.parallel {
                let iter = self
                    .columns
                    .par_iter()
                    .map(|s| s.to_arrow(self.idx, self.compat_level));
                POOL.install(|| iter.collect())
            } else {
                self.columns
                    .iter()
                    .map(|s| s.to_arrow(self.idx, self.compat_level))
                    .collect()
            };
            self.idx += 1;

            Some(RecordBatch::new(batch_cols))
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.n_chunks - self.idx;
        (n, Some(n))
    }
}

pub struct PhysRecordBatchIter<'a> {
    iters: Vec<std::slice::Iter<'a, ArrayRef>>,
}

impl Iterator for PhysRecordBatchIter<'_> {
    type Item = RecordBatch;

    fn next(&mut self) -> Option<Self::Item> {
        self.iters
            .iter_mut()
            .map(|phys_iter| phys_iter.next().cloned())
            .collect::<Option<Vec<_>>>()
            .map(RecordBatch::new)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        if let Some(iter) = self.iters.first() {
            iter.size_hint()
        } else {
            (0, None)
        }
    }
}

impl Default for DataFrame {
    fn default() -> Self {
        DataFrame::empty()
    }
}

impl From<DataFrame> for Vec<Series> {
    fn from(df: DataFrame) -> Self {
        df.columns
    }
}

// utility to test if we can vstack/extend the columns
fn ensure_can_extend(left: &Series, right: &Series) -> PolarsResult<()> {
    polars_ensure!(
        left.name() == right.name(),
        ShapeMismatch: "unable to vstack, column names don't match: {:?} and {:?}",
        left.name(), right.name(),
    );
    Ok(())
}

#[cfg(test)]
mod test {
    use super::*;

    fn create_frame() -> DataFrame {
        let s0 = Series::new("days", [0, 1, 2].as_ref());
        let s1 = Series::new("temp", [22.1, 19.9, 7.].as_ref());
        DataFrame::new(vec![s0, s1]).unwrap()
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_recordbatch_iterator() {
        let df = df!(
            "foo" => &[1, 2, 3, 4, 5]
        )
        .unwrap();
        let mut iter = df.iter_chunks(CompatLevel::newest(), false);
        assert_eq!(5, iter.next().unwrap().len());
        assert!(iter.next().is_none());
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_select() {
        let df = create_frame();
        assert_eq!(df.column("days").unwrap().equal(1).unwrap().sum(), Some(1));
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_filter_broadcast_on_string_col() {
        let col_name = "some_col";
        let v = vec!["test".to_string()];
        let s0 = Series::new(col_name, v);
        let mut df = DataFrame::new(vec![s0]).unwrap();

        df = df
            .filter(&df.column(col_name).unwrap().equal("").unwrap())
            .unwrap();
        assert_eq!(df.column(col_name).unwrap().n_chunks(), 1);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_filter_broadcast_on_list_col() {
        let s1 = Series::new("", &[true, false, true]);
        let ll: ListChunked = [&s1].iter().copied().collect();

        let mask = BooleanChunked::from_slice("", &[false]);
        let new = ll.filter(&mask).unwrap();

        assert_eq!(new.chunks.len(), 1);
        assert_eq!(new.len(), 0);
    }

    #[test]
    fn slice() {
        let df = create_frame();
        let sliced_df = df.slice(0, 2);
        assert_eq!(sliced_df.shape(), (2, 2));
    }

    #[test]
    fn rechunk_false() {
        let df = create_frame();
        assert!(!df.should_rechunk())
    }

    #[test]
    fn rechunk_true() -> PolarsResult<()> {
        let mut base = df!(
            "a" => [1, 2, 3],
            "b" => [1, 2, 3]
        )?;

        // Create a series with multiple chunks
        let mut s = Series::new("foo", 0..2);
        let s2 = Series::new("bar", 0..1);
        s.append(&s2)?;

        // Append series to frame
        let out = base.with_column(s)?;

        // Now we should rechunk
        assert!(out.should_rechunk());
        Ok(())
    }

    #[test]
    fn test_duplicate_column() {
        let mut df = df! {
            "foo" => &[1, 2, 3]
        }
        .unwrap();
        // check if column is replaced
        assert!(df.with_column(Series::new("foo", &[1, 2, 3])).is_ok());
        assert!(df.with_column(Series::new("bar", &[1, 2, 3])).is_ok());
        assert!(df.column("bar").is_ok())
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn distinct() {
        let df = df! {
            "flt" => [1., 1., 2., 2., 3., 3.],
            "int" => [1, 1, 2, 2, 3, 3, ],
            "str" => ["a", "a", "b", "b", "c", "c"]
        }
        .unwrap();
        let df = df
            .unique_stable(None, UniqueKeepStrategy::First, None)
            .unwrap()
            .sort(["flt"], SortMultipleOptions::default())
            .unwrap();
        let valid = df! {
            "flt" => [1., 2., 3.],
            "int" => [1, 2, 3],
            "str" => ["a", "b", "c"]
        }
        .unwrap();
        assert!(df.equals(&valid));
    }

    #[test]
    fn test_vstack() {
        // check that it does not accidentally rechunks
        let mut df = df! {
            "flt" => [1., 1., 2., 2., 3., 3.],
            "int" => [1, 1, 2, 2, 3, 3, ],
            "str" => ["a", "a", "b", "b", "c", "c"]
        }
        .unwrap();

        df.vstack_mut(&df.slice(0, 3)).unwrap();
        assert_eq!(df.n_chunks(), 2)
    }

    #[test]
    #[cfg(feature = "zip_with")]
    #[cfg_attr(miri, ignore)]
    fn test_horizontal_agg() {
        let a = Series::new("a", &[1, 2, 6]);
        let b = Series::new("b", &[Some(1), None, None]);
        let c = Series::new("c", &[Some(4), None, Some(3)]);

        let df = DataFrame::new(vec![a, b, c]).unwrap();
        assert_eq!(
            Vec::from(
                df.mean_horizontal(NullStrategy::Ignore)
                    .unwrap()
                    .unwrap()
                    .f64()
                    .unwrap()
            ),
            &[Some(2.0), Some(2.0), Some(4.5)]
        );
        assert_eq!(
            Vec::from(
                df.sum_horizontal(NullStrategy::Ignore)
                    .unwrap()
                    .unwrap()
                    .i32()
                    .unwrap()
            ),
            &[Some(6), Some(2), Some(9)]
        );
        assert_eq!(
            Vec::from(df.min_horizontal().unwrap().unwrap().i32().unwrap()),
            &[Some(1), Some(2), Some(3)]
        );
        assert_eq!(
            Vec::from(df.max_horizontal().unwrap().unwrap().i32().unwrap()),
            &[Some(4), Some(2), Some(6)]
        );
    }

    #[test]
    fn test_replace_or_add() -> PolarsResult<()> {
        let mut df = df!(
            "a" => [1, 2, 3],
            "b" => [1, 2, 3]
        )?;

        // check that the new column is "c" and not "bar".
        df.replace_or_add("c", Series::new("bar", [1, 2, 3]))?;

        assert_eq!(df.get_column_names(), &["a", "b", "c"]);
        Ok(())
    }

    #[test]
    fn test_empty_df_hstack() -> PolarsResult<()> {
        let mut base = df!(
            "a" => [1, 2, 3],
            "b" => [1, 2, 3]
        )?;

        // has got columns, but no rows
        let mut df = base.clear();
        let out = df.with_column(Series::new("c", [1]))?;
        assert_eq!(out.shape(), (0, 3));
        assert!(out.iter().all(|s| s.len() == 0));

        // no columns
        base.columns = vec![];
        let out = base.with_column(Series::new("c", [1]))?;
        assert_eq!(out.shape(), (1, 1));

        Ok(())
    }
}
