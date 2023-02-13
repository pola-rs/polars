//! DataFrame module.
use std::borrow::Cow;
use std::iter::{FromIterator, Iterator};
use std::{mem, ops};

use ahash::AHashSet;
use polars_arrow::prelude::QuantileInterpolOptions;
use rayon::prelude::*;

use crate::chunked_array::ops::unique::is_unique_helper;
use crate::prelude::*;
#[cfg(feature = "describe")]
use crate::utils::concat_df_unchecked;
use crate::utils::{split_ca, split_df, try_get_supertype, NoNull};

#[cfg(feature = "dataframe_arithmetic")]
mod arithmetic;
#[cfg(feature = "asof_join")]
pub(crate) mod asof_join;
mod chunks;
#[cfg(feature = "cross_join")]
pub(crate) mod cross_join;
pub mod explode;
mod from;
pub mod groupby;
pub mod hash_join;
#[cfg(feature = "rows")]
pub mod row;
mod upstream_traits;

pub use chunks::*;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::frame::groupby::GroupsIndicator;
#[cfg(feature = "sort_multiple")]
use crate::prelude::sort::prepare_arg_sort;
use crate::series::IsSorted;
#[cfg(feature = "row_hash")]
use crate::vector_hasher::df_rows_to_hashes_threaded;
use crate::POOL;

#[derive(Copy, Clone, Debug)]
pub enum NullStrategy {
    Ignore,
    Propagate,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum UniqueKeepStrategy {
    First,
    Last,
    None,
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
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DataFrame {
    pub(crate) columns: Vec<Series>,
}

pub fn _duplicate_err(name: &str) -> PolarsResult<()> {
    Err(PolarsError::Duplicate(
        format!("Column with name: '{name}' has more than one occurrences").into(),
    ))
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

    // reduce monomorphization
    fn apply_columns(&self, func: &(dyn Fn(&Series) -> Series)) -> Vec<Series> {
        self.columns.iter().map(|s| func(s)).collect()
    }

    // reduce monomorphization
    fn apply_columns_par(&self, func: &(dyn Fn(&Series) -> Series + Send + Sync)) -> Vec<Series> {
        POOL.install(|| self.columns.par_iter().map(|s| func(s)).collect())
    }

    // reduce monomorphization
    fn try_apply_columns_par(
        &self,
        func: &(dyn Fn(&Series) -> PolarsResult<Series> + Send + Sync),
    ) -> PolarsResult<Vec<Series>> {
        POOL.install(|| self.columns.par_iter().map(|s| func(s)).collect())
    }

    // reduce monomorphization
    fn try_apply_columns(
        &self,
        func: &(dyn Fn(&Series) -> PolarsResult<Series> + Send + Sync),
    ) -> PolarsResult<Vec<Series>> {
        self.columns.iter().map(|s| func(s)).collect()
    }

    /// Get the index of the column.
    fn check_name_to_idx(&self, name: &str) -> PolarsResult<usize> {
        self.find_idx_by_name(name)
            .ok_or_else(|| PolarsError::ColumnNotFound(name.to_string().into()))
    }

    fn check_already_present(&self, name: &str) -> PolarsResult<()> {
        if self.columns.iter().any(|s| s.name() == name) {
            Err(PolarsError::Duplicate(
                format!("column with name: '{name}' already present in DataFrame").into(),
            ))
        } else {
            Ok(())
        }
    }

    /// Reserve additional slots into the chunks of the series.
    pub(crate) fn reserve_chunks(&mut self, additional: usize) {
        for s in &mut self.columns {
            // Safety
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
    pub fn new<S: IntoSeries>(columns: Vec<S>) -> PolarsResult<Self> {
        let mut first_len = None;

        let shape_err = |&first_name, &first_len, &name, &len| {
            let msg = format!(
                "Could not create a new DataFrame from Series. The Series have different lengths: \
                found length {first_len:?} for Series named {first_name:?} and length {len:?} for Series named {name:?}."
            );
            Err(PolarsError::ShapeMisMatch(msg.into()))
        };

        let series_cols = if S::is_series() {
            // Safety:
            // we are guarded by the type system here.
            #[allow(clippy::transmute_undefined_repr)]
            let series_cols = unsafe { std::mem::transmute::<Vec<S>, Vec<Series>>(columns) };
            let mut names = PlHashSet::with_capacity(series_cols.len());

            for s in &series_cols {
                let name = s.name();

                match first_len {
                    Some(len) => {
                        if s.len() != len {
                            let first_series = &series_cols.first().unwrap();
                            return shape_err(
                                &first_series.name(),
                                &first_series.len(),
                                &name,
                                &s.len(),
                            );
                        }
                    }
                    None => first_len = Some(s.len()),
                }

                if names.contains(name) {
                    _duplicate_err(name)?
                }

                names.insert(name);
            }
            // we drop early as the brchk thinks the &str borrows are used when calling the drop
            // of both `series_cols` and `names`
            drop(names);
            series_cols
        } else {
            let mut series_cols: Vec<Series> = Vec::with_capacity(columns.len());
            let mut names = PlHashSet::with_capacity(columns.len());

            // check for series length equality and convert into series in one pass
            for s in columns {
                let series = s.into_series();
                // we have aliasing borrows so we must allocate a string
                let name = series.name().to_string();

                match first_len {
                    Some(len) => {
                        if series.len() != len {
                            let first_series = &series_cols.first().unwrap();
                            return shape_err(
                                &first_series.name(),
                                &first_series.len(),
                                &name.as_str(),
                                &series.len(),
                            );
                        }
                    }
                    None => first_len = Some(series.len()),
                }

                if names.contains(&name) {
                    _duplicate_err(&name)?
                }

                series_cols.push(series);
                names.insert(name);
            }
            drop(names);
            series_cols
        };

        Ok(DataFrame {
            columns: series_cols,
        })
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
        DataFrame::new_no_checks(Vec::new())
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
    /// let df2: DataFrame = df1.with_row_count("Id", None)?;
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
    pub fn with_row_count(&self, name: &str, offset: Option<IdxSize>) -> PolarsResult<Self> {
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

    /// Add a row count in place.
    pub fn with_row_count_mut(&mut self, name: &str, offset: Option<IdxSize>) -> &mut Self {
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
    /// It is advised to use [Series::new](Series::new) in favor of this method.
    ///
    /// # Panic
    /// It is the callers responsibility to uphold the contract of all `Series`
    /// having an equal length, if not this may panic down the line.
    pub const fn new_no_checks(columns: Vec<Series>) -> DataFrame {
        DataFrame { columns }
    }

    /// Aggregate all chunks to contiguous memory.
    #[must_use]
    pub fn agg_chunks(&self) -> Self {
        // Don't parallelize this. Memory overhead
        let f = |s: &Series| s.rechunk();
        let cols = self.columns.iter().map(f).collect();
        DataFrame::new_no_checks(cols)
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
        if self.columns.iter().any(|s| s.n_chunks() > 1) {
            self.columns = self.apply_columns_par(&|s| s.rechunk());
        }
        self
    }

    /// Returns true if the chunks of the columns do not align and re-chunking should be done
    pub fn should_rechunk(&self) -> bool {
        let mut chunk_lenghts = self.columns.iter().map(|s| s.chunk_lengths());
        match chunk_lenghts.next() {
            None => false,
            Some(first_column_chunk_lengths) => {
                // Fast Path for single Chunk Series
                if first_column_chunk_lengths.len() == 1 {
                    return chunk_lenghts.any(|cl| cl.len() != 1);
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
                for cl in chunk_lenghts {
                    if cl.enumerate().any(|(idx, el)| Some(&el) != v.get(idx)) {
                        return true;
                    }
                }
                false
            }
        }
    }

    /// Ensure all the chunks in the DataFrame are aligned.
    pub fn rechunk(&mut self) -> &mut Self {
        if self.should_rechunk() {
            self.as_single_chunk_par()
        } else {
            self
        }
    }

    /// Get the `DataFrame` schema.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let df: DataFrame = df!("Thing" => &["Observable universe", "Human stupidity"],
    ///                         "Diameter (m)" => &[8.8e26, f64::INFINITY])?;
    ///
    /// let f1: Field = Field::new("Thing", DataType::Utf8);
    /// let f2: Field = Field::new("Diameter (m)", DataType::Float64);
    /// let sc: Schema = Schema::from(vec![f1, f2].into_iter());
    ///
    /// assert_eq!(df.schema(), sc);
    /// # Ok::<(), PolarsError>(())
    /// ```
    pub fn schema(&self) -> Schema {
        Schema::from(self.iter().map(|s| s.field().into_owned()))
    }

    /// Get a reference to the `DataFrame` columns.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let df: DataFrame = df!("Name" => &["Adenine", "Cytosine", "Guanine", "Thymine"],
    ///                         "Symbol" => &["A", "C", "G", "T"])?;
    /// let columns: &Vec<Series> = df.get_columns();
    ///
    /// assert_eq!(columns[0].name(), "Name");
    /// assert_eq!(columns[1].name(), "Symbol");
    /// # Ok::<(), PolarsError>(())
    /// ```
    #[inline]
    pub fn get_columns(&self) -> &Vec<Series> {
        &self.columns
    }

    #[cfg(feature = "private")]
    #[inline]
    pub fn get_columns_mut(&mut self) -> &mut Vec<Series> {
        &mut self.columns
    }

    /// Iterator over the columns as `Series`.
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

    /// Get the `Vec<String>` representing the column names.
    pub fn get_column_names_owned(&self) -> Vec<String> {
        self.columns.iter().map(|s| s.name().to_string()).collect()
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
        if names.len() != self.columns.len() {
            return Err(PolarsError::ShapeMisMatch("the provided slice with column names has not the same size as the DataFrame's width".into()));
        }
        let unique_names: AHashSet<&str, ahash::RandomState> =
            AHashSet::from_iter(names.iter().map(|name| name.as_ref()));
        if unique_names.len() != self.columns.len() {
            return Err(PolarsError::SchemaMisMatch(
                "duplicate column names found".into(),
            ));
        }

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

    /// Get the data types of the columns in the DataFrame.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let venus_air: DataFrame = df!("Element" => &["Carbon dioxide", "Nitrogen"],
    ///                                "Fraction" => &[0.965, 0.035])?;
    ///
    /// assert_eq!(venus_air.dtypes(), &[DataType::Utf8, DataType::Float64]);
    /// # Ok::<(), PolarsError>(())
    /// ```
    pub fn dtypes(&self) -> Vec<DataType> {
        self.columns.iter().map(|s| s.dtype().clone()).collect()
    }

    /// The number of chunks per column
    pub fn n_chunks(&self) -> usize {
        match self.columns.get(0) {
            None => 0,
            Some(s) => s.n_chunks(),
        }
    }

    /// Get a reference to the schema fields of the `DataFrame`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let earth: DataFrame = df!("Surface type" => &["Water", "Land"],
    ///                            "Fraction" => &[0.708, 0.292])?;
    ///
    /// let f1: Field = Field::new("Surface type", DataType::Utf8);
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

    /// Get (height, width) of the `DataFrame`.
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

    /// Get the width of the `DataFrame` which is the number of columns.
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

    /// Get the height of the `DataFrame` which is the number of rows.
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

    /// Check if the `DataFrame` is empty.
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
        self.columns.is_empty()
    }

    pub(crate) fn hstack_mut_no_checks(&mut self, columns: &[Series]) -> &mut Self {
        for col in columns {
            self.columns.push(col.clone());
        }
        self
    }

    /// Add multiple `Series` to a `DataFrame`.
    /// The added `Series` are required to have the same length.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// fn stack(df: &mut DataFrame, columns: &[Series]) {
    ///     df.hstack_mut(columns);
    /// }
    /// ```
    pub fn hstack_mut(&mut self, columns: &[Series]) -> PolarsResult<&mut Self> {
        let mut names = PlHashSet::with_capacity(self.columns.len());
        for s in &self.columns {
            names.insert(s.name());
        }

        let height = self.height();
        // first loop check validity. We don't do this in a single pass otherwise
        // this DataFrame is already modified when an error occurs.
        for col in columns {
            if col.len() != height && height != 0 {
                return Err(PolarsError::ShapeMisMatch(
                    format!("Could not horizontally stack Series. The Series length {} differs from the DataFrame height: {height}", col.len()).into()));
            }

            let name = col.name();
            if names.contains(name) {
                return Err(PolarsError::Duplicate(
                    format!("Cannot do hstack operation. Column with name: {name} already exists",)
                        .into(),
                ));
            }
            names.insert(name);
        }
        drop(names);
        Ok(self.hstack_mut_no_checks(columns))
    }

    /// Add multiple `Series` to a `DataFrame`.
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

    /// Concatenate a `DataFrame` to this `DataFrame` and return as newly allocated `DataFrame`.
    ///
    /// If many `vstack` operations are done, it is recommended to call [`DataFrame::rechunk`].
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

    /// Concatenate a DataFrame to this DataFrame
    ///
    /// If many `vstack` operations are done, it is recommended to call [`DataFrame::rechunk`].
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
            if self.width() == 0 {
                self.columns = other.columns.clone();
                return Ok(self);
            }

            return Err(PolarsError::ShapeMisMatch(
                format!("Could not vertically stack DataFrame. The DataFrames appended width {} differs from the parent DataFrames width {}", self.width(), other.width()).into()
            ));
        }

        self.columns
            .iter_mut()
            .zip(other.columns.iter())
            .try_for_each::<_, PolarsResult<_>>(|(left, right)| {
                can_extend(left, right)?;
                left.append(right).expect("should not fail");
                Ok(())
            })?;
        Ok(self)
    }

    /// Does not check if schema is correct
    pub(crate) fn vstack_mut_unchecked(&mut self, other: &DataFrame) {
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
    /// of `append` operations with a [`rechunk`](Self::rechunk).
    pub fn extend(&mut self, other: &DataFrame) -> PolarsResult<()> {
        if self.width() != other.width() {
            return Err(PolarsError::ShapeMisMatch(
                format!("Could not extend DataFrame. The DataFrames extended width {} differs from the parent DataFrames width {}", self.width(), other.width()).into()
            ));
        }

        self.columns
            .iter_mut()
            .zip(other.columns.iter())
            .try_for_each::<_, PolarsResult<_>>(|(left, right)| {
                can_extend(left, right)?;
                left.extend(right).unwrap();
                Ok(())
            })?;
        Ok(())
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

    /// Return a new `DataFrame` where all null values are dropped.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use polars_core::prelude::*;
    /// let df1: DataFrame = df!("Country" => ["Malta", "Liechtenstein", "North Korea"],
    ///                         "Tax revenue (% GDP)" => [Some(32.7), None, None])?;
    /// assert_eq!(df1.shape(), (3, 2));
    ///
    /// let df2: DataFrame = df1.drop_nulls(None)?;
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
    pub fn drop_nulls(&self, subset: Option<&[String]>) -> PolarsResult<Self> {
        let selected_series;

        let mut iter = match subset {
            Some(cols) => {
                selected_series = self.select_series(cols)?;
                selected_series.iter()
            }
            None => self.columns.iter(),
        };

        // fast path for no nulls in df
        if iter.clone().all(|s| !s.has_validity()) {
            return Ok(self.clone());
        }

        let mask = iter
            .next()
            .ok_or_else(|| PolarsError::NoData("No data to drop nulls from".into()))?;
        let mut mask = mask.is_not_null();

        for s in iter {
            mask = mask & s.is_not_null();
        }
        self.filter(&mask)
    }

    /// Drop a column by name.
    /// This is a pure method and will return a new `DataFrame` instead of modifying
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

        Ok(DataFrame::new_no_checks(new_cols))
    }

    pub fn drop_many<S: AsRef<str>>(&self, names: &[S]) -> Self {
        let names = names.iter().map(|s| s.as_ref()).collect();
        fn inner(df: &DataFrame, names: Vec<&str>) -> DataFrame {
            let mut new_cols = Vec::with_capacity(df.columns.len() - names.len());
            df.columns.iter().for_each(|s| {
                if !names.contains(&s.name()) {
                    new_cols.push(s.clone())
                }
            });

            DataFrame::new_no_checks(new_cols)
        }
        inner(self, names)
    }

    fn insert_at_idx_no_name_check(
        &mut self,
        index: usize,
        series: Series,
    ) -> PolarsResult<&mut Self> {
        if series.len() == self.height() {
            self.columns.insert(index, series);
            Ok(self)
        } else {
            Err(PolarsError::ShapeMisMatch(
                format!(
                    "Could not add column. The Series length {} differs from the DataFrame height: {}",
                    series.len(),
                    self.height()
                )
                .into(),
            ))
        }
    }

    /// Insert a new column at a given index.
    pub fn insert_at_idx<S: IntoSeries>(
        &mut self,
        index: usize,
        column: S,
    ) -> PolarsResult<&mut Self> {
        let series = column.into_series();
        self.check_already_present(series.name())?;
        self.insert_at_idx_no_name_check(index, series)
    }

    fn add_column_by_search(&mut self, series: Series) -> PolarsResult<()> {
        if let Some(idx) = self.find_idx_by_name(series.name()) {
            self.replace_at_idx(idx, series)?;
        } else {
            self.columns.push(series);
        }
        Ok(())
    }

    /// Add a new column to this `DataFrame` or replace an existing one.
    pub fn with_column<S: IntoSeries>(&mut self, column: S) -> PolarsResult<&mut Self> {
        fn inner(df: &mut DataFrame, mut series: Series) -> PolarsResult<&mut DataFrame> {
            let height = df.height();
            if series.len() == 1 && height > 1 {
                series = series.new_from_index(0, height);
            }

            if series.len() == height || df.is_empty() {
                df.add_column_by_search(series)?;
                Ok(df)
            }
            // special case for literals
            else if height == 0 && series.len() == 1 {
                let s = series.slice(0, 0);
                df.add_column_by_search(s)?;
                Ok(df)
            } else {
                Err(PolarsError::ShapeMisMatch(
                    format!(
                        "Could not add column. The Series length {} differs from the DataFrame height: {}",
                        series.len(),
                        df.height()
                    )
                        .into(),
                ))
            }
        }
        let series = column.into_series();
        inner(self, series)
    }

    fn add_column_by_schema(&mut self, s: Series, schema: &Schema) -> PolarsResult<()> {
        let name = s.name();
        if let Some((idx, _, _)) = schema.get_full(name) {
            // schema is incorrect fallback to search
            if self.columns.get(idx).map(|s| s.name()) != Some(name) {
                self.add_column_by_search(s)?;
            } else {
                self.replace_at_idx(idx, s)?;
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

    /// Add a new column to this `DataFrame` or replace an existing one.
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

        if series.len() == height || self.is_empty() {
            self.add_column_by_schema(series, schema)?;
            Ok(self)
        }
        // special case for literals
        else if height == 0 && series.len() == 1 {
            let s = series.slice(0, 0);
            self.add_column_by_schema(s, schema)?;
            Ok(self)
        } else {
            Err(PolarsError::ShapeMisMatch(
                format!(
                    "Could not add column. The Series length {} differs from the DataFrame height: {}",
                    series.len(),
                    self.height()
                )
                    .into(),
            ))
        }
    }

    /// Get a row in the `DataFrame`. Beware this is slow.
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
        match self.columns.get(0) {
            Some(s) => {
                if s.len() <= idx {
                    return None;
                }
            }
            None => return None,
        }
        // safety: we just checked bounds
        unsafe { Some(self.columns.iter().map(|s| s.get_unchecked(idx)).collect()) }
    }

    /// Select a `Series` by index.
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

    /// Select column(s) from this `DataFrame` by range and return a new DataFrame
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
    /// assert!(df.select(&["0", "1"])?.frame_equal(&df.select_by_range(0..=1)?));
    /// assert!(df.frame_equal(&df.select_by_range(..)?));
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

        self.select_impl(&colnames[range])
    }

    /// Get column index of a `Series` by name.
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let df: DataFrame = df!("Name" => &["Player 1", "Player 2", "Player 3"],
    ///                         "Health" => &[100, 200, 500],
    ///                         "Mana" => &[250, 100, 0],
    ///                         "Strength" => &[30, 150, 300])?;
    ///
    /// assert_eq!(df.find_idx_by_name("Name"), Some(0));
    /// assert_eq!(df.find_idx_by_name("Health"), Some(1));
    /// assert_eq!(df.find_idx_by_name("Mana"), Some(2));
    /// assert_eq!(df.find_idx_by_name("Strength"), Some(3));
    /// assert_eq!(df.find_idx_by_name("Haste"), None);
    /// # Ok::<(), PolarsError>(())
    /// ```
    pub fn find_idx_by_name(&self, name: &str) -> Option<usize> {
        self.columns.iter().position(|s| s.name() == name)
    }

    /// Get column index of a `Series` by name.
    pub fn try_find_idx_by_name(&self, name: &str) -> PolarsResult<usize> {
        self.find_idx_by_name(name)
            .ok_or_else(|| PolarsError::ColumnNotFound(name.to_string().into()))
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
        let idx = self
            .find_idx_by_name(name)
            .ok_or_else(|| PolarsError::ColumnNotFound(name.to_string().into()))?;
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

    /// Select column(s) from this `DataFrame` and return a new `DataFrame`.
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
            .map(|s| s.as_ref().to_string())
            .collect::<Vec<_>>();
        self.select_impl(&cols)
    }

    fn select_impl(&self, cols: &[String]) -> PolarsResult<Self> {
        self.select_check_duplicates(cols)?;
        let selected = self.select_series_impl(cols)?;
        Ok(DataFrame::new_no_checks(selected))
    }

    pub fn select_physical<I, S>(&self, selection: I) -> PolarsResult<Self>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let cols = selection
            .into_iter()
            .map(|s| s.as_ref().to_string())
            .collect::<Vec<_>>();
        self.select_physical_impl(&cols)
    }

    fn select_physical_impl(&self, cols: &[String]) -> PolarsResult<Self> {
        self.select_check_duplicates(cols)?;
        let selected = self.select_series_physical_impl(cols)?;
        Ok(DataFrame::new_no_checks(selected))
    }

    fn select_check_duplicates(&self, cols: &[String]) -> PolarsResult<()> {
        let mut names = PlHashSet::with_capacity(cols.len());
        for name in cols {
            if !names.insert(name.as_str()) {
                _duplicate_err(name)?
            }
        }
        Ok(())
    }

    /// Select column(s) from this `DataFrame` and return them into a `Vec`.
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
    pub fn select_series(&self, selection: impl IntoVec<String>) -> PolarsResult<Vec<Series>> {
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
    fn select_series_physical_impl(&self, cols: &[String]) -> PolarsResult<Vec<Series>> {
        let selected = if cols.len() > 1 && self.columns.len() > 10 {
            let name_to_idx = self._names_to_idx_map();
            cols.iter()
                .map(|name| {
                    let idx = *name_to_idx
                        .get(name.as_str())
                        .ok_or_else(|| PolarsError::ColumnNotFound(name.to_string().into()))?;
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
    fn select_series_impl(&self, cols: &[String]) -> PolarsResult<Vec<Series>> {
        let selected = if cols.len() > 1 && self.columns.len() > 10 {
            // we hash, because there are user that having millions of columns.
            // # https://github.com/pola-rs/polars/issues/1023
            let name_to_idx = self._names_to_idx_map();

            cols.iter()
                .map(|name| {
                    let idx = *name_to_idx
                        .get(name.as_str())
                        .ok_or_else(|| PolarsError::ColumnNotFound(name.to_string().into()))?;
                    Ok(self.select_at_idx(idx).unwrap().clone())
                })
                .collect::<PolarsResult<Vec<_>>>()?
        } else {
            cols.iter()
                .map(|c| self.column(c).map(|s| s.clone()))
                .collect::<PolarsResult<Vec<_>>>()?
        };

        Ok(selected)
    }

    /// Select a mutable series by name.
    /// *Note: the length of the Series should remain the same otherwise the DataFrame is invalid.*
    /// For this reason the method is not public
    fn select_mut(&mut self, name: &str) -> Option<&mut Series> {
        let opt_idx = self.find_idx_by_name(name);

        opt_idx.and_then(|idx| self.select_at_idx_mut(idx))
    }

    /// Does a filter but splits thread chunks vertically instead of horizontally
    /// This yields a DataFrame with `n_chunks == n_threads`.
    fn filter_vertical(&mut self, mask: &BooleanChunked) -> PolarsResult<Self> {
        let n_threads = POOL.current_num_threads();

        let masks = split_ca(mask, n_threads).unwrap();
        let dfs = split_df(self, n_threads).unwrap();
        let dfs: PolarsResult<Vec<_>> = POOL.install(|| {
            masks
                .par_iter()
                .zip(dfs)
                .map(|(mask, df)| {
                    let cols = df
                        .columns
                        .iter()
                        .map(|s| s.filter(mask))
                        .collect::<PolarsResult<_>>()?;
                    Ok(DataFrame::new_no_checks(cols))
                })
                .collect()
        });

        let mut iter = dfs?.into_iter();
        let first = iter.next().unwrap();
        Ok(iter.fold(first, |mut acc, df| {
            acc.vstack_mut(&df).unwrap();
            acc
        }))
    }

    /// Take the `DataFrame` rows by a boolean mask.
    ///
    /// # Example
    ///
    /// ```
    /// # use polars_core::prelude::*;
    /// fn example(df: &DataFrame) -> PolarsResult<DataFrame> {
    ///     let mask = df.column("sepal.width")?.is_not_null();
    ///     df.filter(&mask)
    /// }
    /// ```
    pub fn filter(&self, mask: &BooleanChunked) -> PolarsResult<Self> {
        if std::env::var("POLARS_VERT_PAR").is_ok() {
            return self.clone().filter_vertical(mask);
        }
        let new_col = self.try_apply_columns_par(&|s| match s.dtype() {
            DataType::Utf8 => s.filter_threaded(mask, true),
            _ => s.filter(mask),
        })?;
        Ok(DataFrame::new_no_checks(new_col))
    }

    /// Same as `filter` but does not parallelize.
    pub fn _filter_seq(&self, mask: &BooleanChunked) -> PolarsResult<Self> {
        let new_col = self.try_apply_columns(&|s| s.filter(mask))?;
        Ok(DataFrame::new_no_checks(new_col))
    }

    /// Take `DataFrame` value by indexes from an iterator.
    ///
    /// # Example
    ///
    /// ```
    /// # use polars_core::prelude::*;
    /// fn example(df: &DataFrame) -> PolarsResult<DataFrame> {
    ///     let iterator = (0..9).into_iter();
    ///     df.take_iter(iterator)
    /// }
    /// ```
    pub fn take_iter<I>(&self, iter: I) -> PolarsResult<Self>
    where
        I: Iterator<Item = usize> + Clone + Sync + TrustedLen,
    {
        let new_col = self.try_apply_columns_par(&|s| {
            let mut i = iter.clone();
            s.take_iter(&mut i)
        })?;

        Ok(DataFrame::new_no_checks(new_col))
    }

    /// Take `DataFrame` values by indexes from an iterator.
    ///
    /// # Safety
    ///
    /// This doesn't do any bound checking but checks null validity.
    #[must_use]
    pub unsafe fn take_iter_unchecked<I>(&self, mut iter: I) -> Self
    where
        I: Iterator<Item = usize> + Clone + Sync + TrustedLen,
    {
        let n_chunks = self.n_chunks();
        let has_utf8 = self
            .columns
            .iter()
            .any(|s| matches!(s.dtype(), DataType::Utf8));

        if (n_chunks == 1 && self.width() > 1) || has_utf8 {
            let idx_ca: NoNull<IdxCa> = iter.map(|idx| idx as IdxSize).collect();
            let idx_ca = idx_ca.into_inner();
            return self.take_unchecked(&idx_ca);
        }

        let new_col = if self.width() == 1 {
            self.columns
                .iter()
                .map(|s| s.take_iter_unchecked(&mut iter))
                .collect::<Vec<_>>()
        } else {
            self.apply_columns_par(&|s| {
                let mut i = iter.clone();
                s.take_iter_unchecked(&mut i)
            })
        };
        DataFrame::new_no_checks(new_col)
    }

    /// Take `DataFrame` values by indexes from an iterator that may contain None values.
    ///
    /// # Safety
    ///
    /// This doesn't do any bound checking. Out of bounds may access uninitialized memory.
    /// Null validity is checked
    #[must_use]
    pub unsafe fn take_opt_iter_unchecked<I>(&self, mut iter: I) -> Self
    where
        I: Iterator<Item = Option<usize>> + Clone + Sync + TrustedLen,
    {
        let n_chunks = self.n_chunks();

        let has_utf8 = self
            .columns
            .iter()
            .any(|s| matches!(s.dtype(), DataType::Utf8));

        if (n_chunks == 1 && self.width() > 1) || has_utf8 {
            let idx_ca: IdxCa = iter.map(|opt| opt.map(|v| v as IdxSize)).collect();
            return self.take_unchecked(&idx_ca);
        }

        let new_col = if self.width() == 1 {
            self.columns
                .iter()
                .map(|s| s.take_opt_iter_unchecked(&mut iter))
                .collect::<Vec<_>>()
        } else {
            self.apply_columns_par(&|s| {
                let mut i = iter.clone();
                s.take_opt_iter_unchecked(&mut i)
            })
        };

        DataFrame::new_no_checks(new_col)
    }

    /// Take `DataFrame` rows by index values.
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
        let indices = if indices.chunks.len() > 1 {
            Cow::Owned(indices.rechunk())
        } else {
            Cow::Borrowed(indices)
        };
        let new_col = POOL.install(|| {
            self.try_apply_columns_par(&|s| match s.dtype() {
                DataType::Utf8 => s.take_threaded(&indices, true),
                _ => s.take(&indices),
            })
        })?;

        Ok(DataFrame::new_no_checks(new_col))
    }

    pub(crate) unsafe fn take_unchecked(&self, idx: &IdxCa) -> Self {
        self.take_unchecked_impl(idx, true)
    }

    unsafe fn take_unchecked_impl(&self, idx: &IdxCa, allow_threads: bool) -> Self {
        let cols = if allow_threads {
            POOL.install(|| {
                self.apply_columns_par(&|s| match s.dtype() {
                    DataType::Utf8 => s.take_unchecked_threaded(idx, true).unwrap(),
                    _ => s.take_unchecked(idx).unwrap(),
                })
            })
        } else {
            self.columns
                .iter()
                .map(|s| s.take_unchecked(idx).unwrap())
                .collect()
        };
        DataFrame::new_no_checks(cols)
    }

    /// Rename a column in the `DataFrame`.
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
        self.select_mut(column)
            .ok_or_else(|| PolarsError::ColumnNotFound(column.to_string().into()))
            .map(|s| s.rename(name))?;

        let unique_names: AHashSet<&str, ahash::RandomState> =
            AHashSet::from_iter(self.columns.iter().map(|s| s.name()));
        if unique_names.len() != self.columns.len() {
            return Err(PolarsError::SchemaMisMatch(
                "duplicate column names found".into(),
            ));
        }
        Ok(self)
    }

    /// Sort `DataFrame` in place by a column.
    pub fn sort_in_place(
        &mut self,
        by_column: impl IntoVec<String>,
        reverse: impl IntoVec<bool>,
    ) -> PolarsResult<&mut Self> {
        let by_column = self.select_series(by_column)?;
        let reverse = reverse.into_vec();
        self.columns = self
            .sort_impl(by_column, reverse, false, None, true)?
            .columns;
        Ok(self)
    }

    /// This is the dispatch of Self::sort, and exists to reduce compile bloat by monomorphization.
    #[cfg(feature = "private")]
    pub fn sort_impl(
        &self,
        by_column: Vec<Series>,
        reverse: Vec<bool>,
        nulls_last: bool,
        slice: Option<(i64, usize)>,
        parallel: bool,
    ) -> PolarsResult<Self> {
        if self.height() == 0 {
            return Ok(self.clone());
        }
        // a lot of indirection in both sorting and take
        let mut df = self.clone();
        let df = df.as_single_chunk_par();
        // note that the by_column argument also contains evaluated expression from polars-lazy
        // that may not even be present in this dataframe.

        // therefore when we try to set the first columns as sorted, we ignore the error
        // as expressions are not present (they are renamed to _POLARS_SORT_COLUMN_i.
        let first_reverse = reverse[0];
        let first_by_column = by_column[0].name().to_string();
        let mut take = match by_column.len() {
            1 => {
                let s = &by_column[0];
                let options = SortOptions {
                    descending: reverse[0],
                    nulls_last,
                    multithreaded: parallel,
                };
                // fast path for a frame with a single series
                // no need to compute the sort indices and then take by these indices
                // simply sort and return as frame
                if df.width() == 1 && df.check_name_to_idx(s.name()).is_ok() {
                    let mut out = s.sort_with(options);
                    if let Some((offset, len)) = slice {
                        out = out.slice(offset, len);
                    }

                    return Ok(out.into_frame());
                }
                s.arg_sort(options)
            }
            _ => {
                #[cfg(feature = "sort_multiple")]
                {
                    let (first, by_column, reverse) = prepare_arg_sort(by_column, reverse)?;
                    first.arg_sort_multiple(&by_column, &reverse)?
                }
                #[cfg(not(feature = "sort_multiple"))]
                {
                    panic!("activate `sort_multiple` feature gate to enable this functionality");
                }
            }
        };

        if let Some((offset, len)) = slice {
            take = take.slice(offset, len);
        }

        // Safety:
        // the created indices are in bounds
        let mut df = unsafe { df.take_unchecked_impl(&take, parallel) };
        // Mark the first sort column as sorted
        // if the column did not exists it is ok, because we sorted by an expression
        // not present in the dataframe
        let _ = df.apply(&first_by_column, |s| {
            let mut s = s.clone();
            if first_reverse {
                s.set_sorted_flag(IsSorted::Descending)
            } else {
                s.set_sorted_flag(IsSorted::Ascending)
            }
            s
        });
        Ok(df)
    }

    /// Return a sorted clone of this `DataFrame`.
    ///
    /// # Example
    ///
    /// ```
    /// # use polars_core::prelude::*;
    /// fn sort_example(df: &DataFrame, reverse: bool) -> PolarsResult<DataFrame> {
    ///     df.sort(["a"], reverse)
    /// }
    ///
    /// fn sort_by_multiple_columns_example(df: &DataFrame) -> PolarsResult<DataFrame> {
    ///     df.sort(&["a", "b"], vec![false, true])
    /// }
    /// ```
    pub fn sort(
        &self,
        by_column: impl IntoVec<String>,
        reverse: impl IntoVec<bool>,
    ) -> PolarsResult<Self> {
        let mut df = self.clone();
        df.sort_in_place(by_column, reverse)?;
        Ok(df)
    }

    /// Sort the `DataFrame` by a single column with extra options.
    pub fn sort_with_options(&self, by_column: &str, options: SortOptions) -> PolarsResult<Self> {
        let mut df = self.clone();
        let by_column = vec![df.column(by_column)?.clone()];
        let reverse = vec![options.descending];
        df.columns = df
            .sort_impl(
                by_column,
                reverse,
                options.nulls_last,
                None,
                options.multithreaded,
            )?
            .columns;
        Ok(df)
    }

    /// Replace a column with a `Series`.
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

    /// Replace column at index `idx` with a `Series`.
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
    /// df.replace_at_idx(1, df.select_at_idx(1).unwrap() + 32);
    /// # Ok::<(), PolarsError>(())
    /// ```
    pub fn replace_at_idx<S: IntoSeries>(
        &mut self,
        idx: usize,
        new_col: S,
    ) -> PolarsResult<&mut Self> {
        let mut new_column = new_col.into_series();
        if new_column.len() != self.height() {
            return Err(PolarsError::ShapeMisMatch(
                format!("Cannot replace Series at index {}. The shape of Series {} does not match that of the DataFrame {}",
                idx, new_column.len(), self.height()
                ).into()));
        };
        if idx >= self.width() {
            return Err(PolarsError::ComputeError(
                format!(
                    "Column index: {} outside of DataFrame with {} columns",
                    idx,
                    self.width()
                )
                .into(),
            ));
        }
        let old_col = &mut self.columns[idx];
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
    ///     str_val.utf8()
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
            PolarsError::ComputeError(
                format!("Column index: {idx} outside of DataFrame with {width} columns",).into(),
            )
        })?;
        let name = col.name().to_string();
        let new_col = f(col).into_series();
        match new_col.len() {
            1 => {
                let new_col = new_col.new_from_index(0, df_height);
                let _ = mem::replace(col, new_col);
            }
            len if (len == df_height) => {
                let _ = mem::replace(col, new_col);
            }
            len => {
                return Err(PolarsError::ShapeMisMatch(
                    format!(
                        "Result Series has shape {} where the DataFrame has height {}",
                        len,
                        self.height()
                    )
                    .into(),
                ));
            }
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
    ///     s.utf8()?
    ///     .set_at_idx_with(idx, |opt_val| opt_val.map(|string| format!("{}-is-modified", string)))
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
            PolarsError::ComputeError(
                format!("Column index: {idx} outside of DataFrame with {width} columns",).into(),
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
    ///     s.utf8()?
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
        let idx = self
            .find_idx_by_name(column)
            .ok_or_else(|| PolarsError::ColumnNotFound(column.to_string().into()))?;
        self.try_apply_at_idx(idx, f)
    }

    /// Slice the `DataFrame` along the rows.
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
        let col = self
            .columns
            .iter()
            .map(|s| s.slice(offset, length))
            .collect::<Vec<_>>();
        DataFrame::new_no_checks(col)
    }

    #[must_use]
    pub fn slice_par(&self, offset: i64, length: usize) -> Self {
        if offset == 0 && length == self.height() {
            return self.clone();
        }
        DataFrame::new_no_checks(self.apply_columns_par(&|s| s.slice(offset, length)))
    }

    #[must_use]
    pub fn _slice_and_realloc(&self, offset: i64, length: usize) -> Self {
        if offset == 0 && length == self.height() {
            return self.clone();
        }
        DataFrame::new_no_checks(self.apply_columns(&|s| {
            let mut out = s.slice(offset, length);
            out.shrink_to_fit();
            out
        }))
    }

    /// Get the head of the `DataFrame`.
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
        DataFrame::new_no_checks(col)
    }

    /// Get the tail of the `DataFrame`.
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
        DataFrame::new_no_checks(col)
    }

    /// Iterator over the rows in this `DataFrame` as Arrow RecordBatches.
    ///
    /// # Panics
    ///
    /// Panics if the `DataFrame` that is passed is not rechunked.
    ///
    /// This responsibility is left to the caller as we don't want to take mutable references here,
    /// but we also don't want to rechunk here, as this operation is costly and would benefit the caller
    /// as well.
    pub fn iter_chunks(&self) -> RecordBatchIter {
        RecordBatchIter {
            columns: &self.columns,
            idx: 0,
            n_chunks: self.n_chunks(),
        }
    }

    /// Iterator over the rows in this `DataFrame` as Arrow RecordBatches as physical values.
    ///
    /// # Panics
    ///
    /// Panics if the `DataFrame` that is passed is not rechunked.
    ///
    /// This responsibility is left to the caller as we don't want to take mutable references here,
    /// but we also don't want to rechunk here, as this operation is costly and would benefit the caller
    /// as well.
    pub fn iter_chunks_physical(&self) -> PhysRecordBatchIter<'_> {
        PhysRecordBatchIter {
            iters: self.columns.iter().map(|s| s.chunks().iter()).collect(),
        }
    }

    /// Get a `DataFrame` with all the columns in reversed order.
    #[must_use]
    pub fn reverse(&self) -> Self {
        let col = self.columns.iter().map(|s| s.reverse()).collect::<Vec<_>>();
        DataFrame::new_no_checks(col)
    }

    /// Shift the values by a given period and fill the parts that will be empty due to this operation
    /// with `Nones`.
    ///
    /// See the method on [Series](../series/trait.SeriesTrait.html#method.shift) for more info on the `shift` operation.
    #[must_use]
    pub fn shift(&self, periods: i64) -> Self {
        let col = self.apply_columns_par(&|s| s.shift(periods));

        DataFrame::new_no_checks(col)
    }

    /// Replace None values with one of the following strategies:
    /// * Forward fill (replace None with the previous value)
    /// * Backward fill (replace None with the next value)
    /// * Mean fill (replace None with the mean of the whole array)
    /// * Min fill (replace None with the minimum of the whole array)
    /// * Max fill (replace None with the maximum of the whole array)
    ///
    /// See the method on [Series](../series/trait.SeriesTrait.html#method.fill_null) for more info on the `fill_null` operation.
    pub fn fill_null(&self, strategy: FillNullStrategy) -> PolarsResult<Self> {
        let col = self.try_apply_columns_par(&|s| s.fill_null(strategy))?;

        Ok(DataFrame::new_no_checks(col))
    }

    /// Summary statistics for a DataFrame. Only summarizes numeric datatypes at the moment and returns nulls for non numeric datatypes.
    /// Try in keep output similar to pandas
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use polars_core::prelude::*;
    /// let df1: DataFrame = df!("categorical" => &["d","e","f"],
    ///                          "numeric" => &[1, 2, 3],
    ///                          "object" => &["a", "b", "c"])?;
    /// assert_eq!(df1.shape(), (3, 3));
    ///
    /// let df2: DataFrame = df1.describe(None)?;
    /// assert_eq!(df2.shape(), (9, 4));
    /// dbg!(df2);
    /// # Ok::<(), PolarsError>(())
    /// ```
    ///
    /// Output:
    ///
    /// ```text
    /// shape: (8, 4)
    /// ┌────────────┬─────────────┬─────────┬────────┐
    /// │ describe   ┆ categorical ┆ numeric ┆ object │
    /// │ ---        ┆ ---         ┆ ---     ┆ ---    │
    /// │ str        ┆ str         ┆ f64     ┆ str    │
    /// ╞════════════╪═════════════╪═════════╪════════╡
    /// │ count      ┆ 3           ┆ 3.0     ┆ 3      │
    /// │ null_count ┆ 0           ┆ 0.0     ┆ 0      │
    /// │ mean       ┆ null        ┆ 2.0     ┆ null   │
    /// │ std        ┆ null        ┆ 1.0     ┆ null   │
    /// │ min        ┆ d           ┆ 1,0     ┆ a      │
    /// │ 25%        ┆ null        ┆ 1.5     ┆ null   │
    /// │ 50%        ┆ null        ┆ 2.0     ┆ null   │
    /// │ 75%        ┆ null        ┆ 2.5     ┆ null   │
    /// │ max        ┆ f           ┆ 3.0     ┆ c      │
    /// └────────────┴─────────────┴─────────┴────────┘
    /// ```
    #[cfg(feature = "describe")]
    pub fn describe(&self, percentiles: Option<&[f64]>) -> PolarsResult<Self> {
        fn describe_cast(df: &DataFrame, original_schema: &Schema) -> PolarsResult<DataFrame> {
            let columns = df
                .columns
                .iter()
                .zip(original_schema.iter_dtypes())
                .map(|(s, original_dt)| {
                    if original_dt.is_numeric() | matches!(original_dt, DataType::Boolean) {
                        s.cast(&DataType::Float64)
                    }
                    // for dates, strings, etc, we cast to string so that all
                    // statistics can be shown
                    else {
                        s.cast(&DataType::Utf8)
                    }
                })
                .collect::<PolarsResult<Vec<Series>>>()?;

            DataFrame::new(columns)
        }

        fn count(df: &DataFrame) -> DataFrame {
            let columns = df.apply_columns_par(&|s| Series::new(s.name(), [s.len() as IdxSize]));
            DataFrame::new_no_checks(columns)
        }

        let percentiles = percentiles.unwrap_or(&[0.25, 0.5, 0.75]);

        let mut headers: Vec<String> = vec![
            "count".to_string(),
            "null_count".to_string(),
            "mean".to_string(),
            "std".to_string(),
            "min".to_string(),
        ];

        let original_schema = self.schema();

        let mut tmp: Vec<DataFrame> = vec![
            describe_cast(&count(self), &original_schema)?,
            describe_cast(&self.null_count(), &original_schema)?,
            describe_cast(&self.mean(), &original_schema)?,
            describe_cast(&self.std(1), &original_schema)?,
            describe_cast(&self.min(), &original_schema)?,
        ];

        for p in percentiles {
            tmp.push(describe_cast(
                &self.quantile(*p, QuantileInterpolOptions::Linear)?,
                &original_schema,
            )?);
            headers.push(format!("{}%", *p * 100.0));
        }

        // Keep order same as pandas
        tmp.push(describe_cast(&self.max(), &original_schema)?);
        headers.push("max".to_string());

        let mut summary = concat_df_unchecked(&tmp);

        summary.insert_at_idx(0, Series::new("describe", headers))?;

        Ok(summary)
    }

    /// Aggregate the columns to their maximum values.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use polars_core::prelude::*;
    /// let df1: DataFrame = df!("Die n°1" => &[1, 3, 1, 5, 6],
    ///                          "Die n°2" => &[3, 2, 3, 5, 3])?;
    /// assert_eq!(df1.shape(), (5, 2));
    ///
    /// let df2: DataFrame = df1.max();
    /// assert_eq!(df2.shape(), (1, 2));
    /// println!("{}", df2);
    /// # Ok::<(), PolarsError>(())
    /// ```
    ///
    /// Output:
    ///
    /// ```text
    /// shape: (1, 2)
    /// +---------+---------+
    /// | Die n°1 | Die n°2 |
    /// | ---     | ---     |
    /// | i32     | i32     |
    /// +=========+=========+
    /// | 6       | 5       |
    /// +---------+---------+
    /// ```
    #[must_use]
    pub fn max(&self) -> Self {
        let columns = self.apply_columns_par(&|s| s.max_as_series());

        DataFrame::new_no_checks(columns)
    }

    /// Aggregate the columns to their standard deviation values.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use polars_core::prelude::*;
    /// let df1: DataFrame = df!("Die n°1" => &[1, 3, 1, 5, 6],
    ///                          "Die n°2" => &[3, 2, 3, 5, 3])?;
    /// assert_eq!(df1.shape(), (5, 2));
    ///
    /// let df2: DataFrame = df1.std(1);
    /// assert_eq!(df2.shape(), (1, 2));
    /// println!("{}", df2);
    /// # Ok::<(), PolarsError>(())
    /// ```
    ///
    /// Output:
    ///
    /// ```text
    /// shape: (1, 2)
    /// +-------------------+--------------------+
    /// | Die n°1           | Die n°2            |
    /// | ---               | ---                |
    /// | f64               | f64                |
    /// +===================+====================+
    /// | 2.280350850198276 | 1.0954451150103321 |
    /// +-------------------+--------------------+
    /// ```
    #[must_use]
    pub fn std(&self, ddof: u8) -> Self {
        let columns = self.apply_columns_par(&|s| s.std_as_series(ddof));

        DataFrame::new_no_checks(columns)
    }
    /// Aggregate the columns to their variation values.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use polars_core::prelude::*;
    /// let df1: DataFrame = df!("Die n°1" => &[1, 3, 1, 5, 6],
    ///                          "Die n°2" => &[3, 2, 3, 5, 3])?;
    /// assert_eq!(df1.shape(), (5, 2));
    ///
    /// let df2: DataFrame = df1.var(1);
    /// assert_eq!(df2.shape(), (1, 2));
    /// println!("{}", df2);
    /// # Ok::<(), PolarsError>(())
    /// ```
    ///
    /// Output:
    ///
    /// ```text
    /// shape: (1, 2)
    /// +---------+---------+
    /// | Die n°1 | Die n°2 |
    /// | ---     | ---     |
    /// | f64     | f64     |
    /// +=========+=========+
    /// | 5.2     | 1.2     |
    /// +---------+---------+
    /// ```
    #[must_use]
    pub fn var(&self, ddof: u8) -> Self {
        let columns = self.apply_columns_par(&|s| s.var_as_series(ddof));
        DataFrame::new_no_checks(columns)
    }

    /// Aggregate the columns to their minimum values.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use polars_core::prelude::*;
    /// let df1: DataFrame = df!("Die n°1" => &[1, 3, 1, 5, 6],
    ///                          "Die n°2" => &[3, 2, 3, 5, 3])?;
    /// assert_eq!(df1.shape(), (5, 2));
    ///
    /// let df2: DataFrame = df1.min();
    /// assert_eq!(df2.shape(), (1, 2));
    /// println!("{}", df2);
    /// # Ok::<(), PolarsError>(())
    /// ```
    ///
    /// Output:
    ///
    /// ```text
    /// shape: (1, 2)
    /// +---------+---------+
    /// | Die n°1 | Die n°2 |
    /// | ---     | ---     |
    /// | i32     | i32     |
    /// +=========+=========+
    /// | 1       | 2       |
    /// +---------+---------+
    /// ```
    #[must_use]
    pub fn min(&self) -> Self {
        let columns = self.apply_columns_par(&|s| s.min_as_series());
        DataFrame::new_no_checks(columns)
    }

    /// Aggregate the columns to their sum values.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use polars_core::prelude::*;
    /// let df1: DataFrame = df!("Die n°1" => &[1, 3, 1, 5, 6],
    ///                          "Die n°2" => &[3, 2, 3, 5, 3])?;
    /// assert_eq!(df1.shape(), (5, 2));
    ///
    /// let df2: DataFrame = df1.sum();
    /// assert_eq!(df2.shape(), (1, 2));
    /// println!("{}", df2);
    /// # Ok::<(), PolarsError>(())
    /// ```
    ///
    /// Output:
    ///
    /// ```text
    /// shape: (1, 2)
    /// +---------+---------+
    /// | Die n°1 | Die n°2 |
    /// | ---     | ---     |
    /// | i32     | i32     |
    /// +=========+=========+
    /// | 16      | 16      |
    /// +---------+---------+
    /// ```
    #[must_use]
    pub fn sum(&self) -> Self {
        let columns = self.apply_columns_par(&|s| s.sum_as_series());
        DataFrame::new_no_checks(columns)
    }

    /// Aggregate the columns to their mean values.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use polars_core::prelude::*;
    /// let df1: DataFrame = df!("Die n°1" => &[1, 3, 1, 5, 6],
    ///                          "Die n°2" => &[3, 2, 3, 5, 3])?;
    /// assert_eq!(df1.shape(), (5, 2));
    ///
    /// let df2: DataFrame = df1.mean();
    /// assert_eq!(df2.shape(), (1, 2));
    /// println!("{}", df2);
    /// # Ok::<(), PolarsError>(())
    /// ```
    ///
    /// Output:
    ///
    /// ```text
    /// shape: (1, 2)
    /// +---------+---------+
    /// | Die n°1 | Die n°2 |
    /// | ---     | ---     |
    /// | f64     | f64     |
    /// +=========+=========+
    /// | 3.2     | 3.2     |
    /// +---------+---------+
    /// ```
    #[must_use]
    pub fn mean(&self) -> Self {
        let columns = self.apply_columns_par(&|s| s.mean_as_series());
        DataFrame::new_no_checks(columns)
    }

    /// Aggregate the columns to their median values.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use polars_core::prelude::*;
    /// let df1: DataFrame = df!("Die n°1" => &[1, 3, 1, 5, 6],
    ///                          "Die n°2" => &[3, 2, 3, 5, 3])?;
    /// assert_eq!(df1.shape(), (5, 2));
    ///
    /// let df2: DataFrame = df1.median();
    /// assert_eq!(df2.shape(), (1, 2));
    /// println!("{}", df2);
    /// # Ok::<(), PolarsError>(())
    /// ```
    ///
    /// Output:
    ///
    /// ```text
    /// shape: (1, 2)
    /// +---------+---------+
    /// | Die n°1 | Die n°2 |
    /// | ---     | ---     |
    /// | i32     | i32     |
    /// +=========+=========+
    /// | 3       | 3       |
    /// +---------+---------+
    /// ```
    #[must_use]
    pub fn median(&self) -> Self {
        let columns = self.apply_columns_par(&|s| s.median_as_series());
        DataFrame::new_no_checks(columns)
    }

    /// Aggregate the columns to their quantile values.
    pub fn quantile(&self, quantile: f64, interpol: QuantileInterpolOptions) -> PolarsResult<Self> {
        let columns = self.try_apply_columns_par(&|s| s.quantile_as_series(quantile, interpol))?;

        Ok(DataFrame::new_no_checks(columns))
    }

    /// Aggregate the column horizontally to their min values.
    #[cfg(feature = "zip_with")]
    pub fn hmin(&self) -> PolarsResult<Option<Series>> {
        let min_fn = |acc: &Series, s: &Series| {
            let mask = acc.lt(s)? & acc.is_not_null() | s.is_null();
            acc.zip_with(&mask, s)
        };

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
            }
        }
    }

    /// Aggregate the column horizontally to their max values.
    #[cfg(feature = "zip_with")]
    pub fn hmax(&self) -> PolarsResult<Option<Series>> {
        let max_fn = |acc: &Series, s: &Series| {
            let mask = acc.gt(s)? & acc.is_not_null() | s.is_null();
            acc.zip_with(&mask, s)
        };

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
            }
        }
    }

    /// Aggregate the column horizontally to their sum values.
    pub fn hsum(&self, none_strategy: NullStrategy) -> PolarsResult<Option<Series>> {
        let sum_fn =
            |acc: &Series, s: &Series, none_strategy: NullStrategy| -> PolarsResult<Series> {
                let mut acc = acc.clone();
                let mut s = s.clone();
                if let NullStrategy::Ignore = none_strategy {
                    // if has nulls
                    if acc.has_validity() {
                        acc = acc.fill_null(FillNullStrategy::Zero)?;
                    }
                    if s.has_validity() {
                        s = s.fill_null(FillNullStrategy::Zero)?;
                    }
                }
                Ok(&acc + &s)
            };

        match self.columns.len() {
            0 => Ok(None),
            1 => Ok(Some(self.columns[0].clone())),
            2 => sum_fn(&self.columns[0], &self.columns[1], none_strategy).map(Some),
            _ => {
                // the try_reduce_with is a bit slower in parallelism,
                // but I don't think it matters here as we parallelize over columns, not over elements
                POOL.install(|| {
                    self.columns
                        .par_iter()
                        .map(|s| Ok(Cow::Borrowed(s)))
                        .try_reduce_with(|l, r| sum_fn(&l, &r, none_strategy).map(Cow::Owned))
                        // we can unwrap the option, because we are certain there is a column
                        // we started this operation on 3 columns
                        .unwrap()
                        .map(|cow| Some(cow.into_owned()))
                })
            }
        }
    }

    /// Aggregate the column horizontally to their mean values.
    pub fn hmean(&self, none_strategy: NullStrategy) -> PolarsResult<Option<Series>> {
        match self.columns.len() {
            0 => Ok(None),
            1 => Ok(Some(self.columns[0].clone())),
            _ => {
                let columns = self
                    .columns
                    .iter()
                    .cloned()
                    .filter(|s| {
                        let dtype = s.dtype();
                        dtype.is_numeric() || matches!(dtype, DataType::Boolean)
                    })
                    .collect();
                let numeric_df = DataFrame::new_no_checks(columns);

                let sum = || numeric_df.hsum(none_strategy);

                let null_count = || {
                    numeric_df
                        .columns
                        .par_iter()
                        .map(|s| s.is_null().cast(&DataType::UInt32).unwrap())
                        .reduce_with(|l, r| &l + &r)
                        // we can unwrap the option, because we are certain there is a column
                        // we started this operation on 2 columns
                        .unwrap()
                };

                let (sum, null_count) = POOL.install(|| rayon::join(sum, null_count));
                let sum = sum?;

                // value lengths: len - null_count
                let value_length: UInt32Chunked =
                    (numeric_df.width().sub(&null_count)).u32().unwrap().clone();

                // make sure that we do not divide by zero
                // by replacing with None
                let value_length = value_length
                    .set(&value_length.equal(0), None)?
                    .into_series()
                    .cast(&DataType::Float64)?;

                Ok(sum.map(|sum| &sum / &value_length))
            }
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

    /// Drop duplicate rows from a `DataFrame`.
    /// *This fails when there is a column of type List in DataFrame*
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
    /// println!("{}", df.drop_duplicates(true, None)?);
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
    #[deprecated(note = "use DataFrame::unique")]
    pub fn drop_duplicates(
        &self,
        maintain_order: bool,
        subset: Option<&[String]>,
    ) -> PolarsResult<Self> {
        match maintain_order {
            true => self.unique_stable(subset, UniqueKeepStrategy::First),
            false => self.unique(subset, UniqueKeepStrategy::First),
        }
    }

    /// Drop duplicate rows from a `DataFrame`.
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
    /// println!("{}", df.unique_stable(None, UniqueKeepStrategy::First)?);
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
    pub fn unique_stable(
        &self,
        subset: Option<&[String]>,
        keep: UniqueKeepStrategy,
    ) -> PolarsResult<DataFrame> {
        self.unique_impl(true, subset, keep)
    }

    /// Unstable distinct. See [`DataFrame::unique_stable`].
    pub fn unique(
        &self,
        subset: Option<&[String]>,
        keep: UniqueKeepStrategy,
    ) -> PolarsResult<DataFrame> {
        self.unique_impl(false, subset, keep)
    }

    fn unique_impl(
        &self,
        maintain_order: bool,
        subset: Option<&[String]>,
        keep: UniqueKeepStrategy,
    ) -> PolarsResult<Self> {
        let names = match &subset {
            Some(s) => s.iter().map(|s| &**s).collect(),
            None => self.get_column_names(),
        };

        let columns = match (keep, maintain_order) {
            (UniqueKeepStrategy::First, true) => {
                let gb = self.groupby_stable(names)?;
                let groups = gb.get_groups();
                self.apply_columns_par(&|s| unsafe { s.agg_first(groups) })
            }
            (UniqueKeepStrategy::Last, true) => {
                // maintain order by last values, so the sorted groups are not correct as they
                // are sorted by the first value
                let gb = self.groupby(names)?;
                let groups = gb.get_groups();
                let last_idx: NoNull<IdxCa> = groups
                    .iter()
                    .map(|g| match g {
                        GroupsIndicator::Idx((_first, idx)) => idx[idx.len() - 1],
                        GroupsIndicator::Slice([first, len]) => first + len - 1,
                    })
                    .collect();

                let last_idx = last_idx.sort(false);
                return Ok(unsafe { self.take_unchecked(&last_idx) });
            }
            (UniqueKeepStrategy::First, false) => {
                let gb = self.groupby(names)?;
                let groups = gb.get_groups();
                self.apply_columns_par(&|s| unsafe { s.agg_first(groups) })
            }
            (UniqueKeepStrategy::Last, false) => {
                let gb = self.groupby(names)?;
                let groups = gb.get_groups();
                self.apply_columns_par(&|s| unsafe { s.agg_last(groups) })
            }
            (UniqueKeepStrategy::None, _) => {
                let df_part = self.select(names)?;
                let mask = df_part.is_unique()?;
                return self.filter(&mask);
            }
        };
        Ok(DataFrame::new_no_checks(columns))
    }

    /// Get a mask of all the unique rows in the `DataFrame`.
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
    pub fn is_unique(&self) -> PolarsResult<BooleanChunked> {
        let gb = self.groupby(self.get_column_names())?;
        let groups = gb.take_groups();
        Ok(is_unique_helper(
            groups,
            self.height() as IdxSize,
            true,
            false,
        ))
    }

    /// Get a mask of all the duplicated rows in the `DataFrame`.
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
    pub fn is_duplicated(&self) -> PolarsResult<BooleanChunked> {
        let gb = self.groupby(self.get_column_names())?;
        let groups = gb.take_groups();
        Ok(is_unique_helper(
            groups,
            self.height() as IdxSize,
            false,
            true,
        ))
    }

    /// Create a new `DataFrame` that shows the null counts per column.
    #[must_use]
    pub fn null_count(&self) -> Self {
        let cols = self
            .columns
            .iter()
            .map(|s| Series::new(s.name(), &[s.null_count() as IdxSize]))
            .collect();
        Self::new_no_checks(cols)
    }

    /// Hash and combine the row values
    #[cfg(feature = "row_hash")]
    pub fn hash_rows(
        &mut self,
        hasher_builder: Option<ahash::RandomState>,
    ) -> PolarsResult<UInt64Chunked> {
        let dfs = split_df(self, POOL.current_num_threads())?;
        let (cas, _) = df_rows_to_hashes_threaded(&dfs, hasher_builder)?;

        let mut iter = cas.into_iter();
        let mut acc_ca = iter.next().unwrap();
        for ca in iter {
            acc_ca.append(&ca);
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

    #[cfg(feature = "chunked_ids")]
    #[doc(hidden)]
    //// Take elements by a slice of [`ChunkId`]s.
    /// # Safety
    /// Does not do any bound checks.
    /// `sorted` indicates if the chunks are sorted.
    #[doc(hidden)]
    pub unsafe fn _take_chunked_unchecked_seq(&self, idx: &[ChunkId], sorted: IsSorted) -> Self {
        let cols = self.apply_columns(&|s| s._take_chunked_unchecked(idx, sorted));

        DataFrame::new_no_checks(cols)
    }
    #[cfg(feature = "chunked_ids")]
    //// Take elements by a slice of optional [`ChunkId`]s.
    /// # Safety
    /// Does not do any bound checks.
    #[doc(hidden)]
    pub unsafe fn _take_opt_chunked_unchecked_seq(&self, idx: &[Option<ChunkId>]) -> Self {
        let cols = self.apply_columns(&|s| match s.dtype() {
            DataType::Utf8 => s._take_opt_chunked_unchecked_threaded(idx, true),
            _ => s._take_opt_chunked_unchecked(idx),
        });

        DataFrame::new_no_checks(cols)
    }

    #[cfg(feature = "chunked_ids")]
    pub(crate) unsafe fn take_chunked_unchecked(&self, idx: &[ChunkId], sorted: IsSorted) -> Self {
        let cols = self.apply_columns_par(&|s| match s.dtype() {
            DataType::Utf8 => s._take_chunked_unchecked_threaded(idx, sorted, true),
            _ => s._take_chunked_unchecked(idx, sorted),
        });

        DataFrame::new_no_checks(cols)
    }

    #[cfg(feature = "chunked_ids")]
    pub(crate) unsafe fn take_opt_chunked_unchecked(&self, idx: &[Option<ChunkId>]) -> Self {
        let cols = self.apply_columns_par(&|s| match s.dtype() {
            DataType::Utf8 => s._take_opt_chunked_unchecked_threaded(idx, true),
            _ => s._take_opt_chunked_unchecked(idx),
        });

        DataFrame::new_no_checks(cols)
    }

    /// Be careful with allowing threads when calling this in a large hot loop
    /// every thread split may be on rayon stack and lead to SO
    #[doc(hidden)]
    pub unsafe fn _take_unchecked_slice(&self, idx: &[IdxSize], allow_threads: bool) -> Self {
        self._take_unchecked_slice2(idx, allow_threads, IsSorted::Not)
    }

    #[doc(hidden)]
    pub unsafe fn _take_unchecked_slice2(
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
                    }
                    IsSorted::Descending => {
                        assert!(idx[0] >= idx[idx.len() - 1]);
                    }
                    _ => {}
                }
            }
        }
        let ptr = idx.as_ptr() as *mut IdxSize;
        let len = idx.len();

        // create a temporary vec. we will not drop it.
        let mut ca = IdxCa::from_vec("", Vec::from_raw_parts(ptr, len, len));
        ca.set_sorted_flag(sorted);
        let out = self.take_unchecked_impl(&ca, allow_threads);

        // ref count of buffers should be one because we dropped all allocations
        let arr = {
            let arr_ref = std::mem::take(&mut ca.chunks).pop().unwrap();
            arr_ref
                .as_any()
                .downcast_ref::<PrimitiveArray<IdxSize>>()
                .unwrap()
                .clone()
        };
        // the only owned heap allocation is the `Vec` we created and must not be dropped
        let _ = std::mem::ManuallyDrop::new(arr.into_mut().right().unwrap());
        out
    }

    #[cfg(feature = "partition_by")]
    #[doc(hidden)]
    pub fn _partition_by_impl(
        &self,
        cols: &[String],
        stable: bool,
    ) -> PolarsResult<Vec<DataFrame>> {
        let groups = if stable {
            self.groupby_stable(cols)?.take_groups()
        } else {
            self.groupby(cols)?.take_groups()
        };

        // don't parallelize this
        // there is a lot of parallelization in take and this may easily SO
        POOL.install(|| {
            match groups {
                GroupsProxy::Idx(idx) => {
                    Ok(idx
                        .into_par_iter()
                        .map(|(_, group)| {
                            // groups are in bounds
                            unsafe { self._take_unchecked_slice(&group, false) }
                        })
                        .collect())
                }
                GroupsProxy::Slice { groups, .. } => Ok(groups
                    .into_par_iter()
                    .map(|[first, len]| self.slice(first as i64, len as usize))
                    .collect()),
            }
        })
    }

    /// Split into multiple DataFrames partitioned by groups
    #[cfg(feature = "partition_by")]
    pub fn partition_by(&self, cols: impl IntoVec<String>) -> PolarsResult<Vec<DataFrame>> {
        let cols = cols.into_vec();
        self._partition_by_impl(&cols, false)
    }

    /// Split into multiple DataFrames partitioned by groups
    /// Order of the groups are maintained.
    #[cfg(feature = "partition_by")]
    pub fn partition_by_stable(&self, cols: impl IntoVec<String>) -> PolarsResult<Vec<DataFrame>> {
        let cols = cols.into_vec();
        self._partition_by_impl(&cols, true)
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
                let ca = s.struct_()?;
                new_cols.extend_from_slice(ca.fields());
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
                    .ok_or_else(|| PolarsError::ColumnNotFound(col.into()))?;
            }
        }
        DataFrame::new(new_cols)
    }
}

pub struct RecordBatchIter<'a> {
    columns: &'a Vec<Series>,
    idx: usize,
    n_chunks: usize,
}

impl<'a> Iterator for RecordBatchIter<'a> {
    type Item = ArrowChunk;

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx >= self.n_chunks {
            None
        } else {
            // create a batch of the columns with the same chunk no.
            let batch_cols = self.columns.iter().map(|s| s.to_arrow(self.idx)).collect();
            self.idx += 1;

            Some(ArrowChunk::new(batch_cols))
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
    type Item = ArrowChunk;

    fn next(&mut self) -> Option<Self::Item> {
        self.iters
            .iter_mut()
            .map(|phys_iter| phys_iter.next().cloned())
            .collect::<Option<Vec<_>>>()
            .map(ArrowChunk::new)
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
        DataFrame::new_no_checks(vec![])
    }
}

impl From<DataFrame> for Vec<Series> {
    fn from(df: DataFrame) -> Self {
        df.columns
    }
}

// utility to test if we can vstack/extend the columns
fn can_extend(left: &Series, right: &Series) -> PolarsResult<()> {
    if left.dtype() != right.dtype() || left.name() != right.name() {
        if left.dtype() != right.dtype() {
            return Err(PolarsError::SchemaMisMatch(
                format!(
                    "cannot vstack: because column datatypes (dtypes) in the two DataFrames do not match for \
                                left.name='{}' with left.dtype={} != right.dtype={} with right.name='{}'",
                    left.name(),
                    left.dtype(),
                    right.dtype(),
                    right.name()
                )
                    .into(),
            ));
        } else {
            return Err(PolarsError::SchemaMisMatch(
                format!(
                    "cannot vstack: because column names in the two DataFrames do not match for \
                                left.name='{}' != right.name='{}'",
                    left.name(),
                    right.name()
                )
                .into(),
            ));
        }
    };
    Ok(())
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::frame::NullStrategy;

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
        let mut iter = df.iter_chunks();
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
    fn test_filter_broadcast_on_utf8_col() {
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
            .unique_stable(None, UniqueKeepStrategy::First)
            .unwrap()
            .sort(["flt"], false)
            .unwrap();
        let valid = df! {
            "flt" => [1., 2., 3.],
            "int" => [1, 2, 3],
            "str" => ["a", "b", "c"]
        }
        .unwrap();
        assert!(df.frame_equal(&valid));
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
    fn test_h_agg() {
        let a = Series::new("a", &[1, 2, 6]);
        let b = Series::new("b", &[Some(1), None, None]);
        let c = Series::new("c", &[Some(4), None, Some(3)]);

        let df = DataFrame::new(vec![a, b, c]).unwrap();
        assert_eq!(
            Vec::from(
                df.hmean(NullStrategy::Ignore)
                    .unwrap()
                    .unwrap()
                    .f64()
                    .unwrap()
            ),
            &[Some(2.0), Some(2.0), Some(4.5)]
        );
        assert_eq!(
            Vec::from(
                df.hsum(NullStrategy::Ignore)
                    .unwrap()
                    .unwrap()
                    .i32()
                    .unwrap()
            ),
            &[Some(6), Some(2), Some(9)]
        );
        assert_eq!(
            Vec::from(df.hmin().unwrap().unwrap().i32().unwrap()),
            &[Some(1), Some(2), Some(3)]
        );
        assert_eq!(
            Vec::from(df.hmax().unwrap().unwrap().i32().unwrap()),
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
        let mut df = base.slice(0, 0);
        let out = df.with_column(Series::new("c", [1]))?;
        assert_eq!(out.shape(), (0, 3));
        assert!(out.iter().all(|s| s.len() == 0));

        // no columns
        base.columns = vec![];
        let out = base.with_column(Series::new("c", [1]))?;
        assert_eq!(out.shape(), (1, 1));

        Ok(())
    }

    #[test]
    #[cfg(feature = "describe")]
    fn test_df_describe() -> PolarsResult<()> {
        let df1: DataFrame = df!("categorical" => &["d","e","f"],
                                 "numeric" => &[1, 2, 3],
                                 "object" => &["a", "b", "c"])?;

        assert_eq!(df1.shape(), (3, 3));

        let df2: DataFrame = df1.describe(None)?;

        assert_eq!(df2.shape(), (9, 4));

        let expected = df!(
            "describe" => ["count", "null_count", "mean", "std", "min", "25%", "50%", "75%", "max"],
            "categorical" => [Some("3"), Some("0"), None, None, Some("d"), None, None, None, Some("f")],
            "numeric" => [3.0, 0.0, 2.0, 1.0, 1.0, 1.5, 2.0, 2.5, 3.0],
            "object" => [Some("3"), Some("0"), None, None, Some("a"), None, None, None, Some("c")],
        ).unwrap();

        assert_eq!(df2, expected);

        Ok(())
    }
}
