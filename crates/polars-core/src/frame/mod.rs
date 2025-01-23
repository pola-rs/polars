//! DataFrame module.
use std::sync::OnceLock;
use std::{mem, ops};

use arrow::datatypes::ArrowSchemaRef;
use polars_row::ArrayRef;
use polars_schema::schema::debug_ensure_matching_schema_names;
use polars_utils::itertools::Itertools;
use rayon::prelude::*;

use crate::chunked_array::flags::StatisticsFlags;
#[cfg(feature = "algorithm_group_by")]
use crate::chunked_array::ops::unique::is_unique_helper;
use crate::prelude::*;
#[cfg(feature = "row_hash")]
use crate::utils::split_df;
use crate::utils::{slice_offsets, try_get_supertype, Container, NoNull};
use crate::{HEAD_DEFAULT_LENGTH, TAIL_DEFAULT_LENGTH};

#[cfg(feature = "dataframe_arithmetic")]
mod arithmetic;
mod chunks;
pub use chunks::chunk_df_for_writing;
pub mod column;
pub mod explode;
mod from;
#[cfg(feature = "algorithm_group_by")]
pub mod group_by;
pub(crate) mod horizontal;
#[cfg(any(feature = "rows", feature = "object"))]
pub mod row;
mod top_k;
mod upstream_traits;

use arrow::record_batch::{RecordBatch, RecordBatchT};
use polars_utils::pl_str::PlSmallStr;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use strum_macros::IntoStaticStr;

#[cfg(feature = "row_hash")]
use crate::hashing::_df_rows_to_hashes_threaded_vertical;
use crate::prelude::sort::{argsort_multiple_row_fmt, prepare_arg_sort};
use crate::series::IsSorted;
use crate::POOL;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Default, Hash, IntoStaticStr)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[strum(serialize_all = "snake_case")]
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
    F: for<'a> FnMut(&'a T) -> &'a str,
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
/// let s1 = Column::new("Fruit".into(), ["Apple", "Apple", "Pear"]);
/// let s2 = Column::new("Color".into(), ["Red", "Yellow", "Green"]);
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
/// let df: PolarsResult<DataFrame> = df!("Fruit" => ["Apple", "Apple", "Pear"],
///                                       "Color" => ["Red", "Yellow", "Green"]);
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
/// let df = df!("Fruit" => ["Apple", "Apple", "Pear"],
///              "Color" => ["Red", "Yellow", "Green"])?;
///
/// assert_eq!(df[0], Column::new("Fruit".into(), &["Apple", "Apple", "Pear"]));
/// assert_eq!(df[1], Column::new("Color".into(), &["Red", "Yellow", "Green"]));
/// # Ok::<(), PolarsError>(())
/// ```
///
/// ## By a `Series` name
///
/// ```rust
/// # use polars_core::prelude::*;
/// let df = df!("Fruit" => ["Apple", "Apple", "Pear"],
///              "Color" => ["Red", "Yellow", "Green"])?;
///
/// assert_eq!(df["Fruit"], Column::new("Fruit".into(), &["Apple", "Apple", "Pear"]));
/// assert_eq!(df["Color"], Column::new("Color".into(), &["Red", "Yellow", "Green"]));
/// # Ok::<(), PolarsError>(())
/// ```
#[derive(Clone)]
pub struct DataFrame {
    height: usize,
    // invariant: columns[i].len() == height for each 0 >= i > columns.len()
    pub(crate) columns: Vec<Column>,

    /// A cached schema. This might not give correct results if the DataFrame was modified in place
    /// between schema and reading.
    cached_schema: OnceLock<SchemaRef>,
}

impl DataFrame {
    pub fn clear_schema(&mut self) {
        self.cached_schema = OnceLock::new();
    }

    #[inline]
    pub fn materialized_column_iter(&self) -> impl ExactSizeIterator<Item = &Series> {
        self.columns.iter().map(Column::as_materialized_series)
    }

    #[inline]
    pub fn par_materialized_column_iter(&self) -> impl ParallelIterator<Item = &Series> {
        self.columns.par_iter().map(Column::as_materialized_series)
    }

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
        self.columns.iter().map(Column::estimated_size).sum()
    }

    // Reduce monomorphization.
    fn try_apply_columns(
        &self,
        func: &(dyn Fn(&Column) -> PolarsResult<Column> + Send + Sync),
    ) -> PolarsResult<Vec<Column>> {
        self.columns.iter().map(func).collect()
    }
    // Reduce monomorphization.
    pub fn _apply_columns(&self, func: &(dyn Fn(&Column) -> Column)) -> Vec<Column> {
        self.columns.iter().map(func).collect()
    }
    // Reduce monomorphization.
    fn try_apply_columns_par(
        &self,
        func: &(dyn Fn(&Column) -> PolarsResult<Column> + Send + Sync),
    ) -> PolarsResult<Vec<Column>> {
        POOL.install(|| self.columns.par_iter().map(func).collect())
    }
    // Reduce monomorphization.
    pub fn _apply_columns_par(
        &self,
        func: &(dyn Fn(&Column) -> Column + Send + Sync),
    ) -> Vec<Column> {
        POOL.install(|| self.columns.par_iter().map(func).collect())
    }

    /// Get the index of the column.
    fn check_name_to_idx(&self, name: &str) -> PolarsResult<usize> {
        self.get_column_index(name)
            .ok_or_else(|| polars_err!(col_not_found = name))
    }

    fn check_already_present(&self, name: &str) -> PolarsResult<()> {
        polars_ensure!(
            self.columns.iter().all(|s| s.name().as_str() != name),
            Duplicate: "column with name {:?} is already present in the DataFrame", name
        );
        Ok(())
    }

    /// Reserve additional slots into the chunks of the series.
    pub(crate) fn reserve_chunks(&mut self, additional: usize) {
        for s in &mut self.columns {
            if let Column::Series(s) = s {
                // SAFETY:
                // do not modify the data, simply resize.
                unsafe { s.chunks_mut().reserve(additional) }
            }
        }
    }

    /// Create a DataFrame from a Vector of Series.
    ///
    /// # Example
    ///
    /// ```
    /// # use polars_core::prelude::*;
    /// let s0 = Column::new("days".into(), [0, 1, 2].as_ref());
    /// let s1 = Column::new("temp".into(), [22.1, 19.9, 7.].as_ref());
    ///
    /// let df = DataFrame::new(vec![s0, s1])?;
    /// # Ok::<(), PolarsError>(())
    /// ```
    pub fn new(columns: Vec<Column>) -> PolarsResult<Self> {
        ensure_names_unique(&columns, |s| s.name().as_str())?;

        let Some(fst) = columns.first() else {
            return Ok(DataFrame {
                height: 0,
                columns,
                cached_schema: OnceLock::new(),
            });
        };

        let height = fst.len();
        for col in &columns[1..] {
            polars_ensure!(
                col.len() == height,
                ShapeMismatch: "could not create a new DataFrame: series {:?} has length {} while series {:?} has length {}",
                columns[0].name(), height, col.name(), col.len()
            );
        }

        Ok(DataFrame {
            height,
            columns,
            cached_schema: OnceLock::new(),
        })
    }

    /// Converts a sequence of columns into a DataFrame, broadcasting length-1
    /// columns to match the other columns.
    pub fn new_with_broadcast(columns: Vec<Column>) -> PolarsResult<Self> {
        // The length of the longest non-unit length column determines the
        // broadcast length. If all columns are unit-length the broadcast length
        // is one.
        let broadcast_len = columns
            .iter()
            .map(|s| s.len())
            .filter(|l| *l != 1)
            .max()
            .unwrap_or(1);
        Self::new_with_broadcast_len(columns, broadcast_len)
    }

    /// Converts a sequence of columns into a DataFrame, broadcasting length-1
    /// columns to broadcast_len.
    pub fn new_with_broadcast_len(
        columns: Vec<Column>,
        broadcast_len: usize,
    ) -> PolarsResult<Self> {
        ensure_names_unique(&columns, |s| s.name().as_str())?;
        unsafe { Self::new_with_broadcast_no_namecheck(columns, broadcast_len) }
    }

    /// Converts a sequence of columns into a DataFrame, broadcasting length-1
    /// columns to match the other columns.
    ///  
    /// # Safety
    /// Does not check that the column names are unique (which they must be).
    pub unsafe fn new_with_broadcast_no_namecheck(
        mut columns: Vec<Column>,
        broadcast_len: usize,
    ) -> PolarsResult<Self> {
        for col in &mut columns {
            // Length not equal to the broadcast len, needs broadcast or is an error.
            let len = col.len();
            if len != broadcast_len {
                if len != 1 {
                    let name = col.name().to_owned();
                    let extra_info =
                        if let Some(c) = columns.iter().find(|c| c.len() == broadcast_len) {
                            format!(" (matching column '{}')", c.name())
                        } else {
                            String::new()
                        };
                    polars_bail!(
                        ShapeMismatch: "could not create a new DataFrame: series {name:?} has length {len} while trying to broadcast to length {broadcast_len}{extra_info}",
                    );
                }
                *col = col.new_from_index(0, broadcast_len);
            }
        }

        let length = if columns.is_empty() { 0 } else { broadcast_len };

        Ok(unsafe { DataFrame::new_no_checks(length, columns) })
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
        DataFrame {
            height: 0,
            columns: vec![],
            cached_schema: OnceLock::new(),
        }
    }

    /// Create an empty `DataFrame` with empty columns as per the `schema`.
    pub fn empty_with_schema(schema: &Schema) -> Self {
        let cols = schema
            .iter()
            .map(|(name, dtype)| Column::from(Series::new_empty(name.clone(), dtype)))
            .collect();
        unsafe { DataFrame::new_no_checks(0, cols) }
    }

    /// Create an empty `DataFrame` with empty columns as per the `schema`.
    pub fn empty_with_arrow_schema(schema: &ArrowSchema) -> Self {
        let cols = schema
            .iter_values()
            .map(|fld| {
                Column::from(Series::new_empty(
                    fld.name.clone(),
                    &(DataType::from_arrow_field(fld)),
                ))
            })
            .collect();
        unsafe { DataFrame::new_no_checks(0, cols) }
    }

    /// Create a new `DataFrame` with the given schema, only containing nulls.
    pub fn full_null(schema: &Schema, height: usize) -> Self {
        let columns = schema
            .iter_fields()
            .map(|f| Column::full_null(f.name.clone(), height, f.dtype()))
            .collect();
        unsafe { DataFrame::new_no_checks(height, columns) }
    }

    /// Removes the last `Series` from the `DataFrame` and returns it, or [`None`] if it is empty.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let s1 = Column::new("Ocean".into(), ["Atlantic", "Indian"]);
    /// let s2 = Column::new("Area (km²)".into(), [106_460_000, 70_560_000]);
    /// let mut df = DataFrame::new(vec![s1.clone(), s2.clone()])?;
    ///
    /// assert_eq!(df.pop(), Some(s2));
    /// assert_eq!(df.pop(), Some(s1));
    /// assert_eq!(df.pop(), None);
    /// assert!(df.is_empty());
    /// # Ok::<(), PolarsError>(())
    /// ```
    pub fn pop(&mut self) -> Option<Column> {
        self.clear_schema();

        self.columns.pop()
    }

    /// Add a new column at index 0 that counts the rows.
    ///
    /// # Example
    ///
    /// ```
    /// # use polars_core::prelude::*;
    /// let df1: DataFrame = df!("Name" => ["James", "Mary", "John", "Patricia"])?;
    /// assert_eq!(df1.shape(), (4, 1));
    ///
    /// let df2: DataFrame = df1.with_row_index("Id".into(), None)?;
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
    pub fn with_row_index(&self, name: PlSmallStr, offset: Option<IdxSize>) -> PolarsResult<Self> {
        let mut columns = Vec::with_capacity(self.columns.len() + 1);
        let offset = offset.unwrap_or(0);

        let mut ca = IdxCa::from_vec(
            name,
            (offset..(self.height() as IdxSize) + offset).collect(),
        );
        ca.set_sorted_flag(IsSorted::Ascending);
        columns.push(ca.into_series().into());

        columns.extend_from_slice(&self.columns);
        DataFrame::new(columns)
    }

    /// Add a row index column in place.
    pub fn with_row_index_mut(&mut self, name: PlSmallStr, offset: Option<IdxSize>) -> &mut Self {
        let offset = offset.unwrap_or(0);
        let mut ca = IdxCa::from_vec(
            name,
            (offset..(self.height() as IdxSize) + offset).collect(),
        );
        ca.set_sorted_flag(IsSorted::Ascending);

        self.clear_schema();
        self.columns.insert(0, ca.into_series().into());
        self
    }

    /// Create a new `DataFrame` but does not check the length or duplicate occurrence of the
    /// `Series`.
    ///
    /// Calculates the height from the first column or `0` if no columns are given.
    ///
    /// # Safety
    ///
    /// It is the callers responsibility to uphold the contract of all `Series`
    /// having an equal length and a unique name, if not this may panic down the line.
    pub unsafe fn new_no_checks_height_from_first(columns: Vec<Column>) -> DataFrame {
        let height = columns.first().map_or(0, Column::len);
        unsafe { Self::new_no_checks(height, columns) }
    }

    /// Create a new `DataFrame` but does not check the length or duplicate occurrence of the
    /// `Series`.
    ///
    /// It is advised to use [DataFrame::new] in favor of this method.
    ///
    /// # Safety
    ///
    /// It is the callers responsibility to uphold the contract of all `Series`
    /// having an equal length and a unique name, if not this may panic down the line.
    pub unsafe fn new_no_checks(height: usize, columns: Vec<Column>) -> DataFrame {
        if cfg!(debug_assertions) {
            ensure_names_unique(&columns, |s| s.name().as_str()).unwrap();

            for col in &columns {
                assert_eq!(col.len(), height);
            }
        }

        unsafe { Self::_new_no_checks_impl(height, columns) }
    }

    /// This will not panic even in debug mode - there are some (rare) use cases where a DataFrame
    /// is temporarily constructed containing duplicates for dispatching to functions. A DataFrame
    /// constructed with this method is generally highly unsafe and should not be long-lived.
    #[allow(clippy::missing_safety_doc)]
    pub const unsafe fn _new_no_checks_impl(height: usize, columns: Vec<Column>) -> DataFrame {
        DataFrame {
            height,
            columns,
            cached_schema: OnceLock::new(),
        }
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
    pub unsafe fn new_no_length_checks(columns: Vec<Column>) -> PolarsResult<DataFrame> {
        ensure_names_unique(&columns, |s| s.name().as_str())?;

        Ok(if cfg!(debug_assertions) {
            Self::new(columns).unwrap()
        } else {
            let height = Self::infer_height(&columns);
            DataFrame {
                height,
                columns,
                cached_schema: OnceLock::new(),
            }
        })
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
            if let Column::Series(s) = s {
                *s = s.rechunk().into();
            }
        }
        self
    }

    /// Aggregate all the chunks in the DataFrame to a single chunk in parallel.
    /// This may lead to more peak memory consumption.
    pub fn as_single_chunk_par(&mut self) -> &mut Self {
        if self.columns.iter().any(|c| c.n_chunks() > 1) {
            self.columns = self._apply_columns_par(&|s| s.rechunk());
        }
        self
    }

    /// Rechunks all columns to only have a single chunk.
    pub fn rechunk_mut(&mut self) {
        // SAFETY: We never adjust the length or names of the columns.
        let columns = unsafe { self.get_columns_mut() };

        for col in columns.iter_mut().filter(|c| c.n_chunks() > 1) {
            *col = col.rechunk();
        }
    }

    /// Rechunks all columns to only have a single chunk and turns it into a [`RecordBatchT`].
    pub fn rechunk_to_record_batch(
        self,
        compat_level: CompatLevel,
    ) -> RecordBatchT<Box<dyn Array>> {
        let height = self.height();

        let (schema, arrays) = self
            .columns
            .into_iter()
            .map(|col| {
                let mut series = col.take_materialized_series();
                // Rechunk to one chunk if necessary
                if series.n_chunks() > 1 {
                    series = series.rechunk();
                }
                (
                    series.field().to_arrow(compat_level),
                    series.to_arrow(0, compat_level),
                )
            })
            .collect();

        RecordBatchT::new(height, Arc::new(schema), arrays)
    }

    /// Returns true if the chunks of the columns do not align and re-chunking should be done
    pub fn should_rechunk(&self) -> bool {
        // Fast check. It is also needed for correctness, as code below doesn't check if the number
        // of chunks is equal.
        if !self
            .get_columns()
            .iter()
            .filter_map(|c| c.as_series().map(|s| s.n_chunks()))
            .all_equal()
        {
            return true;
        }

        // From here we check chunk lengths.
        let mut chunk_lengths = self.materialized_column_iter().map(|s| s.chunk_lengths());
        match chunk_lengths.next() {
            None => false,
            Some(first_column_chunk_lengths) => {
                // Fast Path for single Chunk Series
                if first_column_chunk_lengths.size_hint().0 == 1 {
                    return chunk_lengths.any(|cl| cl.size_hint().0 != 1);
                }
                // Always rechunk if we have more chunks than rows.
                // except when we have an empty df containing a single chunk
                let height = self.height();
                let n_chunks = first_column_chunk_lengths.size_hint().0;
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
    pub fn align_chunks_par(&mut self) -> &mut Self {
        if self.should_rechunk() {
            self.as_single_chunk_par()
        } else {
            self
        }
    }

    pub fn align_chunks(&mut self) -> &mut Self {
        if self.should_rechunk() {
            self.as_single_chunk()
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
    /// let df: DataFrame = df!("Thing" => ["Observable universe", "Human stupidity"],
    ///                         "Diameter (m)" => [8.8e26, f64::INFINITY])?;
    ///
    /// let f1: Field = Field::new("Thing".into(), DataType::String);
    /// let f2: Field = Field::new("Diameter (m)".into(), DataType::Float64);
    /// let sc: Schema = Schema::from_iter(vec![f1, f2]);
    ///
    /// assert_eq!(&**df.schema(), &sc);
    /// # Ok::<(), PolarsError>(())
    /// ```
    pub fn schema(&self) -> &SchemaRef {
        let out = self.cached_schema.get_or_init(|| {
            Arc::new(
                self.columns
                    .iter()
                    .map(|x| (x.name().clone(), x.dtype().clone()))
                    .collect(),
            )
        });

        debug_assert_eq!(out.len(), self.width());

        out
    }

    /// Get a reference to the [`DataFrame`] columns.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let df: DataFrame = df!("Name" => ["Adenine", "Cytosine", "Guanine", "Thymine"],
    ///                         "Symbol" => ["A", "C", "G", "T"])?;
    /// let columns: &[Column] = df.get_columns();
    ///
    /// assert_eq!(columns[0].name(), "Name");
    /// assert_eq!(columns[1].name(), "Symbol");
    /// # Ok::<(), PolarsError>(())
    /// ```
    #[inline]
    pub fn get_columns(&self) -> &[Column] {
        &self.columns
    }

    #[inline]
    /// Get mutable access to the underlying columns.
    ///
    /// # Safety
    ///
    /// The caller must ensure the length of all [`Series`] remains equal to `height` or
    /// [`DataFrame::set_height`] is called afterwards with the appropriate `height`.
    /// The caller must ensure that the cached schema is cleared if it modifies the schema by
    /// calling [`DataFrame::clear_schema`].
    pub unsafe fn get_columns_mut(&mut self) -> &mut Vec<Column> {
        &mut self.columns
    }

    #[inline]
    /// Remove all the columns in the [`DataFrame`] but keep the `height`.
    pub fn clear_columns(&mut self) {
        unsafe { self.get_columns_mut() }.clear();
        self.clear_schema();
    }

    #[inline]
    /// Extend the columns without checking for name collisions or height.
    ///
    /// # Safety
    ///
    /// The caller needs to ensure that:
    /// - Column names are unique within the resulting [`DataFrame`].
    /// - The length of each appended column matches the height of the [`DataFrame`]. For
    ///   `DataFrame`]s with no columns (ZCDFs), it is important that the height is set afterwards
    ///   with [`DataFrame::set_height`].
    pub unsafe fn column_extend_unchecked(&mut self, iter: impl IntoIterator<Item = Column>) {
        unsafe { self.get_columns_mut() }.extend(iter);
        self.clear_schema();
    }

    /// Take ownership of the underlying columns vec.
    pub fn take_columns(self) -> Vec<Column> {
        self.columns
    }

    /// Iterator over the columns as [`Series`].
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let s1 = Column::new("Name".into(), ["Pythagoras' theorem", "Shannon entropy"]);
    /// let s2 = Column::new("Formula".into(), ["a²+b²=c²", "H=-Σ[P(x)log|P(x)|]"]);
    /// let df: DataFrame = DataFrame::new(vec![s1.clone(), s2.clone()])?;
    ///
    /// let mut iterator = df.iter();
    ///
    /// assert_eq!(iterator.next(), Some(s1.as_materialized_series()));
    /// assert_eq!(iterator.next(), Some(s2.as_materialized_series()));
    /// assert_eq!(iterator.next(), None);
    /// # Ok::<(), PolarsError>(())
    /// ```
    pub fn iter(&self) -> impl ExactSizeIterator<Item = &Series> {
        self.materialized_column_iter()
    }

    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let df: DataFrame = df!("Language" => ["Rust", "Python"],
    ///                         "Designer" => ["Graydon Hoare", "Guido van Rossum"])?;
    ///
    /// assert_eq!(df.get_column_names(), &["Language", "Designer"]);
    /// # Ok::<(), PolarsError>(())
    /// ```
    pub fn get_column_names(&self) -> Vec<&PlSmallStr> {
        self.columns.iter().map(|s| s.name()).collect()
    }

    /// Get the [`Vec<PlSmallStr>`] representing the column names.
    pub fn get_column_names_owned(&self) -> Vec<PlSmallStr> {
        self.columns.iter().map(|s| s.name().clone()).collect()
    }

    pub fn get_column_names_str(&self) -> Vec<&str> {
        self.columns.iter().map(|s| s.name().as_str()).collect()
    }

    /// Set the column names.
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let mut df: DataFrame = df!("Mathematical set" => ["ℕ", "ℤ", "𝔻", "ℚ", "ℝ", "ℂ"])?;
    /// df.set_column_names(["Set"])?;
    ///
    /// assert_eq!(df.get_column_names(), &["Set"]);
    /// # Ok::<(), PolarsError>(())
    /// ```
    pub fn set_column_names<I, S>(&mut self, names: I) -> PolarsResult<()>
    where
        I: IntoIterator<Item = S>,
        S: Into<PlSmallStr>,
    {
        let names = names.into_iter().map(Into::into).collect::<Vec<_>>();
        self._set_column_names_impl(names.as_slice())
    }

    fn _set_column_names_impl(&mut self, names: &[PlSmallStr]) -> PolarsResult<()> {
        polars_ensure!(
            names.len() == self.width(),
            ShapeMismatch: "{} column names provided for a DataFrame of width {}",
            names.len(), self.width()
        );
        ensure_names_unique(names, |s| s.as_str())?;

        let columns = mem::take(&mut self.columns);
        self.columns = columns
            .into_iter()
            .zip(names)
            .map(|(s, name)| {
                let mut s = s;
                s.rename(name.clone());
                s
            })
            .collect();
        self.clear_schema();
        Ok(())
    }

    /// Get the data types of the columns in the [`DataFrame`].
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let venus_air: DataFrame = df!("Element" => ["Carbon dioxide", "Nitrogen"],
    ///                                "Fraction" => [0.965, 0.035])?;
    ///
    /// assert_eq!(venus_air.dtypes(), &[DataType::String, DataType::Float64]);
    /// # Ok::<(), PolarsError>(())
    /// ```
    pub fn dtypes(&self) -> Vec<DataType> {
        self.columns.iter().map(|s| s.dtype().clone()).collect()
    }

    pub(crate) fn first_series_column(&self) -> Option<&Series> {
        self.columns.iter().find_map(|col| col.as_series())
    }

    /// The number of chunks for the first column.
    pub fn first_col_n_chunks(&self) -> usize {
        match self.first_series_column() {
            None if self.columns.is_empty() => 0,
            None => 1,
            Some(s) => s.n_chunks(),
        }
    }

    /// The highest number of chunks for any column.
    pub fn max_n_chunks(&self) -> usize {
        self.columns
            .iter()
            .map(|s| s.as_series().map(|s| s.n_chunks()).unwrap_or(1))
            .max()
            .unwrap_or(0)
    }

    /// Get a reference to the schema fields of the [`DataFrame`].
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let earth: DataFrame = df!("Surface type" => ["Water", "Land"],
    ///                            "Fraction" => [0.708, 0.292])?;
    ///
    /// let f1: Field = Field::new("Surface type".into(), DataType::String);
    /// let f2: Field = Field::new("Fraction".into(), DataType::Float64);
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
    /// let df1: DataFrame = df!("1" => [1, 2, 3, 4, 5])?;
    /// let df2: DataFrame = df!("1" => [1, 2, 3, 4, 5],
    ///                          "2" => [1, 2, 3, 4, 5])?;
    ///
    /// assert_eq!(df0.shape(), (0 ,0));
    /// assert_eq!(df1.shape(), (5, 1));
    /// assert_eq!(df2.shape(), (5, 2));
    /// # Ok::<(), PolarsError>(())
    /// ```
    pub fn shape(&self) -> (usize, usize) {
        (self.height, self.columns.len())
    }

    /// Get the width of the [`DataFrame`] which is the number of columns.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let df0: DataFrame = DataFrame::default();
    /// let df1: DataFrame = df!("Series 1" => [0; 0])?;
    /// let df2: DataFrame = df!("Series 1" => [0; 0],
    ///                          "Series 2" => [0; 0])?;
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
    /// let df1: DataFrame = df!("Currency" => ["€", "$"])?;
    /// let df2: DataFrame = df!("Currency" => ["€", "$", "¥", "£", "₿"])?;
    ///
    /// assert_eq!(df0.height(), 0);
    /// assert_eq!(df1.height(), 2);
    /// assert_eq!(df2.height(), 5);
    /// # Ok::<(), PolarsError>(())
    /// ```
    pub fn height(&self) -> usize {
        self.height
    }

    /// Returns the size as number of rows * number of columns
    pub fn size(&self) -> usize {
        let s = self.shape();
        s.0 * s.1
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
    /// let df2: DataFrame = df!("First name" => ["Forever"],
    ///                          "Last name" => ["Alone"])?;
    /// assert!(!df2.is_empty());
    /// # Ok::<(), PolarsError>(())
    /// ```
    pub fn is_empty(&self) -> bool {
        matches!(self.shape(), (0, _) | (_, 0))
    }

    /// Set the height (i.e. number of rows) of this [`DataFrame`].
    ///
    /// # Safety
    ///
    /// This needs to be equal to the length of all the columns.
    pub unsafe fn set_height(&mut self, height: usize) {
        self.height = height;
    }

    /// Add multiple [`Series`] to a [`DataFrame`].
    /// The added `Series` are required to have the same length.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let df1: DataFrame = df!("Element" => ["Copper", "Silver", "Gold"])?;
    /// let s1 = Column::new("Proton".into(), [29, 47, 79]);
    /// let s2 = Column::new("Electron".into(), [29, 47, 79]);
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
    pub fn hstack(&self, columns: &[Column]) -> PolarsResult<Self> {
        let mut new_cols = self.columns.clone();
        new_cols.extend_from_slice(columns);
        DataFrame::new(new_cols)
    }

    /// Concatenate a [`DataFrame`] to this [`DataFrame`] and return as newly allocated [`DataFrame`].
    ///
    /// If many `vstack` operations are done, it is recommended to call [`DataFrame::align_chunks_par`].
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let df1: DataFrame = df!("Element" => ["Copper", "Silver", "Gold"],
    ///                          "Melting Point (K)" => [1357.77, 1234.93, 1337.33])?;
    /// let df2: DataFrame = df!("Element" => ["Platinum", "Palladium"],
    ///                          "Melting Point (K)" => [2041.4, 1828.05])?;
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
    /// If many `vstack` operations are done, it is recommended to call [`DataFrame::align_chunks_par`].
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let mut df1: DataFrame = df!("Element" => ["Copper", "Silver", "Gold"],
    ///                          "Melting Point (K)" => [1357.77, 1234.93, 1337.33])?;
    /// let df2: DataFrame = df!("Element" => ["Platinum", "Palladium"],
    ///                          "Melting Point (K)" => [2041.4, 1828.05])?;
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
            self.height = other.height;
            return Ok(self);
        }

        self.columns
            .iter_mut()
            .zip(other.columns.iter())
            .try_for_each::<_, PolarsResult<_>>(|(left, right)| {
                ensure_can_extend(&*left, right)?;
                left.append(right).map_err(|e| {
                    e.context(format!("failed to vstack column '{}'", right.name()).into())
                })?;
                Ok(())
            })?;
        self.height += other.height;
        Ok(self)
    }

    /// Concatenate a [`DataFrame`] to this [`DataFrame`]
    ///
    /// If many `vstack` operations are done, it is recommended to call [`DataFrame::align_chunks_par`].
    ///
    /// # Panics
    /// Panics if the schema's don't match.
    pub fn vstack_mut_unchecked(&mut self, other: &DataFrame) {
        self.columns
            .iter_mut()
            .zip(other.columns.iter())
            .for_each(|(left, right)| {
                left.append(right)
                    .map_err(|e| {
                        e.context(format!("failed to vstack column '{}'", right.name()).into())
                    })
                    .expect("should not fail");
            });
        self.height += other.height;
    }

    /// Concatenate a [`DataFrame`] to this [`DataFrame`]
    ///
    /// If many `vstack` operations are done, it is recommended to call [`DataFrame::align_chunks_par`].
    ///
    /// # Panics
    /// Panics if the schema's don't match.
    pub fn vstack_mut_owned_unchecked(&mut self, other: DataFrame) {
        self.columns
            .iter_mut()
            .zip(other.columns)
            .for_each(|(left, right)| {
                left.append_owned(right).expect("should not fail");
            });
        self.height += other.height;
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
    /// of `append` operations with a [`rechunk`](Self::align_chunks_par).
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
                ensure_can_extend(&*left, right)?;
                left.extend(right).map_err(|e| {
                    e.context(format!("failed to extend column '{}'", right.name()).into())
                })?;
                Ok(())
            })?;
        self.height += other.height;
        self.clear_schema();
        Ok(())
    }

    /// Remove a column by name and return the column removed.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let mut df: DataFrame = df!("Animal" => ["Tiger", "Lion", "Great auk"],
    ///                             "IUCN" => ["Endangered", "Vulnerable", "Extinct"])?;
    ///
    /// let s1: PolarsResult<Column> = df.drop_in_place("Average weight");
    /// assert!(s1.is_err());
    ///
    /// let s2: Column = df.drop_in_place("Animal")?;
    /// assert_eq!(s2, Column::new("Animal".into(), &["Tiger", "Lion", "Great auk"]));
    /// # Ok::<(), PolarsError>(())
    /// ```
    pub fn drop_in_place(&mut self, name: &str) -> PolarsResult<Column> {
        let idx = self.check_name_to_idx(name)?;
        self.clear_schema();
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
    pub fn drop_nulls<S>(&self, subset: Option<&[S]>) -> PolarsResult<Self>
    where
        for<'a> &'a S: Into<PlSmallStr>,
    {
        if let Some(v) = subset {
            let v = self.select_columns(v)?;
            self._drop_nulls_impl(v.as_slice())
        } else {
            self._drop_nulls_impl(self.columns.as_slice())
        }
    }

    fn _drop_nulls_impl(&self, subset: &[Column]) -> PolarsResult<Self> {
        // fast path for no nulls in df
        if subset.iter().all(|s| !s.has_nulls()) {
            return Ok(self.clone());
        }

        let mut iter = subset.iter();

        let mask = iter
            .next()
            .ok_or_else(|| polars_err!(NoData: "no data to drop nulls from"))?;
        let mut mask = mask.is_not_null();

        for c in iter {
            mask = mask & c.is_not_null();
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
    /// let df1: DataFrame = df!("Ray type" => ["α", "β", "X", "γ"])?;
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

        Ok(unsafe { DataFrame::new_no_checks(self.height(), new_cols) })
    }

    /// Drop columns that are in `names`.
    pub fn drop_many<I, S>(&self, names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<PlSmallStr>,
    {
        let names: PlHashSet<PlSmallStr> = names.into_iter().map(|s| s.into()).collect();
        self.drop_many_amortized(&names)
    }

    /// Drop columns that are in `names` without allocating a [`HashSet`](std::collections::HashSet).
    pub fn drop_many_amortized(&self, names: &PlHashSet<PlSmallStr>) -> DataFrame {
        if names.is_empty() {
            return self.clone();
        }
        let mut new_cols = Vec::with_capacity(self.columns.len().saturating_sub(names.len()));
        self.columns.iter().for_each(|s| {
            if !names.contains(s.name()) {
                new_cols.push(s.clone())
            }
        });

        unsafe { DataFrame::new_no_checks(self.height(), new_cols) }
    }

    /// Insert a new column at a given index without checking for duplicates.
    /// This can leave the [`DataFrame`] at an invalid state
    fn insert_column_no_name_check(
        &mut self,
        index: usize,
        column: Column,
    ) -> PolarsResult<&mut Self> {
        polars_ensure!(
            self.width() == 0 || column.len() == self.height(),
            ShapeMismatch: "unable to add a column of length {} to a DataFrame of height {}",
            column.len(), self.height(),
        );

        if self.width() == 0 {
            self.height = column.len();
        }

        self.columns.insert(index, column);
        self.clear_schema();
        Ok(self)
    }

    /// Insert a new column at a given index.
    pub fn insert_column<S: IntoColumn>(
        &mut self,
        index: usize,
        column: S,
    ) -> PolarsResult<&mut Self> {
        let column = column.into_column();
        self.check_already_present(column.name().as_str())?;
        self.insert_column_no_name_check(index, column)
    }

    fn add_column_by_search(&mut self, column: Column) -> PolarsResult<()> {
        if let Some(idx) = self.get_column_index(column.name().as_str()) {
            self.replace_column(idx, column)?;
        } else {
            if self.width() == 0 {
                self.height = column.len();
            }

            self.columns.push(column);
            self.clear_schema();
        }
        Ok(())
    }

    /// Add a new column to this [`DataFrame`] or replace an existing one.
    pub fn with_column<C: IntoColumn>(&mut self, column: C) -> PolarsResult<&mut Self> {
        fn inner(df: &mut DataFrame, mut column: Column) -> PolarsResult<&mut DataFrame> {
            let height = df.height();
            if column.len() == 1 && height > 1 {
                column = column.new_from_index(0, height);
            }

            if column.len() == height || df.get_columns().is_empty() {
                df.add_column_by_search(column)?;
                Ok(df)
            }
            // special case for literals
            else if height == 0 && column.len() == 1 {
                let s = column.clear();
                df.add_column_by_search(s)?;
                Ok(df)
            } else {
                polars_bail!(
                    ShapeMismatch: "unable to add a column of length {} to a DataFrame of height {}",
                    column.len(), height,
                );
            }
        }
        let column = column.into_column();
        inner(self, column)
    }

    /// Adds a column to the [`DataFrame`] without doing any checks
    /// on length or duplicates.
    ///
    /// # Safety
    /// The caller must ensure `self.width() == 0 || column.len() == self.height()` .
    pub unsafe fn with_column_unchecked(&mut self, column: Column) -> &mut Self {
        debug_assert!(self.width() == 0 || self.height() == column.len());
        debug_assert!(self.get_column_index(column.name().as_str()).is_none());

        // SAFETY: Invariant of function guarantees for case `width` > 0. We set the height
        // properly for `width` == 0.
        if self.width() == 0 {
            unsafe { self.set_height(column.len()) };
        }
        unsafe { self.get_columns_mut() }.push(column);
        self.clear_schema();

        self
    }

    // Note: Schema can be both input or output_schema
    fn add_column_by_schema(&mut self, c: Column, schema: &Schema) -> PolarsResult<()> {
        let name = c.name();
        if let Some((idx, _, _)) = schema.get_full(name.as_str()) {
            if self.columns.get(idx).map(|s| s.name()) != Some(name) {
                // Given schema is output_schema and we can push.
                if idx == self.columns.len() {
                    if self.width() == 0 {
                        self.height = c.len();
                    }

                    self.columns.push(c);
                    self.clear_schema();
                }
                // Schema is incorrect fallback to search
                else {
                    debug_assert!(false);
                    self.add_column_by_search(c)?;
                }
            } else {
                self.replace_column(idx, c)?;
            }
        } else {
            if self.width() == 0 {
                self.height = c.len();
            }

            self.columns.push(c);
            self.clear_schema();
        }

        Ok(())
    }

    // Note: Schema can be both input or output_schema
    pub fn _add_series(&mut self, series: Vec<Series>, schema: &Schema) -> PolarsResult<()> {
        for (i, s) in series.into_iter().enumerate() {
            // we need to branch here
            // because users can add multiple columns with the same name
            if i == 0 || schema.get(s.name().as_str()).is_some() {
                self.with_column_and_schema(s.into_column(), schema)?;
            } else {
                self.with_column(s.clone().into_column())?;
            }
        }
        Ok(())
    }

    pub fn _add_columns(&mut self, columns: Vec<Column>, schema: &Schema) -> PolarsResult<()> {
        for (i, s) in columns.into_iter().enumerate() {
            // we need to branch here
            // because users can add multiple columns with the same name
            if i == 0 || schema.get(s.name().as_str()).is_some() {
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
    ///
    /// Note: Schema can be both input or output_schema
    pub fn with_column_and_schema<C: IntoColumn>(
        &mut self,
        column: C,
        schema: &Schema,
    ) -> PolarsResult<&mut Self> {
        let mut column = column.into_column();

        let height = self.height();
        if column.len() == 1 && height > 1 {
            column = column.new_from_index(0, height);
        }

        if column.len() == height || self.columns.is_empty() {
            self.add_column_by_schema(column, schema)?;
            Ok(self)
        }
        // special case for literals
        else if height == 0 && column.len() == 1 {
            let s = column.clear();
            self.add_column_by_schema(s, schema)?;
            Ok(self)
        } else {
            polars_bail!(
                ShapeMismatch: "unable to add a column of length {} to a DataFrame of height {}",
                column.len(), height,
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
        unsafe { Some(self.columns.iter().map(|c| c.get_unchecked(idx)).collect()) }
    }

    /// Select a [`Series`] by index.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let df: DataFrame = df!("Star" => ["Sun", "Betelgeuse", "Sirius A", "Sirius B"],
    ///                         "Absolute magnitude" => [4.83, -5.85, 1.42, 11.18])?;
    ///
    /// let s1: Option<&Column> = df.select_at_idx(0);
    /// let s2 = Column::new("Star".into(), ["Sun", "Betelgeuse", "Sirius A", "Sirius B"]);
    ///
    /// assert_eq!(s1, Some(&s2));
    /// # Ok::<(), PolarsError>(())
    /// ```
    pub fn select_at_idx(&self, idx: usize) -> Option<&Column> {
        self.columns.get(idx)
    }

    /// Select column(s) from this [`DataFrame`] by range and return a new [`DataFrame`]
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let df = df! {
    ///     "0" => [0, 0, 0],
    ///     "1" => [1, 1, 1],
    ///     "2" => [2, 2, 2]
    /// }?;
    ///
    /// assert!(df.select(["0", "1"])?.equals(&df.select_by_range(0..=1)?));
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
    /// let df: DataFrame = df!("Name" => ["Player 1", "Player 2", "Player 3"],
    ///                         "Health" => [100, 200, 500],
    ///                         "Mana" => [250, 100, 0],
    ///                         "Strength" => [30, 150, 300])?;
    ///
    /// assert_eq!(df.get_column_index("Name"), Some(0));
    /// assert_eq!(df.get_column_index("Health"), Some(1));
    /// assert_eq!(df.get_column_index("Mana"), Some(2));
    /// assert_eq!(df.get_column_index("Strength"), Some(3));
    /// assert_eq!(df.get_column_index("Haste"), None);
    /// # Ok::<(), PolarsError>(())
    /// ```
    pub fn get_column_index(&self, name: &str) -> Option<usize> {
        let schema = self.schema();
        if let Some(idx) = schema.index_of(name) {
            if self
                .get_columns()
                .get(idx)
                .is_some_and(|c| c.name() == name)
            {
                return Some(idx);
            }
        }

        self.columns.iter().position(|s| s.name().as_str() == name)
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
    /// let s1 = Column::new("Password".into(), ["123456", "[]B$u$g$s$B#u#n#n#y[]{}"]);
    /// let s2 = Column::new("Robustness".into(), ["Weak", "Strong"]);
    /// let df: DataFrame = DataFrame::new(vec![s1.clone(), s2])?;
    ///
    /// assert_eq!(df.column("Password")?, &s1);
    /// # Ok::<(), PolarsError>(())
    /// ```
    pub fn column(&self, name: &str) -> PolarsResult<&Column> {
        let idx = self.try_get_column_index(name)?;
        Ok(self.select_at_idx(idx).unwrap())
    }

    /// Selected multiple columns by name.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let df: DataFrame = df!("Latin name" => ["Oncorhynchus kisutch", "Salmo salar"],
    ///                         "Max weight (kg)" => [16.0, 35.89])?;
    /// let sv: Vec<&Column> = df.columns(["Latin name", "Max weight (kg)"])?;
    ///
    /// assert_eq!(&df[0], sv[0]);
    /// assert_eq!(&df[1], sv[1]);
    /// # Ok::<(), PolarsError>(())
    /// ```
    pub fn columns<I, S>(&self, names: I) -> PolarsResult<Vec<&Column>>
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
        S: Into<PlSmallStr>,
    {
        let cols = selection.into_iter().map(|s| s.into()).collect::<Vec<_>>();
        self._select_impl(cols.as_slice())
    }

    pub fn _select_impl(&self, cols: &[PlSmallStr]) -> PolarsResult<Self> {
        ensure_names_unique(cols, |s| s.as_str())?;
        self._select_impl_unchecked(cols)
    }

    pub fn _select_impl_unchecked(&self, cols: &[PlSmallStr]) -> PolarsResult<Self> {
        let selected = self.select_columns_impl(cols)?;
        Ok(unsafe { DataFrame::new_no_checks(self.height(), selected) })
    }

    /// Select with a known schema. The schema names must match the column names of this DataFrame.
    pub fn select_with_schema<I, S>(&self, selection: I, schema: &SchemaRef) -> PolarsResult<Self>
    where
        I: IntoIterator<Item = S>,
        S: Into<PlSmallStr>,
    {
        let cols = selection.into_iter().map(|s| s.into()).collect::<Vec<_>>();
        self._select_with_schema_impl(&cols, schema, true)
    }

    /// Select with a known schema without checking for duplicates in `selection`.
    /// The schema names must match the column names of this DataFrame.
    pub fn select_with_schema_unchecked<I, S>(
        &self,
        selection: I,
        schema: &Schema,
    ) -> PolarsResult<Self>
    where
        I: IntoIterator<Item = S>,
        S: Into<PlSmallStr>,
    {
        let cols = selection.into_iter().map(|s| s.into()).collect::<Vec<_>>();
        self._select_with_schema_impl(&cols, schema, false)
    }

    /// * The schema names must match the column names of this DataFrame.
    pub fn _select_with_schema_impl(
        &self,
        cols: &[PlSmallStr],
        schema: &Schema,
        check_duplicates: bool,
    ) -> PolarsResult<Self> {
        if check_duplicates {
            ensure_names_unique(cols, |s| s.as_str())?;
        }

        let selected = self.select_columns_impl_with_schema(cols, schema)?;
        Ok(unsafe { DataFrame::new_no_checks(self.height(), selected) })
    }

    /// A non generic implementation to reduce compiler bloat.
    fn select_columns_impl_with_schema(
        &self,
        cols: &[PlSmallStr],
        schema: &Schema,
    ) -> PolarsResult<Vec<Column>> {
        debug_ensure_matching_schema_names(schema, self.schema())?;

        cols.iter()
            .map(|name| {
                let index = schema.try_get_full(name.as_str())?.0;
                Ok(self.columns[index].clone())
            })
            .collect()
    }

    pub fn select_physical<I, S>(&self, selection: I) -> PolarsResult<Self>
    where
        I: IntoIterator<Item = S>,
        S: Into<PlSmallStr>,
    {
        let cols = selection.into_iter().map(|s| s.into()).collect::<Vec<_>>();
        self.select_physical_impl(&cols)
    }

    fn select_physical_impl(&self, cols: &[PlSmallStr]) -> PolarsResult<Self> {
        ensure_names_unique(cols, |s| s.as_str())?;
        let selected = self.select_columns_physical_impl(cols)?;
        Ok(unsafe { DataFrame::new_no_checks(self.height(), selected) })
    }

    /// Select column(s) from this [`DataFrame`] and return them into a [`Vec`].
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let df: DataFrame = df!("Name" => ["Methane", "Ethane", "Propane"],
    ///                         "Carbon" => [1, 2, 3],
    ///                         "Hydrogen" => [4, 6, 8])?;
    /// let sv: Vec<Column> = df.select_columns(["Carbon", "Hydrogen"])?;
    ///
    /// assert_eq!(df["Carbon"], sv[0]);
    /// assert_eq!(df["Hydrogen"], sv[1]);
    /// # Ok::<(), PolarsError>(())
    /// ```
    pub fn select_columns(&self, selection: impl IntoVec<PlSmallStr>) -> PolarsResult<Vec<Column>> {
        let cols = selection.into_vec();
        self.select_columns_impl(&cols)
    }

    fn _names_to_idx_map(&self) -> PlHashMap<&str, usize> {
        self.columns
            .iter()
            .enumerate()
            .map(|(i, s)| (s.name().as_str(), i))
            .collect()
    }

    /// A non generic implementation to reduce compiler bloat.
    fn select_columns_physical_impl(&self, cols: &[PlSmallStr]) -> PolarsResult<Vec<Column>> {
        let selected = if cols.len() > 1 && self.columns.len() > 10 {
            let name_to_idx = self._names_to_idx_map();
            cols.iter()
                .map(|name| {
                    let idx = *name_to_idx
                        .get(name.as_str())
                        .ok_or_else(|| polars_err!(col_not_found = name))?;
                    Ok(self.select_at_idx(idx).unwrap().to_physical_repr())
                })
                .collect::<PolarsResult<Vec<_>>>()?
        } else {
            cols.iter()
                .map(|c| self.column(c.as_str()).map(|s| s.to_physical_repr()))
                .collect::<PolarsResult<Vec<_>>>()?
        };

        Ok(selected)
    }

    /// A non generic implementation to reduce compiler bloat.
    fn select_columns_impl(&self, cols: &[PlSmallStr]) -> PolarsResult<Vec<Column>> {
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
                .map(|c| self.column(c.as_str()).cloned())
                .collect::<PolarsResult<Vec<_>>>()?
        };

        Ok(selected)
    }

    fn filter_height(&self, filtered: &[Column], mask: &BooleanChunked) -> usize {
        // If there is a filtered column just see how many columns there are left.
        if let Some(fst) = filtered.first() {
            return fst.len();
        }

        // Otherwise, count the number of values that would be filtered and return that height.
        let num_trues = mask.num_trues();
        if mask.len() == self.height() {
            num_trues
        } else {
            // This is for broadcasting masks
            debug_assert!(num_trues == 0 || num_trues == 1);
            self.height() * num_trues
        }
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
        let height = self.filter_height(&new_col, mask);

        Ok(unsafe { DataFrame::new_no_checks(height, new_col) })
    }

    /// Same as `filter` but does not parallelize.
    pub fn _filter_seq(&self, mask: &BooleanChunked) -> PolarsResult<Self> {
        let new_col = self.try_apply_columns(&|s| s.filter(mask))?;
        let height = self.filter_height(&new_col, mask);

        Ok(unsafe { DataFrame::new_no_checks(height, new_col) })
    }

    /// Take [`DataFrame`] rows by index values.
    ///
    /// # Example
    ///
    /// ```
    /// # use polars_core::prelude::*;
    /// fn example(df: &DataFrame) -> PolarsResult<DataFrame> {
    ///     let idx = IdxCa::new("idx".into(), [0, 1, 9]);
    ///     df.take(&idx)
    /// }
    /// ```
    pub fn take(&self, indices: &IdxCa) -> PolarsResult<Self> {
        let new_col = POOL.install(|| self.try_apply_columns_par(&|s| s.take(indices)))?;

        Ok(unsafe { DataFrame::new_no_checks(indices.len(), new_col) })
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
            POOL.install(|| self._apply_columns_par(&|c| c.take_unchecked(idx)))
        } else {
            self._apply_columns(&|s| s.take_unchecked(idx))
        };
        unsafe { DataFrame::new_no_checks(idx.len(), cols) }
    }

    /// # Safety
    /// The indices must be in-bounds.
    pub unsafe fn take_slice_unchecked(&self, idx: &[IdxSize]) -> Self {
        self.take_slice_unchecked_impl(idx, true)
    }

    /// # Safety
    /// The indices must be in-bounds.
    pub unsafe fn take_slice_unchecked_impl(&self, idx: &[IdxSize], allow_threads: bool) -> Self {
        let cols = if allow_threads {
            POOL.install(|| self._apply_columns_par(&|s| s.take_slice_unchecked(idx)))
        } else {
            self._apply_columns(&|s| s.take_slice_unchecked(idx))
        };
        unsafe { DataFrame::new_no_checks(idx.len(), cols) }
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
    ///     df.rename(original_name, new_name.into())
    /// }
    /// ```
    pub fn rename(&mut self, column: &str, name: PlSmallStr) -> PolarsResult<&mut Self> {
        if column == name.as_str() {
            return Ok(self);
        }
        polars_ensure!(
            !self.schema().contains(&name),
            Duplicate: "column rename attempted with already existing name \"{name}\""
        );

        self.get_column_index(column)
            .and_then(|idx| self.columns.get_mut(idx))
            .ok_or_else(|| polars_err!(col_not_found = column))
            .map(|c| c.rename(name))?;
        Ok(self)
    }

    /// Sort [`DataFrame`] in place.
    ///
    /// See [`DataFrame::sort`] for more instruction.
    pub fn sort_in_place(
        &mut self,
        by: impl IntoVec<PlSmallStr>,
        sort_options: SortMultipleOptions,
    ) -> PolarsResult<&mut Self> {
        let by_column = self.select_columns(by)?;
        self.columns = self.sort_impl(by_column, sort_options, None)?.columns;
        Ok(self)
    }

    #[doc(hidden)]
    /// This is the dispatch of Self::sort, and exists to reduce compile bloat by monomorphization.
    pub fn sort_impl(
        &self,
        by_column: Vec<Column>,
        mut sort_options: SortMultipleOptions,
        slice: Option<(i64, usize)>,
    ) -> PolarsResult<Self> {
        if by_column.is_empty() {
            // If no columns selected, any order (including original order) is correct.
            return if let Some((offset, len)) = slice {
                Ok(self.slice(offset, len))
            } else {
                Ok(self.clone())
            };
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
            if k < self.len() {
                return self.bottom_k_impl(k, by_column, sort_options);
            }
        }
        // Check if the required column is already sorted; if so we can exit early
        // We can do so when there is only one column to sort by, for multiple columns
        // it will be complicated to do so
        #[cfg(feature = "dtype-categorical")]
        let is_not_categorical_enum =
            !(matches!(by_column[0].dtype(), DataType::Categorical(_, _))
                || matches!(by_column[0].dtype(), DataType::Enum(_, _)));

        #[cfg(not(feature = "dtype-categorical"))]
        #[allow(non_upper_case_globals)]
        const is_not_categorical_enum: bool = true;

        if by_column.len() == 1 && is_not_categorical_enum {
            let required_sorting = if sort_options.descending[0] {
                IsSorted::Descending
            } else {
                IsSorted::Ascending
            };
            // If null count is 0 then nulls_last doesnt matter
            // Safe to get value at last position since the dataframe is not empty (taken care above)
            let no_sorting_required = (by_column[0].is_sorted_flag() == required_sorting)
                && ((by_column[0].null_count() == 0)
                    || by_column[0].get(by_column[0].len() - 1).unwrap().is_null()
                        == sort_options.nulls_last[0]);

            if no_sorting_required {
                return if let Some((offset, len)) = slice {
                    Ok(self.slice(offset, len))
                } else {
                    Ok(self.clone())
                };
            }
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
                    limit: sort_options.limit,
                };
                // fast path for a frame with a single series
                // no need to compute the sort indices and then take by these indices
                // simply sort and return as frame
                if df.width() == 1 && df.check_name_to_idx(s.name().as_str()).is_ok() {
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
                    first
                        .as_materialized_series()
                        .arg_sort_multiple(&other, &sort_options)?
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

    /// Create a `DataFrame` that has fields for all the known runtime metadata for each column.
    ///
    /// This dataframe does not necessarily have a specified schema and may be changed at any
    /// point. It is primarily used for debugging.
    pub fn _to_metadata(&self) -> DataFrame {
        let num_columns = self.columns.len();

        let mut column_names =
            StringChunkedBuilder::new(PlSmallStr::from_static("column_name"), num_columns);
        let mut repr_ca = StringChunkedBuilder::new(PlSmallStr::from_static("repr"), num_columns);
        let mut sorted_asc_ca =
            BooleanChunkedBuilder::new(PlSmallStr::from_static("sorted_asc"), num_columns);
        let mut sorted_dsc_ca =
            BooleanChunkedBuilder::new(PlSmallStr::from_static("sorted_dsc"), num_columns);
        let mut fast_explode_list_ca =
            BooleanChunkedBuilder::new(PlSmallStr::from_static("fast_explode_list"), num_columns);
        let mut materialized_at_ca =
            StringChunkedBuilder::new(PlSmallStr::from_static("materialized_at"), num_columns);

        for col in &self.columns {
            let flags = col.get_flags();

            let (repr, materialized_at) = match col {
                Column::Series(s) => ("series", s.materialized_at()),
                Column::Partitioned(_) => ("partitioned", None),
                Column::Scalar(_) => ("scalar", None),
            };
            let sorted_asc = flags.contains(StatisticsFlags::IS_SORTED_ASC);
            let sorted_dsc = flags.contains(StatisticsFlags::IS_SORTED_DSC);
            let fast_explode_list = flags.contains(StatisticsFlags::CAN_FAST_EXPLODE_LIST);

            column_names.append_value(col.name().clone());
            repr_ca.append_value(repr);
            sorted_asc_ca.append_value(sorted_asc);
            sorted_dsc_ca.append_value(sorted_dsc);
            fast_explode_list_ca.append_value(fast_explode_list);
            materialized_at_ca.append_option(materialized_at.map(|v| format!("{v:#?}")));
        }

        unsafe {
            DataFrame::new_no_checks(
                self.width(),
                vec![
                    column_names.finish().into_column(),
                    repr_ca.finish().into_column(),
                    sorted_asc_ca.finish().into_column(),
                    sorted_dsc_ca.finish().into_column(),
                    fast_explode_list_ca.finish().into_column(),
                    materialized_at_ca.finish().into_column(),
                ],
            )
        }
    }

    /// Return a sorted clone of this [`DataFrame`].
    ///
    /// In many cases the output chunks will be continuous in memory but this is not guaranteed
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
    ///         ["sepal_width", "sepal_length"],
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
        by: impl IntoVec<PlSmallStr>,
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
    /// let mut df: DataFrame = df!("Country" => ["United States", "China"],
    ///                         "Area (km²)" => [9_833_520, 9_596_961])?;
    /// let s: Series = Series::new("Country".into(), ["USA", "PRC"]);
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
        column: PlSmallStr,
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
    /// let s0 = Series::new("foo".into(), ["ham", "spam", "egg"]);
    /// let s1 = Series::new("ascii".into(), [70, 79, 79]);
    /// let mut df = DataFrame::new(vec![s0, s1])?;
    ///
    /// // Add 32 to get lowercase ascii values
    /// df.replace_column(1, df.select_at_idx(1).unwrap() + 32);
    /// # Ok::<(), PolarsError>(())
    /// ```
    pub fn replace_column<C: IntoColumn>(
        &mut self,
        index: usize,
        new_column: C,
    ) -> PolarsResult<&mut Self> {
        polars_ensure!(
            index < self.width(),
            ShapeMismatch:
            "unable to replace at index {}, the DataFrame has only {} columns",
            index, self.width(),
        );
        let mut new_column = new_column.into_column();
        polars_ensure!(
            new_column.len() == self.height(),
            ShapeMismatch:
            "unable to replace a column, series length {} doesn't match the DataFrame height {}",
            new_column.len(), self.height(),
        );
        let old_col = &mut self.columns[index];
        mem::swap(old_col, &mut new_column);
        self.clear_schema();
        Ok(self)
    }

    /// Apply a closure to a column. This is the recommended way to do in place modification.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let s0 = Column::new("foo".into(), ["ham", "spam", "egg"]);
    /// let s1 = Column::new("names".into(), ["Jean", "Claude", "van"]);
    /// let mut df = DataFrame::new(vec![s0, s1])?;
    ///
    /// fn str_to_len(str_val: &Column) -> Column {
    ///     str_val.str()
    ///         .unwrap()
    ///         .into_iter()
    ///         .map(|opt_name: Option<&str>| {
    ///             opt_name.map(|name: &str| name.len() as u32)
    ///          })
    ///         .collect::<UInt32Chunked>()
    ///         .into_column()
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
    pub fn apply<F, C>(&mut self, name: &str, f: F) -> PolarsResult<&mut Self>
    where
        F: FnOnce(&Column) -> C,
        C: IntoColumn,
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
    /// let s0 = Column::new("foo".into(), ["ham", "spam", "egg"]);
    /// let s1 = Column::new("ascii".into(), [70, 79, 79]);
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
    pub fn apply_at_idx<F, C>(&mut self, idx: usize, f: F) -> PolarsResult<&mut Self>
    where
        F: FnOnce(&Column) -> C,
        C: IntoColumn,
    {
        let df_height = self.height();
        let width = self.width();
        let col = self.columns.get_mut(idx).ok_or_else(|| {
            polars_err!(
                ComputeError: "invalid column index: {} for a DataFrame with {} columns",
                idx, width
            )
        })?;
        let name = col.name().clone();
        let new_col = f(col).into_column();
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
            col.rename(name);
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
    /// let s0 = Column::new("foo".into(), ["ham", "spam", "egg", "bacon", "quack"]);
    /// let s1 = Column::new("values".into(), [1, 2, 3, 4, 5]);
    /// let mut df = DataFrame::new(vec![s0, s1])?;
    ///
    /// let idx = vec![0, 1, 4];
    ///
    /// df.try_apply("foo", |c| {
    ///     c.str()?
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
    pub fn try_apply_at_idx<F, C>(&mut self, idx: usize, f: F) -> PolarsResult<&mut Self>
    where
        F: FnOnce(&Column) -> PolarsResult<C>,
        C: IntoColumn,
    {
        let width = self.width();
        let col = self.columns.get_mut(idx).ok_or_else(|| {
            polars_err!(
                ComputeError: "invalid column index: {} for a DataFrame with {} columns",
                idx, width
            )
        })?;
        let name = col.name().clone();

        let _ = mem::replace(col, f(col).map(|c| c.into_column())?);

        // make sure the name remains the same after applying the closure
        unsafe {
            let col = self.columns.get_unchecked_mut(idx);
            col.rename(name);
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
    /// let s0 = Column::new("foo".into(), ["ham", "spam", "egg", "bacon", "quack"]);
    /// let s1 = Column::new("values".into(), [1, 2, 3, 4, 5]);
    /// let mut df = DataFrame::new(vec![s0, s1])?;
    ///
    /// // create a mask
    /// let values = df.column("values")?.as_materialized_series();
    /// let mask = values.lt_eq(1)? | values.gt_eq(5_i32)?;
    ///
    /// df.try_apply("foo", |c| {
    ///     c.str()?
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
    pub fn try_apply<F, C>(&mut self, column: &str, f: F) -> PolarsResult<&mut Self>
    where
        F: FnOnce(&Series) -> PolarsResult<C>,
        C: IntoColumn,
    {
        let idx = self.try_get_column_index(column)?;
        self.try_apply_at_idx(idx, |c| f(c.as_materialized_series()))
    }

    /// Slice the [`DataFrame`] along the rows.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let df: DataFrame = df!("Fruit" => ["Apple", "Grape", "Grape", "Fig", "Fig"],
    ///                         "Color" => ["Green", "Red", "White", "White", "Red"])?;
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

        let height = if let Some(fst) = col.first() {
            fst.len()
        } else {
            let (_, length) = slice_offsets(offset, length, self.height());
            length
        };

        unsafe { DataFrame::new_no_checks(height, col) }
    }

    /// Split [`DataFrame`] at the given `offset`.
    pub fn split_at(&self, offset: i64) -> (Self, Self) {
        let (a, b) = self.columns.iter().map(|s| s.split_at(offset)).unzip();

        let (idx, _) = slice_offsets(offset, 0, self.height());

        let a = unsafe { DataFrame::new_no_checks(idx, a) };
        let b = unsafe { DataFrame::new_no_checks(self.height() - idx, b) };
        (a, b)
    }

    pub fn clear(&self) -> Self {
        let col = self.columns.iter().map(|s| s.clear()).collect::<Vec<_>>();
        unsafe { DataFrame::new_no_checks(0, col) }
    }

    #[must_use]
    pub fn slice_par(&self, offset: i64, length: usize) -> Self {
        if offset == 0 && length == self.height() {
            return self.clone();
        }
        let columns = self._apply_columns_par(&|s| s.slice(offset, length));
        unsafe { DataFrame::new_no_checks(length, columns) }
    }

    #[must_use]
    pub fn _slice_and_realloc(&self, offset: i64, length: usize) -> Self {
        if offset == 0 && length == self.height() {
            return self.clone();
        }
        // @scalar-opt
        let columns = self._apply_columns(&|s| {
            let mut out = s.slice(offset, length);
            out.shrink_to_fit();
            out
        });
        unsafe { DataFrame::new_no_checks(length, columns) }
    }

    /// Get the head of the [`DataFrame`].
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let countries: DataFrame =
    ///     df!("Rank by GDP (2021)" => [1, 2, 3, 4, 5],
    ///         "Continent" => ["North America", "Asia", "Asia", "Europe", "Europe"],
    ///         "Country" => ["United States", "China", "Japan", "Germany", "United Kingdom"],
    ///         "Capital" => ["Washington", "Beijing", "Tokyo", "Berlin", "London"])?;
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
            .map(|c| c.head(length))
            .collect::<Vec<_>>();

        let height = length.unwrap_or(HEAD_DEFAULT_LENGTH);
        let height = usize::min(height, self.height());
        unsafe { DataFrame::new_no_checks(height, col) }
    }

    /// Get the tail of the [`DataFrame`].
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let countries: DataFrame =
    ///     df!("Rank (2021)" => [105, 106, 107, 108, 109],
    ///         "Apple Price (€/kg)" => [0.75, 0.70, 0.70, 0.65, 0.52],
    ///         "Country" => ["Kosovo", "Moldova", "North Macedonia", "Syria", "Turkey"])?;
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
            .map(|c| c.tail(length))
            .collect::<Vec<_>>();

        let height = length.unwrap_or(TAIL_DEFAULT_LENGTH);
        let height = usize::min(height, self.height());
        unsafe { DataFrame::new_no_checks(height, col) }
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
        debug_assert!(!self.should_rechunk(), "expected equal chunks");
        // If any of the columns is binview and we don't convert `compat_level` we allow parallelism
        // as we must allocate arrow strings/binaries.
        let must_convert = compat_level.0 == 0;
        let parallel = parallel
            && must_convert
            && self.columns.len() > 1
            && self
                .columns
                .iter()
                .any(|s| matches!(s.dtype(), DataType::String | DataType::Binary));

        RecordBatchIter {
            columns: &self.columns,
            schema: Arc::new(
                self.columns
                    .iter()
                    .map(|c| c.field().to_arrow(compat_level))
                    .collect(),
            ),
            idx: 0,
            n_chunks: self.first_col_n_chunks(),
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
            schema: Arc::new(
                self.get_columns()
                    .iter()
                    .map(|c| c.field().to_arrow(CompatLevel::newest()))
                    .collect(),
            ),
            arr_iters: self
                .materialized_column_iter()
                .map(|s| s.chunks().iter())
                .collect(),
        }
    }

    /// Get a [`DataFrame`] with all the columns in reversed order.
    #[must_use]
    pub fn reverse(&self) -> Self {
        let col = self.columns.iter().map(|s| s.reverse()).collect::<Vec<_>>();
        unsafe { DataFrame::new_no_checks(self.height(), col) }
    }

    /// Shift the values by a given period and fill the parts that will be empty due to this operation
    /// with `Nones`.
    ///
    /// See the method on [Series](crate::series::SeriesTrait::shift) for more info on the `shift` operation.
    #[must_use]
    pub fn shift(&self, periods: i64) -> Self {
        let col = self._apply_columns_par(&|s| s.shift(periods));
        unsafe { DataFrame::new_no_checks(self.height(), col) }
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

        Ok(unsafe { DataFrame::new_no_checks(self.height(), col) })
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
        self.unique_impl(
            true,
            subset.map(|v| v.iter().map(|x| PlSmallStr::from_str(x.as_str())).collect()),
            keep,
            slice,
        )
    }

    /// Unstable distinct. See [`DataFrame::unique_stable`].
    #[cfg(feature = "algorithm_group_by")]
    pub fn unique<I, S>(
        &self,
        subset: Option<&[String]>,
        keep: UniqueKeepStrategy,
        slice: Option<(i64, usize)>,
    ) -> PolarsResult<DataFrame> {
        self.unique_impl(
            false,
            subset.map(|v| v.iter().map(|x| PlSmallStr::from_str(x.as_str())).collect()),
            keep,
            slice,
        )
    }

    #[cfg(feature = "algorithm_group_by")]
    pub fn unique_impl(
        &self,
        maintain_order: bool,
        subset: Option<Vec<PlSmallStr>>,
        keep: UniqueKeepStrategy,
        slice: Option<(i64, usize)>,
    ) -> PolarsResult<Self> {
        let names = subset.unwrap_or_else(|| self.get_column_names_owned());
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

        let height = Self::infer_height(&columns);
        Ok(unsafe { DataFrame::new_no_checks(height, columns) })
    }

    /// Get a mask of all the unique rows in the [`DataFrame`].
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use polars_core::prelude::*;
    /// let df: DataFrame = df!("Company" => ["Apple", "Microsoft"],
    ///                         "ISIN" => ["US0378331005", "US5949181045"])?;
    /// let ca: ChunkedArray<BooleanType> = df.is_unique()?;
    ///
    /// assert!(ca.all());
    /// # Ok::<(), PolarsError>(())
    /// ```
    #[cfg(feature = "algorithm_group_by")]
    pub fn is_unique(&self) -> PolarsResult<BooleanChunked> {
        let gb = self.group_by(self.get_column_names_owned())?;
        let groups = gb.get_groups();
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
    /// let df: DataFrame = df!("Company" => ["Alphabet", "Alphabet"],
    ///                         "ISIN" => ["US02079K3059", "US02079K1079"])?;
    /// let ca: ChunkedArray<BooleanType> = df.is_duplicated()?;
    ///
    /// assert!(!ca.all());
    /// # Ok::<(), PolarsError>(())
    /// ```
    #[cfg(feature = "algorithm_group_by")]
    pub fn is_duplicated(&self) -> PolarsResult<BooleanChunked> {
        let gb = self.group_by(self.get_column_names_owned())?;
        let groups = gb.get_groups();
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
            .map(|c| Column::new(c.name().clone(), [c.null_count() as IdxSize]))
            .collect();
        unsafe { Self::new_no_checks(1, cols) }
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
        let mut ca = IdxCa::mmap_slice(PlSmallStr::EMPTY, idx);
        ca.set_sorted_flag(sorted);
        self.take_unchecked_impl(&ca, allow_threads)
    }

    #[cfg(all(feature = "partition_by", feature = "algorithm_group_by"))]
    #[doc(hidden)]
    pub fn _partition_by_impl(
        &self,
        cols: &[PlSmallStr],
        stable: bool,
        include_key: bool,
    ) -> PolarsResult<Vec<DataFrame>> {
        let groups = if stable {
            self.group_by_stable(cols.iter().cloned())?.take_groups()
        } else {
            self.group_by(cols.iter().cloned())?.take_groups()
        };

        // drop key columns prior to calculation if requested
        let df = if include_key {
            self.clone()
        } else {
            self.drop_many(cols.iter().cloned())
        };

        // don't parallelize this
        // there is a lot of parallelization in take and this may easily SO
        POOL.install(|| {
            match groups.as_ref() {
                GroupsType::Idx(idx) => {
                    // Rechunk as the gather may rechunk for every group #17562.
                    let mut df = df.clone();
                    df.as_single_chunk_par();
                    Ok(idx
                        .into_par_iter()
                        .map(|(_, group)| {
                            // groups are in bounds
                            unsafe {
                                df._take_unchecked_slice_sorted(group, false, IsSorted::Ascending)
                            }
                        })
                        .collect())
                },
                GroupsType::Slice { groups, .. } => Ok(groups
                    .into_par_iter()
                    .map(|[first, len]| df.slice(*first as i64, *len as usize))
                    .collect()),
            }
        })
    }

    /// Split into multiple DataFrames partitioned by groups
    #[cfg(feature = "partition_by")]
    pub fn partition_by<I, S>(&self, cols: I, include_key: bool) -> PolarsResult<Vec<DataFrame>>
    where
        I: IntoIterator<Item = S>,
        S: Into<PlSmallStr>,
    {
        let cols = cols
            .into_iter()
            .map(Into::into)
            .collect::<Vec<PlSmallStr>>();
        self._partition_by_impl(cols.as_slice(), false, include_key)
    }

    /// Split into multiple DataFrames partitioned by groups
    /// Order of the groups are maintained.
    #[cfg(feature = "partition_by")]
    pub fn partition_by_stable<I, S>(
        &self,
        cols: I,
        include_key: bool,
    ) -> PolarsResult<Vec<DataFrame>>
    where
        I: IntoIterator<Item = S>,
        S: Into<PlSmallStr>,
    {
        let cols = cols
            .into_iter()
            .map(Into::into)
            .collect::<Vec<PlSmallStr>>();
        self._partition_by_impl(cols.as_slice(), true, include_key)
    }

    /// Unnest the given `Struct` columns. This means that the fields of the `Struct` type will be
    /// inserted as columns.
    #[cfg(feature = "dtype-struct")]
    pub fn unnest<I: IntoVec<PlSmallStr>>(&self, cols: I) -> PolarsResult<DataFrame> {
        let cols = cols.into_vec();
        self.unnest_impl(cols.into_iter().collect())
    }

    #[cfg(feature = "dtype-struct")]
    fn unnest_impl(&self, cols: PlHashSet<PlSmallStr>) -> PolarsResult<DataFrame> {
        let mut new_cols = Vec::with_capacity(std::cmp::min(self.width() * 2, self.width() + 128));
        let mut count = 0;
        for s in &self.columns {
            if cols.contains(s.name()) {
                let ca = s.struct_()?.clone();
                new_cols.extend(ca.fields_as_series().into_iter().map(Column::from));
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
                    .get(col.as_str())
                    .ok_or_else(|| polars_err!(col_not_found = col))?;
            }
        }
        DataFrame::new(new_cols)
    }

    pub(crate) fn infer_height(cols: &[Column]) -> usize {
        cols.first().map_or(0, Column::len)
    }

    pub fn append_record_batch(&mut self, rb: RecordBatchT<ArrayRef>) -> PolarsResult<()> {
        polars_ensure!(
            rb.arrays().len() == self.width(),
            InvalidOperation: "attempt to extend dataframe of width {} with record batch of width {}",
            self.width(),
            rb.arrays().len(),
        );

        if rb.height() == 0 {
            return Ok(());
        }

        // SAFETY:
        // - we don't adjust the names of the columns
        // - each column gets appended the same number of rows, which is an invariant of
        //   record_batch.
        let columns = unsafe { self.get_columns_mut() };
        for (col, arr) in columns.iter_mut().zip(rb.into_arrays()) {
            let arr_series = Series::from_arrow_chunks(PlSmallStr::EMPTY, vec![arr])?.into_column();
            col.append(&arr_series)?;
        }

        Ok(())
    }
}

pub struct RecordBatchIter<'a> {
    columns: &'a Vec<Column>,
    schema: ArrowSchemaRef,
    idx: usize,
    n_chunks: usize,
    compat_level: CompatLevel,
    parallel: bool,
}

impl Iterator for RecordBatchIter<'_> {
    type Item = RecordBatch;

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx >= self.n_chunks {
            return None;
        }

        // Create a batch of the columns with the same chunk no.
        let batch_cols: Vec<ArrayRef> = if self.parallel {
            let iter = self
                .columns
                .par_iter()
                .map(Column::as_materialized_series)
                .map(|s| s.to_arrow(self.idx, self.compat_level));
            POOL.install(|| iter.collect())
        } else {
            self.columns
                .iter()
                .map(Column::as_materialized_series)
                .map(|s| s.to_arrow(self.idx, self.compat_level))
                .collect()
        };
        self.idx += 1;

        let length = batch_cols.first().map_or(0, |arr| arr.len());
        Some(RecordBatch::new(length, self.schema.clone(), batch_cols))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.n_chunks - self.idx;
        (n, Some(n))
    }
}

pub struct PhysRecordBatchIter<'a> {
    schema: ArrowSchemaRef,
    arr_iters: Vec<std::slice::Iter<'a, ArrayRef>>,
}

impl Iterator for PhysRecordBatchIter<'_> {
    type Item = RecordBatch;

    fn next(&mut self) -> Option<Self::Item> {
        let arrs = self
            .arr_iters
            .iter_mut()
            .map(|phys_iter| phys_iter.next().cloned())
            .collect::<Option<Vec<_>>>()?;

        let length = arrs.first().map_or(0, |arr| arr.len());
        Some(RecordBatch::new(length, self.schema.clone(), arrs))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        if let Some(iter) = self.arr_iters.first() {
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

impl From<DataFrame> for Vec<Column> {
    fn from(df: DataFrame) -> Self {
        df.columns
    }
}

// utility to test if we can vstack/extend the columns
fn ensure_can_extend(left: &Column, right: &Column) -> PolarsResult<()> {
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
        let s0 = Column::new("days".into(), [0, 1, 2].as_ref());
        let s1 = Column::new("temp".into(), [22.1, 19.9, 7.].as_ref());
        DataFrame::new(vec![s0, s1]).unwrap()
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_recordbatch_iterator() {
        let df = df!(
            "foo" => [1, 2, 3, 4, 5]
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
        assert_eq!(
            df.column("days")
                .unwrap()
                .as_series()
                .unwrap()
                .equal(1)
                .unwrap()
                .sum(),
            Some(1)
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_filter_broadcast_on_string_col() {
        let col_name = "some_col";
        let v = vec!["test".to_string()];
        let s0 = Column::new(PlSmallStr::from_str(col_name), v);
        let mut df = DataFrame::new(vec![s0]).unwrap();

        df = df
            .filter(
                &df.column(col_name)
                    .unwrap()
                    .as_materialized_series()
                    .equal("")
                    .unwrap(),
            )
            .unwrap();
        assert_eq!(
            df.column(col_name)
                .unwrap()
                .as_materialized_series()
                .n_chunks(),
            1
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_filter_broadcast_on_list_col() {
        let s1 = Series::new(PlSmallStr::EMPTY, [true, false, true]);
        let ll: ListChunked = [&s1].iter().copied().collect();

        let mask = BooleanChunked::from_slice(PlSmallStr::EMPTY, &[false]);
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
        let mut s = Series::new("foo".into(), 0..2);
        let s2 = Series::new("bar".into(), 0..1);
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
            "foo" => [1, 2, 3]
        }
        .unwrap();
        // check if column is replaced
        assert!(df
            .with_column(Series::new("foo".into(), &[1, 2, 3]))
            .is_ok());
        assert!(df
            .with_column(Series::new("bar".into(), &[1, 2, 3]))
            .is_ok());
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
        assert_eq!(df.first_col_n_chunks(), 2)
    }

    #[test]
    fn test_vstack_on_empty_dataframe() {
        let mut df = DataFrame::empty();

        let df_data = df! {
            "flt" => [1., 1., 2., 2., 3., 3.],
            "int" => [1, 1, 2, 2, 3, 3, ],
            "str" => ["a", "a", "b", "b", "c", "c"]
        }
        .unwrap();

        df.vstack_mut(&df_data).unwrap();
        assert_eq!(df.height, 6)
    }

    #[test]
    fn test_replace_or_add() -> PolarsResult<()> {
        let mut df = df!(
            "a" => [1, 2, 3],
            "b" => [1, 2, 3]
        )?;

        // check that the new column is "c" and not "bar".
        df.replace_or_add("c".into(), Series::new("bar".into(), [1, 2, 3]))?;

        assert_eq!(df.get_column_names(), &["a", "b", "c"]);
        Ok(())
    }
}
