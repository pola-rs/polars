use std::sync::{Arc, OnceLock};

use polars_error::PolarsResult;

use super::broadcast::{broadcast_columns, infer_broadcast_height};
use super::validation::validate_columns_slice;
use crate::frame::column::Column;
use crate::schema::{Schema, SchemaRef};

/// A contiguous growable collection of [`Column`]s that have the same length.
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
/// let df = DataFrame::empty();
/// assert_eq!(df.shape(), (0, 0));
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
/// let df: PolarsResult<DataFrame> = DataFrame::new_infer_height(vec![s1, s2]);
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
    /// All columns must have length equal to `self.height`.
    columns: Vec<Column>,
    /// Cached schema. Must be cleared if column names / dtypes in `self.columns` change.
    cached_schema: OnceLock<SchemaRef>,
}

impl Default for DataFrame {
    fn default() -> Self {
        DataFrame::empty()
    }
}

impl DataFrame {
    /// Creates an empty `DataFrame` usable in a compile time context (such as static initializers).
    ///
    /// # Example
    ///
    /// ```rust
    /// use polars_core::prelude::DataFrame;
    /// static EMPTY: DataFrame = DataFrame::empty();
    /// ```
    pub const fn empty() -> Self {
        DataFrame::empty_with_height(0)
    }

    pub const fn empty_with_height(height: usize) -> Self {
        DataFrame {
            height,
            columns: vec![],
            cached_schema: OnceLock::new(),
        }
    }

    pub fn new(height: usize, columns: Vec<Column>) -> PolarsResult<Self> {
        validate_columns_slice(height, &columns)
            .map_err(|e| e.wrap_msg(|e| format!("could not create a new DataFrame: {e}")))?;

        Ok(unsafe { DataFrame::_new_unchecked_impl(height, columns) })
    }

    /// Height is sourced from first column.
    pub fn new_infer_height(columns: Vec<Column>) -> PolarsResult<Self> {
        DataFrame::new(columns.first().map_or(0, |c| c.len()), columns)
    }

    /// Create a new `DataFrame` but does not check the length or duplicate occurrence of the
    /// [`Column`]s.
    ///
    /// # Safety
    /// [`Column`]s must have unique names and matching lengths.
    pub unsafe fn new_unchecked(height: usize, columns: Vec<Column>) -> DataFrame {
        if cfg!(debug_assertions) {
            validate_columns_slice(height, &columns).unwrap();
        }

        unsafe { DataFrame::_new_unchecked_impl(height, columns) }
    }

    /// Height is sourced from first column. Does not check for matching height / duplicate names.
    ///
    /// # Safety
    /// [`Column`]s must have unique names and matching lengths.
    pub unsafe fn new_unchecked_infer_height(columns: Vec<Column>) -> DataFrame {
        DataFrame::new_unchecked(columns.first().map_or(0, |c| c.len()), columns)
    }

    /// This will not panic even in debug mode - there are some (rare) use cases where a DataFrame
    /// is temporarily constructed containing duplicates for dispatching to functions. A DataFrame
    /// constructed with this method is generally highly unsafe and should not be long-lived.
    #[expect(clippy::missing_safety_doc)]
    pub const unsafe fn _new_unchecked_impl(height: usize, columns: Vec<Column>) -> DataFrame {
        DataFrame {
            height,
            columns,
            cached_schema: OnceLock::new(),
        }
    }

    /// Broadcasts unit-length columns to `height`. Errors if a column has height that is non-unit
    /// length and not equal to `self.height()`.
    pub fn new_with_broadcast(height: usize, mut columns: Vec<Column>) -> PolarsResult<Self> {
        broadcast_columns(height, &mut columns)?;
        DataFrame::new(height, columns)
    }

    /// Infers height as the first non-unit length column or 1 if not found.
    pub fn new_infer_broadcast(columns: Vec<Column>) -> PolarsResult<Self> {
        DataFrame::new_with_broadcast(infer_broadcast_height(&columns), columns)
    }

    /// Broadcasts unit-length columns to `height`. Errors if a column has height that is non-unit
    /// length and not equal to `self.height()`.
    ///
    /// # Safety
    /// [`Column`]s must have unique names.
    pub unsafe fn new_unchecked_with_broadcast(
        height: usize,
        mut columns: Vec<Column>,
    ) -> PolarsResult<Self> {
        broadcast_columns(height, &mut columns)?;
        Ok(unsafe { DataFrame::new_unchecked(height, columns) })
    }

    /// # Safety
    /// [`Column`]s must have unique names.
    pub unsafe fn new_unchecked_infer_broadcast(columns: Vec<Column>) -> PolarsResult<Self> {
        DataFrame::new_unchecked_with_broadcast(infer_broadcast_height(&columns), columns)
    }

    /// Create a `DataFrame` 0 height and columns as per the `schema`.
    pub fn empty_with_schema(schema: &Schema) -> Self {
        let cols = schema
            .iter()
            .map(|(name, dtype)| Column::new_empty(name.clone(), dtype))
            .collect();

        unsafe { DataFrame::_new_unchecked_impl(0, cols) }
    }

    /// Create an empty `DataFrame` with empty columns as per the `schema`.
    pub fn empty_with_arc_schema(schema: SchemaRef) -> Self {
        let mut df = DataFrame::empty_with_schema(&schema);
        unsafe { df.set_schema(schema) };
        df
    }

    /// Set the height (i.e. number of rows) of this [`DataFrame`].
    ///
    /// # Safety
    ///
    /// This needs to be equal to the length of all the columns, or `self.width()` must be 0.
    #[inline]
    pub unsafe fn set_height(&mut self, height: usize) -> &mut Self {
        self.height = height;
        self
    }

    /// Get the height of the [`DataFrame`] which is the number of rows.
    #[inline]
    pub fn height(&self) -> usize {
        self.height
    }

    /// Get the number of columns in this [`DataFrame`].
    #[inline]
    pub fn width(&self) -> usize {
        self.columns.len()
    }

    /// Get (height, width) of the [`DataFrame`].
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let df0: DataFrame = DataFrame::empty();
    /// let df1: DataFrame = df!("1" => [1, 2, 3, 4, 5])?;
    /// let df2: DataFrame = df!("1" => [1, 2, 3, 4, 5],
    ///                          "2" => [1, 2, 3, 4, 5])?;
    ///
    /// assert_eq!(df0.shape(), (0 ,0));
    /// assert_eq!(df1.shape(), (5, 1));
    /// assert_eq!(df2.shape(), (5, 2));
    /// # Ok::<(), PolarsError>(())
    /// ```
    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        (self.height(), self.width())
    }

    /// 0 width or height.
    #[inline]
    pub fn shape_has_zero(&self) -> bool {
        matches!(self.shape(), (0, _) | (_, 0))
    }

    #[inline]
    pub fn columns(&self) -> &[Column] {
        self.columns.as_slice()
    }

    #[inline]
    pub fn into_columns(self) -> Vec<Column> {
        self.columns
    }

    /// # Safety
    ///
    /// The caller must ensure the length of all [`Column`]s remains equal to `self.height`, or
    /// that [`DataFrame::set_height`] is called afterwards with the new `height`.
    #[inline]
    pub unsafe fn columns_mut(&mut self) -> &mut Vec<Column> {
        self.clear_schema();
        &mut self.columns
    }

    /// # Safety
    /// Adheres to all safety requirements of [`DataFrame::columns_mut`], and that the list of column
    /// names remains unchanged.
    #[inline]
    pub unsafe fn columns_mut_retain_schema(&mut self) -> &mut Vec<Column> {
        &mut self.columns
    }

    /// Get the schema of this [`DataFrame`].
    ///
    /// # Panics
    /// Panics if there are duplicate column names.
    pub fn schema(&self) -> &SchemaRef {
        let out = self.cached_schema.get_or_init(|| {
            Arc::new(
                Schema::from_iter_check_duplicates(
                    self.columns
                        .iter()
                        .map(|x| (x.name().clone(), x.dtype().clone())),
                )
                .unwrap(),
            )
        });

        assert_eq!(out.len(), self.width());

        out
    }

    #[inline]
    pub fn cached_schema(&self) -> Option<&SchemaRef> {
        self.cached_schema.get()
    }

    /// Set the cached schema
    ///
    /// # Safety
    /// Schema must match the columns in `self`.
    #[inline]
    pub unsafe fn set_schema(&mut self, schema: SchemaRef) -> &mut Self {
        self.cached_schema = schema.into();
        self
    }

    /// Set the cached schema
    ///
    /// # Safety
    /// Schema must match the columns in `self`.
    #[inline]
    pub unsafe fn with_schema(mut self, schema: SchemaRef) -> Self {
        self.cached_schema = schema.into();
        self
    }

    /// Set the cached schema if `schema` is `Some()`.
    ///
    /// # Safety
    /// Schema must match the columns in `self`.
    #[inline]
    pub unsafe fn set_opt_schema(&mut self, schema: Option<SchemaRef>) -> &mut Self {
        if let Some(schema) = schema {
            unsafe { self.set_schema(schema) };
        }

        self
    }

    /// Clones the cached schema from `from` to `self.cached_schema` if there is one.
    ///
    /// # Safety
    /// Schema must match the columns in `self`.
    #[inline]
    pub unsafe fn set_schema_from(&mut self, from: &DataFrame) -> &mut Self {
        self.set_opt_schema(from.cached_schema().cloned());
        self
    }

    /// Clones the cached schema from `from` to `self.cached_schema` if there is one.
    ///
    /// # Safety
    /// Schema must match the columns in `self`.
    #[inline]
    pub unsafe fn with_schema_from(mut self, from: &DataFrame) -> Self {
        self.set_opt_schema(from.cached_schema().cloned());
        self
    }

    #[inline]
    fn clear_schema(&mut self) -> &mut Self {
        self.cached_schema = OnceLock::new();
        self
    }
}
