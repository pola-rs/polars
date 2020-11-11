//! DataFrame module.
use crate::chunked_array::ops::unique::is_unique_helper;
use crate::frame::select::Selection;
use crate::prelude::*;
use arrow::datatypes::{Field, Schema};
use arrow::record_batch::RecordBatch;
use rayon::prelude::*;
use std::marker::Sized;
use std::mem;
use std::sync::Arc;

pub mod explode;
pub mod group_by;
pub mod hash_join;
pub mod select;
pub mod ser;
mod upstream_traits;

pub trait IntoSeries {
    fn into_series(self) -> Series
    where
        Self: Sized;
}

impl IntoSeries for Series {
    fn into_series(self) -> Series {
        self
    }
}

impl<T: PolarsDataType> IntoSeries for ChunkedArray<T> {
    fn into_series(self) -> Series {
        Series::from_chunked_array(self)
    }
}

impl Default for DataFrame {
    fn default() -> Self {
        DataFrame::new_no_checks(Vec::with_capacity(0))
    }
}

type DfSchema = Arc<Schema>;
type DfSeries = Series;
type DfColumns = Vec<DfSeries>;

#[derive(Clone)]
pub struct DataFrame {
    columns: DfColumns,
}

impl DataFrame {
    /// Get the index of the column.
    fn name_to_idx(&self, name: &str) -> Result<usize> {
        let mut idx = 0;
        for column in &self.columns {
            if column.name() == name {
                break;
            }
            idx += 1;
        }
        if idx == self.columns.len() {
            Err(PolarsError::NotFound(name.into()))
        } else {
            Ok(idx)
        }
    }

    /// Create a DataFrame from a Vector of Series.
    ///
    /// # Example
    ///
    /// ```
    /// use polars::prelude::*;
    /// let s0 = Series::new("days", [0, 1, 2].as_ref());
    /// let s1 = Series::new("temp", [22.1, 19.9, 7.].as_ref());
    /// let df = DataFrame::new(vec![s0, s1]).unwrap();
    /// ```
    pub fn new<S: IntoSeries>(columns: Vec<S>) -> Result<Self> {
        let mut first_len = None;
        let mut series_cols = Vec::with_capacity(columns.len());

        // check for series length equality and convert into series in one pass
        for s in columns {
            let series = s.into_series();
            match first_len {
                Some(len) => {
                    if series.len() != len {
                        return Err(PolarsError::ShapeMisMatch("Could not create a new DataFrame from Series. The Series have different lengths".into()));
                    }
                }
                None => first_len = Some(series.len()),
            }
            series_cols.push(series)
        }
        let mut df = DataFrame {
            columns: series_cols,
        };
        df.rechunk()?;
        Ok(df)
    }

    // doesn't check Series sizes.
    pub(crate) fn new_no_checks(columns: Vec<Series>) -> DataFrame {
        DataFrame { columns }
    }

    /// Aggregate all chunks to contiguous memory.
    pub fn agg_chunks(&self) -> Self {
        let cols = self
            .columns
            .par_iter()
            .map(|s| s.rechunk(Some(&[1])).expect("can always rechunk to single"))
            .collect();
        DataFrame::new_no_checks(cols)
    }

    /// Ensure all the chunks in the DataFrame are aligned.
    fn rechunk(&mut self) -> Result<&mut Self> {
        let mut all_equal = true;

        let mut it = self.columns.iter();
        let first_s = it
            .next()
            .ok_or(PolarsError::NoData("no data to rechunk".into()))?;
        let id = first_s.chunk_lengths();

        while let Some(s) = it.next() {
            let current_id = s.chunk_lengths();
            if current_id != id {
                all_equal = false;
                break;
            }
        }

        // fast path
        if all_equal {
            Ok(self)
        } else {
            self.columns = self
                .columns
                .iter()
                .map(|s| {
                    s.rechunk(Some(&[1]))
                        .expect("can always aggregate to single chunk")
                })
                .collect();
            Ok(self)
        }
    }

    /// Get a reference to the DataFrame schema.
    pub fn schema(&self) -> Schema {
        let fields = Self::create_fields(&self.columns);
        Schema::new(fields)
    }

    /// Get a reference to the DataFrame columns.
    #[inline]
    pub fn get_columns(&self) -> &DfColumns {
        &self.columns
    }

    /// Get the column labels of the DataFrame.
    #[deprecated(since = "0.9.0", note = "please use get_column_names")]
    pub fn columns(&self) -> Vec<&str> {
        self.get_column_names()
    }

    pub fn get_column_names(&self) -> Vec<&str> {
        self.columns.iter().map(|s| s.name()).collect()
    }

    /// Set the column names.
    pub fn set_column_names<S: AsRef<str>>(&mut self, names: &[S]) -> Result<()> {
        if names.len() != self.columns.len() {
            return Err(PolarsError::ShapeMisMatch("the provided slice with column names has not the same size as the DataFrame's width".into()));
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
    pub fn dtypes(&self) -> Vec<ArrowDataType> {
        self.columns.iter().map(|s| s.dtype().clone()).collect()
    }

    /// The number of chunks per column
    pub fn n_chunks(&self) -> Result<usize> {
        Ok(self
            .columns
            .get(0)
            .ok_or(PolarsError::NoData(
                "Can not determine number of chunks if there is no data".into(),
            ))?
            .chunks()
            .len())
    }

    /// Get fields from the columns.
    fn create_fields(columns: &DfColumns) -> Vec<Field> {
        columns.iter().map(|s| s.field().clone()).collect()
    }

    /// This method should be called after every mutable addition/ deletion of columns
    fn register_mutation(&mut self) -> Result<()> {
        self.rechunk()?;
        Ok(())
    }

    /// Get a reference to the schema fields of the DataFrame.
    pub fn fields(&self) -> Vec<Field> {
        self.columns.iter().map(|s| s.field().clone()).collect()
    }

    /// Get (width x height)
    ///
    /// # Example
    ///
    /// ```
    /// use polars::prelude::*;
    /// fn assert_shape(df: &DataFrame, shape: (usize, usize)) {
    ///     assert_eq!(df.shape(), shape)
    /// }
    /// ```
    pub fn shape(&self) -> (usize, usize) {
        let columns = self.columns.len();
        if columns > 0 {
            (self.columns[0].len(), columns)
        } else {
            (0, 0)
        }
    }

    /// Get width of DataFrame
    ///
    /// # Example
    ///
    /// ```
    /// use polars::prelude::*;
    /// fn assert_width(df: &DataFrame, width: usize) {
    ///     assert_eq!(df.width(), width)
    /// }
    /// ```
    pub fn width(&self) -> usize {
        self.columns.len()
    }

    /// Get height of DataFrame
    ///
    /// # Example
    ///
    /// ```
    /// use polars::prelude::*;
    /// fn assert_height(df: &DataFrame, height: usize) {
    ///     assert_eq!(df.height(), height)
    /// }
    /// ```
    pub fn height(&self) -> usize {
        self.shape().0
    }

    /// Add multiple Series to a DataFrame
    /// This expects the Series to have the same length.
    ///
    /// # Example
    ///
    /// ```
    /// use polars::prelude::*;
    /// fn stack(df: &mut DataFrame, columns: &[Series]) {
    ///     df.hstack(columns);
    /// }
    /// ```
    pub fn hstack(&mut self, columns: &[DfSeries]) -> Result<&mut Self> {
        let height = self.height();
        for col in columns {
            if col.len() != height {
                return Err(PolarsError::ShapeMisMatch(
                    format!("Could not horizontally stack Series. The Series length {} differs from the DataFrame height: {}", col.len(), height).into()));
            } else {
                self.columns.push(col.clone());
            }
        }
        self.register_mutation()?;
        Ok(self)
    }

    /// Concatenate a DataFrame to this DataFrame
    pub fn vstack(&mut self, df: &DataFrame) -> Result<&mut Self> {
        if self.width() != df.width() {
            return Err(PolarsError::ShapeMisMatch(
                format!("Could not vertically stack DataFrame. The DataFrames appended width {} differs from the parent DataFrames width {}", self.width(), df.width()).into()
            ));
        }

        if self.dtypes() != df.dtypes() {
            return Err(PolarsError::DataTypeMisMatch(
                format!(
                    "cannot vstack: data types don't match of {:?} {:?}",
                    self.head(Some(2)),
                    df.head(Some(2))
                )
                .into(),
            ));
        }
        self.columns
            .iter_mut()
            .zip(df.columns.iter())
            .for_each(|(left, right)| {
                left.append(right).expect("should not fail");
            });
        self.register_mutation()?;
        Ok(self)
    }

    /// Remove column by name
    ///
    /// # Example
    ///
    /// ```
    /// use polars::prelude::*;
    /// fn drop_column(df: &mut DataFrame, name: &str) -> Result<Series> {
    ///     df.drop_in_place(name)
    /// }
    /// ```
    pub fn drop_in_place(&mut self, name: &str) -> Result<DfSeries> {
        let idx = self.name_to_idx(name)?;
        let result = Ok(self.columns.remove(idx));
        self.register_mutation()?;
        result
    }

    /// Return a new DataFrame where all null values are dropped
    pub fn drop_nulls(&self) -> Result<Self> {
        let mut iter = self.columns.iter();
        let mask = iter
            .next()
            .ok_or(PolarsError::NoData("No data to drop nulls from".into()))?;
        let mut mask = mask.is_not_null();

        while let Some(s) = iter.next() {
            mask = mask & s.is_not_null();
        }
        self.filter(&mask)
    }

    /// Drop a column by name.
    /// This is a pure method and will return a new DataFrame instead of modifying
    /// the current one in place.
    pub fn drop(&self, name: &str) -> Result<Self> {
        let idx = self.name_to_idx(name)?;
        let mut new_cols = Vec::with_capacity(self.columns.len() - 1);

        self.columns.iter().enumerate().for_each(|(i, s)| {
            if i != idx {
                new_cols.push(s.clone())
            }
        });

        Ok(DataFrame::new_no_checks(new_cols))
    }

    /// Insert a new column at a given index
    pub fn insert_at_idx<S: IntoSeries>(&mut self, index: usize, column: S) -> Result<&mut Self> {
        let series = column.into_series();
        if series.len() == self.height() {
            self.columns.insert(index, series);
            Ok(self)
        } else {
            Err(PolarsError::ShapeMisMatch(
                format!(
                    "Could add column. The Series length {} differs from the DataFrame height: {}",
                    series.len(),
                    self.height()
                )
                .into(),
            ))
        }
    }

    /// Add a new column to this `DataFrame`.
    pub fn add_column<S: IntoSeries>(&mut self, column: S) -> Result<&mut Self> {
        let series = column.into_series();
        if series.len() == self.height() {
            self.columns.push(series);
            Ok(self)
        } else {
            Err(PolarsError::ShapeMisMatch(
                format!(
                    "Could add column. The Series length {} differs from the DataFrame height: {}",
                    series.len(),
                    self.height()
                )
                .into(),
            ))
        }
    }

    /// Create a new `DataFrame` with the column added.
    pub fn with_column<S: IntoSeries>(&self, column: S) -> Result<Self> {
        let mut df = self.clone();
        df.add_column(column)?;
        Ok(df)
    }

    /// Get a row in the `DataFrame` Beware this is slow.
    ///
    /// # Example
    ///
    /// ```
    /// use polars::prelude::*;
    /// fn example(df: &mut DataFrame, idx: usize) -> Option<Vec<AnyType>> {
    ///     df.get(idx)
    /// }
    /// ```
    pub fn get(&self, idx: usize) -> Option<Vec<AnyType>> {
        match self.columns.get(0) {
            Some(s) => {
                if s.len() <= idx {
                    return None;
                }
            }
            None => return None,
        }
        Some(self.columns.iter().map(|s| s.get(idx)).collect())
    }

    /// Select a series by index.
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

    /// Get column index of a series by name.
    pub fn find_idx_by_name(&self, name: &str) -> Option<usize> {
        self.columns
            .iter()
            .enumerate()
            .filter(|(_idx, series)| series.name() == name)
            .map(|(idx, _)| idx)
            .next()
    }

    /// Select a single column by name.
    pub fn column(&self, name: &str) -> Result<&Series> {
        let idx = self
            .find_idx_by_name(name)
            .ok_or(PolarsError::NotFound(name.into()))?;
        Ok(self.select_at_idx(idx).unwrap())
    }

    /// Select column(s) from this DataFrame and return a new DataFrame.
    ///
    /// # Examples
    ///
    /// ```
    /// use polars::prelude::*;
    ///
    /// fn example(df: &DataFrame, possible: &str) -> Result<DataFrame> {
    ///     match possible {
    ///         "by_str" => df.select("my-column"),
    ///         "by_tuple" => df.select(("col_1", "col_2")),
    ///         "by_vec" => df.select(vec!["col_a", "col_b"]),
    ///          _ => unimplemented!()
    ///     }
    /// }
    /// ```
    pub fn select<'a, S, J>(&self, selection: S) -> Result<Self>
    where
        S: Selection<'a, J>,
    {
        let selected = self.select_series(selection)?;
        let df = DataFrame::new_no_checks(selected);
        Ok(df)
    }

    /// Select column(s) from this DataFrame and return them into a Vector.
    pub fn select_series<'a, S, J>(&self, selection: S) -> Result<Vec<Series>>
    where
        S: Selection<'a, J>,
    {
        let cols = selection.to_selection_vec();
        let selected = cols
            .iter()
            .map(|c| self.column(c).map(|s| s.clone()))
            .collect::<Result<Vec<_>>>()?;
        Ok(selected)
    }

    /// Select a mutable series by name.
    /// *Note: the length of the Series should remain the same otherwise the DataFrame is invalid.*
    /// For this reason the method is not public
    fn select_mut(&mut self, name: &str) -> Option<&mut Series> {
        let opt_idx = self.find_idx_by_name(name);

        match opt_idx {
            Some(idx) => self.select_at_idx_mut(idx),
            None => None,
        }
    }

    /// Take DataFrame rows by a boolean mask.
    pub fn filter(&self, mask: &BooleanChunked) -> Result<Self> {
        let new_col = self
            .columns
            .par_iter()
            .map(|col| col.filter(mask))
            .collect::<Result<Vec<_>>>()?;
        Ok(DataFrame::new_no_checks(new_col))
    }

    /// Take DataFrame value by indexes from an iterator.
    ///
    /// # Example
    ///
    /// ```
    /// use polars::prelude::*;
    /// fn example(df: &DataFrame) -> Result<DataFrame> {
    ///     let iterator = (0..9).into_iter();
    ///     df.take_iter(iterator, None)
    /// }
    /// ```
    pub fn take_iter<I>(&self, iter: I, capacity: Option<usize>) -> Result<Self>
    where
        I: Iterator<Item = usize> + Clone + Sync,
    {
        let new_col = self
            .columns
            .par_iter()
            .map(|s| {
                let mut i = iter.clone();
                s.take_iter(&mut i, capacity)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(DataFrame::new_no_checks(new_col))
    }

    /// Take DataFrame values by indexes from an iterator. This doesn't do any bound checking.
    ///
    /// # Example
    ///
    /// ```
    /// use polars::prelude::*;
    /// unsafe fn example(df: &DataFrame) -> DataFrame {
    ///     let iterator = (0..9).into_iter();
    ///     df.take_iter_unchecked(iterator, None)
    /// }
    /// ```
    pub unsafe fn take_iter_unchecked<I>(&self, iter: I, capacity: Option<usize>) -> Self
    where
        I: Iterator<Item = usize> + Clone + Sync,
    {
        let new_col = self
            .columns
            .par_iter()
            .map(|s| {
                let mut i = iter.clone();
                s.take_iter_unchecked(&mut i, capacity)
            })
            .collect::<Vec<_>>();
        DataFrame::new_no_checks(new_col)
    }

    /// Take DataFrame values by indexes from an iterator that may contain None values.
    ///
    /// # Example
    ///
    /// ```
    /// use polars::prelude::*;
    /// fn example(df: &DataFrame) -> Result<DataFrame> {
    ///     let iterator = (0..9).into_iter().map(Some);
    ///     df.take_opt_iter(iterator, None)
    /// }
    /// ```
    pub fn take_opt_iter<I>(&self, iter: I, capacity: Option<usize>) -> Result<Self>
    where
        I: Iterator<Item = Option<usize>> + Clone + Sync,
    {
        let new_col = self
            .columns
            .par_iter()
            .map(|s| {
                let mut i = iter.clone();
                s.take_opt_iter(&mut i, capacity)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(DataFrame::new_no_checks(new_col))
    }

    /// Take DataFrame values by indexes from an iterator that may contain None values.
    /// This doesn't do any bound checking.
    ///
    /// # Example
    ///
    /// ```
    /// use polars::prelude::*;
    /// unsafe fn example(df: &DataFrame) -> DataFrame {
    ///     let iterator = (0..9).into_iter().map(Some);
    ///     df.take_opt_iter_unchecked(iterator, None)
    /// }
    /// ```
    pub unsafe fn take_opt_iter_unchecked<I>(&self, iter: I, capacity: Option<usize>) -> Self
    where
        I: Iterator<Item = Option<usize>> + Clone + Sync,
    {
        let new_col = self
            .columns
            .par_iter()
            .map(|s| {
                let mut i = iter.clone();
                s.take_opt_iter_unchecked(&mut i, capacity)
            })
            .collect::<Vec<_>>();
        DataFrame::new_no_checks(new_col)
    }

    /// Take DataFrame rows by index values.
    ///
    /// # Example
    ///
    /// ```
    /// use polars::prelude::*;
    /// fn example(df: &DataFrame) -> Result<DataFrame> {
    ///     let idx = vec![0, 1, 9];
    ///     df.take(&idx)
    /// }
    /// ```
    pub fn take<T: AsTakeIndex + Sync>(&self, indices: &T) -> Result<Self> {
        let new_col = self
            .columns
            .par_iter()
            .map(|s| s.take(indices))
            .collect::<Result<Vec<_>>>()?;

        Ok(DataFrame::new_no_checks(new_col))
    }

    /// Rename a column in the DataFrame
    ///
    /// # Example
    ///
    /// ```
    /// use polars::prelude::*;
    /// fn example(df: &mut DataFrame) -> Result<&mut DataFrame> {
    ///     let original_name = "foo";
    ///     let new_name = "bar";
    ///     df.rename(original_name, new_name)
    /// }
    /// ```
    pub fn rename(&mut self, column: &str, name: &str) -> Result<&mut Self> {
        self.select_mut(column)
            .ok_or(PolarsError::NotFound(name.to_string()))
            .map(|s| s.rename(name))?;
        Ok(self)
    }

    /// Sort DataFrame in place by a column.
    pub fn sort_in_place(&mut self, by_column: &str, reverse: bool) -> Result<&mut Self> {
        let s = self.column(by_column)?;

        let take = s.argsort(reverse);

        self.columns = self
            .columns
            .par_iter()
            .map(|s| s.take(&take))
            .collect::<Result<Vec<_>>>()?;
        Ok(self)
    }

    /// Return a sorted clone of this DataFrame.
    pub fn sort(&self, by_column: &str, reverse: bool) -> Result<Self> {
        let s = self.column(by_column)?;

        let take = s.argsort(reverse);
        self.take(&take)
    }

    /// Replace a column with a series.
    pub fn replace<S: IntoSeries>(&mut self, column: &str, new_col: S) -> Result<&mut Self> {
        self.apply(column, |_| new_col.into_series())
    }

    /// Replace column at index `idx` with a series.
    ///
    /// # Example
    ///
    /// ```rust
    /// use polars::prelude::*;
    /// let s0 = Series::new("foo", &["ham", "spam", "egg"]);
    /// let s1 = Series::new("ascii", &[70, 79, 79]);
    /// let mut df = DataFrame::new(vec![s0, s1]).unwrap();
    ///
    /// // Add 32 to get lowercase ascii values
    /// df.replace_at_idx(1, df.select_at_idx(1).unwrap() + 32);
    /// ```
    pub fn replace_at_idx<S: IntoSeries>(&mut self, idx: usize, new_col: S) -> Result<&mut Self> {
        let mut new_column = new_col.into_series();
        if new_column.len() != self.height() {
            return Err(PolarsError::ShapeMisMatch(
                format!("Cannot replace Series at index {}. The shape of Series {} does not match that of the DataFrame {}",
                idx, new_column.len(), self.height()
                ).into()));
        };
        if idx >= self.width() {
            return Err(PolarsError::OutOfBounds(
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
    /// use polars::prelude::*;
    /// let s0 = Series::new("foo", &["ham", "spam", "egg"]);
    /// let s1 = Series::new("names", &["Jean", "Claude", "van"]);
    /// let mut df = DataFrame::new(vec![s0, s1]).unwrap();
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
    pub fn apply<F, S>(&mut self, column: &str, f: F) -> Result<&mut Self>
    where
        F: FnOnce(&Series) -> S,
        S: IntoSeries,
    {
        let idx = self
            .find_idx_by_name(column)
            .ok_or(PolarsError::NotFound(column.to_string()))?;
        self.apply_at_idx(idx, f)
    }

    /// Apply a closure to a column at index `idx`. This is the recommended way to do in place
    /// modification.
    ///
    /// # Example
    ///
    /// ```rust
    /// use polars::prelude::*;
    /// let s0 = Series::new("foo", &["ham", "spam", "egg"]);
    /// let s1 = Series::new("ascii", &[70, 79, 79]);
    /// let mut df = DataFrame::new(vec![s0, s1]).unwrap();
    ///
    /// // Add 32 to get lowercase ascii values
    /// df.apply_at_idx(1, |s| s + 32);
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
    pub fn apply_at_idx<F, S>(&mut self, idx: usize, f: F) -> Result<&mut Self>
    where
        F: FnOnce(&Series) -> S,
        S: IntoSeries,
    {
        let width = self.width();
        let col = self.columns.get_mut(idx).ok_or(PolarsError::OutOfBounds(
            format!(
                "Column index: {} outside of DataFrame with {} columns",
                idx, width
            )
            .into(),
        ))?;
        let name = col.name().to_string();
        let _ = mem::replace(col, f(col).into_series());

        // make sure the name remains the same after applying the closure
        unsafe {
            let col = self.columns.get_unchecked_mut(idx);
            col.rename(&name);
        }
        self.register_mutation()?;
        Ok(self)
    }

    /// Apply a closure that may fail to a column at index `idx`. This is the recommended way to do in place
    /// modification.
    ///
    /// # Example
    ///
    /// This is the idomatic way to replace some values a column of a `DataFrame` given range of indexes.
    ///
    /// ```
    /// # use polars::prelude::*;
    /// let s0 = Series::new("foo", &["ham", "spam", "egg", "bacon", "quack"]);
    /// let s1 = Series::new("values", &[1, 2, 3, 4, 5]);
    /// let mut df = DataFrame::new(vec![s0, s1]).unwrap();
    ///
    /// let idx = &[0, 1, 4];
    ///
    /// df.may_apply("foo", |s| {
    ///     s.utf8()?
    ///     .set_at_idx_with(idx, |opt_val| opt_val.map(|string| format!("{}-is-modified", string)))
    /// });
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
    pub fn may_apply_at_idx<F, S>(&mut self, idx: usize, f: F) -> Result<&mut Self>
    where
        F: FnOnce(&Series) -> Result<S>,
        S: IntoSeries,
    {
        let width = self.width();
        let col = self.columns.get_mut(idx).ok_or(PolarsError::OutOfBounds(
            format!(
                "Column index: {} outside of DataFrame with {} columns",
                idx, width
            )
            .into(),
        ))?;
        let name = col.name().to_string();

        let _ = mem::replace(col, f(col).map(|s| s.into_series())?);

        // make sure the name remains the same after applying the closure
        unsafe {
            let col = self.columns.get_unchecked_mut(idx);
            col.rename(&name);
        }
        self.register_mutation()?;
        Ok(self)
    }

    /// Apply a closure that may fail to a column. This is the recommended way to do in place
    /// modification.
    ///
    /// # Example
    ///
    /// This is the idomatic way to replace some values a column of a `DataFrame` given a boolean mask.
    ///
    /// ```
    /// # use polars::prelude::*;
    /// let s0 = Series::new("foo", &["ham", "spam", "egg", "bacon", "quack"]);
    /// let s1 = Series::new("values", &[1, 2, 3, 4, 5]);
    /// let mut df = DataFrame::new(vec![s0, s1]).unwrap();
    ///
    /// // create a mask
    /// let values = df.column("values").unwrap();
    /// let mask = values.lt_eq(1) | values.gt_eq(5);
    ///
    /// df.may_apply("foo", |s| {
    ///     s.utf8()?
    ///     .set(&mask, Some("not_within_bounds"))
    /// });
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
    pub fn may_apply<F, S>(&mut self, column: &str, f: F) -> Result<&mut Self>
    where
        F: FnOnce(&Series) -> Result<S>,
        S: IntoSeries,
    {
        let idx = self
            .find_idx_by_name(column)
            .ok_or(PolarsError::NotFound(column.to_string()))?;
        self.may_apply_at_idx(idx, f)
    }

    /// Slice the DataFrame along the rows.
    pub fn slice(&self, offset: usize, length: usize) -> Result<Self> {
        let col = self
            .columns
            .par_iter()
            .map(|s| s.slice(offset, length))
            .collect::<Result<Vec<_>>>()?;
        Ok(DataFrame::new_no_checks(col))
    }

    /// Get the head of the DataFrame
    pub fn head(&self, length: Option<usize>) -> Self {
        let col = self
            .columns
            .iter()
            .map(|s| s.head(length))
            .collect::<Vec<_>>();
        DataFrame::new_no_checks(col)
    }

    /// Get the tail of the DataFrame
    pub fn tail(&self, length: Option<usize>) -> Self {
        let col = self
            .columns
            .iter()
            .map(|s| s.tail(length))
            .collect::<Vec<_>>();
        DataFrame::new_no_checks(col)
    }

    /// Transform the underlying chunks in the DataFrame to Arrow RecordBatches
    pub fn as_record_batches(&self) -> Result<Vec<RecordBatch>> {
        let n_chunks = self.n_chunks()?;
        let width = self.width();

        let schema = Arc::new(self.schema());

        let mut record_batches = Vec::with_capacity(n_chunks);
        for i in 0..n_chunks {
            // the columns of a single recorbatch
            let mut rb_cols = Vec::with_capacity(width);

            for col in &self.columns {
                rb_cols.push(Arc::clone(&col.chunks()[i]))
            }
            let rb = RecordBatch::try_new(Arc::clone(&schema), rb_cols)?;
            record_batches.push(rb)
        }
        Ok(record_batches)
    }

    /// Iterator over the rows in this DataFrame as Arrow RecordBatches.
    pub fn iter_record_batches(
        &mut self,
        buffer_size: usize,
    ) -> impl Iterator<Item = RecordBatch> + '_ {
        match self.n_chunks() {
            Ok(1) => {}
            Ok(_) => {
                self.columns = self
                    .columns
                    .iter()
                    .map(|s| s.rechunk(None).unwrap())
                    .collect();
            }
            Err(_) => {} // no data. So iterator will be empty
        }
        RecordBatchIter {
            columns: &self.columns,
            schema: Arc::new(self.schema()),
            buffer_size,
            idx: 0,
            len: self.height(),
        }
    }

    /// Get a DataFrame with all the columns in reversed order
    pub fn reverse(&self) -> Self {
        let col = self.columns.iter().map(|s| s.reverse()).collect::<Vec<_>>();
        DataFrame::new_no_checks(col)
    }

    /// Shift the values by a given period and fill the parts that will be empty due to this operation
    /// with `Nones`.
    ///
    /// See the method on [Series](../series/enum.Series.html#method.shift) for more info on the `shift` operation.
    pub fn shift(&self, periods: i32) -> Result<Self> {
        let col = self
            .columns
            .par_iter()
            .map(|s| s.shift(periods))
            .collect::<Result<Vec<_>>>()?;
        Ok(DataFrame::new_no_checks(col))
    }

    /// Replace None values with one of the following strategies:
    /// * Forward fill (replace None with the previous value)
    /// * Backward fill (replace None with the next value)
    /// * Mean fill (replace None with the mean of the whole array)
    /// * Min fill (replace None with the minimum of the whole array)
    /// * Max fill (replace None with the maximum of the whole array)
    ///
    /// See the method on [Series](../series/enum.Series.html#method.fill_none) for more info on the `fill_none` operation.
    pub fn fill_none(&self, strategy: FillNoneStrategy) -> Result<Self> {
        let col = self
            .columns
            .par_iter()
            .map(|s| s.fill_none(strategy))
            .collect::<Result<Vec<_>>>()?;
        Ok(DataFrame::new_no_checks(col))
    }

    /// Aggregate the columns to their maximum values.
    pub fn max(&self) -> Self {
        let columns = self.columns.par_iter().map(|s| s.max_as_series()).collect();
        DataFrame::new_no_checks(columns)
    }

    /// Aggregate the columns to their minimum values.
    pub fn min(&self) -> Self {
        let columns = self.columns.par_iter().map(|s| s.min_as_series()).collect();
        DataFrame::new_no_checks(columns)
    }

    /// Aggregate the columns to their sum values.
    pub fn sum(&self) -> Self {
        let columns = self.columns.par_iter().map(|s| s.sum_as_series()).collect();
        DataFrame::new_no_checks(columns)
    }

    /// Aggregate the columns to their mean values.
    pub fn mean(&self) -> Self {
        let columns = self
            .columns
            .par_iter()
            .map(|s| s.mean_as_series())
            .collect();
        DataFrame::new_no_checks(columns)
    }

    /// Aggregate the columns to their median values.
    pub fn median(&self) -> Self {
        let columns = self
            .columns
            .par_iter()
            .map(|s| s.median_as_series())
            .collect();
        DataFrame::new_no_checks(columns)
    }

    /// Aggregate the columns to their quantile values.
    pub fn quantile(&self, quantile: f64) -> Result<Self> {
        let columns = self
            .columns
            .par_iter()
            .map(|s| s.quantile_as_series(quantile))
            .collect::<Result<Vec<_>>>()?;
        Ok(DataFrame::new_no_checks(columns))
    }

    /// Pipe different functions/ closure operations that work on a DataFrame together.
    pub fn pipe<F, B>(self, f: F) -> Result<B>
    where
        F: Fn(DataFrame) -> Result<B>,
    {
        f(self)
    }

    /// Pipe different functions/ closure operations that work on a DataFrame together.
    pub fn pipe_mut<F, B>(&mut self, f: F) -> Result<B>
    where
        F: Fn(&mut DataFrame) -> Result<B>,
    {
        f(self)
    }

    /// Pipe different functions/ closure operations that work on a DataFrame together.
    pub fn pipe_with_args<F, B, Args>(self, f: F, args: Args) -> Result<B>
    where
        F: Fn(DataFrame, Args) -> Result<B>,
    {
        f(self, args)
    }

    /// Create dummy variables.
    ///
    /// # Example
    ///
    /// ```rust
    ///
    /// # #[macro_use] extern crate polars;
    /// # fn main() {
    ///
    ///  use polars::prelude::*;
    ///
    ///  let df = df! {
    ///       "id" => &[1, 2, 3, 1, 2, 3, 1, 1],
    ///       "type" => &["A", "B", "B", "B", "C", "C", "C", "B"],
    ///       "code" => &["X1", "X2", "X3", "X3", "X2", "X2", "X1", "X1"]
    ///   }.unwrap();
    ///
    ///   let dummies = df.to_dummies().unwrap();
    ///   dbg!(dummies);
    /// # }
    /// ```
    /// Outputs:
    /// ```text
    ///  +------+------+------+--------+--------+--------+---------+---------+---------+
    ///  | id_1 | id_3 | id_2 | type_A | type_B | type_C | code_X1 | code_X2 | code_X3 |
    ///  | ---  | ---  | ---  | ---    | ---    | ---    | ---     | ---     | ---     |
    ///  | u32  | u32  | u32  | u32    | u32    | u32    | u32     | u32     | u32     |
    ///  +======+======+======+========+========+========+=========+=========+=========+
    ///  | 1    | 0    | 0    | 1      | 0      | 0      | 1       | 0       | 0       |
    ///  +------+------+------+--------+--------+--------+---------+---------+---------+
    ///  | 0    | 0    | 1    | 0      | 1      | 0      | 0       | 1       | 0       |
    ///  +------+------+------+--------+--------+--------+---------+---------+---------+
    ///  | 0    | 1    | 0    | 0      | 1      | 0      | 0       | 0       | 1       |
    ///  +------+------+------+--------+--------+--------+---------+---------+---------+
    ///  | 1    | 0    | 0    | 0      | 1      | 0      | 0       | 0       | 1       |
    ///  +------+------+------+--------+--------+--------+---------+---------+---------+
    ///  | 0    | 0    | 1    | 0      | 0      | 1      | 0       | 1       | 0       |
    ///  +------+------+------+--------+--------+--------+---------+---------+---------+
    ///  | 0    | 1    | 0    | 0      | 0      | 1      | 0       | 1       | 0       |
    ///  +------+------+------+--------+--------+--------+---------+---------+---------+
    ///  | 1    | 0    | 0    | 0      | 0      | 1      | 1       | 0       | 0       |
    ///  +------+------+------+--------+--------+--------+---------+---------+---------+
    ///  | 1    | 0    | 0    | 0      | 1      | 0      | 1       | 0       | 0       |
    ///  +------+------+------+--------+--------+--------+---------+---------+---------+
    /// ```
    pub fn to_dummies(&self) -> Result<Self> {
        let mut df = self.clone();
        let df_index_name = "--__CUSTOM_INDEX_FOR_TO_DUMMIES__--";
        let s = UInt32Chunked::new_from_aligned_vec(
            df_index_name,
            (0u32..self.height() as u32).collect(),
        );
        df.add_column(s).unwrap();

        let mut gb = df.groupby(df_index_name)?;

        let mut columns = Vec::with_capacity(self.columns.len() * 256);
        for col in &self.columns {
            let pivot_col = col.name();
            // value column is not important in count
            let value_col = pivot_col;
            let mut pivot_df = gb
                .pivot(pivot_col, value_col)
                .count()?
                .sort(df_index_name, false)?;
            pivot_df.drop_in_place(df_index_name).unwrap();

            // rename columns
            for i in 0..(pivot_df.width()) {
                let s = pivot_df.select_at_idx_mut(i).unwrap();
                s.rename(&format!("{}_{}", pivot_col, s.name()));
                columns.push(s.clone())
            }
        }
        Ok(DataFrame::new_no_checks(columns))
    }

    /// Drop duplicate rows from a DataFrame.
    /// *This fails when there is a column of type List in DataFrame*
    ///
    /// # Example
    ///
    /// ```rust
    ///
    /// # #[macro_use] extern crate polars;
    /// # fn main() {
    ///  use polars::prelude::*;
    ///
    ///  fn example() -> Result<DataFrame> {
    ///      let df = df! {
    ///                    "flt" => [1., 1., 2., 2., 3., 3.],
    ///                    "int" => [1, 1, 2, 2, 3, 3, ],
    ///                    "str" => ["a", "a", "b", "b", "c", "c"]
    ///                }?;
    ///      df.drop_duplicates()?
    ///          .sort("flt", false)
    ///  }
    /// # }
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
    pub fn drop_duplicates(&self) -> Result<Self> {
        let gb = self.groupby(self.get_column_names())?;
        let groups = gb.get_groups().into_iter().map(|v| v.0);
        let cap = Some(groups.size_hint().0);
        let df = unsafe { self.take_iter_unchecked(groups, cap) };
        Ok(df)
    }

    /// Get a mask of all the unique rows in the DataFrame.
    pub fn is_unique(&self) -> Result<BooleanChunked> {
        let mut gb = self.groupby(self.get_column_names())?;
        let groups = std::mem::take(&mut gb.groups);
        is_unique_helper(groups.into_iter(), self.height(), true, false)
    }

    /// Get a mask of all the duplicated rows in the DataFrame.
    pub fn is_duplicated(&self) -> Result<BooleanChunked> {
        let mut gb = self.groupby(self.get_column_names())?;
        let groups = std::mem::take(&mut gb.groups);
        is_unique_helper(groups.into_iter(), self.height(), false, true)
    }
}

pub struct RecordBatchIter<'a> {
    columns: &'a Vec<Series>,
    schema: Arc<Schema>,
    buffer_size: usize,
    idx: usize,
    len: usize,
}

impl<'a> Iterator for RecordBatchIter<'a> {
    type Item = RecordBatch;

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx >= self.len {
            return None;
        }
        // most iterations the slice length will be buffer_size, except for the last. That one
        // may be shorter
        let length = if self.idx + self.buffer_size < self.len {
            self.buffer_size
        } else {
            self.len - self.idx
        };

        let mut rb_cols = Vec::with_capacity(self.columns.len());
        // take a slice from all columns and add the the current RecordBatch
        self.columns.iter().for_each(|s| {
            let slice = s.slice(self.idx, length).unwrap();
            rb_cols.push(Arc::clone(&slice.chunks()[0]))
        });
        let rb = RecordBatch::try_new(Arc::clone(&self.schema), rb_cols).unwrap();
        self.idx += length;
        Some(rb)
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::*;

    fn create_frame() -> DataFrame {
        let s0 = Series::new("days", [0, 1, 2].as_ref());
        let s1 = Series::new("temp", [22.1, 19.9, 7.].as_ref());
        DataFrame::new(vec![s0, s1]).unwrap()
    }

    #[test]
    fn test_select() {
        let df = create_frame();
        assert_eq!(df.column("days").unwrap().eq(1).sum(), Some(1));
    }

    #[test]
    fn test_filter() {
        let df = create_frame();
        println!("{}", df.column("days").unwrap());
        println!("{:?}", df);
        println!("{:?}", df.filter(&df.column("days").unwrap().eq(0)))
    }

    #[test]
    fn test_sort() {
        let mut df = create_frame();
        df.sort_in_place("temp", false).unwrap();
        println!("{:?}", df);
    }

    #[test]
    fn slice() {
        let df = create_frame();
        let sliced_df = df.slice(0, 2).expect("slice");
        assert_eq!(sliced_df.shape(), (2, 2));
        println!("{:?}", df)
    }

    #[test]
    fn get_dummies() {
        let df = df! {
            "id" => &[1, 2, 3, 1, 2, 3, 1, 1],
            "type" => &["A", "B", "B", "B", "C", "C", "C", "B"],
            "code" => &["X1", "X2", "X3", "X3", "X2", "X2", "X1", "X1"]
        }
        .unwrap();
        let dummies = df.to_dummies().unwrap();
        assert_eq!(
            Vec::from(dummies.column("id_1").unwrap().u32().unwrap()),
            &[
                Some(1),
                Some(0),
                Some(0),
                Some(1),
                Some(0),
                Some(0),
                Some(1),
                Some(1)
            ]
        );
        dbg!(dummies);
    }

    #[test]
    fn drop_duplicates() {
        let df = df! {
            "flt" => [1., 1., 2., 2., 3., 3.],
            "int" => [1, 1, 2, 2, 3, 3, ],
            "str" => ["a", "a", "b", "b", "c", "c"]
        }
        .unwrap();
        dbg!(&df);
        let df = df.drop_duplicates().unwrap().sort("flt", false).unwrap();
        let valid = df! {
            "flt" => [1., 2., 3.],
            "int" => [1, 2, 3],
            "str" => ["a", "b", "c"]
        }
        .unwrap();
        dbg!(&df);
        assert!(df.frame_equal(&valid));
    }
}
