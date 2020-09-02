//! DataFrame module.
use crate::frame::select::Selection;
use crate::prelude::*;
use arrow::datatypes::{Field, Schema};
use arrow::record_batch::RecordBatch;
use itertools::Itertools;
use itertools::__std_iter::FromIterator;
use rayon::prelude::*;
use std::marker::Sized;
use std::mem;
use std::sync::Arc;

pub mod group_by;
pub mod hash_join;
pub mod select;
pub mod ser;

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

type DfSchema = Arc<Schema>;
type DfSeries = Series;
type DfColumns = Vec<DfSeries>;

#[derive(Clone)]
pub struct DataFrame {
    schema: DfSchema,
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
            Err(PolarsError::NotFound)
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
    pub fn new(columns: Vec<Series>) -> Result<Self> {
        let fields = Self::create_fields(&columns);
        let schema = Arc::new(Schema::new(fields));

        let mut df = Self::new_with_schema(schema, columns)?;
        df.rechunk()?;
        Ok(df)
    }

    /// Create a new DataFrame from a schema and a vec of series.
    /// Only for crate use as schema is not checked.
    fn new_with_schema(schema: DfSchema, columns: DfColumns) -> Result<Self> {
        if !columns.iter().map(|s| s.len()).all_equal() {
            return Err(PolarsError::ShapeMisMatch);
        }
        Ok(DataFrame { schema, columns })
    }

    /// Ensure all the chunks in the DataFrame are aligned.
    fn rechunk(&mut self) -> Result<&mut Self> {
        let chunk_lens = self
            .columns
            .iter()
            .map(|s| s.n_chunks())
            .collect::<Vec<_>>();

        let argmin = chunk_lens
            .iter()
            .position_min()
            .ok_or(PolarsError::NoData)?;
        let min_chunks = chunk_lens[argmin];

        let to_rechunk = chunk_lens
            .into_iter()
            .enumerate()
            .filter_map(|(idx, len)| if len > min_chunks { Some(idx) } else { None })
            .collect::<Vec<_>>();

        // clone shouldn't be too expensive as we expect the nr. of chunks to be close to 1.
        let chunk_id = self.columns[argmin].chunk_lengths().clone();

        for idx in to_rechunk {
            let col = &self.columns[idx];
            let new_col = col.rechunk(Some(&chunk_id))?;
            self.columns[idx] = new_col;
        }
        Ok(self)
    }

    /// Get a reference to the DataFrame schema.
    pub fn schema(&self) -> &DfSchema {
        &self.schema
    }

    /// Get a reference to the DataFrame columns.
    pub fn get_columns(&self) -> &DfColumns {
        &self.columns
    }

    /// Get the column labels of the DataFrame.
    pub fn columns(&self) -> Vec<&str> {
        self.columns.iter().map(|s| s.name()).collect()
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
            .ok_or(PolarsError::NoData)?
            .chunks()
            .len())
    }

    /// Get fields from the columns.
    fn create_fields(columns: &DfColumns) -> Vec<Field> {
        columns.iter().map(|s| s.field().clone()).collect()
    }

    /// This method should be called after every mutable addition/ deletion of columns
    fn register_mutation(&mut self) -> Result<()> {
        let fields = Self::create_fields(&self.columns);
        self.schema = Arc::new(Schema::new(fields));
        self.rechunk()?;
        Ok(())
    }

    /// Get a reference the schema fields of the DataFrame.
    pub fn fields(&self) -> &Vec<Field> {
        self.schema.fields()
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
        let rows = self.columns.len();
        if rows > 0 {
            (self.columns[0].len(), rows)
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
        self.shape().1
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
                return Err(PolarsError::ShapeMisMatch);
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
            return Err(PolarsError::ShapeMisMatch);
        }

        if self.dtypes() != df.dtypes() {
            return Err(PolarsError::DataTypeMisMatch);
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

        DataFrame::new(new_cols)
    }

    /// Get a row in the dataframe. Beware this is slow.
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
    fn select_idx_mut(&mut self, idx: usize) -> Option<&mut Series> {
        self.columns.get_mut(idx)
    }

    /// Get column index of a series by name.
    pub fn find_idx_by_name(&self, name: &str) -> Option<usize> {
        self.schema
            .fields()
            .iter()
            .enumerate()
            .filter(|(_idx, field)| field.name() == name)
            .map(|(idx, _)| idx)
            .next()
    }

    /// Select a single column by name.
    pub fn column(&self, name: &str) -> Result<&Series> {
        let idx = self.find_idx_by_name(name).ok_or(PolarsError::NotFound)?;
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
    pub fn select<'a, S>(&self, selection: S) -> Result<Self>
    where
        S: Selection<'a>,
    {
        let selected = self.select_series(selection)?;
        let df = DataFrame::new(selected)?;
        Ok(df)
    }

    /// Select column(s) from this DataFrame and return them into a Vector.
    pub fn select_series<'a, S>(&self, selection: S) -> Result<Vec<Series>>
    where
        S: Selection<'a>,
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
            Some(idx) => self.select_idx_mut(idx),
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
        DataFrame::new(new_col)
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
        DataFrame::new_with_schema(self.schema.clone(), new_col)
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
        DataFrame::new_with_schema(self.schema.clone(), new_col).unwrap()
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
        DataFrame::new_with_schema(self.schema.clone(), new_col)
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
        DataFrame::new_with_schema(self.schema.clone(), new_col).unwrap()
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

        DataFrame::new_with_schema(self.schema.clone(), new_col)
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
            .ok_or(PolarsError::NotFound)
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
        self.apply_at_idx(idx, |_| new_col)
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
        let idx = self.find_idx_by_name(column).ok_or(PolarsError::NotFound)?;
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
        let col = self.columns.get_mut(idx).ok_or(PolarsError::OutOfBounds)?;
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
        let col = self.columns.get_mut(idx).ok_or(PolarsError::OutOfBounds)?;
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
    /// let mask = || {
    ///     (df.column("values")?.lt_eq(1) | df.column("values")?.gt_eq(5))
    /// };
    /// let mask = mask().unwrap();
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
        let idx = self.find_idx_by_name(column).ok_or(PolarsError::NotFound)?;
        self.may_apply_at_idx(idx, f)
    }

    /// Slice the DataFrame along the rows.
    pub fn slice(&self, offset: usize, length: usize) -> Result<Self> {
        let col = self
            .columns
            .par_iter()
            .map(|s| s.slice(offset, length))
            .collect::<Result<Vec<_>>>()?;
        DataFrame::new_with_schema(self.schema.clone(), col)
    }

    /// Get the head of the DataFrame
    pub fn head(&self, length: Option<usize>) -> Self {
        let col = self
            .columns
            .iter()
            .map(|s| s.head(length))
            .collect::<Vec<_>>();
        DataFrame::new_with_schema(self.schema.clone(), col).unwrap()
    }

    /// Get the tail of the DataFrame
    pub fn tail(&self, length: Option<usize>) -> Self {
        let col = self
            .columns
            .iter()
            .map(|s| s.tail(length))
            .collect::<Vec<_>>();
        DataFrame::new_with_schema(self.schema.clone(), col).unwrap()
    }

    /// Transform the underlying chunks in the DataFrame to Arrow RecordBatches
    pub fn as_record_batches(&self) -> Result<Vec<RecordBatch>> {
        let n_chunks = self.n_chunks()?;
        let width = self.width();

        let mut record_batches = Vec::with_capacity(n_chunks);
        for i in 0..n_chunks {
            // the columns of a single recorbatch
            let mut rb_cols = Vec::with_capacity(width);

            for col in &self.columns {
                rb_cols.push(Arc::clone(&col.chunks()[i]))
            }
            let rb = RecordBatch::try_new(Arc::clone(&self.schema), rb_cols)?;
            record_batches.push(rb)
        }
        Ok(record_batches)
    }

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
            schema: &self.schema,
            buffer_size,
            idx: 0,
            len: self.height(),
        }
    }

    /// Get a DataFrame with all the columns in reversed order
    pub fn reverse(&self) -> Self {
        let col = self.columns.iter().map(|s| s.reverse()).collect::<Vec<_>>();
        DataFrame::new_with_schema(self.schema.clone(), col).unwrap()
    }

    /// Shift the values by a given period and fill the parts that will be empty due to this operation
    /// with `Nones`.
    ///
    /// See the method on [Series](../series/enum.Series.html#method.shift) for more info on the `shift` operation.
    pub fn shift(&self, periods: i32) -> Result<Self> {
        let col = self
            .columns
            .iter()
            .map(|s| s.shift(periods))
            .collect::<Result<Vec<_>>>()?;
        DataFrame::new_with_schema(self.schema.clone(), col)
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
            .iter()
            .map(|s| s.fill_none(strategy))
            .collect::<Result<Vec<_>>>()?;
        DataFrame::new_with_schema(self.schema.clone(), col)
    }
}

pub struct RecordBatchIter<'a> {
    columns: &'a Vec<Series>,
    schema: &'a Arc<Schema>,
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
        let rb = RecordBatch::try_new(Arc::clone(self.schema), rb_cols).unwrap();
        self.idx += length;
        Some(rb)
    }
}

impl FromIterator<Series> for DataFrame {
    /// # Panics
    ///
    /// Panics if Series have different lengths.
    fn from_iter<T: IntoIterator<Item = Series>>(iter: T) -> Self {
        let v = iter.into_iter().collect();
        DataFrame::new(v).expect("could not create DataFrame from iterator")
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
}
