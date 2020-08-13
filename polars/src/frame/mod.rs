//! DataFrame module
use crate::frame::select::Selection;
use crate::prelude::*;
use arrow::datatypes::{Field, Schema};
use arrow::record_batch::RecordBatch;
use itertools::Itertools;
use rayon::prelude::*;
use std::mem;
use std::sync::Arc;

pub mod group_by;
pub mod hash_join;
pub mod select;
pub mod ser;

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
            return Err(PolarsError::LengthMismatch);
        }
        Ok(DataFrame { schema, columns })
    }

    /// Ensure all the chunks in the DataFrame are aligned.
    fn rechunk(&mut self) -> Result<()> {
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
        Ok(())
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

    /// Add series column to DataFrame
    ///
    /// # Example
    ///
    /// ```
    /// use polars::prelude::*;
    /// fn stack(df: &mut DataFrame, columns: &[Series]) {
    ///     df.hstack(columns);
    /// }
    /// ```
    pub fn hstack(&mut self, columns: &[DfSeries]) -> Result<()> {
        let height = self.height();
        for col in columns {
            if col.len() != height {
                return Err(PolarsError::LengthMismatch);
            } else {
                self.columns.push(col.clone());
            }
        }
        self.register_mutation()?;
        Ok(())
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
    pub fn drop(&self, name: &str) -> Result<DataFrame> {
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
    pub fn select_idx(&self, idx: usize) -> Option<&Series> {
        self.columns.get(idx)
    }

    /// Force select.
    pub fn f_select_idx(&self, idx: usize) -> &Series {
        self.select_idx(idx).expect("out of bounds")
    }

    /// Select a mutable series by index.
    pub fn select_idx_mut(&mut self, idx: usize) -> Option<&mut Series> {
        self.columns.get_mut(idx)
    }

    /// Force select.
    pub fn f_select_idx_mut(&mut self, idx: usize) -> &mut Series {
        self.select_idx_mut(idx).expect("out of bounds")
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
    pub fn column(&self, name: &str) -> Option<&Series> {
        let opt_idx = self.find_idx_by_name(name);

        match opt_idx {
            Some(idx) => self.select_idx(idx),
            None => None,
        }
    }

    /// Force select a single column.
    pub fn f_column(&self, name: &str) -> &Series {
        self.column(name)
            .expect(&format!("name {} does not exist on dataframe", name))
    }

    /// Select column(s) from this DataFrame.
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
    pub fn select<'a, S>(&self, selection: S) -> Result<DataFrame>
    where
        S: Selection<'a>,
    {
        let cols = selection.to_selection_vec();
        let selected = cols
            .iter()
            .map(|c| {
                self.column(c)
                    .map(|s| s.clone())
                    .ok_or(PolarsError::NotFound)
            })
            .collect::<Result<Vec<_>>>()?;
        let df = DataFrame::new(selected)?;
        Ok(df)
    }

    /// Select a mutable series by name.
    pub fn select_mut(&mut self, name: &str) -> Option<&mut Series> {
        let opt_idx = self.find_idx_by_name(name);

        match opt_idx {
            Some(idx) => self.select_idx_mut(idx),
            None => None,
        }
    }

    /// Force select.
    pub fn f_select_mut(&mut self, name: &str) -> &mut Series {
        self.select_mut(name)
            .expect(&format!("name {} does not exist on dataframe", name))
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

    /// Force filter
    pub fn f_filter(&self, mask: &BooleanChunked) -> Self {
        self.filter(mask).expect("could not filter")
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

    /// Force take
    pub fn f_take<T: AsTakeIndex + Sync>(&self, indices: &T) -> Self {
        self.take(indices).expect("could not take")
    }

    /// Rename a column in the DataFrame
    ///
    /// # Example
    ///
    /// ```
    /// use polars::prelude::*;
    /// fn example(df: &mut DataFrame) -> Result<()> {
    ///     let original_name = "foo";
    ///     let new_name = "bar";
    ///     df.rename(original_name, new_name)
    /// }
    /// ```
    pub fn rename(&mut self, column: &str, name: &str) -> Result<()> {
        self.select_mut(column)
            .ok_or(PolarsError::NotFound)
            .map(|s| s.rename(name))
    }

    /// Sort DataFrame in place by a column.
    pub fn sort_in_place(&mut self, by_column: &str, reverse: bool) -> Result<()> {
        let s = match self.column(by_column) {
            Some(s) => s,
            None => return Err(PolarsError::NotFound),
        };

        let take = s.argsort(reverse);

        self.columns = self
            .columns
            .par_iter()
            .map(|s| s.take(&take))
            .collect::<Result<Vec<_>>>()?;
        Ok(())
    }

    pub fn sort(&self, by_column: &str, reverse: bool) -> Result<Self> {
        let s = match self.column(by_column) {
            Some(s) => s,
            None => return Err(PolarsError::NotFound),
        };

        let take = s.argsort(reverse);
        self.take(&take)
    }

    /// Replace a column with a series.
    pub fn replace(&mut self, column: &str, new_col: DfSeries) -> Result<()> {
        let idx = self.find_idx_by_name(column).ok_or(PolarsError::NotFound)?;
        let _ = mem::replace(&mut self.columns[idx], new_col);
        self.register_mutation()?;
        Ok(())
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

    /// Get a DataFrame with all the columns in reversed order
    pub fn reverse(&self) -> Self {
        let col = self.columns.iter().map(|s| s.reverse()).collect::<Vec<_>>();
        DataFrame::new_with_schema(self.schema.clone(), col).unwrap()
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
        assert_eq!(df.f_column("days").eq(1).sum(), Some(1));
    }

    #[test]
    fn test_filter() {
        let df = create_frame();
        println!("{}", df.f_column("days"));
        println!("{:?}", df);
        println!("{:?}", df.filter(&df.f_column("days").eq(0)))
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
        assert_eq!(sliced_df.shape(), (2, 2))
    }
}
