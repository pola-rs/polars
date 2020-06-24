use crate::prelude::*;
use arrow::datatypes::{Field, Schema};
use arrow::{compute::TakeOptions, record_batch::RecordBatch};
use itertools::Itertools;
use std::sync::Arc;

pub mod csv;
mod group_by;
mod hash_join;

type DfSchema = Arc<Schema>;
type DfSeries = Series;
type DfColumns = Vec<DfSeries>;

#[derive(Clone)]
pub struct DataFrame {
    pub schema: DfSchema,
    pub columns: DfColumns,
}

impl DataFrame {
    /// Create a new DataFrame from a schema and a vec of series.
    pub fn new(schema: DfSchema, columns: DfColumns) -> Result<Self> {
        if !columns.iter().map(|s| s.len()).all_equal() {
            return Err(PolarsError::LengthMismatch);
        }
        Ok(DataFrame { schema, columns })
    }

    /// Get fields from the columns.
    fn create_fields(columns: &DfColumns) -> Vec<Field> {
        columns.iter().map(|s| s.field().clone()).collect()
    }

    /// This method should be called after every mutable addition/ deletion of columns
    fn update_schema(&mut self) {
        let fields = Self::create_fields(&self.columns);
        self.schema = Arc::new(Schema::new(fields));
    }

    /// Create a DataFrame from a Vector of Series.
    pub fn new_from_columns(columns: Vec<Series>) -> Result<Self> {
        let fields = Self::create_fields(&columns);
        let schema = Arc::new(Schema::new(fields));
        Self::new(schema, columns)
    }

    /// Get a reference the schema fields of the DataFrame.
    pub fn fields(&self) -> &Vec<Field> {
        self.schema.fields()
    }

    /// Get (width x height)
    pub fn shape(&self) -> (usize, usize) {
        let width = self.columns.len();
        if width > 0 {
            (width, self.columns[0].len())
        } else {
            (0, 0)
        }
    }

    /// Get width of DataFrame
    pub fn width(&self) -> usize {
        self.shape().0
    }

    /// Get height of DataFrame
    pub fn height(&self) -> usize {
        self.shape().1
    }

    /// Add series column to DataFrame
    pub fn hstack(&mut self, columns: &[DfSeries]) -> Result<()> {
        columns
            .iter()
            .for_each(|column| self.columns.push(column.clone()));
        self.update_schema();
        Ok(())
    }

    /// Remove column by name
    pub fn drop(&mut self, name: &str) -> Option<DfSeries> {
        let mut idx = 0;
        for column in &self.columns {
            if column.name() == name {
                break;
            }
            idx += 1;
        }
        if idx == self.columns.len() {
            None
        } else {
            let result = Some(self.columns.remove(idx));
            self.update_schema();
            result
        }
    }

    /// Get a row in the dataframe. Beware this is slow.
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

    /// Select a series by name.
    pub fn select(&self, name: &str) -> Option<&Series> {
        let opt_idx = self.find_idx_by_name(name);

        match opt_idx {
            Some(idx) => self.select_idx(idx),
            None => None,
        }
    }

    /// Force select.
    pub fn f_select(&self, name: &str) -> &Series {
        self.select(name)
            .expect(&format!("name {} does not exist on dataframe", name))
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
        let mut new_col = Vec::with_capacity(self.columns.len());
        for col in &self.columns {
            new_col.push(col.filter(mask)?)
        }
        DataFrame::new_from_columns(new_col)
    }

    /// Force filter
    pub fn f_filter(&self, mask: &BooleanChunked) -> Self {
        self.filter(mask).expect("could not filter")
    }

    pub fn take_iter<I>(
        &self,
        iter: I,
        options: Option<TakeOptions>,
        capacity: Option<usize>,
    ) -> Result<Self>
    where
        I: Iterator<Item = Option<usize>> + Clone,
    {
        let new_col = self
            .columns
            .iter()
            .map(|s| {
                let mut i = iter.clone();
                s.take_iter(&mut i, options.clone(), capacity)
            })
            .collect::<Result<Vec<_>>>()?;
        DataFrame::new(self.schema.clone(), new_col)
    }

    /// Take DataFrame rows by index values.
    pub fn take<T: TakeIndex>(&self, indices: &T, options: Option<TakeOptions>) -> Result<Self> {
        let new_col = self
            .columns
            .iter()
            .map(|s| s.take(indices, options.clone()))
            .collect::<Result<Vec<_>>>()?;

        DataFrame::new(self.schema.clone(), new_col)
    }

    /// Force take
    pub fn f_take<T: TakeIndex>(&self, indices: &T, options: Option<TakeOptions>) -> Self {
        self.take(indices, options).expect("could not take")
    }

    /// Rename a column in the DataFrame
    pub fn rename(&mut self, column: &str, name: &str) -> Result<()> {
        self.select_mut(column)
            .ok_or(PolarsError::NotFound)
            .map(|s| s.rename(name))
    }

    /// Sort DataFrame in place by a column.
    pub fn sort(&mut self, by_column: &str) -> Result<()> {
        let s = match self.select(by_column) {
            Some(s) => s,
            None => return Err(PolarsError::NotFound),
        };

        let take = s.argsort();

        self.columns = self
            .columns
            .iter()
            .map(|s| s.take(&take, None))
            .collect::<Result<Vec<_>>>()?;
        Ok(())
    }

    pub fn as_record_batch(&self, _offset: usize, _length: usize) -> RecordBatch {
        todo!()
    }
}

pub struct DataFrameBuilder {
    schema: Option<DfSchema>,
    columns: DfColumns,
}

impl DataFrameBuilder {
    pub fn new(columns: DfColumns) -> Self {
        DataFrameBuilder {
            schema: None,
            columns,
        }
    }

    pub fn schema(mut self, schema: DfSchema) -> Self {
        self.schema = Some(schema);
        self
    }

    pub fn finish(self) -> Result<DataFrame> {
        match self.schema {
            Some(schema) => DataFrame::new(schema, self.columns),
            None => DataFrame::new_from_columns(self.columns),
        }
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::*;
    use std::fs::File;

    fn create_frame() -> DataFrame {
        let s0 = Series::init("days", [0, 1, 2].as_ref());
        let s1 = Series::init("temp", [22.1, 19.9, 7.].as_ref());
        DataFrame::new_from_columns(vec![s0, s1]).unwrap()
    }

    #[test]
    fn test_select() {
        let df = create_frame();
        assert_eq!(df.f_select("days").f_eq(1).sum(), Some(1));
    }

    #[test]
    fn test_filter() {
        let df = create_frame();
        println!("{}", df.f_select("days"));
        println!("{:?}", df);
        println!("{:?}", df.filter(&df.f_select("days").f_eq(0)))
    }

    #[test]
    fn read_csv() {
        let file = File::open("../data/iris.csv").unwrap();
        let df = CsvReader::new(file)
            .infer_schema(Some(100))
            .has_header(true)
            .with_batch_size(100)
            .finish()
            .unwrap();

        assert_eq!("sepal.length", df.schema.fields()[0].name());
        assert_eq!(1, df.f_select("sepal.length").chunks().len());
        println!("{:?}", df)
    }

    #[test]
    fn test_sort() {
        let mut df = create_frame();
        df.sort("temp").unwrap();
        println!("{:?}", df);
    }
}
