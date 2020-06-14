use crate::datatypes::{
    AnyType, BooleanChunked, Float32Chunked, Float64Chunked, Int32Chunked, Int64Chunked,
    UInt32Chunked, Utf8Chunked,
};
use crate::prelude::*;
use crate::series::chunked_array::SeriesOps;
use crate::{
    datatypes,
    datatypes::ArrowDataType,
    error::Result,
    series,
    series::{chunked_array::ChunkedArray, series::Series},
};
use arrow::compute::TakeOptions;
use arrow::datatypes::{Field, Schema};
use itertools::Itertools;
use std::borrow::Borrow;
use std::io::Read;
use std::sync::Arc;

mod hash_join;

type CSVReader<R> = arrow::csv::Reader<R>;

type DfSchema = Arc<Schema>;
type DfSeries = Series;
type DfColumns = Vec<DfSeries>;

pub struct DataFrame {
    pub schema: DfSchema,
    pub columns: DfColumns,
}

impl DataFrame {
    pub fn new(schema: DfSchema, columns: DfColumns) -> Result<Self> {
        if !columns.iter().map(|s| s.len()).all_equal() {
            return Err(PolarsError::LengthMismatch);
        }
        Ok(DataFrame { schema, columns })
    }

    fn create_fields(columns: &DfColumns) -> Vec<Field> {
        columns.iter().map(|s| s.field().clone()).collect()
    }

    fn update_schema(&mut self) {
        let fields = Self::create_fields(&self.columns);
        self.schema = Arc::new(Schema::new(fields));
    }

    pub fn new_from_columns(columns: Vec<Series>) -> Result<Self> {
        let fields = Self::create_fields(&columns);
        let schema = Arc::new(Schema::new(fields));
        Self::new(schema, columns)
    }

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

    pub fn select_idx(&self, idx: usize) -> Option<&Series> {
        self.columns.get(idx)
    }

    pub fn select(&self, name: &str) -> Option<&Series> {
        let opt_idx = self
            .schema
            .fields()
            .iter()
            .enumerate()
            .filter(|(_idx, field)| field.name() == name)
            .map(|(idx, _)| idx)
            .next();

        match opt_idx {
            Some(idx) => self.select_idx(idx),
            None => None,
        }
    }

    pub fn f_select(&self, name: &str) -> &Series {
        self.select(name)
            .expect(&format!("name {} does not exist on dataframe", name))
    }

    pub fn filter(&self, mask: &BooleanChunked) -> Result<Self> {
        let mut new_col = Vec::with_capacity(self.columns.len());
        for col in &self.columns {
            new_col.push(col.filter(mask)?)
        }
        DataFrame::new_from_columns(new_col)
    }

    pub fn f_filter(&self, mask: &BooleanChunked) -> Self {
        self.filter(mask).expect("could not filter")
    }

    pub fn take<T: AsRef<UInt32Chunked>>(
        &self,
        indices: T,
        options: Option<TakeOptions>,
    ) -> Result<Self> {
        let new_col = self
            .columns
            .iter()
            .map(|s| s.take(indices.as_ref(), options.clone()))
            .collect::<Result<Vec<_>>>()?;

        DataFrame::new(self.schema.clone(), new_col)
    }
}

struct DataFrameBuilder {
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

struct DataFrameCsvBuilder<'a, R>
where
    R: Read,
{
    reader: &'a mut CSVReader<R>,
    rechunk: bool,
}

impl<'a, R> DataFrameCsvBuilder<'a, R>
where
    R: Read,
{
    fn new_from_csv(reader: &'a mut CSVReader<R>) -> Self {
        DataFrameCsvBuilder {
            reader,
            rechunk: true,
        }
    }

    fn finish(&mut self) -> Result<DataFrame> {
        let mut columns = self
            .reader
            .schema()
            .fields()
            .iter()
            .map(|field| match field.data_type() {
                ArrowDataType::UInt32 => {
                    Series::UInt32(UInt32Chunked::new_from_chunks(field.name(), vec![]))
                }
                ArrowDataType::Int32 => {
                    Series::Int32(Int32Chunked::new_from_chunks(field.name(), vec![]))
                }
                ArrowDataType::Int64 => {
                    Series::Int64(Int64Chunked::new_from_chunks(field.name(), vec![]))
                }
                ArrowDataType::Float32 => {
                    Series::Float32(Float32Chunked::new_from_chunks(field.name(), vec![]))
                }
                ArrowDataType::Float64 => {
                    Series::Float64(Float64Chunked::new_from_chunks(field.name(), vec![]))
                }
                ArrowDataType::Utf8 => {
                    Series::Utf8(Utf8Chunked::new_from_chunks(field.name(), vec![]))
                }
                ArrowDataType::Boolean => {
                    Series::Bool(BooleanChunked::new_from_chunks(field.name(), vec![]))
                }
                _ => unimplemented!(),
            })
            .collect::<Vec<_>>();

        while let Some(batch) = self.reader.next()? {
            batch
                .columns()
                .into_iter()
                .zip(&mut columns)
                .map(|(arr, ser)| ser.append_array(arr.clone()))
                .collect::<Result<Vec<_>>>()?;
        }

        Ok(DataFrame {
            schema: self.reader.schema(),
            columns,
        })
    }
}

mod test {
    use super::*;
    use crate::series::series::NamedFrom;
    use arrow::csv;
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
        let file = File::open("data/iris.csv").unwrap();
        let builder = csv::ReaderBuilder::new()
            .infer_schema(None)
            .has_header(true);
        let mut reader = builder.build(file).unwrap();

        let df = DataFrameCsvBuilder::new_from_csv(&mut reader)
            .finish()
            .unwrap();
        assert_eq!(reader.schema(), df.schema);
        println!("{:?}", df.schema)
    }
}
