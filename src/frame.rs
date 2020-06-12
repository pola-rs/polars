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
use arrow::datatypes::{Field, Schema};
use std::io::Read;
use std::sync::Arc;

type CSVReader<R> = arrow::csv::Reader<R>;

pub struct DataFrame {
    schema: Arc<Schema>,
    columns: Vec<Series>,
}

impl DataFrame {
    pub fn new(columns: Vec<Series>) -> Self {
        let fields = columns
            .iter()
            .map(|s| s.field().clone())
            .collect::<Vec<_>>();
        let schema = Arc::new(Schema::new(fields));

        DataFrame { schema, columns }
    }

    pub fn fields(&self) -> &Vec<Field> {
        self.schema.fields()
    }

    pub fn select_row_idx(&self, idx: usize) -> Option<Vec<AnyType>> {
        if self.columns.len() == 0 {
            return None;
        }
        unsafe {
            if self.columns.get_unchecked(0).len() <= idx {
                return None;
            }
        };
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
            .filter(|(idx, field)| field.name() == name)
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
        Ok(DataFrame::new(new_col))
    }

    pub fn f_filter(&self, mask: &BooleanChunked) -> Self {
        self.filter(mask).expect("could not filter")
    }
}

struct DataFrameBuilder<'a, R>
where
    R: Read,
{
    reader: &'a mut CSVReader<R>,
    rechunk: bool,
}

impl<'a, R> DataFrameBuilder<'a, R>
where
    R: Read,
{
    fn new_from_csv(reader: &'a mut CSVReader<R>) -> Self {
        DataFrameBuilder {
            reader,
            rechunk: true,
        }
    }

    fn build(&mut self) -> Result<DataFrame> {
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
        DataFrame::new(vec![s0, s1])
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
        println!("{:?}", df.filter(&df.f_select("days").f_eq(0)))
    }

    #[test]
    fn read_csv() {
        let file = File::open("data/iris.csv").unwrap();
        let builder = csv::ReaderBuilder::new()
            .infer_schema(None)
            .has_header(true);
        let mut reader = builder.build(file).unwrap();

        let df = DataFrameBuilder::new_from_csv(&mut reader).build().unwrap();
        assert_eq!(reader.schema(), df.schema);
        println!("{:?}", df.schema)
    }
}
