use crate::series::chunked_array::SeriesOps;
use crate::{
    datatypes,
    datatypes::ArrowDataType,
    error::Result,
    series,
    series::{chunked_array::ChunkedArray, series::Series},
};
use arrow::datatypes::{ArrowPrimitiveType, Field, Schema};
use std::io::Read;
use std::sync::Arc;

type CSVReader<R> = arrow::csv::Reader<R>;

struct DataFrame {
    schema: Arc<Schema>,
    columns: Vec<Series>,
}

impl DataFrame {
    fn fields(&self) -> &Vec<Field> {
        self.schema.fields()
    }

    fn select_series_ops_by_idx(&self, idx: usize) -> &dyn SeriesOps {
        match &self.columns[idx] {
            Series::Int32(arr) => arr,
            Series::Int64(arr) => arr,
            Series::Float32(arr) => arr,
            Series::Float64(arr) => arr,
            Series::Utf8(arr) => arr,
            _ => unimplemented!(),
        }
    }

    fn select_type<N>(&self, name: &str) -> ChunkedArray<N> {
        unimplemented!()
    }

    fn select_type_by_idx<N>(&self, idx: usize) -> ChunkedArray<N> {
        unimplemented!()
    }

    fn select_by_idx(&self, idx: usize) -> Series {
        unimplemented!()
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
                ArrowDataType::Int32 => Series::Int32(
                    ChunkedArray::<datatypes::Int32Type>::new_from_chunks(field.name(), vec![]),
                ),
                ArrowDataType::Int64 => Series::Int64(
                    ChunkedArray::<datatypes::Int64Type>::new_from_chunks(field.name(), vec![]),
                ),
                ArrowDataType::Float32 => Series::Float32(
                    ChunkedArray::<datatypes::Float32Type>::new_from_chunks(field.name(), vec![]),
                ),
                ArrowDataType::Float64 => Series::Float64(
                    ChunkedArray::<datatypes::Float64Type>::new_from_chunks(field.name(), vec![]),
                ),
                ArrowDataType::Utf8 => Series::Utf8(
                    ChunkedArray::<datatypes::Utf8Type>::new_from_chunks(field.name(), vec![]),
                ),
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
    use arrow::csv;
    use std::fs::File;

    #[test]
    fn read_csv() {
        let file = File::open("data/iris.csv").unwrap();
        let builder = csv::ReaderBuilder::new()
            .infer_schema(None)
            .has_headers(true);
        let mut reader = builder.build(file).unwrap();

        let df = DataFrameBuilder::new_from_csv(&mut reader).build().unwrap();
        assert_eq!(reader.schema(), df.schema);
        println!("{:?}", df.schema)
    }
}
