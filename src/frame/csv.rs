use crate::prelude::*;
pub use arrow::csv::{ReaderBuilder as csvReaderBuilder, WriterBuilder as csvWriterBuilder};
use std::io::Read;

type CSVReader<R> = arrow::csv::Reader<R>;

pub struct DataFrameCsvBuilder<'a, R>
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
    pub fn new_from_csv(reader: &'a mut CSVReader<R>) -> Self {
        DataFrameCsvBuilder {
            reader,
            rechunk: true,
        }
    }

    pub fn finish(&mut self) -> Result<DataFrame> {
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
