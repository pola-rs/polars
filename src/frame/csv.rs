use crate::prelude::*;
pub use arrow::csv::ReaderBuilder;
use arrow::datatypes::Schema;
use std::io::{Read, Seek};
use std::sync::Arc;

/// Creates a DataFrame after reading a csv.
pub struct CsvReader<R>
where
    R: Read + Seek,
{
    reader: R,
    reader_builder: ReaderBuilder,
    rechunk: bool,
}

impl<R> CsvReader<R>
where
    R: Read + Seek,
{
    /// Create a new CsvReader from a file/ stream
    pub fn new(reader: R) -> Self {
        CsvReader {
            reader,
            reader_builder: ReaderBuilder::new(),
            rechunk: true,
        }
    }

    /// Rechunk to one contiguous chunk of memory after all data is read
    pub fn set_rechunk(mut self, rechunk: bool) -> Self {
        self.rechunk = rechunk;
        self
    }

    /// Set the CSV file's schema
    pub fn with_schema(mut self, schema: Arc<Schema>) -> Self {
        self.reader_builder = self.reader_builder.with_schema(schema);
        self
    }

    /// Set whether the CSV file has headers
    pub fn has_header(mut self, has_header: bool) -> Self {
        self.reader_builder = self.reader_builder.has_header(has_header);
        self
    }

    /// Set the CSV file's column delimiter as a byte character
    pub fn with_delimiter(mut self, delimiter: u8) -> Self {
        self.reader_builder = self.reader_builder.with_delimiter(delimiter);
        self
    }

    /// Set the CSV reader to infer the schema of the file
    pub fn infer_schema(mut self, max_records: Option<usize>) -> Self {
        self.reader_builder = self.reader_builder.infer_schema(max_records);
        self
    }

    /// Set the batch size (number of records to load at one time)
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.reader_builder = self.reader_builder.with_batch_size(batch_size);
        self
    }

    /// Set the reader's column projection
    pub fn with_projection(mut self, projection: Vec<usize>) -> Self {
        self.reader_builder = self.reader_builder.with_projection(projection);
        self
    }

    /// Read the file and create the DataFrame.
    pub fn finish(self) -> Result<DataFrame> {
        let mut csv_reader = self.reader_builder.build(self.reader)?;
        let mut columns = csv_reader
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
                // TODO: We've got more types
                _ => unimplemented!(),
            })
            .collect::<Vec<_>>();

        while let Some(batch) = csv_reader.next()? {
            batch
                .columns()
                .into_iter()
                .zip(&mut columns)
                .map(|(arr, ser)| ser.append_array(arr.clone()))
                .collect::<Result<Vec<_>>>()?;
        }

        Ok(DataFrame {
            schema: csv_reader.schema(),
            columns,
        })
    }
}
