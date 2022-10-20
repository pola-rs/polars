use std::io::{Read, Seek};

use arrow::io::avro::{self, read};
use polars_core::prelude::*;

use super::{finish_reader, ArrowChunk, ArrowReader, ArrowResult};
use crate::avro::convert_err;
use crate::prelude::*;

/// Read Apache Avro format into a DataFrame
///
/// # Example
/// ```
/// use std::fs::File;
/// use polars_core::prelude::*;
/// use polars_io::avro::AvroReader;
/// use polars_io::SerReader;
///
/// fn example() -> PolarsResult<DataFrame> {
///     let file = File::open("file.avro").expect("file not found");
///
///     AvroReader::new(file)
///             .finish()
/// }
/// ```
#[must_use]
pub struct AvroReader<R> {
    reader: R,
    rechunk: bool,
    n_rows: Option<usize>,
    columns: Option<Vec<String>>,
    projection: Option<Vec<usize>>,
}

impl<R: Read + Seek> AvroReader<R> {
    /// Get schema of the Avro File
    pub fn schema(&mut self) -> PolarsResult<Schema> {
        let schema = self.arrow_schema()?;
        Ok((schema.fields.iter()).into())
    }

    /// Get arrow schema of the avro File, this is faster than a polars schema.
    pub fn arrow_schema(&mut self) -> PolarsResult<ArrowSchema> {
        let metadata =
            avro::avro_schema::read::read_metadata(&mut self.reader).map_err(convert_err)?;
        let schema = read::infer_schema(&metadata.record)?;
        Ok(schema)
    }

    /// Stop reading when `n` rows are read.
    pub fn with_n_rows(mut self, num_rows: Option<usize>) -> Self {
        self.n_rows = num_rows;
        self
    }

    /// Set the reader's column projection. This counts from 0, meaning that
    /// `vec![0, 4]` would select the 1st and 5th column.
    pub fn with_projection(mut self, projection: Option<Vec<usize>>) -> Self {
        self.projection = projection;
        self
    }

    /// Columns to select/ project
    pub fn with_columns(mut self, columns: Option<Vec<String>>) -> Self {
        self.columns = columns;
        self
    }
}

impl<R> ArrowReader for read::Reader<R>
where
    R: Read + Seek,
{
    fn next_record_batch(&mut self) -> ArrowResult<Option<ArrowChunk>> {
        self.next().map_or(Ok(None), |v| v.map(Some))
    }
}

impl<R> SerReader<R> for AvroReader<R>
where
    R: Read + Seek,
{
    fn new(reader: R) -> Self {
        AvroReader {
            reader,
            rechunk: true,
            n_rows: None,
            columns: None,
            projection: None,
        }
    }

    fn set_rechunk(mut self, rechunk: bool) -> Self {
        self.rechunk = rechunk;
        self
    }

    fn finish(mut self) -> PolarsResult<DataFrame> {
        let rechunk = self.rechunk;
        let metadata =
            avro::avro_schema::read::read_metadata(&mut self.reader).map_err(convert_err)?;
        let schema = read::infer_schema(&metadata.record)?;

        if let Some(columns) = &self.columns {
            self.projection = Some(columns_to_projection(columns, &schema)?);
        }

        let (projection, projected_schema) = if let Some(projection) = self.projection {
            let mut prj = vec![false; schema.fields.len()];
            for &index in projection.iter() {
                prj[index] = true;
            }
            (Some(prj), apply_projection(&schema, &projection))
        } else {
            (None, schema.clone())
        };

        let avro_reader =
            avro::read::Reader::new(&mut self.reader, metadata, schema.fields, projection);

        finish_reader(
            avro_reader,
            rechunk,
            self.n_rows,
            None,
            &projected_schema,
            None,
        )
    }
}
