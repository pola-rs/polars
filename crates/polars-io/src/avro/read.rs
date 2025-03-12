use std::io::{Read, Seek};

use arrow::io::avro::{self, read};
use arrow::record_batch::RecordBatch;
use polars_core::error::to_compute_err;
use polars_core::prelude::*;

use crate::RowIndex;
use crate::predicates::PhysicalIoExpr;
use crate::prelude::*;
use crate::shared::{ArrowReader, finish_reader};

/// Read [Apache Avro] format into a [`DataFrame`]
///
/// [Apache Avro]: https://avro.apache.org
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
pub struct AvroReader<'a, R> {
    reader: R,
    rechunk: bool,
    n_rows: Option<usize>,
    columns: Option<Arc<[PlSmallStr]>>,
    projection: Option<Arc<[usize]>>,
    row_index: Option<&'a mut RowIndex>,
    predicate: Option<Arc<dyn PhysicalIoExpr>>,
}

impl<'a, R: Read + Seek> AvroReader<'a, R> {
    /// Get schema of the Avro File
    pub fn schema(&mut self) -> PolarsResult<Schema> {
        let schema = self.arrow_schema()?;
        Ok(Schema::from_arrow_schema(&schema))
    }

    /// Get arrow schema of the avro File, this is faster than a polars schema.
    pub fn arrow_schema(&mut self) -> PolarsResult<ArrowSchema> {
        let metadata =
            avro::avro_schema::read::read_metadata(&mut self.reader).map_err(to_compute_err)?;
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
    pub fn with_projection(mut self, projection: Option<Arc<[usize]>>) -> Self {
        self.projection = projection;
        self
    }

    /// Columns to select/ project
    ///
    /// This will take precedence over projection
    pub fn with_columns(mut self, columns: Option<Arc<[PlSmallStr]>>) -> Self {
        self.columns = columns;
        self
    }

    /// Column to for sequential index
    pub fn with_row_index(mut self, row_index: Option<&'a mut RowIndex>) -> Self {
        self.row_index = row_index;
        self
    }

    /// Predicate to apply
    pub fn with_predicate(mut self, predicate: Option<Arc<dyn PhysicalIoExpr>>) -> Self {
        self.predicate = predicate;
        self
    }

    /// Count the number of rows without the predicate
    pub fn unfiltered_count(mut self) -> PolarsResult<usize> {
        let metadata =
            avro::avro_schema::read::read_metadata(&mut self.reader).map_err(to_compute_err)?;
        let schema = read::infer_schema(&metadata.record)?;

        let avro_reader = avro::read::Reader::new(&mut self.reader, metadata, schema, None);
        let mut num_rows = 0;
        for batch in avro_reader {
            num_rows += batch?.len();
        }
        Ok(num_rows)
    }
}

impl<R> ArrowReader for read::Reader<R>
where
    R: Read + Seek,
{
    fn next_record_batch(&mut self) -> PolarsResult<Option<RecordBatch>> {
        self.next().map_or(Ok(None), |v| v.map(Some))
    }
}

impl<R> SerReader<R> for AvroReader<'_, R>
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
            row_index: None,
            predicate: None,
        }
    }

    fn set_rechunk(mut self, rechunk: bool) -> Self {
        self.rechunk = rechunk;
        self
    }

    fn finish(mut self) -> PolarsResult<DataFrame> {
        let rechunk = self.rechunk;
        let metadata =
            avro::avro_schema::read::read_metadata(&mut self.reader).map_err(to_compute_err)?;
        let schema = read::infer_schema(&metadata.record)?;

        if let Some(columns) = &self.columns {
            self.projection = Some(
                columns_to_projection(columns, &schema)?
                    .into_boxed_slice()
                    .into(),
            );
        }

        let (projection, projected_schema) = if let Some(projection) = self.projection {
            let mut prj = vec![false; schema.len()];
            for &index in projection.iter() {
                prj[index] = true;
            }
            (Some(prj), apply_projection(&schema, &projection))
        } else {
            (None, schema.clone())
        };

        let avro_reader = avro::read::Reader::new(&mut self.reader, metadata, schema, projection);

        finish_reader(
            avro_reader,
            rechunk,
            self.n_rows,
            self.predicate,
            &projected_schema,
            self.row_index,
        )
    }
}
