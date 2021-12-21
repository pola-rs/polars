use super::{finish_reader, ArrowReader, ArrowResult, RecordBatch};
use crate::mmap::MmapBytesReader;
use crate::parquet::read_par::parallel_read;
use crate::prelude::*;
use crate::{PhysicalIoExpr, ScanAggregation};
use arrow::io::parquet::read;
use polars_arrow::io::read_parquet;
use polars_core::prelude::*;
use polars_core::utils::accumulate_dataframes_vertical;
use std::convert::TryFrom;
use std::io::{Read, Seek};
use std::sync::Arc;

/// Read Apache parquet format into a DataFrame.
pub struct ParquetReader<R: Read + Seek> {
    reader: R,
    rechunk: bool,
    n_rows: Option<usize>,
    columns: Option<Vec<String>>,
    projection: Option<Vec<usize>>,
    parallel: bool,
}

impl<R: MmapBytesReader> ParquetReader<R> {
    #[cfg(feature = "lazy")]
    // todo! hoist to lazy crate
    pub fn finish_with_scan_ops(
        mut self,
        predicate: Option<Arc<dyn PhysicalIoExpr>>,
        aggregate: Option<&[ScanAggregation]>,
        projection: Option<&[usize]>,
    ) -> Result<DataFrame> {
        if aggregate.is_none() {
            self.finish()
        } else {
            let rechunk = self.rechunk;

            let reader = read::RecordReader::try_new(
                &mut self.reader,
                projection.map(|x| x.to_vec()),
                self.n_rows,
                None,
                None,
            )?;

            finish_reader(reader, rechunk, self.n_rows, predicate, aggregate)
        }
    }

    /// Read the parquet file in parallel (default). The single threaded reader consumes less memory.
    pub fn read_parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }

    /// Stop parsing when `n` rows are parsed. By settings this parameter the csv will be parsed
    /// sequentially.
    pub fn with_n_rows(mut self, num_rows: Option<usize>) -> Self {
        self.n_rows = num_rows;
        self
    }

    /// Columns to select/ project
    pub fn with_columns(mut self, columns: Option<Vec<String>>) -> Self {
        self.columns = columns;
        self
    }

    /// Set the reader's column projection. This counts from 0, meaning that
    /// `vec![0, 4]` would select the 1st and 5th column.
    pub fn with_projection(mut self, projection: Option<Vec<usize>>) -> Self {
        self.projection = projection;
        self
    }

    pub fn schema(mut self) -> Result<Schema> {
        let metadata = read::read_metadata(&mut self.reader)?;

        let schema = read::get_schema(&metadata)?;
        Ok(schema.into())
    }
}

impl<R: Read + Seek> ArrowReader for read::RecordReader<R> {
    fn next_record_batch(&mut self) -> ArrowResult<Option<RecordBatch>> {
        self.next().map_or(Ok(None), |v| v.map(Some))
    }

    fn schema(&self) -> Arc<Schema> {
        Arc::new((&*self.schema().clone()).into())
    }
}

impl<R: MmapBytesReader> SerReader<R> for ParquetReader<R> {
    fn new(reader: R) -> Self {
        ParquetReader {
            reader,
            rechunk: false,
            n_rows: None,
            columns: None,
            projection: None,
            parallel: true,
        }
    }

    fn set_rechunk(mut self, rechunk: bool) -> Self {
        self.rechunk = rechunk;
        self
    }

    fn finish(mut self) -> Result<DataFrame> {
        let metadata = read::read_metadata(&mut self.reader)?;
        let schema = read::schema::get_schema(&metadata)?;

        if let Some(cols) = self.columns {
            let mut prj = Vec::with_capacity(cols.len());
            for col in cols.iter() {
                let i = schema.index_of(col)?;
                prj.push(i);
            }

            self.projection = Some(prj);
        }

        if self.parallel {
            let rechunk = self.rechunk;
            return parallel_read(
                self.reader,
                self.n_rows.unwrap_or(usize::MAX),
                self.projection.as_deref(),
                &schema,
                Some(metadata),
            )
            .map(|mut df| {
                if rechunk {
                    df.rechunk();
                };
                df
            });
        }

        let chunks = read_parquet(
            &mut self.reader,
            self.n_rows.unwrap_or(usize::MAX),
            self.projection.as_deref(),
            &schema,
            Some(metadata),
        )?;
        let projection = self.projection.take();
        let mut df = accumulate_dataframes_vertical(chunks.into_iter().map(|cols| {
            DataFrame::new_no_checks(
                cols.into_iter()
                    .enumerate()
                    .map(|(mut i, arr)| {
                        if let Some(projection) = &projection {
                            i = projection[i]
                        }
                        Series::try_from((schema.field(i).name().as_str(), arr)).unwrap()
                    })
                    .collect(),
            )
        }))?;
        if self.rechunk {
            df.rechunk();
        }

        Ok(df)
    }
}
