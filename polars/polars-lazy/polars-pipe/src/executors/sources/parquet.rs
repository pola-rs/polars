use std::path::PathBuf;

use polars_core::cloud::CloudOptions;
use polars_core::error::PolarsResult;
use polars_core::schema::*;
use polars_core::POOL;
use polars_io::parquet::{BatchedParquetReader, ParquetReader};
#[cfg(feature = "async")]
use polars_io::prelude::ParquetAsyncReader;
use polars_io::{is_cloud_url, SerReader};
use polars_plan::prelude::ParquetOptions;
use polars_utils::IdxSize;

use crate::operators::{DataChunk, PExecutionContext, Source, SourceResult};
use crate::pipeline::determine_chunk_size;

pub struct ParquetSource {
    batched_reader: BatchedParquetReader,
    n_threads: usize,
    chunk_index: IdxSize,
}

impl ParquetSource {
    #[allow(unused_variables)]
    pub(crate) fn new(
        path: PathBuf,
        options: ParquetOptions,
        cloud_options: Option<CloudOptions>,
        schema: &Schema,
        verbose: bool,
    ) -> PolarsResult<Self> {
        let projection: Option<Vec<_>> = options.with_columns.map(|with_columns| {
            with_columns
                .iter()
                .map(|name| schema.index_of(name).unwrap())
                .collect()
        });

        let n_cols = projection.as_ref().map(|v| v.len()).unwrap_or(schema.len());
        let n_threads = POOL.current_num_threads();
        let chunk_size = determine_chunk_size(n_cols, n_threads)?;

        if verbose {
            eprintln!("STREAMING CHUNK SIZE: {chunk_size} rows")
        }

        let batched_reader = if is_cloud_url(&path) {
            #[cfg(not(feature = "async"))]
            {
                panic!(
                    "Feature 'async' (or more likely one of the cloud provider features) is required to access parquet files on cloud storage."
                )
            }
            #[cfg(feature = "async")]
            {
                let uri = path.to_string_lossy();
                ParquetAsyncReader::from_uri(&uri, cloud_options.as_ref())?
                    .with_n_rows(options.n_rows)
                    .with_row_count(options.row_count)
                    .with_projection(projection)
                    .use_statistics(options.use_statistics)
                    .batched(chunk_size)?
            }
        } else {
            let file = std::fs::File::open(path).unwrap();

            ParquetReader::new(file)
                .with_n_rows(options.n_rows)
                .with_row_count(options.row_count)
                .with_projection(projection)
                .use_statistics(options.use_statistics)
                .batched(chunk_size)?
        };

        Ok(ParquetSource {
            batched_reader,
            n_threads,
            chunk_index: 0,
        })
    }
}

impl Source for ParquetSource {
    fn get_batches(&mut self, _context: &PExecutionContext) -> PolarsResult<SourceResult> {
        let batches = self.batched_reader.next_batches(self.n_threads)?;
        Ok(match batches {
            None => SourceResult::Finished,
            Some(batches) => SourceResult::GotMoreData(
                batches
                    .into_iter()
                    .map(|data| {
                        let chunk_index = self.chunk_index;
                        self.chunk_index += 1;
                        DataChunk { chunk_index, data }
                    })
                    .collect(),
            ),
        })
    }
    fn fmt(&self) -> &str {
        "parquet"
    }
}
