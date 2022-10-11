use std::path::PathBuf;

use polars_core::error::PolarsResult;
use polars_core::schema::*;
use polars_core::POOL;
use polars_io::parquet::{BatchedParquetReader, ParquetReader};
use polars_io::SerReader;
use polars_plan::prelude::ParquetOptions;
use polars_utils::IdxSize;

use crate::operators::{DataChunk, PExecutionContext, Source, SourceResult};

pub struct ParquetSource {
    batched_reader: BatchedParquetReader,
    n_threads: usize,
    chunk_index: IdxSize,
}

impl ParquetSource {
    pub(crate) fn new(
        path: PathBuf,
        options: ParquetOptions,
        schema: &Schema,
    ) -> PolarsResult<Self> {
        let projection: Option<Vec<_>> = options.with_columns.map(|with_columns| {
            with_columns
                .iter()
                .map(|name| schema.index_of(name).unwrap())
                .collect()
        });

        let file = std::fs::File::open(path).unwrap();
        let batched_reader = ParquetReader::new(file)
            .with_n_rows(options.n_rows)
            .with_row_count(options.row_count)
            .with_projection(projection)
            .batched()?;

        Ok(ParquetSource {
            batched_reader,
            n_threads: POOL.current_num_threads(),
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
}
