use std::sync::Arc;

use polars_core::prelude::ArrowSchema;
use polars_error::PolarsResult;
use polars_io::predicates::ScanIOPredicate;
use polars_io::prelude::{FileMetadata, ParquetOptions};
use polars_io::utils::byte_source::DynByteSource;
use polars_plan::dsl::ScanSources;
use polars_utils::index::AtomicIdxSize;
use polars_utils::pl_str::PlSmallStr;

use super::multi_file_reader::reader_interface::output::FileReaderOutputRecv;
use crate::utils::task_handles_ext;

mod init;
pub mod metadata_utils;
mod row_group_data_fetch;
mod row_group_decode;

type AsyncTaskData = (
    FileReaderOutputRecv,
    task_handles_ext::AbortOnDropHandle<PolarsResult<()>>,
);

#[allow(clippy::type_complexity)]
pub struct ParquetReadImpl {
    pub scan_sources: ScanSources,
    pub predicate: Option<ScanIOPredicate>,
    pub options: ParquetOptions,
    pub byte_source: Arc<DynByteSource>,
    pub normalized_pre_slice: Option<(usize, usize)>,
    pub metadata: Arc<FileMetadata>,
    // Run-time vars
    pub config: Config,
    pub verbose: bool,
    pub schema: Arc<ArrowSchema>,
    pub projected_arrow_schema: Arc<ArrowSchema>,
    pub memory_prefetch_func: fn(&[u8]) -> (),
    /// The offset is an AtomicIdxSize, as in the negative slice case, the row
    /// offset becomes relative to the starting point in the list of files,
    /// so the row index offset needs to be updated by the initializer to
    /// reflect this (https://github.com/pola-rs/polars/issues/19607).
    pub row_index: Option<Arc<(PlSmallStr, AtomicIdxSize)>>,
}

#[derive(Debug)]
pub struct Config {
    pub num_pipelines: usize,
    /// Number of row groups to pre-fetch concurrently, this can be across files
    pub row_group_prefetch_size: usize,
    /// Minimum number of values for a parallel spawned task to process to amortize
    /// parallelism overhead.
    pub min_values_per_thread: usize,
}

impl ParquetReadImpl {
    pub fn run(mut self) -> AsyncTaskData {
        if self.verbose {
            eprintln!("[ParquetSource]: {:?}", &self.config);
        }

        self.init_projected_arrow_schema();
        self.init_morsel_distributor()
    }
}
