//! Generic driver loop that should work for all file types.

use std::collections::VecDeque;

use futures::StreamExt;
use polars_error::PolarsResult;
use polars_io::pl_async;
use polars_utils::IdxSize;
use polars_utils::slice_enum::Slice;

use super::DriverState;
use crate::async_executor::{self, AbortOnDropHandle, JoinHandle, TaskPriority};
use crate::async_primitives::connector;
use crate::nodes::io_sources::multi_file_reader::bridge::spawn_bridge;
use crate::nodes::io_sources::multi_file_reader::extra_ops::ExtraOperations;
use crate::nodes::io_sources::multi_file_reader::extra_ops::apply::ApplyExtraOps;
use crate::nodes::io_sources::multi_file_reader::extra_ops::cast_columns::CastColumnsPolicy;
use crate::nodes::io_sources::multi_file_reader::extra_ops::missing_columns::MissingColumnsPolicy;
use crate::nodes::io_sources::multi_file_reader::extra_ops::pipeline::PostApplyPipeline;
use crate::nodes::io_sources::multi_file_reader::initialization::MultiScanTaskInitializer;
use crate::nodes::io_sources::multi_file_reader::initialization::slice::{
    ResolvedPositiveSliceInfo, resolve_to_positive_slice,
};
use crate::nodes::io_sources::multi_file_reader::reader_interface::builder::FileReaderType;
use crate::nodes::io_sources::multi_file_reader::reader_interface::{
    FileReader, FileReaderCallbacks,
};
use crate::utils::task_handles_ext;

impl MultiScanTaskInitializer {
    /// Generic driver loop:
    /// * Negative slices are translated to positive by performing a row count on the files list in reverse.
    pub async fn init_and_run_generic_loop(mut self) -> PolarsResult<JoinHandle<PolarsResult<()>>> {
        todo!()
    }
}

async fn run_loop(driver_state: DriverState) -> PolarsResult<()> {
    todo!()
}
