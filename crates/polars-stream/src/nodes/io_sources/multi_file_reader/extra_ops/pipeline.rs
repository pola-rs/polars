use std::sync::Arc;

use polars_error::PolarsResult;
use polars_utils::IdxSize;

use super::apply::ApplyExtraOps;
use crate::async_executor::{self, JoinHandle, TaskPriority};
use crate::async_primitives::distributor_channel::distributor_channel;
use crate::morsel::Morsel;
use crate::nodes::io_sources::multi_file_reader::reader_interface::output::{
    FileReaderOutputRecv, FileReaderOutputSend,
};

pub struct PostApplyPipeline {
    pub ops_applier: ApplyExtraOps,
    pub reader_output_port: FileReaderOutputRecv,
    pub num_pipelines: usize,
}

impl PostApplyPipeline {
    /// Spawn a pipeline to apply extra operations on top of a FileReaderOutputPort.
    ///
    /// Returns the join handle of the pipeline and the new output port.
    pub fn spawn(self) -> (FileReaderOutputRecv, JoinHandle<PolarsResult<()>>) {
        todo!()
    }
}
