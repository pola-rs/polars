use polars_core::config::verbose;
use polars_core::prelude::*;

use crate::executors::sinks::io::IOThread;
use crate::executors::sources::IpcSourceOneShot;
use crate::operators::{DataChunk, FinalizedSink, PExecutionContext, Sink, Source, SourceResult};
use crate::pipeline::PipeLine;

pub(super) struct GroupBySource {
    // holding this keeps the lockfile in place
    _io_thread: IOThread,
    already_finished: Option<DataFrame>,
    partitions: std::fs::ReadDir,
    groupby_sink: Box<dyn Sink>,
    chunk_idx: IdxSize,
}

impl GroupBySource {
    pub(super) fn new(
        io_thread: IOThread,
        already_finished: DataFrame,
        groupby_sink: Box<dyn Sink>,
    ) -> PolarsResult<Self> {
        let partitions = std::fs::read_dir(&io_thread.dir)?;

        Ok(Self {
            _io_thread: io_thread,
            already_finished: Some(already_finished),
            partitions,
            groupby_sink,
            chunk_idx: 0,
        })
    }
}

impl Source for GroupBySource {
    fn get_batches(&mut self, context: &PExecutionContext) -> PolarsResult<SourceResult> {
        let chunk_idx = self.chunk_idx;
        self.chunk_idx += 1;

        if let Some(df) = self.already_finished.take() {
            return Ok(SourceResult::GotMoreData(vec![DataChunk::new(
                chunk_idx, df,
            )]));
        }

        match self.partitions.next() {
            None => Ok(SourceResult::Finished),
            Some(dir) => {
                let partition_dir = dir?;
                if partition_dir.path().ends_with(".lock") {
                    return self.get_batches(context);
                }

                // read the files in the partition into sources
                let sources = std::fs::read_dir(partition_dir.path())?
                    .map(|entry| {
                        let path = entry?.path();
                        Ok(Box::new(IpcSourceOneShot::new(path.as_path())?) as Box<dyn Source>)
                    })
                    .collect::<PolarsResult<Vec<_>>>()?;

                // create a pipeline with a the files as sources and the groupby as sink
                let mut pipe =
                    PipeLine::new_simple(sources, vec![], self.groupby_sink.split(0), verbose());

                match pipe.run_pipeline(context)? {
                    FinalizedSink::Finished(df) => {
                        Ok(SourceResult::GotMoreData(vec![DataChunk::new(
                            chunk_idx, df,
                        )]))
                    }
                    _ => unreachable!(),
                }
            }
        }
    }

    fn fmt(&self) -> &str {
        "ooc-groupby-source"
    }
}
