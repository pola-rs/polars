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
    slice: Option<(usize, usize)>,
}

impl GroupBySource {
    pub(super) fn new(
        io_thread: IOThread,
        already_finished: DataFrame,
        groupby_sink: Box<dyn Sink>,
        slice: Option<(i64, usize)>,
    ) -> PolarsResult<Self> {
        let partitions = std::fs::read_dir(&io_thread.dir)?;

        if let Some(slice) = slice {
            if slice.0 < 0 {
                polars_bail!(ComputeError: "negative slice not supported with out-of-core groupby")
            }
        }

        Ok(Self {
            _io_thread: io_thread,
            already_finished: Some(already_finished),
            partitions,
            groupby_sink,
            chunk_idx: 0,
            slice: slice.map(|slice| (slice.0 as usize, slice.1)),
        })
    }
}

impl Source for GroupBySource {
    fn get_batches(&mut self, context: &PExecutionContext) -> PolarsResult<SourceResult> {
        let chunk_idx = self.chunk_idx;
        self.chunk_idx += 1;

        if self.slice == Some((0, 0)) {
            return Ok(SourceResult::Finished);
        }

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
                    FinalizedSink::Finished(mut df) => {
                        if let Some(slice) = &mut self.slice {
                            let height = df.height();
                            if slice.0 >= height {
                                slice.0 -= height;
                                return self.get_batches(context);
                            } else {
                                df = df.slice(slice.0 as i64, slice.1);
                                slice.0 = 0;
                                slice.1 = slice.1.saturating_sub(height);
                            }
                        }

                        // TODO! repartition if df is > chunk_size
                        Ok(SourceResult::GotMoreData(vec![DataChunk::new(
                            chunk_idx, df,
                        )]))
                    }
                    // recursively out of core path
                    FinalizedSink::Source(mut src) => src.get_batches(context),
                    _ => unreachable!(),
                }
            }
        }
    }

    fn fmt(&self) -> &str {
        "ooc-groupby-source"
    }
}
