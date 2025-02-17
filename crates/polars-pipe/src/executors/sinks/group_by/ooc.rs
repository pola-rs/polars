use polars_core::config::verbose;
use polars_core::prelude::*;
use polars_core::utils::split_df;

use crate::executors::sinks::io::IOThread;
use crate::executors::sources::IpcSourceOneShot;
use crate::operators::{DataChunk, FinalizedSink, PExecutionContext, Sink, Source, SourceResult};
use crate::pipeline::{morsels_per_sink, PipeLine};

pub(super) struct GroupBySource {
    // Holding this keeps the lockfile in place
    io_thread: IOThread,
    already_finished: Option<DataFrame>,
    partitions: std::fs::ReadDir,
    group_by_sink: Box<dyn Sink>,
    chunk_idx: IdxSize,
    morsels_per_sink: usize,
    slice: Option<(usize, usize)>,
}

impl GroupBySource {
    pub(super) fn new(
        io_thread: IOThread,
        already_finished: DataFrame,
        group_by_sink: Box<dyn Sink>,
        slice: Option<(i64, usize)>,
    ) -> PolarsResult<Self> {
        let partitions = std::fs::read_dir(&io_thread.dir)?;

        if let Some(slice) = slice {
            if slice.0 < 0 {
                polars_bail!(ComputeError: "negative slice not supported with out-of-core group_by")
            }
        }

        Ok(Self {
            io_thread,
            already_finished: Some(already_finished),
            partitions,
            group_by_sink,
            chunk_idx: 0,
            morsels_per_sink: morsels_per_sink(),
            slice: slice.map(|slice| (slice.0 as usize, slice.1)),
        })
    }
}

impl Source for GroupBySource {
    fn get_batches(&mut self, context: &PExecutionContext) -> PolarsResult<SourceResult> {
        if self.slice == Some((0, 0)) {
            return Ok(SourceResult::Finished);
        }

        if let Some(df) = self.already_finished.take() {
            let chunk_idx = self.chunk_idx;
            self.chunk_idx += 1;
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
                // ensure we read in the right order
                let mut files = std::fs::read_dir(partition_dir.path())?
                    .map(|e| e.map(|e| e.path()))
                    .collect::<Result<Vec<_>, _>>()?;
                files.sort_unstable();

                let sources = files
                    .iter()
                    .map(|path| {
                        Ok(Box::new(IpcSourceOneShot::new(path.as_path())?) as Box<dyn Source>)
                    })
                    .collect::<PolarsResult<Vec<_>>>()?;

                // create a pipeline with a the files as sources and the group_by as sink
                let mut pipe =
                    PipeLine::new_simple(sources, vec![], self.group_by_sink.split(0), verbose());

                let out = match pipe.run_pipeline(context, &mut vec![])?.unwrap() {
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

                        let dfs = split_df(&mut df, self.morsels_per_sink, false);
                        let chunks = dfs
                            .into_iter()
                            .map(|data| {
                                let chunk = DataChunk {
                                    chunk_index: self.chunk_idx,
                                    data,
                                };
                                self.chunk_idx += 1;

                                chunk
                            })
                            .collect::<Vec<_>>();

                        Ok(SourceResult::GotMoreData(chunks))
                    },
                    // recursively out of core path
                    FinalizedSink::Source(mut src) => src.get_batches(context),
                    _ => unreachable!(),
                };
                for path in files {
                    self.io_thread.clean(path)
                }

                out
            },
        }
    }

    fn fmt(&self) -> &str {
        "ooc-group_by-source"
    }
}
