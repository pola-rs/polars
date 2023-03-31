use polars_core::config::verbose;
use polars_core::prelude::*;
use polars_core::utils::split_df;

use crate::executors::sinks::io::IOThread;
use crate::executors::sources::IpcSourceOneShot;
use crate::operators::{DataChunk, FinalizedSink, PExecutionContext, Sink, Source, SourceResult};
use crate::pipeline::{morsels_per_sink, PipeLine};
use std::iter::Enumerate;

pub(super) type PartitionSink = Box<dyn Fn(u32) -> Box<dyn Sink> + Send + Sync>;

pub(super) struct GroupBySource {
    // holding this keeps the lockfile in place
    _io_thread: IOThread,
    already_finished: Option<DataFrame>,
    partitions: std::fs::ReadDir,
    groupby_sink: Box<dyn Sink>,
    chunk_idx: IdxSize,
    morsels_per_sink: usize,
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

                        let dfs = split_df(&mut df, self.morsels_per_sink).unwrap();
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

pub(super) struct GroupBySource2 {
    // holding this keeps the lockfile in place
    _io_thread: IOThread,
    already_finished: Option<DataFrame>,
    partitions: std::fs::ReadDir,
    groupby_sink: PartitionSink,
    chunk_idx: IdxSize,
    morsels_per_sink: usize,
    slice: Option<(usize, usize)>,
}

impl GroupBySource2 {
    pub(super) fn new(
        io_thread: IOThread,
        already_finished: DataFrame,
        groupby_sink: PartitionSink,
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
            morsels_per_sink: morsels_per_sink(),
            slice: slice.map(|slice| (slice.0 as usize, slice.1)),
        })
    }
}

impl Source for GroupBySource2 {
    fn get_batches(&mut self, context: &PExecutionContext) -> PolarsResult<SourceResult> {
        dbg!("run source");
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
                let partition_path = partition_dir.path();
                // can be a lockfile or serialized hashmap
                if partition_path.is_file() && !partition_path.ends_with(".ipc") {
                    return self.get_batches(context);
                }

                // get the partition number of this partition directory
                // we can use that to restore the state oft the hashmap
                let partition = partition_path.file_name().unwrap();
                let partition = partition.to_str().unwrap().parse::<u32>().unwrap();

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

                // create a pipeline with a the files as sources and the groupby as sink
                let mut pipe =
                    PipeLine::new_simple(sources, vec![], (self.groupby_sink)(partition), verbose());

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

                        let dfs = split_df(&mut df, self.morsels_per_sink).unwrap();
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
