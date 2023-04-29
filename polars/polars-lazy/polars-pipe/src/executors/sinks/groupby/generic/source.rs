use polars_core::utils::flatten::flatten_df_iter;
use polars_io::ipc::IpcReader;
use polars_io::SerReader;

use super::*;
use crate::executors::sinks::groupby::generic::global::GlobalTable;
use crate::executors::sinks::io::{block_thread_until_io_thread_done, IOThread};
use crate::operators::{Source, SourceResult};

pub(super) struct GroupBySource {
    // holding this keeps the lockfile in place
    _io_thread: IOThread,
    partitions: std::fs::ReadDir,
    global_table: Arc<GlobalTable>,
    slice: Option<(usize, usize)>,
    chunk_idx: IdxSize,
}

impl GroupBySource {
    pub(super) fn new(
        io_thread: &IOThreadRef,
        slice: Option<(i64, usize)>,
        global_table: Arc<GlobalTable>,
    ) -> PolarsResult<Self> {
        let mut io_thread = io_thread.lock().unwrap();
        let io_thread = io_thread.take().unwrap();

        if let Some(slice) = slice {
            polars_ensure!(slice.0 >= 0, ComputeError: "negative slice not supported with out-of-core groupby")
        }

        block_thread_until_io_thread_done(&io_thread);
        let partitions = std::fs::read_dir(&io_thread.dir)?;
        Ok(Self {
            _io_thread: io_thread,
            partitions,
            slice: slice.map(|slice| (slice.0 as usize, slice.1)),
            global_table,
            chunk_idx: 0,
        })
    }
}

impl Source for GroupBySource {
    fn get_batches(&mut self, context: &PExecutionContext) -> PolarsResult<SourceResult> {
        if self.slice == Some((0, 0)) {
            return Ok(SourceResult::Finished);
        }

        let partition_dir = if let Some(part) = self.partitions.next() {
            part?.path()
        } else {
            // otherwise no files have been processed
            assert!(self.chunk_idx > 0);
            return Ok(SourceResult::Finished);
        };
        if partition_dir.ends_with(".lock") {
            return self.get_batches(context);
        }

        let partition_name = partition_dir.file_name().unwrap().to_str().unwrap();
        let partition_no = partition_name.parse::<usize>().unwrap();

        if context.verbose {
            eprintln!("process {partition_no} during {}", self.fmt())
        }

        for file in std::fs::read_dir(partition_dir).expect("should be there") {
            let spilled = file.unwrap().path();
            let file = std::fs::File::open(spilled)?;
            let reader = IpcReader::new(file);
            let spilled = reader.finish().unwrap();
            if spilled.n_chunks() > 1 {
                for spilled in flatten_df_iter(&spilled) {
                    self.global_table
                        .process_partition_from_dumped(partition_no, &spilled)
                }
            } else {
                self.global_table
                    .process_partition_from_dumped(partition_no, &spilled)
            }
        }
        let df = self.global_table.finalize_partition(partition_no);
        let chunk_idx = self.chunk_idx;
        self.chunk_idx += 1;
        Ok(SourceResult::GotMoreData(vec![DataChunk::new(
            chunk_idx, df,
        )]))
    }
    fn fmt(&self) -> &str {
        "generic-groupby-source"
    }
}
