use polars_core::utils::flatten::flatten_df_iter;
use polars_io::ipc::IpcReader;
use polars_io::SerReader;

use super::*;
use crate::executors::sinks::group_by::generic::global::GlobalTable;
use crate::executors::sinks::io::block_thread_until_io_thread_done;
use crate::operators::{Source, SourceResult};
use crate::pipeline::PARTITION_SIZE;

pub(super) struct GroupBySource {
    // holding this keeps the lockfile in place
    _io_thread: IOThread,
    global_table: Arc<GlobalTable>,
    slice: Option<(i64, usize)>,
    partition_processed: usize,
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
            polars_ensure!(slice.0 >= 0, ComputeError: "negative slice not supported with out-of-core group_by")
        }

        block_thread_until_io_thread_done(&io_thread);
        Ok(Self {
            _io_thread: io_thread,
            slice,
            global_table,
            partition_processed: 0,
        })
    }
}

impl Source for GroupBySource {
    fn get_batches(&mut self, context: &PExecutionContext) -> PolarsResult<SourceResult> {
        if self.slice == Some((0, 0)) {
            return Ok(SourceResult::Finished);
        }

        let partition = self.partition_processed;
        self.partition_processed += 1;

        if partition >= PARTITION_SIZE {
            return Ok(SourceResult::Finished);
        }
        let mut partition_dir = self._io_thread.dir.clone();
        partition_dir.push(format!("{partition}"));

        if context.verbose {
            eprintln!("process partition {partition} during {}", self.fmt())
        }

        // merge the dumped tables
        // if no tables are spilled we simply skip
        // this and finalize the in memory state
        if partition_dir.exists() {
            for file in std::fs::read_dir(partition_dir).expect("should be there") {
                let spilled = file.unwrap().path();
                let file = polars_utils::open_file(&spilled)?;
                let reader = IpcReader::new(file);
                let spilled = reader.finish().unwrap();
                if spilled.n_chunks() > 1 {
                    for spilled in flatten_df_iter(&spilled) {
                        self.global_table
                            .process_partition_from_dumped(partition, &spilled)
                    }
                } else {
                    self.global_table
                        .process_partition_from_dumped(partition, &spilled)
                }
            }
        }

        let df = self
            .global_table
            .finalize_partition(partition, &mut self.slice);

        let chunk_idx = self.partition_processed as IdxSize;
        Ok(SourceResult::GotMoreData(vec![DataChunk::new(
            chunk_idx, df,
        )]))
    }
    fn fmt(&self) -> &str {
        "generic-group_by-source"
    }
}
