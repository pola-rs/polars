use std::any::Any;
use std::collections::VecDeque;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

use polars_core::error::PolarsResult;
use polars_core::frame::DataFrame;
use polars_core::prelude::SortOptions;
use polars_core::utils::accumulate_dataframes_vertical_unchecked;
use polars_utils::atomic::SyncCounter;
use polars_utils::sys::MEMINFO;

use crate::executors::sinks::sort::io::IOThread;
use crate::operators::{DataChunk, FinalizedSink, PExecutionContext, Sink, SinkResult};

pub struct SortSink {
    chunks: VecDeque<DataFrame>,
    free_mem: SyncCounter,
    // Total memory used by sort sink
    mem_total: SyncCounter,
    // sort in memory or out of core
    ooc: bool,
    // when ooc, we write to disk using an IO thread
    io_thread: Arc<Mutex<Option<IOThread>>>,
    // location in the dataframe of the columns to sort by
    sort_idx: Vec<usize>,
    reverse: Vec<bool>,
}

impl SortSink {
    pub(crate) fn new(sort_idx: Vec<usize>, reverse: Vec<bool>) -> Self {
        Self {
            chunks: Default::default(),
            free_mem: SyncCounter::new(0),
            mem_total: SyncCounter::new(0),
            ooc: false,
            io_thread: Default::default(),
            sort_idx,
            reverse,
        }
    }

    fn refresh_memory(&self) {
        if !self.free_mem.load(Ordering::Relaxed) == 0 {
            self.free_mem
                .store(MEMINFO.free() as usize, Ordering::Relaxed);
        }
    }

    fn store_chunk(&mut self, chunk: DataChunk) -> PolarsResult<()> {
        let chunk_bytes = chunk.data.estimated_size();

        if !self.ooc {
            let used = self.mem_total.fetch_add(chunk_bytes, Ordering::Relaxed);
            let free = self.free_mem.load(Ordering::Relaxed);

            // we need some free memory to be able to sort
            // so we keep 3x the sort data size before we go out of core
            if used * 3 > free {
                self.ooc = true;

                // start IO thread
                let mut iot = self.io_thread.lock().unwrap();
                if iot.is_none() {
                    *iot = Some(IOThread::try_new()?)
                }
            }
        }
        self.chunks.push_back(chunk.data);
        Ok(())
    }

    fn sort_and_dump(&mut self) -> PolarsResult<()> {
        // take from the front so that sorted data remains sorted in writing order
        while let Some(mut df) = self.chunks.pop_front() {
            let cols = df.get_columns();
            let sort_cols = self
                .sort_idx
                .iter()
                .map(|i| cols[*i].clone())
                .collect::<Vec<_>>();

            df = df.sort_impl(sort_cols, self.reverse.clone(), false, None, false)?;

            let iot = self.io_thread.lock().unwrap();
            let iot = iot.as_ref().unwrap();
            iot.dump_chunk(df)
        }
        Ok(())
    }
}

impl Sink for SortSink {
    fn sink(&mut self, _context: &PExecutionContext, chunk: DataChunk) -> PolarsResult<SinkResult> {
        if !self.ooc {
            self.refresh_memory();
        }
        self.store_chunk(chunk)?;

        if self.ooc {
            self.sort_and_dump()?;
        }
        Ok(SinkResult::CanHaveMoreInput)
    }

    fn combine(&mut self, mut other: Box<dyn Sink>) {
        let mut other = other.as_any().downcast_mut::<Self>().unwrap();
        self.chunks.extend(std::mem::take(&mut other.chunks));
        self.sort_and_dump().unwrap()
    }

    fn split(&self, _thread_no: usize) -> Box<dyn Sink> {
        Box::new(Self {
            chunks: Default::default(),
            free_mem: self.free_mem.clone(),
            mem_total: self.mem_total.clone(),
            ooc: self.ooc.clone(),
            io_thread: self.io_thread.clone(),
            sort_idx: self.sort_idx.clone(),
            reverse: self.reverse.clone(),
        })
    }

    fn finalize(&mut self, context: &PExecutionContext) -> PolarsResult<FinalizedSink> {
        if self.ooc {
            todo!()
        } else {
            let df = accumulate_dataframes_vertical_unchecked(std::mem::take(&mut self.chunks));
            Ok(FinalizedSink::Finished(df))
        }
    }

    fn as_any(&mut self) -> &mut dyn Any {
        self
    }

    fn fmt(&self) -> &str {
        "sort"
    }
}
