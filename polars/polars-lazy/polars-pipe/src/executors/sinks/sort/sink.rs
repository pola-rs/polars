use std::any::Any;
use std::collections::VecDeque;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use polars_core::error::PolarsResult;
use polars_core::frame::DataFrame;
use polars_core::prelude::{AnyValue, SchemaRef, Series, SortOptions};
use polars_core::utils::accumulate_dataframes_vertical_unchecked;
use polars_utils::atomic::SyncCounter;
use polars_utils::sys::MEMINFO;

use crate::executors::sinks::sort::io::IOThread;
use crate::executors::sinks::sort::ooc::sort_ooc;
use crate::operators::{DataChunk, FinalizedSink, PExecutionContext, Sink, SinkResult};

pub struct SortSink {
    schema: SchemaRef,
    chunks: VecDeque<DataFrame>,
    free_mem: SyncCounter,
    // Total memory used by sort sink
    mem_total: SyncCounter,
    // sort in memory or out of core
    ooc: bool,
    // when ooc, we write to disk using an IO thread
    io_thread: Arc<Mutex<Option<IOThread>>>,
    // location in the dataframe of the columns to sort by
    sort_idx: usize,
    reverse: bool,
    // sampled values so we can find the distribution.
    dist_sample: Vec<AnyValue<'static>>
}

impl SortSink {
    pub(crate) fn new(sort_idx: usize, reverse: bool, schema: SchemaRef) -> Self {
        Self {
            schema,
            chunks: Default::default(),
            free_mem: SyncCounter::new(0),
            mem_total: SyncCounter::new(0),
            ooc: false,
            io_thread: Default::default(),
            sort_idx,
            reverse,
            dist_sample: vec![]
        }
    }

    fn refresh_memory(&self) {
        if self.free_mem.load(Ordering::Relaxed) == 0 {
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
                    *iot = Some(IOThread::try_new(self.schema.clone())?)
                }
            }

            // TODO! remove, only for testing
            if used > 200_000 {
                dbg!("Start ooc");
                self.ooc = true;

                // start IO thread
                let mut iot = self.io_thread.lock().unwrap();
                if iot.is_none() {
                    *iot = Some(IOThread::try_new(self.schema.clone())?)
                }
            }
        }
        self.chunks.push_back(chunk.data);
        Ok(())
    }

    fn dump(&mut self) -> PolarsResult<()> {
        // take from the front so that sorted data remains sorted in writing order
        while let Some(mut df) = self.chunks.pop_front() {
            if df.height() > 0 {
                // safety: we just asserted height > 0
                let sample = unsafe { df.get_columns()[self.sort_idx].get_unchecked(0) };
                self.dist_sample.push(sample.into_static().unwrap());

                let iot = self.io_thread.lock().unwrap();
                let iot = iot.as_ref().unwrap();
                iot.dump_chunk(df)
            }
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
            self.dump()?;
        }
        Ok(SinkResult::CanHaveMoreInput)
    }

    fn combine(&mut self, mut other: Box<dyn Sink>) {
        let mut other = other.as_any().downcast_mut::<Self>().unwrap();
        self.chunks.extend(std::mem::take(&mut other.chunks));
        self.ooc |= other.ooc;
        self.dist_sample.extend(std::mem::take(&mut other.dist_sample));

        if self.ooc {
            self.dump().unwrap()
        }
    }

    fn split(&self, _thread_no: usize) -> Box<dyn Sink> {
        Box::new(Self {
            schema: self.schema.clone(),
            chunks: Default::default(),
            free_mem: self.free_mem.clone(),
            mem_total: self.mem_total.clone(),
            ooc: self.ooc.clone(),
            io_thread: self.io_thread.clone(),
            sort_idx: self.sort_idx,
            reverse: self.reverse,
            dist_sample: vec![],
        })
    }

    fn finalize(&mut self, _context: &PExecutionContext) -> PolarsResult<FinalizedSink> {
        if self.ooc {
            let lock = self.io_thread.lock().unwrap();
            let io_thread = lock.as_ref().unwrap();
            let all_processed = io_thread.all_processed.clone();

            let dist = Series::from_any_values("", &self.dist_sample).unwrap();
            let dist = dist.sort(self.reverse);


            // get number sent
            let sent = io_thread.sent.load(Ordering::Relaxed);
            // set total sent
            io_thread.total.fetch_add(sent, Ordering::Relaxed);

            // then the io thread will check if it has written all files, and if it has
            // it will set the condvar so we can continue on this thread

            // we don't really need the mutex for our case, but the condvar needs one
            let cond_lock = io_thread.all_processed.1.lock().unwrap();
            all_processed.0.wait(cond_lock).unwrap();

            sort_ooc(io_thread, dist, self.sort_idx, &self.schema).map(FinalizedSink::Finished)
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
