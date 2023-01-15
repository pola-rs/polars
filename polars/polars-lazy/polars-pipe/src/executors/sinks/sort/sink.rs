use std::any::Any;
use std::collections::VecDeque;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

use polars_core::error::PolarsResult;
use polars_core::frame::DataFrame;
use polars_core::prelude::{AnyValue, SchemaRef, Series};
use polars_core::utils::accumulate_dataframes_vertical_unchecked;
use polars_utils::atomic::SyncCounter;
use polars_utils::sys::MEMINFO;

use crate::executors::sinks::sort::io::{block_thread_until_io_thread_done, IOThread};
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
    slice: Option<(i64, usize)>,
    // sampled values so we can find the distribution.
    dist_sample: Vec<AnyValue<'static>>,
}

impl SortSink {
    pub(crate) fn new(
        sort_idx: usize,
        reverse: bool,
        schema: SchemaRef,
        slice: Option<(i64, usize)>,
    ) -> Self {
        // for testing purposes
        let ooc = std::env::var("POLARS_FORCE_OOC_SORT").is_ok();

        let mut out = Self {
            schema,
            chunks: Default::default(),
            free_mem: SyncCounter::new(0),
            mem_total: SyncCounter::new(0),
            ooc,
            io_thread: Default::default(),
            sort_idx,
            reverse,
            slice,
            dist_sample: vec![],
        };
        if ooc {
            eprintln!("OOC sort forced");
            out.init_ooc().unwrap();
        }
        out
    }

    fn refresh_memory(&self) {
        if self.free_mem.load(Ordering::Relaxed) == 0 {
            self.free_mem
                .store(MEMINFO.free() as usize, Ordering::Relaxed);
        }
    }

    fn init_ooc(&mut self) -> PolarsResult<()> {
        self.ooc = true;

        // start IO thread
        let mut iot = self.io_thread.lock().unwrap();
        if iot.is_none() {
            *iot = Some(IOThread::try_new(self.schema.clone())?)
        }
        Ok(())
    }

    fn store_chunk(&mut self, chunk: DataChunk) -> PolarsResult<()> {
        let chunk_bytes = chunk.data.estimated_size();

        if !self.ooc {
            let used = self.mem_total.fetch_add(chunk_bytes, Ordering::Relaxed);
            let free = self.free_mem.load(Ordering::Relaxed);

            // we need some free memory to be able to sort
            // so we keep 3x the sort data size before we go out of core
            if used * 3 > free {
                self.init_ooc()?;
            }
        }
        self.chunks.push_back(chunk.data);
        Ok(())
    }

    fn dump(&mut self) -> PolarsResult<()> {
        // take from the front so that sorted data remains sorted in writing order
        while let Some(df) = self.chunks.pop_front() {
            if df.height() > 0 {
                // safety: we just asserted height > 0
                let sample = unsafe {
                    let s = &df.get_columns()[self.sort_idx];
                    s.to_physical_repr().get_unchecked(0).into_static().unwrap()
                };
                self.dist_sample.push(sample);

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
        let other = other.as_any().downcast_mut::<Self>().unwrap();
        self.chunks.extend(std::mem::take(&mut other.chunks));
        self.ooc |= other.ooc;
        self.dist_sample
            .extend(std::mem::take(&mut other.dist_sample));

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
            ooc: self.ooc,
            io_thread: self.io_thread.clone(),
            sort_idx: self.sort_idx,
            reverse: self.reverse,
            dist_sample: vec![],
            slice: self.slice,
        })
    }

    fn finalize(&mut self, _context: &PExecutionContext) -> PolarsResult<FinalizedSink> {
        // safety: we are the final thread and will drop only once.
        unsafe {
            self.mem_total.manual_drop();
            self.free_mem.manual_drop();
        }

        if self.ooc {
            let lock = self.io_thread.lock().unwrap();
            let io_thread = lock.as_ref().unwrap();

            let dist = Series::from_any_values("", &self.dist_sample).unwrap();
            let dist = dist.sort(self.reverse);

            block_thread_until_io_thread_done(io_thread);

            sort_ooc(io_thread, dist, self.sort_idx, self.reverse, self.slice)
        } else {
            let chunks = std::mem::take(&mut self.chunks);
            let df = accumulate_dataframes_vertical_unchecked(chunks);
            let df = sort_accumulated(df, self.sort_idx, self.reverse, self.slice)?;
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

pub(super) fn sort_accumulated(
    df: DataFrame,
    sort_idx: usize,
    reverse: bool,
    slice: Option<(i64, usize)>,
) -> PolarsResult<DataFrame> {
    let sort_column = df.get_columns()[sort_idx].clone();
    df.sort_impl(vec![sort_column], vec![reverse], false, slice, true)
}
