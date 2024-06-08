use std::any::Any;
use std::sync::{Arc, RwLock};
use std::time::Instant;

use polars_core::chunked_array::ops::SortMultipleOptions;
use polars_core::config::verbose;
use polars_core::error::PolarsResult;
use polars_core::frame::DataFrame;
use polars_core::prelude::{AnyValue, SchemaRef, Series, SortOptions};
use polars_core::utils::accumulate_dataframes_vertical_unchecked;

use crate::executors::sinks::io::{block_thread_until_io_thread_done, IOThread};
use crate::executors::sinks::memory::MemTracker;
use crate::executors::sinks::sort::ooc::sort_ooc;
use crate::operators::{DataChunk, FinalizedSink, PExecutionContext, Sink, SinkResult};
use crate::pipeline::{morsels_per_sink, FORCE_OOC};

pub struct SortSink {
    schema: SchemaRef,
    chunks: Vec<DataFrame>,
    // Stores available memory in the system at the start of this sink.
    // and stores the memory used by this this sink.
    mem_track: MemTracker,
    // sort in-memory or out-of-core
    ooc: bool,
    // when ooc, we write to disk using an IO thread
    // RwLock as we want to have multiple readers at once.
    io_thread: Arc<RwLock<Option<IOThread>>>,
    // location in the dataframe of the columns to sort by
    sort_idx: usize,
    slice: Option<(i64, usize)>,
    sort_options: SortMultipleOptions,
    // Statistics
    // sampled values so we can find the distribution.
    dist_sample: Vec<AnyValue<'static>>,
    // total rows accumulated in current chunk
    current_chunk_rows: usize,
    // total bytes of tables in current chunks
    current_chunks_size: usize,
    // Start time of OOC phase.
    ooc_start: Option<Instant>,
}

impl SortSink {
    pub(crate) fn new(
        sort_idx: usize,
        slice: Option<(i64, usize)>,
        sort_options: SortMultipleOptions,
        schema: SchemaRef,
    ) -> Self {
        // for testing purposes
        let ooc = std::env::var(FORCE_OOC).is_ok();
        let n_morsels_per_sink = morsels_per_sink();

        let mut out = Self {
            schema,
            chunks: Default::default(),
            mem_track: MemTracker::new(n_morsels_per_sink),
            ooc,
            io_thread: Default::default(),
            sort_idx,
            slice,
            sort_options,
            dist_sample: vec![],
            current_chunk_rows: 0,
            current_chunks_size: 0,
            ooc_start: None,
        };
        if ooc {
            if verbose() {
                eprintln!("OOC sort forced");
            }
            out.init_ooc().unwrap();
        }
        out
    }

    fn init_ooc(&mut self) -> PolarsResult<()> {
        if verbose() {
            eprintln!("OOC sort started");
        }
        self.ooc_start = Some(Instant::now());
        self.ooc = true;

        // start IO thread
        let mut iot = self.io_thread.write().unwrap();
        if iot.is_none() {
            *iot = Some(IOThread::try_new(self.schema.clone(), "sort")?)
        }
        Ok(())
    }

    fn store_chunk(&mut self, chunk: DataChunk) -> PolarsResult<()> {
        let chunk_bytes = chunk.data.estimated_size();
        if !self.ooc {
            let used = self.mem_track.fetch_add(chunk_bytes);
            let free = self.mem_track.get_available();

            // we need some free memory to be able to sort
            // so we keep 3x the sort data size before we go out of core
            if used * 3 > free {
                self.init_ooc()?;
                self.dump(true)?;
            }
        };
        // don't add empty dataframes
        if chunk.data.height() > 0 || self.chunks.is_empty() {
            self.current_chunks_size += chunk_bytes;
            self.current_chunk_rows += chunk.data.height();
            self.chunks.push(chunk.data);
        }
        Ok(())
    }

    fn dump(&mut self, force: bool) -> PolarsResult<()> {
        let larger_than_32_mb = self.current_chunks_size > (1 << 25);
        if (force || larger_than_32_mb) && !self.chunks.is_empty() {
            // into a single chunk because multiple file IO's is expensive
            // and may lead to many smaller files in ooc-sort later, which is exponentially
            // expensive
            let df = accumulate_dataframes_vertical_unchecked(self.chunks.drain(..));
            if df.height() > 0 {
                // SAFETY: we just asserted height > 0
                let sample = unsafe {
                    let s = &df.get_columns()[self.sort_idx];
                    s.to_physical_repr().get_unchecked(0).into_static().unwrap()
                };
                self.dist_sample.push(sample);

                let iot = self.io_thread.read().unwrap();
                let iot = iot.as_ref().unwrap();

                iot.dump_chunk(df);

                // reset sizes
                self.current_chunk_rows = 0;
                self.current_chunks_size = 0;
            }
        }
        Ok(())
    }
}

impl Sink for SortSink {
    fn sink(&mut self, _context: &PExecutionContext, chunk: DataChunk) -> PolarsResult<SinkResult> {
        self.store_chunk(chunk)?;

        if self.ooc {
            self.dump(false)?;
        }
        Ok(SinkResult::CanHaveMoreInput)
    }

    fn combine(&mut self, other: &mut dyn Sink) {
        let other = other.as_any().downcast_mut::<Self>().unwrap();
        if let Some(ooc_start) = other.ooc_start {
            self.ooc_start = Some(ooc_start);
        }
        self.chunks.extend(std::mem::take(&mut other.chunks));
        self.ooc |= other.ooc;
        self.dist_sample
            .extend(std::mem::take(&mut other.dist_sample));

        if self.ooc {
            self.dump(false).unwrap()
        }
    }

    fn split(&self, _thread_no: usize) -> Box<dyn Sink> {
        Box::new(Self {
            schema: self.schema.clone(),
            chunks: Default::default(),
            mem_track: self.mem_track.clone(),
            ooc: self.ooc,
            io_thread: self.io_thread.clone(),
            sort_idx: self.sort_idx,
            slice: self.slice,
            sort_options: self.sort_options.clone(),
            dist_sample: vec![],
            current_chunk_rows: 0,
            current_chunks_size: 0,
            ooc_start: self.ooc_start,
        })
    }

    fn finalize(&mut self, context: &PExecutionContext) -> PolarsResult<FinalizedSink> {
        if self.ooc {
            // spill everything
            self.dump(true).unwrap();
            let mut lock = self.io_thread.write().unwrap();
            let io_thread = lock.take().unwrap();

            let dist = Series::from_any_values("", &self.dist_sample, true).unwrap();
            let dist = dist.sort_with(SortOptions::from(&self.sort_options))?;

            let instant = self.ooc_start.unwrap();
            if context.verbose {
                eprintln!("finished sinking into OOC sort in {:?}", instant.elapsed());
            }
            block_thread_until_io_thread_done(&io_thread);
            if context.verbose {
                eprintln!("full file dump of OOC sort took {:?}", instant.elapsed());
            }

            sort_ooc(
                io_thread,
                dist,
                self.sort_idx,
                self.sort_options.descending[0],
                self.sort_options.nulls_last[0],
                self.slice,
                context.verbose,
                self.mem_track.clone(),
                instant,
            )
        } else {
            let chunks = std::mem::take(&mut self.chunks);
            let df = accumulate_dataframes_vertical_unchecked(chunks);
            let df = sort_accumulated(
                df,
                self.sort_idx,
                self.slice,
                SortOptions::from(&self.sort_options),
            )?;
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
    mut df: DataFrame,
    sort_idx: usize,
    slice: Option<(i64, usize)>,
    sort_options: SortOptions,
) -> PolarsResult<DataFrame> {
    // This is needed because we can have empty blocks and we require chunks to have single chunks.
    df.as_single_chunk_par();
    let sort_column = df.get_columns()[sort_idx].clone();
    df.sort_impl(
        vec![sort_column],
        SortMultipleOptions::from(&sort_options),
        slice,
    )
}
