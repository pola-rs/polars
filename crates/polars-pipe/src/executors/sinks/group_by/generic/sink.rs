use std::cell::UnsafeCell;

use polars_core::utils::accumulate_dataframes_vertical_unchecked;

use super::*;
use crate::executors::sinks::group_by::generic::global::GlobalTable;
use crate::executors::sinks::group_by::generic::ooc_state::{OocState, SpillAction};
use crate::executors::sinks::group_by::generic::source::GroupBySource;
use crate::executors::sources::DataFrameSource;
use crate::expressions::PhysicalPipedExpr;

pub(crate) struct GenericGroupby2 {
    thread_local_table: UnsafeCell<ThreadLocalTable>,
    global_table: Arc<GlobalTable>,
    eval: Eval,
    slice: Option<(i64, usize)>,
    ooc_state: OocState,
}

impl GenericGroupby2 {
    pub(crate) fn new(
        key_columns: Arc<Vec<Arc<dyn PhysicalPipedExpr>>>,
        aggregation_columns: Arc<Vec<Arc<dyn PhysicalPipedExpr>>>,
        agg_constructors: Arc<[AggregateFunction]>,
        output_schema: SchemaRef,
        agg_input_dtypes: Vec<DataType>,
        slice: Option<(i64, usize)>,
    ) -> Self {
        let key_dtypes: Arc<[DataType]> = Arc::from(
            output_schema
                .iter_dtypes()
                .take(key_columns.len())
                .cloned()
                .collect::<Vec<_>>(),
        );

        let agg_dtypes: Arc<[DataType]> = Arc::from(agg_input_dtypes);

        let global_map = GlobalTable::new(
            agg_constructors.clone(),
            key_dtypes.as_ref(),
            output_schema.clone(),
        );

        Self {
            thread_local_table: UnsafeCell::new(ThreadLocalTable::new(
                agg_constructors,
                key_dtypes,
                agg_dtypes,
                output_schema,
            )),
            global_table: Arc::new(global_map),
            eval: Eval::new(key_columns, aggregation_columns),
            slice,
            ooc_state: Default::default(),
        }
    }
}

impl Sink for GenericGroupby2 {
    fn sink(&mut self, context: &PExecutionContext, chunk: DataChunk) -> PolarsResult<SinkResult> {
        if chunk.is_empty() {
            return Ok(SinkResult::CanHaveMoreInput);
        }
        // load data and hashes
        unsafe {
            // SAFETY: we don't hold mutable refs
            self.eval.evaluate_keys_aggs_and_hashes(context, &chunk)?;
        }
        // SAFETY: eval is alive for the duration of keys
        let keys = unsafe { self.eval.get_keys_iter() };
        // SAFETY: we don't hold mutable refs
        let mut aggs = unsafe { self.eval.get_aggs_iters() };

        let chunk_idx = chunk.chunk_index;
        unsafe {
            // SAFETY: the mutable borrows are not aliasing
            let table = &mut *self.thread_local_table.get();

            for (hash, row) in self.eval.hashes().iter().zip(keys.values_iter()) {
                if let Some((partition, spill_payload)) =
                    table.insert(*hash, row, &mut aggs, chunk_idx)
                {
                    self.global_table.spill(partition, spill_payload)
                }
            }
        }

        // clear memory
        unsafe {
            drop(aggs);
            // SAFETY: we don't hold mutable refs, we just dropped them
            self.eval.clear()
        };

        // indicates if we should early merge a partition
        // other scenario could be that we must spill to disk
        match self
            .ooc_state
            .check_memory_usage(&|| self.global_table.get_ooc_dump_schema())?
        {
            SpillAction::None => {},
            SpillAction::EarlyMerge => self.global_table.early_merge(),
            SpillAction::Dump => {
                if let Some((partition_no, spill)) = self.global_table.get_ooc_dump() {
                    self.ooc_state.dump(partition_no, spill)
                } else {
                    // do nothing
                }
            },
        }
        Ok(SinkResult::CanHaveMoreInput)
    }

    fn combine(&mut self, other: &mut dyn Sink) {
        let other = other.as_any().downcast_mut::<Self>().unwrap();
        unsafe {
            let map = &mut *self.thread_local_table.get();
            let other_map = &mut *other.thread_local_table.get();
            map.combine(other_map);
        }
    }

    fn split(&self, _thread_no: usize) -> Box<dyn Sink> {
        // SAFETY: no mutable refs at this point
        let map = unsafe { (*self.thread_local_table.get()).split() };
        Box::new(Self {
            eval: self.eval.split(),
            thread_local_table: UnsafeCell::new(map),
            global_table: self.global_table.clone(),
            slice: self.slice,
            ooc_state: self.ooc_state.clone(),
        })
    }

    fn finalize(&mut self, context: &PExecutionContext) -> PolarsResult<FinalizedSink> {
        let map = unsafe { &mut *self.thread_local_table.get() };

        // only succeeds if it hasn't spilled to global
        if let Some(out) = map.finalize(&mut self.slice) {
            if context.verbose {
                eprintln!("finish streaming aggregation with local in-memory table")
            }
            Ok(FinalizedSink::Finished(out))
        } else {
            // ensure the global map gets all overflow buckets
            for (partition, payload) in map.get_all_spilled() {
                self.global_table.spill(partition, payload);
            }
            // ensure the global map update the partitioned hash tables with keys from local map
            self.global_table.merge_local_map(map.get_inner_map_mut());

            // all data is in memory
            // finalize
            if !self.ooc_state.ooc {
                if context.verbose {
                    eprintln!("finish streaming aggregation with global in-memory table")
                }

                let out = self.global_table.finalize(&mut self.slice);
                let src = DataFrameSource::from_df(accumulate_dataframes_vertical_unchecked(out));
                Ok(FinalizedSink::Source(Box::new(src)))
            }
            // create an ooc source
            else {
                Ok(FinalizedSink::Source(Box::new(GroupBySource::new(
                    &self.ooc_state.io_thread,
                    self.slice,
                    self.global_table.clone(),
                )?)))
            }
        }
    }

    fn as_any(&mut self) -> &mut dyn Any {
        self
    }

    fn fmt(&self) -> &str {
        "generic-group_by"
    }
}

unsafe impl Sync for GenericGroupby2 {}
