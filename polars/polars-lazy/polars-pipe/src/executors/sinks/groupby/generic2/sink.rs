use std::arch::x86_64::_mm_undefined_si128;
use std::cell::UnsafeCell;

use super::*;
use crate::executors::sinks::groupby::generic2::global::GlobalTable;
use crate::executors::sinks::groupby::generic2::ooc_state::{OocState, SpillAction};
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
        slice: Option<(i64, usize)>,
    ) -> Self {
        let key_dtypes = output_schema
            .iter_dtypes()
            .take(key_columns.len())
            .map(|dt| dt.clone())
            .collect::<Vec<_>>();

        let global_map =
            GlobalTable::new(agg_constructors.clone(), &key_dtypes, output_schema.clone());

        Self {
            thread_local_table: UnsafeCell::new(ThreadLocalTable::new(
                agg_constructors,
                &key_dtypes,
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
        // load data and hashes
        unsafe {
            // safety: we don't hold mutable refs
            self.eval.evaluate_keys_aggs_and_hashes(context, &chunk)?;
        }
        // safety: we don't hold mutable refs
        let mut keys = unsafe { self.eval.get_keys_iters() };
        // safety: we don't hold mutable refs
        let mut aggs = unsafe { self.eval.get_aggs_iters() };

        let chunk_idx = chunk.chunk_index;
        unsafe {
            // safety: the mutable borrows are not aliasing
            let table = &mut *self.thread_local_table.get();

            for hash in self.eval.hashes() {
                if let Some((partition, spill_payload)) =
                    table.insert(*hash, &mut keys, &mut aggs, chunk_idx)
                {
                    // append payload to global spills
                    self.global_table.spill(partition, spill_payload)
                }
            }
        }

        // clear memory
        unsafe {
            drop(keys);
            drop(aggs);
            // safety: we don't hold mutable refs, we just dropped them
            self.eval.clear()
        };

        // indicates if we should early merge a partition
        // other scenario could be that we must spill to disk
        match self
            .ooc_state
            .check_memory_usage(&|| self.global_table.get_ooc_dump_schema())?
        {
            SpillAction::None => {}
            SpillAction::EarlyMerge => self.global_table.early_merge(),
            SpillAction::Dump => {
                if let Some((partition_no, spill)) = self.global_table.get_ooc_dump() {
                    self.ooc_state.dump(partition_no, spill)
                } else {
                    // do nothing
                }
            }
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
        // safety: no mutable refs at this point
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
        let map = unsafe { (&mut *self.thread_local_table.get()) };
        let (out, spilled) = map.finalize(&mut self.slice);

        // TODO: make source
        Ok(FinalizedSink::Finished(out.unwrap()))
    }

    fn as_any(&mut self) -> &mut dyn Any {
        self
    }

    fn fmt(&self) -> &str {
        "generic-groupby"
    }
}

unsafe impl Sync for GenericGroupby2 {}
