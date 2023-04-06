use std::cell::UnsafeCell;

use super::*;
use crate::expressions::PhysicalPipedExpr;

pub(crate) struct GenericGroupby {
    thread_local_map: UnsafeCell<HashTbl>,
    eval: Eval,
    input_schema: SchemaRef,
    output_schema: SchemaRef,
    slice: Option<(i64, usize)>,
}

impl GenericGroupby {
    pub(crate) fn new(
        key_columns: Arc<Vec<Arc<dyn PhysicalPipedExpr>>>,
        aggregation_columns: Arc<Vec<Arc<dyn PhysicalPipedExpr>>>,
        agg_constructors: Vec<AggregateFunction>,
        input_schema: SchemaRef,
        output_schema: SchemaRef,
        slice: Option<(i64, usize)>,
    ) -> Self {
        let key_dtypes = key_columns
            .iter()
            .map(|e| e.field(&input_schema).unwrap().dtype)
            .collect::<Vec<_>>();
        Self {
            thread_local_map: UnsafeCell::new(HashTbl::new(agg_constructors, &key_dtypes)),
            eval: Eval::new(key_columns, aggregation_columns),
            input_schema,
            output_schema,
            slice,
        }
    }
}

impl Sink for GenericGroupby {
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

        // clear memory
        unsafe {
            drop(keys);
            drop(aggs);
            // safety: we don't hold mutable refs, we just dropped them
            self.eval.clear()
        };
        Ok(SinkResult::CanHaveMoreInput)
    }

    fn combine(&mut self, other: &mut dyn Sink) {
        todo!()
    }

    fn split(&self, thread_no: usize) -> Box<dyn Sink> {
        // safety: no mutable refs at this point
        let map = unsafe { (*self.thread_local_map.get()).split() };
        Box::new(Self {
            eval: self.eval.split(),
            thread_local_map: UnsafeCell::new(map),
            input_schema: self.input_schema.clone(),
            output_schema: self.output_schema.clone(),
            slice: self.slice,
        })
    }

    fn finalize(&mut self, context: &PExecutionContext) -> PolarsResult<FinalizedSink> {
        todo!()
    }

    fn as_any(&mut self) -> &mut dyn Any {
        todo!()
    }

    fn fmt(&self) -> &str {
        todo!()
    }
}

unsafe impl Sync for GenericGroupby {}
