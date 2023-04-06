use std::cell::UnsafeCell;

use super::*;

pub(crate) struct GenericGroupby {
    thread_local_map: UnsafeCell<HashTbl>,
    eval: Eval,
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
        todo!()
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
