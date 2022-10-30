use std::any::Any;
use std::sync::Arc;

use polars_core::error::PolarsResult;
use polars_core::frame::DataFrame;
use polars_core::POOL;
use rayon::prelude::*;
use polars_core::prelude::{PlHashMap, PlHashSet};
use polars_utils::arena::Node;
use crate::executors::sinks::OrderedSink;

use crate::operators::{DataChunk, FinalizedSink, Operator, OperatorResult, PExecutionContext, Sink, SinkResult, Source, SourceResult};

pub struct PipeLine {
    sources: Vec<Box<dyn Source>>,
    operators: Vec<Arc<dyn Operator>>,
    sink: Vec<Box<dyn Sink>>,
    rh_sides: Vec<PipeLine>
}

impl PipeLine {
    pub fn new(
        sources: Vec<Box<dyn Source>>,
        operators: Vec<Arc<dyn Operator>>,
        sink: Box<dyn Sink>,
    ) -> PipeLine {
        let n_threads = POOL.current_num_threads();
        let sink = (0..n_threads).map(|i| sink.split(i)).collect();

        PipeLine {
            sources,
            operators,
            sink,
            rh_sides: vec![]
        }
    }

    /// Add a parent
    /// This should be in the right order
    pub fn with_rhs(mut self, rhs: PipeLine) -> Self {
        self.rh_sides.push(rhs);
        self
    }

    fn par_process_chunks(
        &self,
        chunks: Vec<DataChunk>,
        sink: &mut [Box<dyn Sink>],
        ec: &PExecutionContext,
    ) -> PolarsResult<Vec<SinkResult>> {
        debug_assert!(chunks.len() <= sink.len());
        POOL.install(|| {
            chunks
                .into_par_iter()
                .zip(sink.par_iter_mut())
                .map(|(chunk, sink)| {
                    let chunk = match self.push_operators(chunk, ec)? {
                        OperatorResult::Finished(chunk) => chunk,
                        _ => todo!(),
                    };
                    sink.sink(ec, chunk)
                })
                // only collect failed and finished messages as there should be acted upon those
                // the other ones (e.g. success and can have more input) can be ignored
                // this saves a lot of allocations.
                .filter(|result| match result {
                    Ok(sink_result) => matches!(sink_result, SinkResult::Finished),
                    Err(_) => true,
                })
                .collect()
        })
    }

    fn push_operators(
        &self,
        chunk: DataChunk,
        ec: &PExecutionContext,
    ) -> PolarsResult<OperatorResult> {
        let mut in_process = vec![];
        let mut op_iter = self.operators.iter();

        if let Some(op) = op_iter.next() {
            in_process.push((op, chunk));

            while let Some((op, chunk)) = in_process.pop() {
                match op.execute(ec, &chunk)? {
                    OperatorResult::Finished(chunk) => {
                        if let Some(op) = op_iter.next() {
                            in_process.push((op, chunk))
                        } else {
                            return Ok(OperatorResult::Finished(chunk));
                        }
                    }
                    OperatorResult::HaveMoreOutPut(output_chunk) => {
                        if let Some(op) = op_iter.next() {
                            in_process.push((op, output_chunk))
                        }
                        // this operator first at the top of the stack
                        in_process.push((op, chunk))
                    }
                    OperatorResult::NeedMoreInput => {
                        // current chunk will be used again
                        in_process.push((op, chunk))
                    }
                }
            }
            unreachable!()
        } else {
            Ok(OperatorResult::Finished(chunk))
        }
    }

    pub fn execute(&mut self, state: Box<dyn Any + Send + Sync>) -> PolarsResult<DataFrame> {
        let ec = PExecutionContext::new(state);
        let mut sink = std::mem::take(&mut self.sink);

        for src in &mut std::mem::take(&mut self.sources) {
            while let SourceResult::GotMoreData(chunks) = src.get_batches(&ec)? {
                let results = self.par_process_chunks(chunks, &mut sink, &ec)?;

                if results
                    .iter()
                    .any(|sink_result| matches!(sink_result, SinkResult::Finished))
                {
                    break;
                }
            }
        }

        let mut reduced_sink = sink
            .into_par_iter()
            .reduce_with(|mut a, b| {
                a.combine(b);
                a
            })
            .unwrap();

        match reduced_sink.finalize()? {
            FinalizedSink::Finished(df) => Ok(df),
            FinalizedSink::Operator(op) => {
                // for parent in &mut self.parents {
                //     todo!()
                // }
                return Err(todo!());
            }
        }
    }
}
