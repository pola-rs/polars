use std::any::Any;
use std::io::BufRead;
use std::sync::Arc;

use polars_core::error::PolarsResult;
use polars_core::frame::DataFrame;
use polars_core::POOL;
use polars_utils::arena::Node;
use rayon::prelude::*;

use crate::operators::{
    DataChunk, FinalizedSink, Operator, OperatorResult, PExecutionContext, Sink, SinkResult,
    Source, SourceResult,
};

pub struct PipeLine {
    sources: Vec<Box<dyn Source>>,
    operators: Vec<Vec<Box<dyn Operator>>>,
    operator_nodes: Vec<Node>,
    sink: Vec<Box<dyn Sink>>,
    sink_node: Option<Node>,
    rh_sides: Vec<PipeLine>,
    operator_offset: usize,
    n_threads: usize
}

impl PipeLine {
    pub fn new(
        sources: Vec<Box<dyn Source>>,
        operators: Vec<Box<dyn Operator>>,
        operator_nodes: Vec<Node>,
        sink: Box<dyn Sink>,
        sink_node: Option<Node>,
        operator_offset: usize,
    ) -> PipeLine {
        debug_assert_eq!(operators.len(), operator_nodes.len() + operator_offset);
        let n_threads = POOL.current_num_threads();

        // We split so that every thread get's an operator
        let sink = (0..n_threads).map(|i| sink.split(i)).collect();

        // every index maps to a chain of operators than can be pushed as a pipeline for one thread
        let operators = (0..n_threads).map(|i|{
            operators.iter().map(|op| op.split(i)).collect()
        }).collect();

        PipeLine {
            sources,
            operators,
            operator_nodes,
            sink,
            sink_node,
            rh_sides: vec![],
            operator_offset,
            n_threads
        }
    }

    /// Add a parent
    /// This should be in the right order
    pub fn with_rhs(mut self, rhs: PipeLine) -> Self {
        self.rh_sides.push(rhs);
        self
    }

    fn replace_operator(&mut self, op: &Box<dyn Operator>, node: Node) {
        if let Some(pos) = self.operator_nodes.iter().position(|n| *n == node) {
            let pos = pos + self.operator_offset;
            for (i, operator_pipe) in &mut self.operators.iter_mut().enumerate() {
                operator_pipe[pos] = op.split(i)
            }
        }
    }

    fn par_process_chunks(
        &mut self,
        chunks: Vec<DataChunk>,
        sink: &mut [Box<dyn Sink>],
        ec: &PExecutionContext,
    ) -> PolarsResult<Vec<SinkResult>> {
        debug_assert!(chunks.len() <= sink.len());
        let mut operators = std::mem::take(&mut self.operators);
        let out = POOL.install(|| {
            chunks
                .into_par_iter()
                .zip(sink.par_iter_mut())
                .zip(operators.par_iter_mut())
                .map(|((chunk, sink), operator_pipe)| {
                    let chunk = match self.push_operators(chunk, ec, operator_pipe)? {
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
        });
        self.operators = operators;
        out
    }

    fn push_operators(
        &self,
        chunk: DataChunk,
        ec: &PExecutionContext,
        operators: &mut Vec<Box<dyn Operator>>
    ) -> PolarsResult<OperatorResult> {
        let mut in_process = vec![];
        let mut op_iter = operators.iter_mut();

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

    pub fn run_pipeline(&mut self, ec: &PExecutionContext) -> PolarsResult<FinalizedSink> {
        let mut sink = std::mem::take(&mut self.sink);

        for src in &mut std::mem::take(&mut self.sources) {
            while let SourceResult::GotMoreData(chunks) = src.get_batches(ec)? {
                let results = self.par_process_chunks(chunks, &mut sink, ec)?;

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
        reduced_sink.finalize()
    }

    pub fn execute(&mut self, state: Box<dyn Any + Send + Sync>) -> PolarsResult<DataFrame> {
        let ec = PExecutionContext::new(state);
        let mut sink_out = self.run_pipeline(&ec)?;
        let mut pipelines = self.rh_sides.iter_mut();
        let mut sink_node = self.sink_node;

        loop {
            match &mut sink_out {
                FinalizedSink::Finished(df) => return Ok(std::mem::take(df)),

                //
                //  1/\
                //   2/\
                //     3\
                // the left hand side of the join has finished and now is an operator
                // we replace the dummy node in the right hand side pipeline with this
                // operator and then we run the pipeline rinse and repeat
                // until the final right hand side pipeline ran
                FinalizedSink::Operator(op) => {
                    // we unwrap, because the latest pipeline should not return an Operator
                    let pipeline = pipelines.next().unwrap();
                    if let Some(sink_node) = sink_node {
                        pipeline.replace_operator(op, sink_node);
                        sink_out = pipeline.run_pipeline(&ec)?;
                    }
                    sink_node = pipeline.sink_node;
                }
            }
        }
    }
}
