use polars_core::error::PolarsResult;
use polars_core::frame::DataFrame;
use polars_core::POOL;
use polars_utils::arena::Node;
use rayon::prelude::*;

use crate::executors::sources::DataFrameSource;
use crate::operators::{
    DataChunk, FinalizedSink, Operator, OperatorResult, PExecutionContext, SExecutionContext, Sink,
    SinkResult, Source, SourceResult,
};
use crate::pipeline::morsels_per_sink;

pub struct PipeLine {
    sources: Vec<Box<dyn Source>>,
    operators: Vec<Vec<Box<dyn Operator>>>,
    operator_nodes: Vec<Node>,
    // offset in the operators vec
    // node of the sink
    sinks: Vec<(usize, Vec<Box<dyn Sink>>)>,
    sink_nodes: Vec<Node>,
    rh_sides: Vec<PipeLine>,
    operator_offset: usize,
}

impl PipeLine {
    pub fn new(
        sources: Vec<Box<dyn Source>>,
        operators: Vec<Box<dyn Operator>>,
        operator_nodes: Vec<Node>,
        sink_and_nodes: Vec<(usize, Node, Box<dyn Sink>)>,
        operator_offset: usize,
    ) -> PipeLine {
        debug_assert_eq!(operators.len(), operator_nodes.len() + operator_offset);
        // we don't use the power of two partition size here
        // we only do that in the sinks itself.
        let n_threads = morsels_per_sink();

        // We split so that every thread gets an operator
        let sink_nodes = sink_and_nodes.iter().map(|(_, node, _)| *node).collect();
        let sinks = sink_and_nodes
            .into_iter()
            .map(|(offset, _, sink)| (offset, (0..n_threads).map(|i| sink.split(i)).collect()))
            .collect();

        // every index maps to a chain of operators than can be pushed as a pipeline for one thread
        let operators = (0..n_threads)
            .map(|i| operators.iter().map(|op| op.split(i)).collect())
            .collect();

        PipeLine {
            sources,
            operators,
            operator_nodes,
            sinks,
            sink_nodes,
            rh_sides: vec![],
            operator_offset,
        }
    }

    /// Add a parent
    /// This should be in the right order
    pub fn with_rhs(mut self, rhs: PipeLine) -> Self {
        self.rh_sides.push(rhs);
        self
    }

    fn replace_operator(&mut self, op: &dyn Operator, node: Node) {
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
        operator_start: usize,
        operator_end: usize,
    ) -> PolarsResult<Vec<SinkResult>> {
        debug_assert!(chunks.len() <= sink.len());
        let mut operators = std::mem::take(&mut self.operators);
        let out = POOL.install(|| {
            chunks
                .into_par_iter()
                .zip(sink.par_iter_mut())
                .zip(operators.par_iter_mut())
                .map(|((chunk, sink), operator_pipe)| {
                    // truncate the operators that should run into the current sink.
                    let operator_pipe = &mut operator_pipe[operator_start..operator_end];

                    if operator_pipe.is_empty() {
                        sink.sink(ec, chunk)
                    } else {
                        self.push_operators(chunk, ec, operator_pipe, sink)
                    }
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
        operators: &mut [Box<dyn Operator>],
        sink: &mut Box<dyn Sink>,
    ) -> PolarsResult<SinkResult> {
        debug_assert!(!operators.is_empty());
        let mut in_process = vec![];

        let operator_offset = 0usize;
        in_process.push((operator_offset, chunk));

        while let Some((op_i, chunk)) = in_process.pop() {
            match operators.get_mut(op_i) {
                None => {
                    if let SinkResult::Finished = sink.sink(ec, chunk)? {
                        return Ok(SinkResult::Finished);
                    }
                }
                Some(op) => {
                    match op.execute(ec, &chunk)? {
                        OperatorResult::Finished(chunk) => in_process.push((op_i + 1, chunk)),
                        OperatorResult::HaveMoreOutPut(output_chunk) => {
                            // first on the stack the next operator call
                            in_process.push((op_i, chunk));

                            // but first push the output in the next operator
                            // is a join can produce many rows, we want the filter to
                            // be executed in between.
                            // or sink into a slice so that we get sink::finished
                            // before we grow the stack with ever more coming chunks
                            in_process.push((op_i + 1, output_chunk));
                        }
                        OperatorResult::NeedsNewData => {
                            // done, take another chunk from the stack
                        }
                    }
                }
            }
        }
        Ok(SinkResult::CanHaveMoreInput)
    }

    fn set_sources(&mut self, df: DataFrame) {
        self.sources.clear();
        self.sources
            .push(Box::new(DataFrameSource::from_df(df)) as Box<dyn Source>);
    }

    pub fn run_pipeline(&mut self, ec: &PExecutionContext) -> PolarsResult<FinalizedSink> {
        let mut out = None;
        let mut operator_start = 0;
        let last_i = self.sinks.len() - 1;
        for (i, (operator_end, mut sink)) in std::mem::take(&mut self.sinks).into_iter().enumerate()
        {
            for src in &mut std::mem::take(&mut self.sources) {
                while let SourceResult::GotMoreData(chunks) = src.get_batches(ec)? {
                    let results = self.par_process_chunks(
                        chunks,
                        &mut sink,
                        ec,
                        operator_start,
                        operator_end,
                    )?;

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
            let sink_result = reduced_sink.finalize(ec)?;
            operator_start = operator_end;

            if i != last_i {
                match sink_result {
                    // turn this sink an a new source
                    FinalizedSink::Finished(df) => self.set_sources(df),
                    // should not happen
                    FinalizedSink::Operator(_) => {
                        unreachable!()
                    }
                }
            } else {
                out = Some(sink_result)
            }
        }
        Ok(out.unwrap())
    }

    pub fn execute(&mut self, state: Box<dyn SExecutionContext>) -> PolarsResult<DataFrame> {
        let ec = PExecutionContext::new(state);
        let mut sink_out = self.run_pipeline(&ec)?;
        let mut pipelines = self.rh_sides.iter_mut();
        let mut sink_nodes = std::mem::take(&mut self.sink_nodes);

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

                    // latest sink_node will be the operator, as the left side of the join
                    // always finishes that branch.
                    if let Some(sink_node) = sink_nodes.pop() {
                        pipeline.replace_operator(op.as_ref(), sink_node);
                    }
                    sink_out = pipeline.run_pipeline(&ec)?;
                    sink_nodes = std::mem::take(&mut pipeline.sink_nodes);
                }
            }
        }
    }
}
