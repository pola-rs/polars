use std::collections::VecDeque;

use polars_core::error::PolarsResult;
use polars_core::frame::DataFrame;
use polars_core::utils::accumulate_dataframes_vertical_unchecked;
use polars_core::POOL;
use polars_utils::arena::Node;
use rayon::prelude::*;

use crate::executors::operators::PlaceHolder;
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
    verbose: bool,
}

impl PipeLine {
    pub fn new(
        sources: Vec<Box<dyn Source>>,
        operators: Vec<Box<dyn Operator>>,
        operator_nodes: Vec<Node>,
        sink_and_nodes: Vec<(usize, Node, Box<dyn Sink>)>,
        operator_offset: usize,
        verbose: bool,
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
            verbose,
        }
    }

    /// Add a parent
    /// This should be in the right order
    pub fn with_rhs(mut self, rhs: PipeLine) -> Self {
        self.rh_sides.push(rhs);
        self
    }

    // returns if operator was successfully replaced
    fn replace_operator(&mut self, op: &dyn Operator, node: Node) -> bool {
        if let Some(pos) = self.operator_nodes.iter().position(|n| *n == node) {
            let pos = pos + self.operator_offset;
            for (i, operator_pipe) in &mut self.operators.iter_mut().enumerate() {
                operator_pipe[pos] = op.split(i)
            }
            true
        } else {
            false
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

    fn set_df_as_sources(&mut self, df: DataFrame) {
        let src = Box::new(DataFrameSource::from_df(df)) as Box<dyn Source>;
        self.set_sources(src)
    }

    fn set_sources(&mut self, src: Box<dyn Source>) {
        self.sources.clear();
        self.sources.push(src);
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

            let mut reduced_sink = POOL
                .install(|| {
                    sink.into_par_iter().reduce_with(|mut a, b| {
                        a.combine(b);
                        a
                    })
                })
                .unwrap();
            let sink_result = reduced_sink.finalize(ec)?;
            operator_start = operator_end;

            if i != last_i {
                match sink_result {
                    // turn this sink an a new source
                    FinalizedSink::Finished(df) => self.set_df_as_sources(df),
                    FinalizedSink::Source(src) => self.set_sources(src),
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

    /// print the branches of the pipeline
    /// in the order they run.
    fn show(&self) {
        let mut fmt = String::new();
        let mut start = 0usize;
        fmt.push_str(self.sources[0].fmt());
        for (offset_end, sink) in &self.sinks {
            fmt.push_str(" -> ");
            // take operators of a single thread
            let ops = &self.operators[0];
            // slice the pipeline
            let ops = &ops[start..*offset_end];
            for op in ops {
                fmt.push_str(op.fmt());
                fmt.push_str(" -> ")
            }
            start = *offset_end;
            fmt.push_str(sink[0].fmt())
        }
        eprintln!("{fmt}");
        for pl in &self.rh_sides {
            pl.show()
        }
    }

    pub fn execute(&mut self, state: Box<dyn SExecutionContext>) -> PolarsResult<DataFrame> {
        let ec = PExecutionContext::new(state);

        if self.verbose {
            self.show();
        }
        let mut sink_out = self.run_pipeline(&ec)?;
        let mut pipelines = self.rh_sides.iter_mut();
        let mut sink_nodes = std::mem::take(&mut self.sink_nodes);

        // This is a stack of operators that should replace the sinks of join nodes
        // If we don't reorder joins, the order we run the pipelines coincide with the
        // order the sinks need to be replaced, however this is not always the case
        // if we reorder joins.
        // This stack ensures we still replace the dummy operators even if they are all in
        // the most right branch
        let mut operators_to_replace: VecDeque<(Box<dyn Operator>, Node)> = VecDeque::new();

        loop {
            match &mut sink_out {
                FinalizedSink::Finished(df) => return Ok(std::mem::take(df)),
                FinalizedSink::Source(src) => return consume_source(&mut **src, &ec),

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

                    // First check the operators
                    // keep a counter as we also push to the front of deque
                    // otherwise we keep iterating
                    let mut remaining = operators_to_replace.len();
                    while let Some((op, sink_node)) = operators_to_replace.pop_back() {
                        if !pipeline.replace_operator(op.as_ref(), sink_node) {
                            operators_to_replace.push_front((op, sink_node))
                        } else {
                        }
                        if remaining == 0 {
                            break;
                        }
                        remaining -= 1;
                    }

                    // latest sink_node will be the operator, as the left side of the join
                    // always finishes that branch.
                    if let Some(sink_node) = sink_nodes.pop() {
                        // if dummy that should be replaces is not found in this branch
                        // we push it to the operators stack that should be replaced
                        // on the next branch of the pipeline we first check this stack.
                        // this only happens if we reorder joins
                        if !pipeline.replace_operator(op.as_ref(), sink_node) {
                            let mut swap = Box::<PlaceHolder>::default() as Box<dyn Operator>;
                            std::mem::swap(op, &mut swap);
                            operators_to_replace.push_back((swap, sink_node));
                        }
                    }
                    sink_out = pipeline.run_pipeline(&ec)?;
                    sink_nodes = std::mem::take(&mut pipeline.sink_nodes);
                }
            }
        }
    }
}

fn consume_source(src: &mut dyn Source, context: &PExecutionContext) -> PolarsResult<DataFrame> {
    let mut frames = Vec::with_capacity(32);

    while let SourceResult::GotMoreData(batch) = src.get_batches(context)? {
        frames.extend(batch.into_iter().map(|chunk| chunk.data))
    }
    Ok(accumulate_dataframes_vertical_unchecked(frames))
}
