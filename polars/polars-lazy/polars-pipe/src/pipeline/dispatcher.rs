use std::any::Any;
use std::sync::atomic::Ordering::{Acquire, SeqCst};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use polars_core::error::{PolarsError, PolarsResult};
use polars_core::frame::DataFrame;
use polars_core::utils::concat_df_unchecked;
use polars_core::POOL;
use polars_utils::arena::Node;
use rayon::prelude::*;

use crate::executors::sources::DataFrameSource;
use crate::operators::{
    DataChunk, FinalizedSink, Operator, OperatorResult, PExecutionContext, Sink, SinkResult,
    Source, SourceResult,
};

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
    sink_error: Arc<Mutex<Option<PolarsError>>>,
    sink_finished: Arc<AtomicBool>,
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
        let n_threads = POOL.current_num_threads();

        // We split so that every thread get's an operator
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
            sink_error: Default::default(),
            sink_finished: Default::default(),
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
    ) {
        debug_assert!(chunks.len() <= sink.len());
        let mut operators = std::mem::take(&mut self.operators);
        POOL.install(|| {
            chunks
                .into_par_iter()
                .zip(sink.par_iter_mut())
                .zip(operators.par_iter_mut())
                .for_each(|((chunk, sink), operator_pipe)| {
                    // truncate the operators that should run into the current sink.
                    let operator_pipe = &mut operator_pipe[operator_start..operator_end];

                    let sink_out = if operator_pipe.is_empty() {
                        sink.sink(ec, chunk)
                    } else {
                        match self.push_operators(chunk, ec, operator_pipe) {
                            Ok(result) => {
                                match result {
                                    OperatorResult::Finished(chunk) => sink.sink(ec, chunk),
                                    // probably empty chunk?
                                    OperatorResult::NeedsNewData => {
                                        Ok(SinkResult::CanHaveMoreInput)
                                    }
                                    _ => todo!(),
                                }
                            }
                            Err(err) => {
                                self.sink_finished.store(true, SeqCst);
                                let mut sink_error = self.sink_error.lock().unwrap();
                                *sink_error = Some(err);
                                return;
                            }
                        }
                    };
                    match sink_out {
                        Ok(SinkResult::Finished) => {
                            self.sink_finished.store(true, SeqCst);
                        }
                        Ok(SinkResult::CanHaveMoreInput) => {
                            // pass
                        }
                        Err(err) => {
                            self.sink_finished.store(true, SeqCst);
                            let mut sink_error = self.sink_error.lock().unwrap();
                            *sink_error = Some(err);
                        }
                    }
                });
        });
        self.operators = operators;
    }

    fn push_operators(
        &self,
        chunk: DataChunk,
        ec: &PExecutionContext,
        operators: &mut [Box<dyn Operator>],
    ) -> PolarsResult<OperatorResult> {
        debug_assert!(!operators.is_empty());
        let mut in_process = vec![];
        let mut out = vec![];

        let operator_offset = 0usize;
        in_process.push((operator_offset, chunk));

        while let Some((op_i, chunk)) = in_process.pop() {
            match operators.get_mut(op_i) {
                None => {
                    if chunk.data.height() > 0 || out.is_empty() {
                        // final chunk of the pipeline
                        out.push(chunk)
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
                            in_process.push((op_i + 1, output_chunk));
                        }
                        OperatorResult::NeedsNewData => {
                            // done, take another chunk from the stack
                        }
                    }
                }
            }
        }
        let out = match out.len() {
            0 => OperatorResult::NeedsNewData,
            1 => OperatorResult::Finished(out.pop().unwrap()),
            _ => {
                let data = concat_df_unchecked(out.iter().map(|chunk| &chunk.data));
                OperatorResult::Finished(out[out.len() - 1].with_data(data))
            }
        };
        Ok(out)
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
                // we allow a single thread to prefetch IO
                let (sender, receiver) = std::sync::mpsc::sync_channel(2);
                let finished = AtomicBool::new(false);

                std::thread::scope(|s| {
                    let _ = s.spawn(|| {
                        // this will block until the previous message has been received
                        while let SourceResult::GotMoreData(chunks) = src.get_batches(ec).unwrap() {
                            if finished.load(Acquire) {
                                break;
                            }
                            sender.send(chunks).unwrap();
                        }
                        drop(sender);
                    });

                    while let Ok(chunks) = receiver.recv() {
                        self.par_process_chunks(
                            chunks,
                            &mut sink,
                            ec,
                            operator_start,
                            operator_end,
                        );
                        let finished = self.sink_finished.load(SeqCst);
                        if finished {
                            // check for errors
                            let mut err = self.sink_error.lock().unwrap();
                            if let Some(err) = err.take() {
                                return Err(err);
                            }
                            break;
                        }
                    }

                    finished.store(true, Ordering::Release);
                    Ok(())
                })?
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

    pub fn execute(&mut self, state: Box<dyn Any + Send + Sync>) -> PolarsResult<DataFrame> {
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
