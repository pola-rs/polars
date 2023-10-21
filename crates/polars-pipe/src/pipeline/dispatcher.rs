use std::cell::RefCell;
use std::collections::VecDeque;
use std::fmt::{Debug, Formatter};
use std::ops::{Deref, DerefMut};
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use polars_core::error::PolarsResult;
use polars_core::frame::DataFrame;
use polars_core::utils::accumulate_dataframes_vertical_unchecked;
use polars_core::POOL;
use polars_utils::arena::Node;
use polars_utils::sync::SyncPtr;
use rayon::prelude::*;

use crate::executors::sources::DataFrameSource;
use crate::operators::{
    DataChunk, FinalizedSink, Operator, OperatorResult, PExecutionContext, SExecutionContext, Sink,
    SinkResult, Source, SourceResult,
};
use crate::pipeline::morsels_per_sink;

/// A pipeline consists of:
///
/// - 1. One or more sources.
///         Sources get pulled and their data is pushed into operators.
/// - 2. Zero or more operators.
///         The operators simply pass through data, modifying it as they need.
///         Operators can work on batches and don't need all data in scope to
///         succeed.
///         Think for example on multiply a few columns, or applying a predicate.
///         Operators can shrink the batches: filter
///         Grow the batches: explode/ melt
///         Keep them the same size: element-wise operations
///         The probe side of join operations is also an operator.
///
///
/// - 3. One or more sinks
///         A sink needs all data in scope to finalize a pipeline branch.
///         Think of sorts, preparing a build phase of a join, group_by + aggregations.
///
/// This struct will have the SOS (source, operators, sinks) of its own pipeline branch, but also
/// the SOS of other branches. The SOS are stored data oriented and the sinks have an offset that
/// indicates the last operator node before that specific sink. We only store the `end offset` and
/// keep track of the starting operator during execution.
///
/// Pipelines branches are shared with other pipeline branches at the join/union nodes.
/// # JOIN
/// Consider this tree:
///         out
///       /
///     /\
///    1  2
///
/// And let's consider that branch 2 runs first. It will run until the join node where it will sink
/// into a build table. Once that is done it will replace the build-phase placeholder operator in
/// branch 1. Branch one can then run completely until out.
pub struct PipeLine {
    /// All the sources of this pipeline
    sources: Vec<Box<dyn Source>>,
    /// All the operators of this pipeline. Some may be placeholders that will be replaced during
    /// execution
    operators: Vec<Vec<Box<dyn Operator>>>,
    /// The nodes of operators. These are used to identify operators between pipelines
    operator_nodes: Vec<Node>,
    /// - offset in the operators vec
    ///   at that point the sink should be called.
    ///   the pipeline will first call the operators on that point and then
    ///   push the result in the sink.
    /// - shared_count
    ///     when that hits 0, the sink will finalize
    /// - node of the sink
    #[allow(clippy::type_complexity)]
    sinks: Vec<(usize, Rc<RefCell<u32>>, Vec<Box<dyn Sink>>)>,
    /// are used to identify the sink shared with other pipeline branches
    sink_nodes: Vec<Node>,
    /// Other branch of the pipeline/tree that must be executed
    /// after this one has executed.
    /// the dispatcher takes care of this.
    other_branches: Rc<RefCell<VecDeque<PipeLine>>>,
    /// this is a correction as there may be more `operators` than nodes
    /// as during construction, source may have inserted operators
    operator_offset: usize,
    /// Log runtime info to stderr
    verbose: bool,
}

impl PipeLine {
    #[allow(clippy::type_complexity)]
    pub fn new(
        sources: Vec<Box<dyn Source>>,
        operators: Vec<Box<dyn Operator>>,
        operator_nodes: Vec<Node>,
        // (offset, node (for identification), sink, shared_counter)
        sink_and_nodes: Vec<(usize, Node, Box<dyn Sink>, Rc<RefCell<u32>>)>,
        operator_offset: usize,
        verbose: bool,
    ) -> PipeLine {
        debug_assert_eq!(operators.len(), operator_nodes.len() + operator_offset);
        // we don't use the power of two partition size here
        // we only do that in the sinks itself.
        let n_threads = morsels_per_sink();

        // We split so that every thread gets an operator
        let sink_nodes = sink_and_nodes.iter().map(|(_, node, _, _)| *node).collect();
        let sinks = sink_and_nodes
            .into_iter()
            .map(|(offset, _, sink, shared_count)| {
                (
                    offset,
                    shared_count,
                    (0..n_threads).map(|i| sink.split(i)).collect(),
                )
            })
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
            other_branches: Default::default(),
            operator_offset,
            verbose,
        }
    }

    /// Create a pipeline only consisting of a single branch that always finishes with a sink
    pub fn new_simple(
        sources: Vec<Box<dyn Source>>,
        operators: Vec<Box<dyn Operator>>,
        sink: Box<dyn Sink>,
        verbose: bool,
    ) -> Self {
        let operators_len = operators.len();
        Self::new(
            sources,
            operators,
            vec![],
            vec![(
                operators_len,
                Node::default(),
                sink,
                Rc::new(RefCell::new(1)),
            )],
            0,
            verbose,
        )
    }

    /// Add a parent
    /// This should be in the right order
    pub fn with_other_branch(self, rhs: PipeLine) -> Self {
        self.other_branches.borrow_mut().push_back(rhs);
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

    /// Take data chunks from the sources and pushes them into the operators + sink. Every operator
    /// works thread local.
    /// The caller passes an `operator_start`/`operator_end` to indicate which part of the pipeline
    /// branch should be executed.
    fn par_process_chunks(
        &mut self,
        chunks: Vec<DataChunk>,
        sink: &mut [Box<dyn Sink>],
        ec: &PExecutionContext,
        operator_start: usize,
        operator_end: usize,
        src: &mut Box<dyn Source>,
    ) -> PolarsResult<(Option<SinkResult>, SourceResult)> {
        debug_assert!(chunks.len() <= sink.len());

        fn run_operator_pipe(
            pipe: &PipeLine,
            operator_start: usize,
            operator_end: usize,
            chunk: DataChunk,
            sink: &mut Box<dyn Sink>,
            operator_pipe: &mut [Box<dyn Operator>],
            ec: &PExecutionContext,
        ) -> PolarsResult<SinkResult> {
            // truncate the operators that should run into the current sink.
            let operator_pipe = &mut operator_pipe[operator_start..operator_end];

            if operator_pipe.is_empty() {
                sink.sink(ec, chunk)
            } else {
                pipe.push_operators(chunk, ec, operator_pipe, sink)
            }
        }
        let sink_results = Arc::new(Mutex::new(None));
        let mut next_batches: Option<PolarsResult<SourceResult>> = None;
        let next_batches_ptr = &mut next_batches as *mut Option<PolarsResult<SourceResult>>;
        let next_batches_ptr = unsafe { SyncPtr::new(next_batches_ptr) };

        // 1. We will iterate the chunks/sinks/operators
        // where every iteration belongs to a single thread
        // 2. Then we will truncate the pipeline by `start`/`end`
        // so that the pipeline represents pipeline that belongs to this sink
        // 3. Then we push the data
        // # Threading
        // Within a rayon scope
        // we spawn the jobs. They don't have to finish in any specific order,
        // this makes it more lightweight than `par_iter`

        // temporarily take to please the borrow checker
        let mut operators = std::mem::take(&mut self.operators);

        // borrow as ref and move into the closure
        let pipeline = &*self;
        POOL.scope(|s| {
            for ((chunk, sink), operator_pipe) in chunks
                .into_iter()
                .zip(sink.iter_mut())
                .zip(operators.iter_mut())
            {
                let sink_results = sink_results.clone();
                s.spawn(move |_| {
                    let out = run_operator_pipe(
                        pipeline,
                        operator_start,
                        operator_end,
                        chunk,
                        sink,
                        operator_pipe,
                        ec,
                    );
                    match out {
                        Ok(SinkResult::Finished) | Err(_) => {
                            let mut lock = sink_results.lock().unwrap();
                            *lock = Some(out)
                        },
                        _ => {},
                    }
                })
            }
            // already get batches on the thread pool
            // if one job is finished earlier we can already start that work
            s.spawn(|_| {
                let out = src.get_batches(ec);
                unsafe {
                    let ptr = next_batches_ptr.get();
                    *ptr = Some(out);
                }
            })
        });
        self.operators = operators;

        let next_batches = next_batches.unwrap()?;
        let mut lock = sink_results.lock().unwrap();
        lock.take()
            .transpose()
            .map(|sink_result| (sink_result, next_batches))
    }

    /// This thread local logic that pushed a data chunk into the operators + sink
    /// It can be that a single operator needs to be called multiple times, this is for instance the
    /// case with joins that produce many tuples, that's why we keep a stack of `in_process`
    /// operators.
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
                },
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
                        },
                        OperatorResult::NeedsNewData => {
                            // done, take another chunk from the stack
                        },
                    }
                },
            }
        }
        Ok(SinkResult::CanHaveMoreInput)
    }

    /// Replace the current sources with a [`DataFrameSource`].
    fn set_df_as_sources(&mut self, df: DataFrame) {
        let src = Box::new(DataFrameSource::from_df(df)) as Box<dyn Source>;
        self.set_sources(src)
    }

    /// Replace the current sources.
    fn set_sources(&mut self, src: Box<dyn Source>) {
        self.sources.clear();
        self.sources.push(src);
    }

    /// Run a single pipeline branch.
    /// This pulls data from the sources and pushes it into the operators which run on a different
    /// thread and finalize in a sink.
    ///
    /// The sink can be finished, but can also become a new source and then rinse and repeat.
    pub fn run_pipeline(
        &mut self,
        ec: &PExecutionContext,
        pipeline_q: Rc<RefCell<VecDeque<PipeLine>>>,
    ) -> PolarsResult<Option<FinalizedSink>> {
        let (sink_shared_count, mut reduced_sink) =
            run_pipeline_no_finalize_iterative(self, ec, pipeline_q)?;
        assert_eq!(sink_shared_count, 0);
        Ok(reduced_sink.finalize(ec).ok())
    }

    /// Executes all branches and replaces operators and sinks during execution to ensure
    /// we materialize.
    pub fn execute(&mut self, state: Box<dyn SExecutionContext>) -> PolarsResult<DataFrame> {
        let ec = PExecutionContext::new(state, self.verbose);

        if self.verbose {
            eprintln!("{self:?}");
            eprintln!("{:?}", &self.other_branches);
        }
        let mut sink_out = self.run_pipeline(&ec, self.other_branches.clone())?;
        let mut sink_nodes = std::mem::take(&mut self.sink_nodes);
        loop {
            match &mut sink_out {
                None => {
                    let mut pipeline = self.other_branches.borrow_mut().pop_front().unwrap();
                    sink_out = pipeline.run_pipeline(&ec, self.other_branches.clone())?;
                    sink_nodes = std::mem::take(&mut pipeline.sink_nodes);
                },
                Some(FinalizedSink::Finished(df)) => return Ok(std::mem::take(df)),
                Some(FinalizedSink::Source(src)) => return consume_source(&mut **src, &ec),

                //
                //  1/\
                //   2/\
                //     3\
                // the left hand side of the join has finished and now is an operator
                // we replace the dummy node in the right hand side pipeline with this
                // operator and then we run the pipeline rinse and repeat
                // until the final right hand side pipeline ran
                Some(FinalizedSink::Operator(op)) => {
                    // we unwrap, because the latest pipeline should not return an Operator
                    let mut pipeline = self.other_branches.borrow_mut().pop_front().unwrap();

                    // latest sink_node will be the operator, as the left side of the join
                    // always finishes that branch.
                    if let Some(sink_node) = sink_nodes.pop() {
                        // we traverse all pipeline
                        pipeline.replace_operator(op.as_ref(), sink_node);
                        // if there are unions, there can be more
                        for pl in self.other_branches.borrow_mut().iter_mut() {
                            pl.replace_operator(op.as_ref(), sink_node);
                        }
                    }
                    sink_out = pipeline.run_pipeline(&ec, self.other_branches.clone())?;
                    sink_nodes = std::mem::take(&mut pipeline.sink_nodes);
                },
            }
        }
    }
}

impl Debug for PipeLine {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut fmt = String::new();
        let mut start = 0usize;
        fmt.push_str(self.sources[0].fmt());
        for (offset_end, _, sink) in &self.sinks {
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
        write!(f, "{fmt}")
    }
}

/// Take a source and materialize it into a [`DataFrame`].
fn consume_source(src: &mut dyn Source, context: &PExecutionContext) -> PolarsResult<DataFrame> {
    let mut frames = Vec::with_capacity(32);

    while let SourceResult::GotMoreData(batch) = src.get_batches(context)? {
        frames.extend(batch.into_iter().map(|chunk| chunk.data))
    }
    Ok(accumulate_dataframes_vertical_unchecked(frames))
}

unsafe impl Send for PipeLine {}
unsafe impl Sync for PipeLine {}

// The algorithm is written in an iterative fashion to avoid stack overflows.
fn run_pipeline_no_finalize_iterative(
    current_pipeline: &mut PipeLine,
    ec: &PExecutionContext,
    pipeline_q: Rc<RefCell<VecDeque<PipeLine>>>,
) -> PolarsResult<(u32, Box<dyn Sink>)> {
    struct StackFrame<'a> {
        parent_index: Option<usize>,
        pipeline: OwnedOrMut<'a, PipeLine>,
        operator_start: usize,
        sink_finished: bool,
        #[allow(clippy::type_complexity)]
        remaining: std::vec::IntoIter<(usize, Rc<RefCell<u32>>, Vec<Box<dyn Sink>>)>,
        // Sink-local data
        shared_sink_count: u32,
        reduced_sink: Box<dyn Sink>,
    }

    impl<'a> StackFrame<'a> {
        // We run the entire non-recursive initialization step.
        fn new(
            mut pipeline: OwnedOrMut<'a, PipeLine>,
            ec: &PExecutionContext,
            parent_index: Option<usize>,
        ) -> PolarsResult<Self> {
            let operator_start = 0;
            // for unions we typically first want to push all pipelines
            // into the union sink before we call `finalize`
            // however if the sink is finished early, (for instance a `head`)
            // we don't want to run the rest of the pipelines and we finalize early
            let mut sink_finished = false;
            let mut sinks = std::mem::take(&mut pipeline.sinks).into_iter();

            let (operator_end, shared_count, sink) = sinks.next().unwrap();
            let (shared_sink_count, reduced_sink) = Self::iteration_init(
                pipeline.deref_mut(),
                ec,
                sink,
                shared_count,
                operator_start,
                operator_end,
                &mut sink_finished,
            )?;

            Ok(StackFrame {
                operator_start: operator_end,
                // for unions we typically first want to push all pipelines
                // into the union sink before we call `finalize`
                // however if the sink is finished early, (for instance a `head`)
                // we don't want to run the rest of the pipelines and we finalize early
                sink_finished,
                remaining: sinks,
                pipeline,
                parent_index,
                // Dummy values, they get set in the non-recursive init
                shared_sink_count,
                reduced_sink,
            })
        }

        /// Try to pull another (sink, shared_count, operator_end) triplet from remaining,
        /// finalizing the current cycle in the process.
        ///
        /// If there are no more triplets, return `Ok(None)`.
        /// If there was at least one more triplet, return `Ok(Some(()))`.
        fn advance(&mut self, ec: &PExecutionContext) -> PolarsResult<Option<()>> {
            let Some((operator_end, shared_count, sink)) = self.remaining.next() else {
                return Ok(None);
            };
            let sink_result = self.reduced_sink.finalize(ec)?;
            match sink_result {
                // turn this sink an a new source
                FinalizedSink::Finished(df) => self.pipeline.set_df_as_sources(df),
                FinalizedSink::Source(src) => self.pipeline.set_sources(src),
                // should not happen
                FinalizedSink::Operator(_) => {
                    unreachable!()
                },
            }
            let (shared_count, reduced_sink) = Self::iteration_init(
                self.pipeline.deref_mut(),
                ec,
                sink,
                shared_count,
                self.operator_start,
                operator_end,
                &mut self.sink_finished,
            )?;
            self.operator_start = operator_end;
            self.shared_sink_count = shared_count;
            self.reduced_sink = reduced_sink;
            Ok(Some(()))
        }

        /// Perform the non-recursive initialization for the current (sink, shared_count, operator_end) triplet.
        fn iteration_init(
            pipeline: &mut PipeLine,
            ec: &PExecutionContext,
            mut sink: Vec<Box<dyn Sink>>,
            shared_count: Rc<RefCell<u32>>,
            operator_start: usize,
            operator_end: usize,
            sink_finished: &mut bool,
        ) -> PolarsResult<(u32, Box<dyn Sink>)> {
            'outer: for src in &mut std::mem::take(&mut pipeline.sources) {
                let mut next_batches = src.get_batches(ec)?;

                while let SourceResult::GotMoreData(chunks) = next_batches {
                    let (sink_result, next_batches2) = pipeline.par_process_chunks(
                        chunks,
                        &mut sink,
                        ec,
                        operator_start,
                        operator_end,
                        src,
                    )?;
                    next_batches = next_batches2;

                    if let Some(SinkResult::Finished) = sink_result {
                        *sink_finished = true;
                        break 'outer;
                    }
                }
            }

            // The sinks have taken all chunks thread locally, now we reduce them into a single
            // result sink.
            let reduced_sink = POOL
                .install(|| {
                    sink.into_par_iter().reduce_with(|mut a, mut b| {
                        a.combine(&mut *b);
                        a
                    })
                })
                .unwrap();

            let shared_sink_count = {
                let mut shared_sink_count = shared_count.borrow_mut();
                *shared_sink_count -= 1;
                *shared_sink_count
            };
            Ok((shared_sink_count, reduced_sink))
        }
    }

    let mut stack = Vec::new();
    stack.push(StackFrame::new(
        OwnedOrMut::Mut(current_pipeline),
        ec,
        None,
    )?);

    'outer: while let Some(mut current_frame) = stack.pop() {
        if current_frame.shared_sink_count > 0 && !current_frame.sink_finished {
            let pipeline = pipeline_q.borrow_mut().pop_front().unwrap();
            let parent_index = {
                stack.push(current_frame);
                stack.len() - 1
            };
            let next_frame = StackFrame::new(OwnedOrMut::Owned(pipeline), ec, Some(parent_index))?;
            stack.push(next_frame);
            continue 'outer;
        }

        if current_frame.advance(ec)?.is_some() {
            // We re-enqueue for execution since we have more work to do.
            stack.push(current_frame);
        } else if let Some(parent_index) = current_frame.parent_index {
            let parent_frame = &mut stack[parent_index];
            parent_frame.shared_sink_count = current_frame.shared_sink_count;
            parent_frame
                .reduced_sink
                .combine(current_frame.reduced_sink.as_mut());
        } else {
            return Ok((current_frame.shared_sink_count, current_frame.reduced_sink));
        }
    }
    unreachable!()
}

// A type that can be either owned or a mutable reference.
enum OwnedOrMut<'a, T> {
    Owned(T),
    Mut(&'a mut T),
}

impl<'a, T> Deref for OwnedOrMut<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        match self {
            OwnedOrMut::Owned(t) => t,
            OwnedOrMut::Mut(t) => t,
        }
    }
}

impl<'a, T> DerefMut for OwnedOrMut<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self {
            OwnedOrMut::Owned(t) => t,
            OwnedOrMut::Mut(t) => t,
        }
    }
}
