use super::*;
use crate::pipeline::*;

/// Take data chunks from the sources and pushes them into the operators + sink. Every operator
/// works thread local.
/// The caller passes an `operator_start`/`operator_end` to indicate which part of the pipeline
/// branch should be executed.
#[allow(clippy::too_many_arguments)]
pub(super) fn par_process_chunks(
    chunks: Vec<DataChunk>,
    sink: ThreadedSinkMut,
    ec: &PExecutionContext,
    operators: &mut [ThreadedOperator],
    operator_start: usize,
    operator_end: usize,
    src: &mut Box<dyn Source>,
    must_flush: &AtomicBool,
) -> PolarsResult<(Option<SinkResult>, SourceResult)> {
    debug_assert!(chunks.len() <= sink.len());
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

    // borrow as ref and move into the closure
    POOL.scope(|s| {
        for ((chunk, sink), operator_pipe) in chunks
            .into_iter()
            .zip(sink.iter_mut())
            .zip(operators.iter_mut())
        {
            let sink_results = sink_results.clone();
            // Truncate the operators that should run into the current sink.
            let operator_pipe = &mut operator_pipe[operator_start..operator_end];

            s.spawn(move |_| {
                let out = if operator_pipe.is_empty() {
                    sink.sink(ec, chunk)
                } else {
                    push_operators_single_thread(chunk, ec, operator_pipe, sink, must_flush)
                };

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
pub(super) fn push_operators_single_thread(
    chunk: DataChunk,
    ec: &PExecutionContext,
    operators: ThreadedOperatorMut,
    sink: &mut Box<dyn Sink>,
    must_flush: &AtomicBool,
) -> PolarsResult<SinkResult> {
    debug_assert!(!operators.is_empty());

    // Stack based operator execution.
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
                let op = op.get_mut();
                match op.execute(ec, &chunk)? {
                    OperatorResult::Finished(chunk) => {
                        must_flush.store(op.must_flush(), Ordering::Relaxed);
                        in_process.push((op_i + 1, chunk))
                    },
                    OperatorResult::HaveMoreOutPut(output_chunk) => {
                        // Push the next operator call with the same chunk on the stack
                        in_process.push((op_i, chunk));

                        // But first push the output in the next operator
                        // If a join can produce many rows, we want the filter to
                        // be executed in between, or sink into a slice so that we get
                        // sink::finished before we grow the stack with ever more coming chunks
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

/// Similar to `par_process_chunks`.
/// The caller passes an `operator_start`/`operator_end` to indicate which part of the pipeline
/// branch should be executed.
pub(super) fn par_flush(
    sink: ThreadedSinkMut,
    ec: &PExecutionContext,
    operators: &mut [ThreadedOperator],
    operator_start: usize,
    operator_end: usize,
) {
    // 1. We will iterate the chunks/sinks/operators
    // where every iteration belongs to a single thread
    // 2. Then we will truncate the pipeline by `start`/`end`
    // so that the pipeline represents pipeline that belongs to this sink
    // 3. Then we push the data
    // # Threading
    // Within a rayon scope
    // we spawn the jobs. They don't have to finish in any specific order,
    // this makes it more lightweight than `par_iter`

    // borrow as ref and move into the closure
    POOL.scope(|s| {
        for (sink, operator_pipe) in sink.iter_mut().zip(operators.iter_mut()) {
            // Truncate the operators that should run into the current sink.
            let operator_pipe = &mut operator_pipe[operator_start..operator_end];

            s.spawn(move |_| {
                flush_operators(ec, operator_pipe, sink).unwrap();
            })
        }
    });
}

pub(super) fn flush_operators(
    ec: &PExecutionContext,
    operators: &mut [PhysOperator],
    sink: &mut Box<dyn Sink>,
) -> PolarsResult<SinkResult> {
    let needs_flush = operators
        .iter_mut()
        .enumerate()
        .filter_map(|(i, op)| {
            if op.get_mut().must_flush() {
                Some(i)
            } else {
                None
            }
        })
        .collect::<Vec<_>>();

    // Stack based flushing + operator execution.
    if !needs_flush.is_empty() {
        let mut in_process = vec![];

        for op_i in needs_flush.into_iter() {
            // Push all operators that need flushing on the stack.
            // The `None` indicates that we have no `chunk` input, so we `flush`.
            // `Some(chunk)` is the pushing branch
            in_process.push((op_i, None));

            // Next we immediately pop and determine the order of execution below.
            // This is to ensure that all operators below upper operators are completely
            // flushed when the `flush` is called in higher operators. As operators can `flush`
            // multiple times.
            while let Some((op_i, chunk)) = in_process.pop() {
                match chunk {
                    // The branch for flushing.
                    None => {
                        let op = operators.get_mut(op_i).unwrap().get_mut();
                        match op.flush()? {
                            OperatorResult::Finished(chunk) => {
                                // Push the chunk in the next operator.
                                in_process.push((op_i + 1, Some(chunk)))
                            },
                            OperatorResult::HaveMoreOutPut(chunk) => {
                                // Ensure it is flushed again
                                in_process.push((op_i, None));
                                // Push the chunk in the next operator.
                                in_process.push((op_i + 1, Some(chunk)))
                            },
                            _ => unreachable!(),
                        }
                    },
                    // The branch for pushing data in the operators.
                    // This is the same as the default stack executor, except now it pushes
                    // `Some(chunk)` instead of `chunk`.
                    Some(chunk) => {
                        match operators.get_mut(op_i) {
                            None => {
                                if let SinkResult::Finished = sink.sink(ec, chunk)? {
                                    return Ok(SinkResult::Finished);
                                }
                            },
                            Some(op) => {
                                let op = op.get_mut();
                                match op.execute(ec, &chunk)? {
                                    OperatorResult::Finished(chunk) => {
                                        in_process.push((op_i + 1, Some(chunk)))
                                    },
                                    OperatorResult::HaveMoreOutPut(output_chunk) => {
                                        // Push the next operator call with the same chunk on the stack
                                        in_process.push((op_i, Some(chunk)));

                                        // But first push the output in the next operator
                                        // If a join can produce many rows, we want the filter to
                                        // be executed in between, or sink into a slice so that we get
                                        // sink::finished before we grow the stack with ever more coming chunks
                                        in_process.push((op_i + 1, Some(output_chunk)));
                                    },
                                    OperatorResult::NeedsNewData => {
                                        // Done, take another chunk from the stack
                                    },
                                }
                            },
                        }
                    },
                }
            }
        }
    }
    Ok(SinkResult::Finished)
}
