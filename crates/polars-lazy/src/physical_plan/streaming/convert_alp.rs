use polars_core::error::PolarsResult;
use polars_core::prelude::*;
use polars_pipe::pipeline::swap_join_order;
use polars_plan::prelude::*;

use super::checks::*;
use crate::physical_plan::streaming::tree::*;

// The index of the pipeline tree we are building at this moment
// if we have a node we cannot do streaming, we have finished that pipeline tree
// and start a new one.
type CurrentIdx = usize;

fn process_non_streamable_node(
    current_idx: &mut CurrentIdx,
    state: &mut Branch,
    stack: &mut Vec<(Node, Branch, CurrentIdx)>,
    scratch: &mut Vec<Node>,
    pipeline_trees: &mut Vec<Vec<Branch>>,
    lp: &ALogicalPlan,
    insert_file_sink: &mut u32,
) {
    if state.streamable {
        *current_idx += 1;
        // create a completely new streaming pipeline
        // maybe we can stream a subsection of the plan
        pipeline_trees.push(vec![]);
    }
    state.streamable = false;
    lp.copy_inputs(scratch);
    while let Some(input) = scratch.pop() {
        *insert_file_sink += 1;
        stack.push((input, Branch::default(), *current_idx))
    }
}

fn insert_file_sink(mut root: Node, lp_arena: &mut Arena<ALogicalPlan>) -> Node {
    // The pipelines need a final sink, we insert that here.
    // this allows us to split at joins/unions and share a sink
    if !matches!(lp_arena.get(root), ALogicalPlan::Sink { .. }) {
        root = lp_arena.add(ALogicalPlan::Sink {
            input: root,
            payload: SinkType::Memory,
        })
    }
    root
}

fn insert_slice(
    root: Node,
    offset: i64,
    len: IdxSize,
    lp_arena: &mut Arena<ALogicalPlan>,
    state: &mut Branch,
) {
    let node = lp_arena.add(ALogicalPlan::Slice {
        input: root,
        offset,
        len: len as IdxSize,
    });
    state.operators_sinks.push(PipelineNode::Sink(node));
}

pub(crate) fn insert_streaming_nodes(
    root: Node,
    lp_arena: &mut Arena<ALogicalPlan>,
    expr_arena: &mut Arena<AExpr>,
    scratch: &mut Vec<Node>,
    fmt: bool,
    // whether the full plan needs to be translated
    // to streaming
    allow_partial: bool,
) -> PolarsResult<bool> {
    scratch.clear();

    // this is needed to determine which side of the joins should be
    // traversed first
    set_estimated_row_counts(root, lp_arena, expr_arena, 0, scratch);

    scratch.clear();

    // The pipelines always need to end in a SINK, we insert that here.
    // this allows us to split at joins/unions and share a sink
    let root = insert_file_sink(root, lp_arena);

    // We use mutation to communicate when we need to insert a file sink.
    // This happens for instance when we
    //
    //     ________*non-streamable part of query
    //   /\
    //     ________*streamable below this line so we must insert
    //    /\        a file sink here so the pipeline can be built
    //     /\
    //
    // when this is positive we should insert a file sink
    let mut insert_file_sink_ptr: u32 = 0;

    let mut stack = Vec::with_capacity(16);

    stack.push((root, Branch::default(), 0 as CurrentIdx));

    // A state holds a full pipeline until the breaker
    //  1/\
    //   2/\
    //     3\
    //
    // so 1 and 2 are short pipelines and 3 goes all the way to the root.
    // but 3 can only run if 1 and 2 have finished and set the join as operator in 3
    // and state are filled with pipeline 1, 2, 3 in that order
    //
    //     / \
    //  /\  3/\
    //  1 2    4\
    // or in this case 1, 2, 3, 4
    // every inner vec contains a branch/pipeline of a complete pipeline tree
    // the outer vec contains whole pipeline trees
    //
    // # Execution order
    // Trees can have arbitrary splits via joins and unions
    // the branches we have accumulated are flattened into a single Vec<Branch>
    // this therefore has lost the information of the tree. To know in which
    // order the branches need to be executed. For this reason we keep track of
    // an `execution_id` which will be incremented on every stack operation.
    // This way we know in which order the stack/tree was traversed and can
    // use that info to determine the execution order of the single branch/pipelines
    let mut pipeline_trees: Vec<Tree> = vec![vec![]];
    // keep the counter global so that the order will match traversal order
    let mut execution_id = 0;

    use ALogicalPlan::*;
    while let Some((mut root, mut state, mut current_idx)) = stack.pop() {
        if insert_file_sink_ptr > 0 {
            root = insert_file_sink(root, lp_arena);
        }
        insert_file_sink_ptr = insert_file_sink_ptr.saturating_sub(1);
        state.execution_id = execution_id;
        execution_id += 1;
        match lp_arena.get(root) {
            Selection { input, predicate }
                if is_streamable(*predicate, expr_arena, Context::Default) =>
            {
                state.streamable = true;
                state.operators_sinks.push(PipelineNode::Operator(root));
                stack.push((*input, state, current_idx))
            },
            HStack { input, exprs, .. } if all_streamable(exprs, expr_arena, Context::Default) => {
                state.streamable = true;
                state.operators_sinks.push(PipelineNode::Operator(root));
                stack.push((*input, state, current_idx))
            },
            Slice { input, offset, .. } if *offset >= 0 => {
                state.streamable = true;
                state.operators_sinks.push(PipelineNode::Sink(root));
                stack.push((*input, state, current_idx))
            },
            Sink { input, .. } => {
                state.streamable = true;
                state.operators_sinks.push(PipelineNode::Sink(root));
                stack.push((*input, state, current_idx))
            },
            Sort {
                input,
                by_column,
                args,
            } if is_streamable_sort(args) && all_column(by_column, expr_arena) => {
                state.streamable = true;
                state.operators_sinks.push(PipelineNode::Sink(root));
                stack.push((*input, state, current_idx))
            },
            Projection { input, expr, .. }
                if all_streamable(expr, expr_arena, Context::Default) =>
            {
                state.streamable = true;
                state.operators_sinks.push(PipelineNode::Operator(root));
                stack.push((*input, state, current_idx))
            },
            // Rechunks are ignored
            MapFunction {
                input,
                function: FunctionNode::Rechunk,
            } => {
                state.streamable = true;
                stack.push((*input, state, current_idx))
            },
            // Streamable functions will be converted
            lp @ MapFunction { input, function } => {
                if function.is_streamable() {
                    state.streamable = true;
                    state.operators_sinks.push(PipelineNode::Operator(root));
                    stack.push((*input, state, current_idx))
                } else {
                    process_non_streamable_node(
                        &mut current_idx,
                        &mut state,
                        &mut stack,
                        scratch,
                        &mut pipeline_trees,
                        lp,
                        &mut insert_file_sink_ptr,
                    )
                }
            },
            Scan {
                file_options: options,
                scan_type,
                ..
            } if scan_type.streamable() => {
                if state.streamable {
                    #[cfg(feature = "csv")]
                    if matches!(scan_type, FileScan::Csv { .. }) {
                        // the batched csv reader doesn't stop exactly at n_rows
                        if let Some(n_rows) = options.n_rows {
                            insert_slice(root, 0, n_rows as IdxSize, lp_arena, &mut state);
                        }
                    }

                    state.sources.push(root);
                    pipeline_trees[current_idx].push(state)
                }
            },
            DataFrameScan { .. } => {
                if state.streamable {
                    state.sources.push(root);
                    pipeline_trees[current_idx].push(state)
                }
            },
            Join {
                input_left,
                input_right,
                options,
                ..
            } if streamable_join(&options.args) => {
                let input_left = *input_left;
                let input_right = *input_right;
                state.streamable = true;
                state.join_count += 1;

                // We swap so that the build phase contains the smallest table
                // and then we stream the larger table
                // *except* for a left join. In a left join we use the right
                // table as build table and we stream the left table. This way
                // we maintain order in the left join.
                let (input_left, input_right) = if swap_join_order(options) {
                    (input_right, input_left)
                } else {
                    (input_left, input_right)
                };
                let mut state_left = state.split();

                // rhs is second, so that is first on the stack
                let mut state_right = state;
                state_right.join_count = 0;
                state_right
                    .operators_sinks
                    .push(PipelineNode::RhsJoin(root));

                // we want to traverse lhs last, so push it first on the stack
                // rhs is a new pipeline
                state_left.operators_sinks.push(PipelineNode::Sink(root));
                stack.push((input_left, state_left, current_idx));
                stack.push((input_right, state_right, current_idx));
            },
            // add globbing patterns
            #[cfg(any(feature = "csv", feature = "parquet"))]
            Union { inputs, options }
                if options.slice.is_none()
                    && inputs.iter().all(|node| match lp_arena.get(*node) {
                        Scan { .. } => true,
                        MapFunction {
                            input,
                            function: FunctionNode::Rechunk,
                        } => matches!(lp_arena.get(*input), Scan { .. }),
                        _ => false,
                    }) =>
            {
                state.sources.push(root);
                pipeline_trees[current_idx].push(state);
            },
            Union {
                options:
                    UnionOptions {
                        slice: Some((offset, len)),
                        ..
                    },
                ..
            } if *offset >= 0 => {
                insert_slice(root, *offset, *len as IdxSize, lp_arena, &mut state);
                state.streamable = true;
                let Union { inputs, .. } = lp_arena.get(root) else {
                    unreachable!()
                };
                for (i, input) in inputs.iter().enumerate() {
                    let mut state = if i == 0 {
                        // note the clone!
                        let mut state = state.clone();
                        state.join_count += inputs.len() as u32 - 1;
                        state
                    } else {
                        let mut state = state.split_from_sink();
                        state.join_count = 0;
                        state
                    };
                    state.operators_sinks.push(PipelineNode::Union(root));
                    stack.push((*input, state, current_idx));
                }
            },
            Union {
                inputs,
                options: UnionOptions { slice: None, .. },
            } => {
                {
                    state.streamable = true;
                    for (i, input) in inputs.iter().enumerate() {
                        let mut state = if i == 0 {
                            // note the clone!
                            let mut state = state.clone();
                            state.join_count += inputs.len() as u32 - 1;
                            state
                        } else {
                            let mut state = state.split_from_sink();
                            state.join_count = 0;
                            state
                        };
                        state.operators_sinks.push(PipelineNode::Union(root));
                        stack.push((*input, state, current_idx));
                    }
                }
            },
            Distinct { input, options }
                if !options.maintain_order
                    && !matches!(options.keep_strategy, UniqueKeepStrategy::None) =>
            {
                state.streamable = true;
                state.operators_sinks.push(PipelineNode::Sink(root));
                stack.push((*input, state, current_idx))
            },
            #[allow(unused_variables)]
            lp @ Aggregate {
                input,
                keys,
                aggs,
                maintain_order: false,
                apply: None,
                schema,
                options,
                ..
            } => {
                #[cfg(feature = "dtype-categorical")]
                let string_cache = polars_core::using_string_cache();
                #[cfg(not(feature = "dtype-categorical"))]
                let string_cache = true;

                #[allow(unused_variables)]
                fn allowed_dtype(dt: &DataType, string_cache: bool) -> bool {
                    match dt {
                        #[cfg(feature = "object")]
                        DataType::Object(_) => false,
                        #[cfg(feature = "dtype-categorical")]
                        DataType::Categorical(_) => string_cache,
                        _ => true,
                    }
                }
                let input_schema = lp_arena.get(*input).schema(lp_arena);
                #[allow(unused_mut)]
                let mut can_stream = true;

                #[cfg(feature = "dynamic_group_by")]
                {
                    if options.rolling.is_some() || options.dynamic.is_some() {
                        can_stream = false
                    }
                }

                let valid_agg = || {
                    aggs.iter().all(|node| {
                        polars_pipe::pipeline::can_convert_to_hash_agg(
                            *node,
                            expr_arena,
                            &input_schema,
                        )
                    })
                };

                let valid_key = || {
                    keys.iter().all(|node| {
                        expr_arena
                            .get(*node)
                            .get_type(schema, Context::Default, expr_arena)
                            // ensure we don't group_by list
                            .map(|dt| !matches!(dt, DataType::List(_)))
                            .unwrap_or(false)
                    })
                };

                let valid_types = || {
                    schema
                        .iter_dtypes()
                        .all(|dt| allowed_dtype(dt, string_cache))
                };

                if can_stream && valid_agg() && valid_key() && valid_types() {
                    state.streamable = true;
                    state.operators_sinks.push(PipelineNode::Sink(root));
                    stack.push((*input, state, current_idx))
                } else if allow_partial {
                    process_non_streamable_node(
                        &mut current_idx,
                        &mut state,
                        &mut stack,
                        scratch,
                        &mut pipeline_trees,
                        lp,
                        &mut insert_file_sink_ptr,
                    )
                } else {
                    return Ok(false);
                }
            },
            lp => {
                if allow_partial {
                    process_non_streamable_node(
                        &mut current_idx,
                        &mut state,
                        &mut stack,
                        scratch,
                        &mut pipeline_trees,
                        lp,
                        &mut insert_file_sink_ptr,
                    )
                } else {
                    return Ok(false);
                }
            },
        }
    }
    let mut inserted = false;
    for tree in pipeline_trees {
        if is_valid_tree(&tree)
            && super::construct_pipeline::construct(tree, lp_arena, expr_arena, fmt)?.is_some()
        {
            inserted = true;
        }
    }

    Ok(inserted)
}
