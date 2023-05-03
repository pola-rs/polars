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
) {
    if state.streamable {
        *current_idx += 1;
        pipeline_trees.push(vec![]);
    }
    state.streamable = false;
    lp.copy_inputs(scratch);
    while let Some(input) = scratch.pop() {
        stack.push((input, Branch::default(), *current_idx))
    }
}

pub(crate) fn insert_streaming_nodes(
    root: Node,
    lp_arena: &mut Arena<ALogicalPlan>,
    expr_arena: &mut Arena<AExpr>,
    scratch: &mut Vec<Node>,
    fmt: bool,
) -> PolarsResult<bool> {
    // this is needed to determine which side of the joins should be
    // traversed first
    set_estimated_row_counts(root, lp_arena, expr_arena, 0);

    scratch.clear();

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
    let mut pipeline_trees: Vec<Tree> = vec![vec![]];

    use ALogicalPlan::*;
    while let Some((root, mut state, mut current_idx)) = stack.pop() {
        match lp_arena.get(root) {
            Selection { input, predicate }
                if is_streamable(*predicate, expr_arena, Context::Default) =>
            {
                state.streamable = true;
                state.operators_sinks.push(PipelineNode::Operator(root));
                stack.push((*input, state, current_idx))
            }
            HStack { input, exprs, .. } if all_streamable(exprs, expr_arena, Context::Default) => {
                state.streamable = true;
                state.operators_sinks.push(PipelineNode::Operator(root));
                stack.push((*input, state, current_idx))
            }
            Slice { input, offset, .. } if *offset >= 0 => {
                state.streamable = true;
                state.operators_sinks.push(PipelineNode::Sink(root));
                stack.push((*input, state, current_idx))
            }
            FileSink { input, .. } => {
                state.streamable = true;
                state.operators_sinks.push(PipelineNode::Sink(root));
                stack.push((*input, state, current_idx))
            }
            Sort {
                input,
                by_column,
                args,
            } if is_streamable_sort(args) && all_column(by_column, expr_arena) => {
                state.streamable = true;
                state.operators_sinks.push(PipelineNode::Sink(root));
                stack.push((*input, state, current_idx))
            }
            Projection { input, expr, .. }
                if all_streamable(expr, expr_arena, Context::Default) =>
            {
                state.streamable = true;
                state.operators_sinks.push(PipelineNode::Operator(root));
                stack.push((*input, state, current_idx))
            }
            // Rechunks are ignored
            MapFunction {
                input,
                function: FunctionNode::Rechunk,
            } => {
                state.streamable = true;
                stack.push((*input, state, current_idx))
            }
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
                    )
                }
            }
            #[cfg(feature = "csv")]
            CsvScan { .. } => {
                if state.streamable {
                    state.sources.push(root);
                    pipeline_trees[current_idx].push(state)
                }
            }
            #[cfg(feature = "parquet")]
            ParquetScan { .. } => {
                if state.streamable {
                    state.sources.push(root);
                    pipeline_trees[current_idx].push(state)
                }
            }
            DataFrameScan { .. } => {
                if state.streamable {
                    state.sources.push(root);
                    pipeline_trees[current_idx].push(state)
                }
            }
            Join {
                input_left,
                input_right,
                options,
                ..
            } if streamable_join(&options.how) => {
                let input_left = *input_left;
                let input_right = *input_right;
                state.streamable = true;
                state.join_count += 1;

                let join_count = state.join_count;
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

                // rhs is second, so that is first on the stack
                let mut state_right = state;
                state_right.join_count = 0;
                state_right
                    .operators_sinks
                    .push(PipelineNode::RhsJoin(root));
                stack.push((input_right, state_right, current_idx));

                // we want to traverse lhs first, so push it latest on the stack
                // rhs is a new pipeline
                let mut state_left = Branch {
                    streamable: true,
                    join_count,
                    ..Default::default()
                };
                state_left.operators_sinks.push(PipelineNode::Sink(root));
                stack.push((input_left, state_left, current_idx));
            }
            // add globbing patterns
            #[cfg(all(feature = "csv", feature = "parquet"))]
            Union { inputs, .. } => {
                if state.streamable
                    && inputs.iter().all(|node| match lp_arena.get(*node) {
                        ParquetScan { .. } => true,
                        CsvScan { .. } => true,
                        MapFunction {
                            input,
                            function: FunctionNode::Rechunk,
                        } => {
                            matches!(lp_arena.get(*input), ParquetScan { .. } | CsvScan { .. })
                        }
                        _ => false,
                    })
                {
                    state.sources.push(root);
                    pipeline_trees[current_idx].push(state);
                }
            }
            Distinct { input, options }
                if !options.maintain_order
                    && !matches!(options.keep_strategy, UniqueKeepStrategy::None) =>
            {
                state.streamable = true;
                state.operators_sinks.push(PipelineNode::Sink(root));
                stack.push((*input, state, current_idx))
            }
            #[allow(unused_variables)]
            Aggregate {
                input,
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

                #[cfg(feature = "dynamic_groupby")]
                {
                    if options.rolling.is_some() || options.dynamic.is_some() {
                        can_stream = false
                    }
                }

                if can_stream
                    && aggs.iter().all(|node| {
                        polars_pipe::pipeline::can_convert_to_hash_agg(
                            *node,
                            expr_arena,
                            &input_schema,
                        )
                    })
                    && schema
                        .iter_dtypes()
                        .all(|dt| allowed_dtype(dt, string_cache))
                {
                    state.streamable = true;
                    state.operators_sinks.push(PipelineNode::Sink(root));
                    stack.push((*input, state, current_idx))
                } else {
                    stack.push((*input, Branch::default(), current_idx))
                }
            }
            lp => process_non_streamable_node(
                &mut current_idx,
                &mut state,
                &mut stack,
                scratch,
                &mut pipeline_trees,
                lp,
            ),
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
