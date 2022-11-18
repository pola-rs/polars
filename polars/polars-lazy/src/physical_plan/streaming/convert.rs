use std::any::Any;
use std::collections::VecDeque;
use std::sync::Arc;

use polars_core::error::PolarsResult;
use polars_core::frame::DataFrame;
use polars_core::prelude::*;
use polars_core::schema::Schema;
use polars_pipe::expressions::PhysicalPipedExpr;
use polars_pipe::operators::chunks::DataChunk;
use polars_pipe::pipeline::{
    create_pipeline, get_dummy_operator, get_operator, swap_join_order, PipeLine,
};
use polars_plan::prelude::*;

use crate::physical_plan::planner::create_physical_expr;
use crate::physical_plan::state::ExecutionState;
use crate::physical_plan::PhysicalExpr;

pub struct Wrap(Arc<dyn PhysicalExpr>);

impl PhysicalPipedExpr for Wrap {
    fn evaluate(&self, chunk: &DataChunk, state: &dyn Any) -> PolarsResult<Series> {
        let state = state.downcast_ref::<ExecutionState>().unwrap();
        self.0.evaluate(&chunk.data, state)
    }
    fn field(&self, input_schema: &Schema) -> PolarsResult<Field> {
        self.0.to_field(input_schema)
    }
}

fn to_physical_piped_expr(
    node: Node,
    expr_arena: &Arena<AExpr>,
) -> PolarsResult<Arc<dyn PhysicalPipedExpr>> {
    // this is a double Arc<dyn> explore if we can create a single of it.
    create_physical_expr(node, Context::Default, expr_arena)
        .map(|e| Arc::new(Wrap(e)) as Arc<dyn PhysicalPipedExpr>)
}

fn is_streamable(node: Node, expr_arena: &Arena<AExpr>) -> bool {
    expr_arena.iter(node).all(|(_, ae)| match ae {
        AExpr::Function { options, .. } | AExpr::AnonymousFunction { options, .. } => {
            matches!(options.collect_groups, ApplyOptions::ApplyFlat)
        }
        AExpr::Cast { .. }
        | AExpr::Column(_)
        | AExpr::Literal(_)
        | AExpr::BinaryExpr { .. }
        | AExpr::Alias(_, _) => true,
        _ => false,
    })
}

fn all_streamable(exprs: &[Node], expr_arena: &Arena<AExpr>) -> bool {
    exprs.iter().all(|node| is_streamable(*node, expr_arena))
}

fn streamable_join(join_type: &JoinType) -> bool {
    match join_type {
        #[cfg(feature = "cross_join")]
        JoinType::Cross => true,
        JoinType::Inner | JoinType::Left => true,
        _ => false,
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

    // The index of the pipeline tree we are building at this moment
    // if we have a node we cannot do streaming, we have finished that pipeline tree
    // and start a new one.
    type CurrentIdx = usize;
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
    let mut pipeline_trees = vec![vec![]];

    use ALogicalPlan::*;
    while let Some((root, mut state, mut current_idx)) = stack.pop() {
        match lp_arena.get(root) {
            Selection { input, predicate } if is_streamable(*predicate, expr_arena) => {
                state.streamable = true;
                state.operators_sinks.push((false, false, root));
                stack.push((*input, state, current_idx))
            }
            HStack { input, exprs, .. } if all_streamable(exprs, expr_arena) => {
                state.streamable = true;
                state.operators_sinks.push((false, false, root));
                stack.push((*input, state, current_idx))
            }
            Slice { input, offset, .. } if *offset >= 0 => {
                state.streamable = true;
                state.operators_sinks.push((true, false, root));
                stack.push((*input, state, current_idx))
            }
            MapFunction {
                input,
                function: FunctionNode::FastProjection { .. },
            } => {
                state.streamable = true;
                state.operators_sinks.push((false, false, root));
                stack.push((*input, state, current_idx))
            }
            Projection { input, expr, .. } if all_streamable(expr, expr_arena) => {
                state.streamable = true;
                state.operators_sinks.push((false, false, root));
                stack.push((*input, state, current_idx))
            }
            MapFunction {
                input,
                function: FunctionNode::Rechunk,
            } => {
                // we ignore a rechunk
                state.streamable = true;
                stack.push((*input, state, current_idx))
            }
            #[cfg(feature = "csv-file")]
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
                let (input_left, input_right) =
                    if swap_join_order(options) || matches!(options.how, JoinType::Left) {
                        (input_right, input_left)
                    } else {
                        (input_left, input_right)
                    };

                // rhs is second, so that is first on the stack
                let mut state_right = state;
                state_right.join_count = 0;
                state_right.operators_sinks.push((true, true, root));
                stack.push((input_right, state_right, current_idx));

                // we want to traverse lhs first, so push it latest on the stack
                // rhs is a new pipeline
                let mut state_left = Branch {
                    streamable: true,
                    join_count,
                    ..Default::default()
                };
                state_left.operators_sinks.push((true, false, root));
                stack.push((input_left, state_left, current_idx));
            }
            // add globbing patterns
            #[cfg(all(feature = "csv-file", feature = "parquet"))]
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
            Aggregate {
                input,
                aggs,
                maintain_order: false,
                apply: None,
                schema,
                ..
            } => {
                #[cfg(feature = "dtype-categorical")]
                let string_cache = polars_core::using_string_cache();
                #[cfg(not(feature = "dtype-categorical"))]
                let string_cache = true;

                fn allowed_dtype(dt: &DataType, string_cache: bool) -> bool {
                    match dt {
                        #[cfg(feature = "object")]
                        DataType::Object(_) => false,
                        #[cfg(feature = "dtype-categorical")]
                        DataType::Categorical(_) => string_cache,
                        _ => true,
                    }
                }

                if aggs
                    .iter()
                    .all(|node| polars_pipe::pipeline::can_convert_to_hash_agg(*node, expr_arena))
                    && schema
                        .iter_dtypes()
                        .all(|dt| allowed_dtype(dt, string_cache))
                {
                    state.streamable = true;
                    state.operators_sinks.push((true, false, root));
                    stack.push((*input, state, current_idx))
                } else {
                    stack.push((*input, Branch::default(), current_idx))
                }
            }
            lp => {
                if state.streamable {
                    current_idx += 1;
                    pipeline_trees.push(vec![]);
                }
                state.streamable = false;
                lp.copy_inputs(scratch);
                while let Some(input) = scratch.pop() {
                    stack.push((input, Branch::default(), current_idx))
                }
            }
        }
    }
    let mut inserted = false;
    for tree in pipeline_trees {
        if !tree.is_empty() {
            let mut pipelines = VecDeque::with_capacity(tree.len());

            // if join is
            //     1
            //    /\2
            //      /\3
            //
            // we are iterating from 3 to 1.

            // the most far right branch will be the latest that sets this
            // variable and thus will point to root
            let mut latest = Default::default();

            let joins_in_tree = tree.iter().map(|branch| branch.join_count).sum::<IdxSize>();
            let branches_in_tree = tree.len() as IdxSize;

            // all join branches should be added, if not we skip the tree, as it is invalid
            if (branches_in_tree - 1) != joins_in_tree {
                continue;
            }

            for branch in tree {
                // should be reset for every branch
                let mut sink_nodes = vec![];

                let mut operators = Vec::with_capacity(branch.operators_sinks.len());
                let mut operator_nodes = Vec::with_capacity(branch.operators_sinks.len());

                // iterate from leaves upwards
                let mut iter = branch.operators_sinks.into_iter().rev();

                for (is_sink, is_rhs_join, node) in &mut iter {
                    latest = node;
                    let operator_offset = operators.len();
                    if is_sink && !is_rhs_join {
                        sink_nodes.push((operator_offset, node))
                    } else {
                        operator_nodes.push(node);

                        // rhs join we create a dummy operator. This operator will
                        // be replaced by the dispatcher for the real rhs join.
                        let op = if is_rhs_join {
                            // if the join has a slice, we add a new slice node
                            // note that we take the offset + 1, because we want to
                            // slice AFTER the join has happened and the join will be an
                            // operator
                            if let Join {
                                options:
                                    JoinOptions {
                                        slice: Some((offset, len)),
                                        ..
                                    },
                                ..
                            } = lp_arena.get(node)
                            {
                                let slice_node = lp_arena.add(ALogicalPlan::Slice {
                                    input: Node::default(),
                                    offset: *offset,
                                    len: *len as IdxSize,
                                });
                                sink_nodes.push((operator_offset + 1, slice_node));
                            }
                            get_dummy_operator()
                        } else {
                            get_operator(node, lp_arena, expr_arena, &to_physical_piped_expr)?
                        };
                        operators.push(op)
                    }
                }

                let pipeline = create_pipeline(
                    &branch.sources,
                    operators,
                    operator_nodes,
                    sink_nodes,
                    lp_arena,
                    expr_arena,
                    to_physical_piped_expr,
                )?;
                pipelines.push_back(pipeline);
            }
            // the most right latest node should be the root of the pipeline
            let schema = lp_arena.get(latest).schema(lp_arena).into_owned();

            if let Some(mut most_left) = pipelines.pop_front() {
                while let Some(rhs) = pipelines.pop_front() {
                    most_left = most_left.with_rhs(rhs)
                }
                // keep the original around for formatting purposes
                let original_lp = if fmt {
                    let original_lp = lp_arena.take(latest);
                    let original_node = lp_arena.add(original_lp);
                    let original_lp = node_to_lp_cloned(original_node, expr_arena, lp_arena);
                    Some(original_lp)
                } else {
                    None
                };

                // replace the part of the logical plan with a `MapFunction` that will execute the pipeline.
                let pipeline_node = get_pipeline_node(lp_arena, most_left, schema, original_lp);
                lp_arena.replace(latest, pipeline_node);
                inserted = true;
            } else {
                panic!()
            }
        }
    }

    Ok(inserted)
}

type IsSink = bool;
// a rhs of a join will be replaced later
type IsRhsJoin = bool;

#[derive(Default, Debug, Clone)]
struct Branch {
    streamable: bool,
    sources: Vec<Node>,
    // joins seen in whole branch (we count a union as joins with multiple counts)
    join_count: IdxSize,
    // node is operator/sink
    operators_sinks: Vec<(IsSink, IsRhsJoin, Node)>,
}

fn get_pipeline_node(
    lp_arena: &mut Arena<ALogicalPlan>,
    mut pipeline: PipeLine,
    schema: SchemaRef,
    original_lp: Option<LogicalPlan>,
) -> ALogicalPlan {
    // create a dummy input as the map function will call the input
    // so we just create a scan that returns an empty df
    let dummy = lp_arena.add(ALogicalPlan::DataFrameScan {
        df: Arc::new(DataFrame::empty()),
        schema: Arc::new(Schema::new()),
        output_schema: None,
        projection: None,
        selection: None,
    });

    ALogicalPlan::MapFunction {
        function: FunctionNode::Pipeline {
            function: Arc::new(move |_df: DataFrame| {
                let state = ExecutionState::new();
                if state.verbose() {
                    eprintln!("RUN STREAMING PIPELINE")
                }
                let state = Box::new(state) as Box<dyn Any + Send + Sync>;
                pipeline.execute(state)
            }),
            schema,
            original: original_lp.map(Arc::new),
        },
        input: dummy,
    }
}
