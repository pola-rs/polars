use std::any::Any;
use std::collections::VecDeque;
use std::sync::Arc;

use polars_core::error::PolarsResult;
use polars_core::frame::DataFrame;
use polars_core::prelude::*;
use polars_core::schema::Schema;
use polars_pipe::expressions::PhysicalPipedExpr;
use polars_pipe::operators::chunks::DataChunk;
use polars_pipe::pipeline::{create_pipeline, get_dummy_operator, get_operator, PipeLine};
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
        JoinType::Inner => true,
        _ => false,
    }
}

#[cfg(any(feature = "csv-file", feature = "parquet"))]
pub(crate) fn insert_streaming_nodes(
    root: Node,
    lp_arena: &mut Arena<ALogicalPlan>,
    expr_arena: &mut Arena<AExpr>,
    scratch: &mut Vec<Node>,
) -> PolarsResult<()> {
    scratch.clear();

    let mut stack = Vec::with_capacity(lp_arena.len() / 3 + 1);
    stack.push((root, State::default()));

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
    let mut states = vec![];

    use ALogicalPlan::*;
    while let Some((root, mut state)) = stack.pop() {
        match lp_arena.get(root) {
            Selection { input, predicate } if is_streamable(*predicate, expr_arena) => {
                state.streamable = true;
                state.operators_sinks.push((false, false, root));
                stack.push((*input, state))
            }
            HStack { input, exprs, .. } if all_streamable(exprs, expr_arena) => {
                state.streamable = true;
                state.operators_sinks.push((false, false, root));
                stack.push((*input, state))
            }
            MapFunction {
                input,
                function: FunctionNode::FastProjection { .. },
            } => {
                state.streamable = true;
                state.operators_sinks.push((false, false, root));
                stack.push((*input, state))
            }
            Projection { input, expr, .. } if all_streamable(expr, expr_arena) => {
                state.streamable = true;
                state.operators_sinks.push((false, false, root));
                stack.push((*input, state))
            }
            MapFunction {
                input,
                function: FunctionNode::Rechunk,
            } => {
                // we ignore a rechunk
                state.streamable = true;
                stack.push((*input, state))
            }
            #[cfg(feature = "csv-file")]
            CsvScan { .. } => {
                if state.streamable {
                    state.sources.push(root);
                    states.push(state)
                }
            }
            #[cfg(feature = "parquet")]
            ParquetScan { .. } => {
                if state.streamable {
                    state.sources.push(root);
                    states.push(state)
                }
            }
            DataFrameScan { .. } => {
                if state.streamable {
                    state.sources.push(root);
                    states.push(state)
                }
            }
            Join {
                input_left,
                input_right,
                options,
                ..
            } if streamable_join(&options.how) => {
                state.streamable = true;
                // rhs
                let mut state_right = state;
                state_right.operators_sinks.push((true, true, root));
                stack.push((*input_right, state_right));

                // we want to traverse lhs first, so push it latest on the stack
                // lhs is a new pipeline
                let mut state_left = State {
                    streamable: true,
                    ..Default::default()
                };
                state_left.operators_sinks.push((true, false, root));
                stack.push((*input_left, state_left));
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
                    states.push(state);
                }
            }
            Aggregate {
                input,
                aggs,
                maintain_order: false,
                apply: None,
                ..
            } => {
                if aggs
                    .iter()
                    .all(|node| polars_pipe::pipeline::can_convert_to_hash_agg(*node, expr_arena))
                {
                    state.streamable = true;
                    state.operators_sinks.push((true, false, root));
                    stack.push((*input, state))
                } else {
                    stack.push((*input, State::default()))
                }
            }
            lp => {
                state.streamable = false;
                lp.copy_inputs(scratch);
                while let Some(input) = scratch.pop() {
                    stack.push((input, State::default()))
                }
            }
        }
    }

    let mut pipelines = VecDeque::with_capacity(states.len());

    // if join is
    //     1
    //    /\2
    //      /\3
    //
    // we are iterating from 3 to 1.

    // the most far right branch will be the latest that sets this
    // variable and thus will point to root
    let mut latest = Default::default();

    for state in states {
        // should be reset for every branch
        let mut sink_node = None;

        let mut operators = Vec::with_capacity(state.operators_sinks.len());
        let mut operator_nodes = Vec::with_capacity(state.operators_sinks.len());
        let mut iter = state.operators_sinks.into_iter().rev();

        for (is_sink, is_rhs_join, node) in &mut iter {
            latest = node;
            if is_sink && !is_rhs_join {
                sink_node = Some(node);
            } else {
                operator_nodes.push(node);
                let op = if is_rhs_join {
                    get_dummy_operator()
                } else {
                    get_operator(node, lp_arena, expr_arena, &to_physical_piped_expr)?
                };
                operators.push(op)
            }
        }

        let pipeline = create_pipeline(
            &state.sources,
            operators,
            operator_nodes,
            sink_node,
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
        // replace the part of the logical plan with a `MapFunction` that will execute the pipeline.
        let pipeline_node = get_pipeline_node(lp_arena, most_left, schema);
        lp_arena.replace(latest, pipeline_node)
    } else {
        panic!()
    }

    Ok(())
}

type IsSink = bool;
// a rhs of a join will be replaced later
type IsRhsJoin = bool;

#[derive(Default, Debug, Clone)]
struct State {
    streamable: bool,
    sources: Vec<Node>,
    // node is operator/sink
    operators_sinks: Vec<(IsSink, IsRhsJoin, Node)>,
}

fn get_pipeline_node(
    lp_arena: &mut Arena<ALogicalPlan>,
    mut pipeline: PipeLine,
    schema: SchemaRef,
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
        },
        input: dummy,
    }
}
