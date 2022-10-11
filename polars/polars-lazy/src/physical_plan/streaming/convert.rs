use std::any::Any;
use std::sync::Arc;

use polars_core::error::PolarsResult;
use polars_core::frame::DataFrame;
use polars_core::prelude::{Field, SchemaRef, Series};
use polars_core::schema::Schema;
use polars_pipe::expressions::PhysicalPipedExpr;
use polars_pipe::operators::chunks::DataChunk;
use polars_pipe::pipeline::{create_pipeline, Pipeline};
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
        AExpr::Cast { .. } | AExpr::Column(_) | AExpr::Literal(_) => true,
        _ => false,
    })
}

fn all_streamable(exprs: &[Node], expr_arena: &Arena<AExpr>) -> bool {
    exprs.iter().all(|node| is_streamable(*node, expr_arena))
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
    let mut states = vec![];

    use ALogicalPlan::*;
    while let Some((root, mut state)) = stack.pop() {
        match lp_arena.get(root) {
            Selection { input, predicate } if is_streamable(*predicate, expr_arena) => {
                state.streamable = true;
                state.operators.push(root);
                stack.push((*input, state))
            }
            MapFunction {
                input,
                function: FunctionNode::FastProjection { .. },
            } => {
                state.streamable = true;
                state.operators.push(root);
                stack.push((*input, state))
            }
            Projection { input, expr, .. } if all_streamable(expr, expr_arena) => {
                state.streamable = true;
                state.operators.push(root);
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
                    states.push(state)
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
                    state.sink = Some(root);
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

    for state in states {
        let latest = match state.sink {
            Some(node) => node,
            None => {
                if state.operators.is_empty() {
                    continue;
                } else {
                    state.operators[state.operators.len() - 1]
                }
            }
        };
        let schema = lp_arena.get(latest).schema(lp_arena).into_owned();
        let pipeline = create_pipeline(
            &state.sources,
            &state.operators,
            state.sink,
            lp_arena,
            expr_arena,
            to_physical_piped_expr,
        )?;

        // replace the part of the logical plan with a `MapFunction` that will execute the pipeline.
        let pipeline_node = get_pipeline_node(lp_arena, pipeline, schema);
        lp_arena.replace(latest, pipeline_node)
    }
    Ok(())
}

#[derive(Default, Debug)]
struct State {
    streamable: bool,
    sources: Vec<Node>,
    operators: Vec<Node>,
    sink: Option<Node>,
}

fn get_pipeline_node(
    lp_arena: &mut Arena<ALogicalPlan>,
    mut pipeline: Pipeline,
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
