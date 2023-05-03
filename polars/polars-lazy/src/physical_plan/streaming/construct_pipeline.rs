use std::any::Any;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::Arc;

use polars_core::config::verbose;
use polars_core::prelude::*;
use polars_pipe::expressions::PhysicalPipedExpr;
use polars_pipe::operators::chunks::DataChunk;
use polars_pipe::pipeline::{create_pipeline, get_dummy_operator, get_operator, PipeLine};
use polars_pipe::SExecutionContext;
use polars_utils::IdxSize;

use crate::physical_plan::planner::create_physical_expr;
use crate::physical_plan::state::ExecutionState;
use crate::physical_plan::streaming::tree::{PipelineNode, Tree};
use crate::prelude::*;

pub struct Wrap(Arc<dyn PhysicalExpr>);

impl PhysicalPipedExpr for Wrap {
    fn evaluate(&self, chunk: &DataChunk, state: &dyn Any) -> PolarsResult<Series> {
        let state = state.downcast_ref::<ExecutionState>().unwrap();
        self.0.evaluate(&chunk.data, state)
    }
    fn field(&self, input_schema: &Schema) -> PolarsResult<Field> {
        self.0.to_field(input_schema)
    }

    fn expression(&self) -> Expr {
        self.0.as_expression().unwrap().clone()
    }
}

fn to_physical_piped_expr(
    node: Node,
    expr_arena: &Arena<AExpr>,
    schema: Option<&SchemaRef>,
) -> PolarsResult<Arc<dyn PhysicalPipedExpr>> {
    // this is a double Arc<dyn> explore if we can create a single of it.
    create_physical_expr(node, Context::Default, expr_arena, schema)
        .map(|e| Arc::new(Wrap(e)) as Arc<dyn PhysicalPipedExpr>)
}

pub(super) fn construct(
    tree: Tree,
    lp_arena: &mut Arena<ALogicalPlan>,
    expr_arena: &mut Arena<AExpr>,
    fmt: bool,
) -> PolarsResult<Option<Node>> {
    use ALogicalPlan::*;

    let mut pipelines = VecDeque::with_capacity(tree.len());

    // if join is
    //     1
    //    /\2
    //      /\3
    //
    //
    // or join/union is
    //
    //   1
    //  /\__
    //    | |
    //    2 3
    // we are iterating from 3 to 1 as the branches are accumulated from left to right
    //
    // the most far right branch will be the latest that sets this
    // variable and thus will point to root
    let mut latest = None;

    let is_verbose = verbose();
    let mut sink_share_count = PlHashMap::new();

    for branch in tree {
        // should be reset for every branch
        let mut sink_nodes = vec![];

        let mut operators = Vec::with_capacity(branch.operators_sinks.len());
        let mut operator_nodes = Vec::with_capacity(branch.operators_sinks.len());

        for sinks in &branch.shared_sinks {
            for sink in sinks.as_ref() {
                let count = sink_share_count.entry(sink.0).or_insert(0u32);
                *count += 1;
            }
        }

        // iterate from leaves upwards
        let mut iter = branch.operators_sinks.into_iter().rev();

        for pipeline_node in &mut iter {
            latest = Some(pipeline_node.node());
            let operator_offset = operators.len();
            match pipeline_node {
                PipelineNode::Sink(node) => {
                    let count = sink_share_count.entry(node.0).or_insert(0u32);
                    *count += 1;
                    sink_nodes.push((operator_offset, node))
                },
                PipelineNode::Operator(node) => {
                    operator_nodes.push(node);
                    let op = get_operator(node, lp_arena, expr_arena, &to_physical_piped_expr)?;
                    operators.push(op);
                }
                PipelineNode::RhsJoin(node) => {
                    operator_nodes.push(node);
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
                        let slice_node = lp_arena.add(Slice {
                            input: Node::default(),
                            offset: *offset,
                            len: *len as IdxSize,
                        });
                        sink_nodes.push((operator_offset + 1, slice_node));
                    }
                    let op = get_dummy_operator();
                    operators.push(op)
                }
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
            is_verbose,
        )?;
        pipelines.push_back(pipeline);
    }

    let mut counts = sink_share_count.into_iter().collect::<Vec<_>>();
    counts.sort_by_key(|tpl| tpl.1);


    // some queries only have source/sources and don't have any
    // operators/sink so no latest
    // if let Some(latest_sink) = counts.first().map(|tpl| Node(tpl.0)).or(latest) {
    if let Some(latest_sink) = latest {
        // the most right latest node should be the root of the pipeline
        let schema = lp_arena.get(latest_sink).schema(lp_arena).into_owned();

        let Some(mut most_left) = pipelines.pop_front() else {unreachable!()};
        while let Some(rhs) = pipelines.pop_front() {
            most_left = most_left.with_rhs(rhs)
        }
        // keep the original around for formatting purposes
        let original_lp = if fmt {
            let original_lp = lp_arena.take(latest_sink);
            let original_node = lp_arena.add(original_lp);
            let original_lp = node_to_lp_cloned(original_node, expr_arena, lp_arena);
            Some(original_lp)
        } else {
            None
        };

        // replace the part of the logical plan with a `MapFunction` that will execute the pipeline.
        let pipeline_node = get_pipeline_node(lp_arena, most_left, schema, original_lp);
        lp_arena.replace(latest_sink, pipeline_node);

        Ok(Some(latest_sink))
    } else {
        Ok(None)
    }
}

impl SExecutionContext for ExecutionState {
    fn as_any(&self) -> &dyn Any {
        self
    }
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
                let state = Box::new(state) as Box<dyn SExecutionContext>;
                pipeline.execute(state)
            }),
            schema,
            original: original_lp.map(Arc::new),
        },
        input: dummy,
    }
}
