use std::any::Any;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

use polars_core::config::verbose;
use polars_core::prelude::*;
use polars_pipe::expressions::PhysicalPipedExpr;
use polars_pipe::operators::chunks::DataChunk;
use polars_pipe::pipeline::{create_pipeline, get_dummy_operator, get_operator, PipeLine};
use polars_pipe::SExecutionContext;
use polars_utils::IdxSize;

use crate::physical_plan::planner::{create_physical_expr, ExpressionConversionState};
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
    create_physical_expr(
        node,
        Context::Default,
        expr_arena,
        schema,
        &mut ExpressionConversionState::new(false),
    )
    .map(|e| Arc::new(Wrap(e)) as Arc<dyn PhysicalPipedExpr>)
}

fn jit_insert_slice(
    node: Node,
    lp_arena: &mut Arena<ALogicalPlan>,
    sink_nodes: &mut Vec<(usize, Node, Rc<RefCell<u32>>)>,
    operator_offset: usize,
) {
    // if the join/union has a slice, we add a new slice node
    // note that we take the offset + 1, because we want to
    // slice AFTER the join has happened and the join will be an
    // operator
    use ALogicalPlan::*;
    let (offset, len) = match lp_arena.get(node) {
        Join { options, .. } if options.args.slice.is_some() => {
            let Some((offset, len)) = options.args.slice else {
                unreachable!()
            };
            (offset, len)
        },
        Union {
            options:
                UnionOptions {
                    slice: Some((offset, len)),
                    ..
                },
            ..
        } => (*offset, *len),
        _ => return,
    };

    let slice_node = lp_arena.add(Slice {
        input: Node::default(),
        offset,
        len: len as IdxSize,
    });
    sink_nodes.push((operator_offset + 1, slice_node, Rc::new(RefCell::new(1))));
}

pub(super) fn construct(
    tree: Tree,
    lp_arena: &mut Arena<ALogicalPlan>,
    expr_arena: &mut Arena<AExpr>,
    fmt: bool,
) -> PolarsResult<Option<Node>> {
    use ALogicalPlan::*;

    let mut pipelines = Vec::with_capacity(tree.len());

    let is_verbose = verbose();

    // first traverse the branches and nodes to determine how often a sink is
    // shared
    // this shared count will be used in the pipeline to determine
    // when the sink can be finalized.
    let mut sink_share_count = PlHashMap::new();
    let n_branches = tree.len();
    if n_branches > 1 {
        for branch in &tree {
            for sink in branch.iter_sinks() {
                let count = sink_share_count
                    .entry(sink.0)
                    .or_insert(Rc::new(RefCell::new(0u32)));
                *count.borrow_mut() += 1;
            }
        }
    }

    // shared sinks are stored in a cache, so that they share info
    let mut sink_cache = PlHashMap::new();
    let mut final_sink = None;

    for branch in tree {
        // the file sink is always to the top of the tree
        // not every branch has a final sink. For instance rhs join branches
        if let Some(node) = branch.get_final_sink() {
            if matches!(lp_arena.get(node), ALogicalPlan::Sink { .. }) {
                final_sink = Some(node)
            }
        }
        // should be reset for every branch
        let mut sink_nodes = vec![];

        let mut operators = Vec::with_capacity(branch.operators_sinks.len());
        let mut operator_nodes = Vec::with_capacity(branch.operators_sinks.len());

        // iterate from leaves upwards
        let mut iter = branch.operators_sinks.into_iter().rev();

        for pipeline_node in &mut iter {
            let operator_offset = operators.len();
            match pipeline_node {
                PipelineNode::Sink(node) => {
                    let shared_count = if n_branches > 1 {
                        // should be here
                        sink_share_count.get(&node.0).unwrap().clone()
                    } else {
                        Rc::new(RefCell::new(1))
                    };
                    sink_nodes.push((operator_offset, node, shared_count))
                },
                PipelineNode::Operator(node) => {
                    operator_nodes.push(node);
                    let op = get_operator(node, lp_arena, expr_arena, &to_physical_piped_expr)?;
                    operators.push(op);
                },
                PipelineNode::Union(node) => {
                    operator_nodes.push(node);
                    jit_insert_slice(node, lp_arena, &mut sink_nodes, operator_offset);
                    let op = get_operator(node, lp_arena, expr_arena, &to_physical_piped_expr)?;
                    operators.push(op);
                },
                PipelineNode::RhsJoin(node) => {
                    operator_nodes.push(node);
                    jit_insert_slice(node, lp_arena, &mut sink_nodes, operator_offset);
                    let op = get_dummy_operator();
                    operators.push(op)
                },
            }
        }
        let execution_id = branch.execution_id;

        let pipeline = create_pipeline(
            &branch.sources,
            operators,
            operator_nodes,
            sink_nodes,
            lp_arena,
            expr_arena,
            to_physical_piped_expr,
            is_verbose,
            &mut sink_cache,
        )?;
        pipelines.push((execution_id, pipeline));
    }

    // We sort to ensure we execute in the stack traversal order.
    // this is important to make unions and joins work as expected
    // also pipelines are not ready to receive inputs otherwise
    pipelines.sort_by(|a, b| a.0.cmp(&b.0));

    let Some(final_sink) = final_sink else {
        return Ok(None);
    };
    let insertion_location = match lp_arena.get(final_sink) {
        // this was inserted only during conversion and does not exist
        // in the original tree, so we take the input, as that's where
        // we connect into the original tree.
        Sink {
            input,
            payload: SinkType::Memory,
        } => *input,
        // Other sinks were not inserted during conversion,
        // so they are returned as-is
        Sink { .. } => final_sink,
        _ => unreachable!(),
    };
    // keep the original around for formatting purposes
    let original_lp = if fmt {
        let original_lp = node_to_lp_cloned(insertion_location, expr_arena, lp_arena);
        Some(original_lp)
    } else {
        None
    };

    let Some((_, mut most_left)) = pipelines.pop() else {
        unreachable!()
    };
    while let Some((_, rhs)) = pipelines.pop() {
        most_left = most_left.with_other_branch(rhs)
    }
    // replace the part of the logical plan with a `MapFunction` that will execute the pipeline.
    let schema = lp_arena
        .get(insertion_location)
        .schema(lp_arena)
        .into_owned();
    let pipeline_node = get_pipeline_node(lp_arena, most_left, schema, original_lp);
    lp_arena.replace(insertion_location, pipeline_node);

    Ok(Some(final_sink))
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
                let mut state = ExecutionState::new();
                if state.verbose() {
                    eprintln!("RUN STREAMING PIPELINE")
                }
                state.set_in_streaming_engine();
                let state = Box::new(state) as Box<dyn SExecutionContext>;
                pipeline.execute(state)
            }),
            schema,
            original: original_lp.map(Arc::new),
        },
        input: dummy,
    }
}
