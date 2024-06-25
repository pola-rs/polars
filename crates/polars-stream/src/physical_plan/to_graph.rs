use polars_error::PolarsResult;
use polars_expr::planner::{create_physical_expr, get_expr_depth_limit, ExpressionConversionState};
use polars_plan::plans::{AExpr, Context};
use polars_utils::arena::Arena;
use recursive::recursive;
use slotmap::{SecondaryMap, SlotMap};

use super::{PhysNode, PhysNodeKey};
use crate::graph::{Graph, GraphNodeKey};
use crate::nodes;

struct GraphConversionContext<'a> {
    phys_sm: &'a SlotMap<PhysNodeKey, PhysNode>,
    expr_arena: &'a Arena<AExpr>,
    graph: Graph,
    phys_to_graph: SecondaryMap<PhysNodeKey, GraphNodeKey>,
    expr_conversion_state: ExpressionConversionState,
}

pub fn physical_plan_to_graph(
    phys_sm: &SlotMap<PhysNodeKey, PhysNode>,
    expr_arena: &Arena<AExpr>,
) -> PolarsResult<(Graph, SecondaryMap<PhysNodeKey, GraphNodeKey>)> {
    let expr_depth_limit = get_expr_depth_limit()?;
    let mut ctx = GraphConversionContext {
        phys_sm,
        expr_arena,
        graph: Graph::with_capacity(phys_sm.len()),
        phys_to_graph: SecondaryMap::with_capacity(phys_sm.len()),
        expr_conversion_state: ExpressionConversionState::new(false, expr_depth_limit),
    };

    for key in phys_sm.keys() {
        to_graph_rec(key, &mut ctx)?;
    }

    Ok((ctx.graph, ctx.phys_to_graph))
}

#[recursive]
fn to_graph_rec<'a>(
    phys_node_key: PhysNodeKey,
    ctx: &mut GraphConversionContext<'a>,
) -> PolarsResult<GraphNodeKey> {
    // This will ensure we create a proper acyclic directed graph instead of a tree.
    if let Some(graph_key) = ctx.phys_to_graph.get(phys_node_key) {
        return Ok(*graph_key);
    }

    use PhysNode::*;
    let graph_key = match &ctx.phys_sm[phys_node_key] {
        InMemorySource { df } => ctx
            .graph
            .add_node(nodes::in_memory_source::InMemorySource::new(df.clone()), []),

        Filter { predicate, input } => {
            let phys_predicate_expr = create_physical_expr(
                predicate,
                Context::Default,
                ctx.expr_arena,
                None,
                &mut ctx.expr_conversion_state,
            )?;
            let input_key = to_graph_rec(*input, ctx)?;
            ctx.graph.add_node(
                nodes::filter::FilterNode::new(phys_predicate_expr),
                [input_key],
            )
        },

        SimpleProjection { schema, input } => {
            let input_key = to_graph_rec(*input, ctx)?;
            ctx.graph.add_node(
                nodes::simple_projection::SimpleProjectionNode::new(schema.clone()),
                [input_key],
            )
        },

        InMemorySink { input } => {
            let input_key = to_graph_rec(*input, ctx)?;
            ctx.graph
                .add_node(nodes::in_memory_sink::InMemorySink::default(), [input_key])
        },

        // Fallback to the in-memory engine.
        Fallback(_node) => {
            todo!()
        },
    };

    ctx.phys_to_graph.insert(phys_node_key, graph_key);
    Ok(graph_key)
}
