use std::sync::Arc;

use parking_lot::Mutex;
use polars_error::PolarsResult;
use polars_expr::planner::{create_physical_expr, get_expr_depth_limit, ExpressionConversionState};
use polars_expr::reduce::into_reduction;
use polars_expr::state::ExecutionState;
use polars_mem_engine::create_physical_plan;
use polars_plan::plans::expr_ir::ExprIR;
use polars_plan::plans::{AExpr, ArenaExprIter, Context, IR};
use polars_plan::prelude::FunctionFlags;
use polars_utils::arena::{Arena, Node};
use recursive::recursive;
use slotmap::{SecondaryMap, SlotMap};

use super::{PhysNode, PhysNodeKey};
use crate::expression::StreamExpr;
use crate::graph::{Graph, GraphNodeKey};
use crate::nodes;
use crate::utils::late_materialized_df::LateMaterializedDataFrame;

fn has_potential_recurring_entrance(node: Node, arena: &Arena<AExpr>) -> bool {
    arena.iter(node).any(|(_n, ae)| match ae {
        AExpr::Function { options, .. } | AExpr::AnonymousFunction { options, .. } => {
            options.flags.contains(FunctionFlags::OPTIONAL_RE_ENTRANT)
        },
        _ => false,
    })
}

fn create_stream_expr(
    expr_ir: &ExprIR,
    ctx: &mut GraphConversionContext<'_>,
) -> PolarsResult<StreamExpr> {
    let reentrant = has_potential_recurring_entrance(expr_ir.node(), ctx.expr_arena);
    let phys = create_physical_expr(
        expr_ir,
        Context::Default,
        ctx.expr_arena,
        None,
        &mut ctx.expr_conversion_state,
    )?;
    Ok(StreamExpr::new(phys, reentrant))
}

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
        InMemorySource { df } => ctx.graph.add_node(
            nodes::in_memory_source::InMemorySourceNode::new(df.clone()),
            [],
        ),

        StreamingSlice {
            input,
            offset,
            length,
        } => {
            let input_key = to_graph_rec(*input, ctx)?;
            ctx.graph.add_node(
                nodes::streaming_slice::StreamingSliceNode::new(*offset, *length),
                [input_key],
            )
        },

        Filter { predicate, input } => {
            let phys_predicate_expr = create_stream_expr(predicate, ctx)?;
            let input_key = to_graph_rec(*input, ctx)?;
            ctx.graph.add_node(
                nodes::filter::FilterNode::new(phys_predicate_expr),
                [input_key],
            )
        },

        Select {
            selectors,
            input,
            output_schema,
            extend_original,
        } => {
            let phys_selectors = selectors
                .iter()
                .map(|selector| create_stream_expr(selector, ctx))
                .collect::<PolarsResult<_>>()?;
            let input_key = to_graph_rec(*input, ctx)?;
            ctx.graph.add_node(
                nodes::select::SelectNode::new(
                    phys_selectors,
                    output_schema.clone(),
                    *extend_original,
                ),
                [input_key],
            )
        },
        Reduce {
            input,
            exprs,
            input_schema,
            output_schema,
        } => {
            let input_key = to_graph_rec(*input, ctx)?;

            let mut reductions = Vec::with_capacity(exprs.len());
            let mut inputs = Vec::with_capacity(reductions.len());

            for e in exprs {
                let (red, input_node) =
                    into_reduction(e.node(), ctx.expr_arena, input_schema.as_ref())?
                        .expect("invariant");
                reductions.push(red);

                let input_phys =
                    create_stream_expr(&ExprIR::from_node(input_node, ctx.expr_arena), ctx)?;

                inputs.push(input_phys)
            }

            ctx.graph.add_node(
                nodes::reduce::ReduceNode::new(inputs, reductions, output_schema.clone()),
                [input_key],
            )
        },
        SimpleProjection {
            input,
            columns,
            input_schema,
        } => {
            let input_key = to_graph_rec(*input, ctx)?;
            ctx.graph.add_node(
                nodes::simple_projection::SimpleProjectionNode::new(
                    columns.clone(),
                    input_schema.clone(),
                ),
                [input_key],
            )
        },

        InMemorySink { input, schema } => {
            let input_key = to_graph_rec(*input, ctx)?;
            ctx.graph.add_node(
                nodes::in_memory_sink::InMemorySinkNode::new(schema.clone()),
                [input_key],
            )
        },

        InMemoryMap {
            input,
            input_schema,
            map,
        } => {
            let input_key = to_graph_rec(*input, ctx)?;
            ctx.graph.add_node(
                nodes::in_memory_map::InMemoryMapNode::new(input_schema.clone(), map.clone()),
                [input_key],
            )
        },

        Map { input, map } => {
            let input_key = to_graph_rec(*input, ctx)?;
            ctx.graph
                .add_node(nodes::map::MapNode::new(map.clone()), [input_key])
        },

        Sort {
            input,
            input_schema,
            by_column,
            slice,
            sort_options,
        } => {
            let lmdf = Arc::new(LateMaterializedDataFrame::default());
            let mut lp_arena = Arena::default();
            let df_node = lp_arena.add(lmdf.clone().as_ir_node(input_schema.clone()));
            let sort_node = lp_arena.add(IR::Sort {
                input: df_node,
                by_column: by_column.clone(),
                slice: *slice,
                sort_options: sort_options.clone(),
            });
            let executor = Mutex::new(create_physical_plan(
                sort_node,
                &mut lp_arena,
                ctx.expr_arena,
            )?);

            let input_key = to_graph_rec(*input, ctx)?;
            ctx.graph.add_node(
                nodes::in_memory_map::InMemoryMapNode::new(
                    input_schema.clone(),
                    Arc::new(move |df| {
                        lmdf.set_materialized_dataframe(df);
                        let mut state = ExecutionState::new();
                        executor.lock().execute(&mut state)
                    }),
                ),
                [input_key],
            )
        },

        OrderedUnion { inputs } => {
            let input_keys = inputs
                .iter()
                .map(|i| to_graph_rec(*i, ctx))
                .collect::<Result<Vec<_>, _>>()?;
            ctx.graph
                .add_node(nodes::ordered_union::OrderedUnionNode::new(), input_keys)
        },

        Zip { inputs } => {
            let input_keys = inputs
                .iter()
                .map(|i| to_graph_rec(*i, ctx))
                .collect::<Result<Vec<_>, _>>()?;
            ctx.graph.add_node(nodes::zip::ZipNode::new(), input_keys)
        },
    };

    ctx.phys_to_graph.insert(phys_node_key, graph_key);
    Ok(graph_key)
}
