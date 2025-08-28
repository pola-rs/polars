#![allow(unused)] // TODO: remove me
use std::cmp::Reverse;

use polars_core::POOL;
use polars_core::prelude::*;
use polars_expr::planner::{ExpressionConversionState, create_physical_expr, get_expr_depth_limit};
use polars_plan::plans::{Context, IR, IRPlan};
use polars_plan::prelude::AExpr;
use polars_plan::prelude::expr_ir::ExprIR;
use polars_utils::arena::{Arena, Node};
use slotmap::{SecondaryMap, SlotMap};

use crate::graph::{Graph, GraphNodeKey};
use crate::physical_plan::{PhysNode, PhysNodeKey, PhysNodeKind, StreamingLowerIRContext};

/// Executes the IR with the streaming engine.
///
/// Unsupported operations can fall back to the in-memory engine.
///
/// Returns:
/// - `Ok(QueryResult::Single(DataFrame))` when collecting to a single sink.
/// - `Ok(QueryResult::Multiple(Vec<DataFrame>))` when collecting to multiple sinks.
/// - `Err` if the IR can't be executed.
///
/// Returned `DataFrame`s contain data only for memory sinks,
/// `DataFrame`s corresponding to file sinks are empty.
pub fn run_query(
    node: Node,
    ir_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
) -> PolarsResult<QueryResult> {
    StreamingQuery::build(node, ir_arena, expr_arena)?.execute()
}

/// Visualizes the physical plan as a dot graph.
pub fn visualize_physical_plan(
    node: Node,
    ir_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
) -> PolarsResult<String> {
    let mut phys_sm = SlotMap::with_capacity_and_key(ir_arena.len());

    let ctx = StreamingLowerIRContext {
        prepare_visualization: true,
    };
    let root_phys_node =
        crate::physical_plan::build_physical_plan(node, ir_arena, expr_arena, &mut phys_sm, ctx)?;

    let out = crate::physical_plan::visualize_plan(root_phys_node, &phys_sm, expr_arena);

    Ok(out)
}

pub struct StreamingQuery {
    top_ir: IR,
    graph: Graph,
    root_phys_node: PhysNodeKey,
    phys_sm: SlotMap<PhysNodeKey, PhysNode>,
    phys_to_graph: SecondaryMap<PhysNodeKey, GraphNodeKey>,
}

impl StreamingQuery {
    pub fn build(
        node: Node,
        ir_arena: &mut Arena<IR>,
        expr_arena: &mut Arena<AExpr>,
    ) -> PolarsResult<Self> {
        if let Ok(visual_path) = std::env::var("POLARS_VISUALIZE_IR") {
            let plan = IRPlan {
                lp_top: node,
                lp_arena: ir_arena.clone(),
                expr_arena: expr_arena.clone(),
            };
            let visualization = plan.display_dot().to_string();
            std::fs::write(visual_path, visualization).unwrap();
        }
        let mut phys_sm = SlotMap::with_capacity_and_key(ir_arena.len());
        let ctx = StreamingLowerIRContext {
            prepare_visualization: false,
        };
        let root_phys_node = crate::physical_plan::build_physical_plan(
            node,
            ir_arena,
            expr_arena,
            &mut phys_sm,
            ctx,
        )?;
        if let Ok(visual_path) = std::env::var("POLARS_VISUALIZE_PHYSICAL_PLAN") {
            let visualization =
                crate::physical_plan::visualize_plan(root_phys_node, &phys_sm, expr_arena);
            std::fs::write(visual_path, visualization).unwrap();
        }

        let (mut graph, phys_to_graph) =
            crate::physical_plan::physical_plan_to_graph(root_phys_node, &phys_sm, expr_arena)?;

        let top_ir = ir_arena.get(node).clone();

        let out = StreamingQuery {
            top_ir,
            graph,
            root_phys_node,
            phys_sm,
            phys_to_graph,
        };

        Ok(out)
    }

    pub fn execute(self) -> PolarsResult<QueryResult> {
        let StreamingQuery {
            top_ir,
            mut graph,
            root_phys_node,
            phys_sm,
            phys_to_graph,
        } = self;

        crate::async_executor::clear_task_wait_statistics();
        let mut results = crate::execute::execute_graph(&mut graph)?;

        if std::env::var("POLARS_TRACK_WAIT_STATS").as_deref() == Ok("1") {
            let mut stats = crate::async_executor::get_task_wait_statistics();
            stats.sort_by_key(|(_l, w)| Reverse(*w));
            eprintln!("Time spent waiting for async tasks:");
            for (loc, wait_time) in stats {
                eprintln!("{}:{} - {:?}", loc.file(), loc.line(), wait_time);
            }
        }

        match top_ir {
            IR::SinkMultiple { inputs } => {
                let phys_node = &phys_sm[root_phys_node];
                let PhysNodeKind::SinkMultiple { sinks } = phys_node.kind() else {
                    unreachable!();
                };

                Ok(QueryResult::Multiple(
                    sinks
                        .iter()
                        .map(|phys_node_key| {
                            results
                                .remove(phys_to_graph[*phys_node_key])
                                .unwrap_or_else(DataFrame::empty)
                        })
                        .collect(),
                ))
            },
            _ => Ok(QueryResult::Single(
                results
                    .remove(phys_to_graph[root_phys_node])
                    .unwrap_or_else(DataFrame::empty),
            )),
        }
    }
}

pub enum QueryResult {
    Single(DataFrame),
    /// Collected to multiple in-memory sinks
    Multiple(Vec<DataFrame>),
}

impl QueryResult {
    pub fn unwrap_single(self) -> DataFrame {
        use QueryResult::*;
        match self {
            Single(df) => df,
            Multiple(_) => panic!(),
        }
    }

    pub fn unwrap_multiple(self) -> Vec<DataFrame> {
        use QueryResult::*;
        match self {
            Single(_) => panic!(),
            Multiple(dfs) => dfs,
        }
    }
}
