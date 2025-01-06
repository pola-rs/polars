#![allow(unused)] // TODO: remove me
use std::cmp::Reverse;

use polars_core::prelude::*;
use polars_core::POOL;
use polars_expr::planner::{create_physical_expr, get_expr_depth_limit, ExpressionConversionState};
use polars_plan::plans::{Context, IRPlan, IR};
use polars_plan::prelude::expr_ir::ExprIR;
use polars_plan::prelude::AExpr;
use polars_utils::arena::{Arena, Node};
use slotmap::{SecondaryMap, SlotMap};

pub fn run_query(
    node: Node,
    mut ir_arena: Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
) -> PolarsResult<Option<DataFrame>> {
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
    let root =
        crate::physical_plan::build_physical_plan(node, &mut ir_arena, expr_arena, &mut phys_sm)?;
    if let Ok(visual_path) = std::env::var("POLARS_VISUALIZE_PHYSICAL_PLAN") {
        let visualization = crate::physical_plan::visualize_plan(root, &phys_sm, expr_arena);
        std::fs::write(visual_path, visualization).unwrap();
    }
    let (mut graph, phys_to_graph) =
        crate::physical_plan::physical_plan_to_graph(root, &phys_sm, expr_arena)?;

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

    Ok(results.remove(phys_to_graph[root]))
}
