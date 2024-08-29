#![allow(unused)] // TODO: remove me
use polars_core::prelude::*;
use polars_core::POOL;
use polars_expr::planner::{create_physical_expr, get_expr_depth_limit, ExpressionConversionState};
use polars_plan::plans::{Context, IRPlan, IR};
use polars_plan::prelude::expr_ir::ExprIR;
use polars_plan::prelude::AExpr;
use polars_utils::arena::{Arena, Node};
use slotmap::{SecondaryMap, SlotMap};

fn is_streamable(node: Node, arena: &Arena<AExpr>) -> bool {
    polars_plan::plans::is_streamable(node, arena, Context::Default)
}

pub fn run_query(
    node: Node,
    mut ir_arena: Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
) -> PolarsResult<DataFrame> {
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
    let mut results = crate::execute::execute_graph(&mut graph)?;
    Ok(results.remove(phys_to_graph[root]).unwrap())
}
