#![allow(unused)] // TODO: remove me
use polars_core::prelude::*;
use polars_core::POOL;
use polars_expr::planner::{create_physical_expr, get_expr_depth_limit, ExpressionConversionState};
use polars_plan::logical_plan::{Context, IR};
use polars_plan::prelude::expr_ir::ExprIR;
use polars_plan::prelude::AExpr;
use polars_utils::arena::{Arena, Node};
use slotmap::SecondaryMap;
use slotmap::SlotMap;

fn is_streamable(node: Node, arena: &Arena<AExpr>) -> bool {
    polars_plan::logical_plan::is_streamable(node, arena, Context::Default)
}

pub fn run_query(
    node: Node,
    mut ir_arena: Arena<IR>,
    mut expr_arena: Arena<AExpr>,
) -> PolarsResult<DataFrame> {
    let mut phys_sm = SlotMap::with_capacity_and_key(ir_arena.len());
    let root = crate::physical_plan::lower_ir(node, &mut ir_arena, &mut expr_arena, &mut phys_sm)?;
    let expr_depth_limit = get_expr_depth_limit()?;
    let mut expr_conversion_state = ExpressionConversionState::new(false, expr_depth_limit);
    let max_threads = POOL.current_num_threads();
    

    // match phys_sm.take(root) {
    //     LogicalPlan::Filter { input, predicate } => {
    //         let phys_expr = create_physical_expr(
    //             &predicate,
    //             Context::Default,
    //             &expr_arena,
    //             None,
    //             &mut expr_conversion_state,
    //         )?;
    //         todo!()
    //     },
    //     _ => todo!(),
    // }
    todo!()
}
