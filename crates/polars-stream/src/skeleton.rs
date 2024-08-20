#![allow(unused)] // TODO: remove me
use polars_core::prelude::*;
use polars_core::POOL;
use polars_expr::planner::{create_physical_expr, get_expr_depth_limit, ExpressionConversionState};
use polars_plan::plans::{Context, IR};
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
    expr_arena: &Arena<AExpr>,
) -> PolarsResult<DataFrame> {
    let mut phys_sm = SlotMap::with_capacity_and_key(ir_arena.len());
    let mut schema_cache = PlHashMap::with_capacity(ir_arena.len());
    let root = crate::physical_plan::lower_ir(
        node,
        &mut ir_arena,
        expr_arena,
        &mut phys_sm,
        &mut schema_cache,
    )?;
    let (mut graph, phys_to_graph) =
        crate::physical_plan::physical_plan_to_graph(&phys_sm, expr_arena)?;
    let mut results = crate::execute::execute_graph(&mut graph)?;
    Ok(results.remove(phys_to_graph[root]).unwrap())
}
