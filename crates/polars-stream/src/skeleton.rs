#![allow(warnings, unused)] // TODO: remove me
use polars_core::prelude::*;
use polars_core::POOL;
use polars_expr::planner::{create_physical_expr, get_expr_depth_limit, ExpressionConversionState};
use polars_plan::logical_plan::{Context, IR};
use polars_plan::prelude::expr_ir::ExprIR;
use polars_plan::prelude::AExpr;
use polars_utils::arena::{Arena, Node};

fn is_streamable(node: Node, arena: &Arena<AExpr>) -> bool {
    polars_plan::logical_plan::is_streamable(node, arena, Context::Default)
}

pub fn run_query(
    node: Node,
    mut ir_arena: Arena<IR>,
    mut expr_arena: Arena<AExpr>,
) -> PolarsResult<DataFrame> {
    let mut lir_arena = Arena::with_capacity(ir_arena.len());
    let root = lower_ir(node, &mut ir_arena, &mut expr_arena, &mut lir_arena)?;
    let expr_depth_limit = get_expr_depth_limit()?;
    let mut expr_conversion_state = ExpressionConversionState::new(false, expr_depth_limit);
    let max_threads = POOL.current_num_threads();

    match lir_arena.take(root) {
        LogicalPlan::Filter { input, predicate } => {
            let phys_expr = create_physical_expr(
                &predicate,
                Context::Default,
                &expr_arena,
                None,
                &mut expr_conversion_state,
            )?;
            todo!()
        },
        _ => todo!(),
    }
    // todo!
}

#[recursive::recursive]
fn lower_ir(
    node: Node,
    ir_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
    lir_arena: &mut Arena<LogicalPlan>,
) -> PolarsResult<Node> {
    let node = match ir_arena.get(node) {
        IR::Filter { input, predicate } if is_streamable(predicate.node(), expr_arena) => {
            let predicate = predicate.clone();
            let input = lower_ir(*input, ir_arena, expr_arena, lir_arena)?;
            lir_arena.add(LogicalPlan::Filter { input, predicate })
        },
        IR::DataFrameScan {
            df,
            schema,
            output_schema,
            projection,
            filter,
        } => {
            if let Some(filter) = filter {
                if !is_streamable(filter.node(), expr_arena) {
                    return Ok(lir_arena.add(LogicalPlan::FallBack(node)));
                }
            }
            lir_arena.add(LogicalPlan::DataFrameScan {
                df: df.clone(),
                schema: schema.clone(),
                output_schema: output_schema.clone(),
                projection: projection.clone(),
                filter: filter.clone(),
            })
        },
        _ => return Ok(lir_arena.add(LogicalPlan::FallBack(node))),
    };
    Ok(node)
}

/// Invariant that all expression are elementwise.
#[derive(Clone, Default)]
enum LogicalPlan {
    DataFrameScan {
        df: Arc<DataFrame>,
        schema: SchemaRef,
        // schema of the projected file
        output_schema: Option<SchemaRef>,
        projection: Option<Arc<[String]>>,
        filter: Option<ExprIR>,
    },
    Filter {
        input: Node,
        predicate: ExprIR,
    },
    // Fallback to in-memory engine
    FallBack(Node),
    #[default]
    Unreachable,
}
