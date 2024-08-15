use std::borrow::Borrow;

use polars_core::prelude::PlHashMap;
use polars_error::PolarsResult;
use polars_plan::plans::expr_ir::ExprIR;
use polars_plan::plans::{AExpr, LiteralValue, IR};
use polars_plan::prelude::*;
use polars_utils::arena::{Arena, Node};
use slotmap::SlotMap;

use super::{PhysNode, PhysNodeKey};

type IRNodeKey = Node;

#[recursive::recursive]
fn is_streamable_rec(
    expr_key: IRNodeKey,
    arena: &Arena<AExpr>,
    cache: &mut PlHashMap<IRNodeKey, bool>,
) -> bool {
    if let Some(ret) = cache.get(&expr_key) {
        return *ret;
    }

    let ret = match arena.get(expr_key) {
        AExpr::Explode(_) => false,
        AExpr::Alias(inner, _) => is_streamable_rec(*inner, arena, cache),
        AExpr::Column(_) => true,
        AExpr::Literal(lit) => !matches!(lit, LiteralValue::Series(_) | LiteralValue::Range { .. }),
        AExpr::BinaryExpr { left, op: _, right } => {
            is_streamable_rec(*left, arena, cache) && is_streamable_rec(*right, arena, cache)
        },
        AExpr::Cast {
            expr,
            data_type: _,
            options: _,
        } => is_streamable_rec(*expr, arena, cache),
        AExpr::Sort { .. } | AExpr::SortBy { .. } | AExpr::Gather { .. } => false,
        AExpr::Filter { input, by } => {
            is_streamable_rec(*input, arena, cache) && is_streamable_rec(*by, arena, cache)
        },
        AExpr::Agg(_) => false,
        AExpr::Ternary {
            predicate,
            truthy,
            falsy,
        } => {
            is_streamable_rec(*predicate, arena, cache)
                && is_streamable_rec(*truthy, arena, cache)
                && is_streamable_rec(*falsy, arena, cache)
        },
        AExpr::AnonymousFunction {
            input: _,
            function: _,
            output_type: _,
            options,
        }
        | AExpr::Function {
            input: _,
            function: _,
            options,
        } => options.is_elementwise(),
        AExpr::Window { .. } => false,
        AExpr::Slice { .. } => false,
        AExpr::Len => false,
    };

    cache.insert(expr_key, ret);
    ret
}

fn is_streamable(expr_key: IRNodeKey, ctx: &mut LowerExprContext) -> bool {
    is_streamable_rec(expr_key, &ctx.expr_arena, &mut ctx.is_streamable_cache)
}

#[recursive::recursive]
fn is_input_independent_rec(
    expr_key: IRNodeKey,
    arena: &Arena<AExpr>,
    cache: &mut PlHashMap<IRNodeKey, bool>,
) -> bool {
    if let Some(ret) = cache.get(&expr_key) {
        return *ret;
    }

    let ret = match arena.get(expr_key) {
        AExpr::Explode(inner)
        | AExpr::Alias(inner, _)
        | AExpr::Cast {
            expr: inner,
            data_type: _,
            options: _,
        }
        | AExpr::Sort {
            expr: inner,
            options: _,
        } => is_input_independent_rec(*inner, arena, cache),
        AExpr::Column(_) => false,
        AExpr::Literal(_) => true,
        AExpr::BinaryExpr { left, op: _, right } => {
            is_input_independent_rec(*left, arena, cache)
                && is_input_independent_rec(*right, arena, cache)
        },
        AExpr::Gather {
            expr,
            idx,
            returns_scalar: _,
        } => {
            is_input_independent_rec(*expr, arena, cache)
                && is_input_independent_rec(*idx, arena, cache)
        },
        AExpr::SortBy {
            expr,
            by,
            sort_options: _,
        } => {
            is_input_independent_rec(*expr, arena, cache)
                && by
                    .iter()
                    .all(|expr| is_input_independent_rec(*expr, arena, cache))
        },
        AExpr::Filter { input, by } => {
            is_input_independent_rec(*input, arena, cache)
                && is_input_independent_rec(*by, arena, cache)
        },
        AExpr::Agg(agg_expr) => match agg_expr.get_input() {
            polars_plan::plans::NodeInputs::Leaf => true,
            polars_plan::plans::NodeInputs::Single(expr) => {
                is_input_independent_rec(expr, arena, cache)
            },
            polars_plan::plans::NodeInputs::Many(exprs) => exprs
                .iter()
                .all(|expr| is_input_independent_rec(*expr, arena, cache)),
        },
        AExpr::Ternary {
            predicate,
            truthy,
            falsy,
        } => {
            is_input_independent_rec(*predicate, arena, cache)
                && is_input_independent_rec(*truthy, arena, cache)
                && is_input_independent_rec(*falsy, arena, cache)
        },
        AExpr::AnonymousFunction {
            input,
            function: _,
            output_type: _,
            options: _,
        }
        | AExpr::Function {
            input,
            function: _,
            options: _,
        } => input
            .iter()
            .all(|expr| is_input_independent_rec(expr.node(), arena, cache)),
        AExpr::Window {
            function,
            partition_by,
            order_by,
            options: _,
        } => {
            is_input_independent_rec(*function, arena, cache)
                && partition_by
                    .iter()
                    .all(|expr| is_input_independent_rec(*expr, arena, cache))
                && order_by
                    .iter()
                    .all(|(expr, _options)| is_input_independent_rec(*expr, arena, cache))
        },
        AExpr::Slice {
            input,
            offset,
            length,
        } => {
            is_input_independent_rec(*input, arena, cache)
                && is_input_independent_rec(*offset, arena, cache)
                && is_input_independent_rec(*length, arena, cache)
        },
        AExpr::Len => false,
    };

    cache.insert(expr_key, ret);
    ret
}

fn is_input_independent(expr_key: IRNodeKey, ctx: &mut LowerExprContext) -> bool {
    is_input_independent_rec(
        expr_key,
        &ctx.expr_arena,
        &mut ctx.is_input_independent_cache,
    )
}

struct LowerExprContext<'a> {
    ir_arena: &'a mut Arena<IR>,
    expr_arena: &'a mut Arena<AExpr>,
    phys_sm: &'a mut SlotMap<PhysNodeKey, PhysNode>,
    is_streamable_cache: PlHashMap<Node, bool>,
    is_input_independent_cache: PlHashMap<Node, bool>,
}

#[recursive::recursive]
fn lower_exprs_rec(
    input: PhysNodeKey,
    exprs: &[ExprIR],
    ctx: &mut LowerExprContext,
) -> PolarsResult<(PhysNodeKey, Vec<ExprIR>)> {
    if exprs.iter().all(|e| is_input_independent(e.node(), ctx)) {
        return Ok((input, exprs.to_vec()));
    }
    
    let streamable_subset: Vec<_> = exprs.iter().filter(|e| is_input_independent(e.node(), ctx)).collect();
    
    todo!()
}

pub fn lower_exprs(
    input: PhysNodeKey,
    exprs: &[ExprIR],
    ir_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
    phys_sm: &mut SlotMap<PhysNodeKey, PhysNode>,
) -> PolarsResult<(PhysNodeKey, Vec<ExprIR>)> {
    todo!()
}
