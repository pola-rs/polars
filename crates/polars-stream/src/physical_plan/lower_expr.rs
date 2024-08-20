use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use polars_core::frame::DataFrame;
use polars_core::prelude::{Field, InitHashMaps, PlHashMap};
use polars_core::schema::Schema;
use polars_error::PolarsResult;
use polars_expr::state::ExecutionState;
use polars_expr::{create_physical_expr, ExpressionConversionState};
use polars_expr::planner::get_expr_depth_limit;
use polars_plan::plans::expr_ir::{ExprIR, OutputName};
use polars_plan::plans::{AExpr, LiteralValue, IR};
use polars_plan::prelude::*;
use polars_utils::arena::{Arena, Node};
use polars_utils::itertools::Itertools;
use slotmap::{Key, SlotMap};

use super::{PhysNode, PhysNodeKey, PhysNodeKind};

type IRNodeKey = Node;

fn unique_column_name() -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let idx = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("_POLARS_STREAM_TMP_{idx}")
}

#[recursive::recursive]
fn is_elementwise_rec(
    expr_key: IRNodeKey,
    arena: &Arena<AExpr>,
    cache: &mut PlHashMap<IRNodeKey, bool>,
) -> bool {
    if let Some(ret) = cache.get(&expr_key) {
        return *ret;
    }

    let ret = match arena.get(expr_key) {
        AExpr::Explode(_) => false,
        AExpr::Alias(inner, _) => is_elementwise_rec(*inner, arena, cache),
        AExpr::Column(_) => true,
        AExpr::Literal(lit) => !matches!(lit, LiteralValue::Series(_) | LiteralValue::Range { .. }),
        AExpr::BinaryExpr { left, op: _, right } => {
            is_elementwise_rec(*left, arena, cache) && is_elementwise_rec(*right, arena, cache)
        },
        AExpr::Cast {
            expr,
            data_type: _,
            options: _,
        } => is_elementwise_rec(*expr, arena, cache),
        AExpr::Sort { .. } | AExpr::SortBy { .. } | AExpr::Gather { .. } => false,
        AExpr::Filter { input, by } => {
            is_elementwise_rec(*input, arena, cache) && is_elementwise_rec(*by, arena, cache)
        },
        AExpr::Agg(_) => false,
        AExpr::Ternary {
            predicate,
            truthy,
            falsy,
        } => {
            is_elementwise_rec(*predicate, arena, cache)
                && is_elementwise_rec(*truthy, arena, cache)
                && is_elementwise_rec(*falsy, arena, cache)
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

fn is_elementwise(expr_key: IRNodeKey, ctx: &mut LowerExprContext) -> bool {
    is_elementwise_rec(expr_key, &ctx.expr_arena, &mut ctx.is_elementwise_cache)
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

fn is_aggregation(expr_key: IRNodeKey, ctx: &mut LowerExprContext) -> bool {
    match ctx.expr_arena.get(expr_key) {
        AExpr::Agg(_) => true,
        _ => false,
    }
}

/// Generates a new expression for the output column of an arbitrary expression.
fn expr_for_output_column(
    expr: &ExprIR,
    ctx: &mut LowerExprContext
) -> ExprIR {
    if matches!(ctx.expr_arena.get(expr.node()), AExpr::Column(_)) {
        return expr.clone();
    }

    let col_name = expr.output_name_inner().clone();
    ExprIR::new(ctx.expr_arena.add(AExpr::Column(col_name.unwrap().clone())), col_name)
}

fn build_input_independent_node_with_ctx(
    exprs: &[ExprIR],
    ctx: &mut LowerExprContext,
) -> PolarsResult<PhysNodeKey> {
    let expr_depth_limit = get_expr_depth_limit()?;
    let mut state = ExpressionConversionState::new(false, expr_depth_limit);
    let empty = DataFrame::empty();
    let execution_state = ExecutionState::new();
    let columns = exprs.iter().map(|expr| {
        let phys_expr = create_physical_expr(
            expr,
            Context::Default,
            ctx.expr_arena,
            None,
            &mut state,
        )?;
        
        phys_expr.evaluate(&empty, &execution_state)
    }).try_collect_vec()?;
    
    let df = Arc::new(DataFrame::new_with_broadcast(columns)?);
    Ok(ctx.phys_sm.insert(PhysNode::new(Arc::new(df.schema()), PhysNodeKind::InMemorySource { df })))
}

struct LowerExprContext<'a> {
    ir_arena: &'a mut Arena<IR>,
    expr_arena: &'a mut Arena<AExpr>,
    phys_sm: &'a mut SlotMap<PhysNodeKey, PhysNode>,
    is_elementwise_cache: PlHashMap<Node, bool>,
    is_input_independent_cache: PlHashMap<Node, bool>,
}

#[recursive::recursive]
fn lower_exprs_with_ctx(
    input: PhysNodeKey,
    exprs: &[ExprIR],
    ctx: &mut LowerExprContext,
) -> PolarsResult<(PhysNodeKey, Vec<ExprIR>)> {
    if exprs.iter().all(|e| is_input_independent(e.node(), ctx)) {
        let node = build_input_independent_node_with_ctx(exprs, ctx)?;
        let exprs = exprs.iter().map(|e| expr_for_output_column(e, ctx)).collect();
        return Ok((node, exprs));
    }

    // Elementwise expressions that can directly be applied to the original input.
    let mut elementwise_subset = Vec::new();
    
    // Aggregation expressions that can directly be applied to the original input.
    let mut agg_subset = Vec::new();

    // Fallback expressions that can directly be applied to the original input.
    // let mut fallback_subset = Vec::new();

    // Expressions that create DataFrames from thin air.
    let mut input_independent = Vec::new();

    // New nodes containing transformed data columns used for executing
    // transformed expressions.
    let mut transformed = Vec::new();

    for expr in exprs {
        if is_elementwise(expr.node(), ctx) {
            elementwise_subset.push(expr.clone());
            continue;
        }

        match ctx.expr_arena.get(expr.node()) {
            AExpr::Explode(_) => {
                // While explode is streamable, it is not elementwise, so we
                // have to transform it to a select node.
                let output_schema = schema_for_select(input, &[expr.clone()], ctx)?;
                let node_kind = PhysNodeKind::Select { input, selectors: vec![expr.clone()], extend_original: false };
                let node_key = ctx.phys_sm.insert(PhysNode::new(output_schema, node_kind));
                transformed.push(node_key);
            }
            AExpr::Alias(_, _) => unreachable!(),
            AExpr::Column(_) => unreachable!("column should always be streamable"),
            AExpr::Literal(_) => input_independent.push(expr.clone()),
            AExpr::BinaryExpr { left, op, right } => {
                // One or both of left and right is not streaming, give them
                // temporary names, recurse to make a streaming input from them.
                let left_new_name = unique_column_name();
                let right_new_name = unique_column_name();
                let left_expr = ExprIR::new(left.clone(), OutputName::Alias(left_new_name.into()));
                let right_expr = ExprIR::new(right.clone(), OutputName::Alias(right_new_name.into()));
                let (trans_input, trans_exprs) = lower_exprs_with_ctx(input, &[left_expr, right_expr], ctx)?;
                todo!()
                
            },
            AExpr::Cast {
                expr,
                data_type,
                options,
            } => {
                todo!()
                
            },
            AExpr::Sort {
                expr: inner,
                options,
            } => {
                let sort_options = options.into();

                // As we'll refer to the column twice, ensure the inner expr is
                // available as a column by selecting first.
                let inner_expr_ir = ExprIR::new(*inner, expr.output_name_inner().clone());
                let select_node = build_select_node_with_ctx(input, &[inner_expr_ir], ctx)?;
                let kind = PhysNodeKind::Sort {
                    input: select_node,
                    by_column: vec![expr_for_output_column(expr, ctx)],
                    slice: None,
                    sort_options,
                };
                let output_schema = ctx.phys_sm[select_node].output_schema.clone();
                let node_key = ctx.phys_sm.insert(PhysNode::new(output_schema, kind));
                transformed.push(node_key);
            },
            AExpr::Gather { .. } => todo!(),
            AExpr::SortBy {
                expr,
                by,
                sort_options,
            } => todo!(),
            AExpr::Filter { input, by } => todo!(),
            AExpr::Agg(agg) => match agg {
                IRAggExpr::Min { .. } => agg_subset.push(expr.clone()),
                IRAggExpr::Max { .. } => agg_subset.push(expr.clone()),
                IRAggExpr::Median(_) => todo!(),
                IRAggExpr::NUnique(_) => todo!(),
                IRAggExpr::First(_) => todo!(),
                IRAggExpr::Last(_) => todo!(),
                IRAggExpr::Mean(_) => agg_subset.push(expr.clone()),
                IRAggExpr::Implode(_) => todo!(),
                IRAggExpr::Quantile {.. } => todo!(),
                IRAggExpr::Sum(_) => agg_subset.push(expr.clone()),
                IRAggExpr::Count(_, _) => todo!(),
                IRAggExpr::Std(_, _) => todo!(),
                IRAggExpr::Var(_, _) => todo!(),
                IRAggExpr::AggGroups(_) => todo!(),
            },
            AExpr::Ternary {
                predicate,
                truthy,
                falsy,
            } => todo!(),
            AExpr::AnonymousFunction {
                input,
                function,
                output_type,
                options,
            } => todo!(),
            AExpr::Function {
                input,
                function,
                options,
            } => todo!(),
            AExpr::Window {
                function,
                partition_by,
                order_by,
                options,
            } => todo!(),
            AExpr::Slice {
                input,
                offset,
                length,
            } => todo!(),
            AExpr::Len => todo!(),
        }
    }

    // let multiplexer = ctx.phys_sm.insert(PhysNode::Multiplexer { input });
    // let mut transformed = Vec::with_capacity(exprs.len());
    // if streamable_subset.len() > 0 {

    // }
    // for expr in exprs {

    // }

    // let multiplexer = PhysNode::Multiplexer { input };

    // let selectors = expr.clone();
    // let output_schema = schema.clone();
    // let input = lower_ir(*input, ir_arena, expr_arena, phys_sm)?;
    // Ok(phys_sm.insert(PhysNode::Select {
    //     input,
    //     selectors,
    //     output_schema,
    //     extend_original: false,
    // }))

    // Replace original input node with multiplexer.
    let input_schema = ctx.phys_sm[input].output_schema.clone();
    let orig_input_node = core::mem::replace(
        &mut ctx.phys_sm[input],
        PhysNode::new(input_schema, PhysNodeKind::Multiplexer { input: PhysNodeKey::null() }),
    );
    let orig_input_key = ctx.phys_sm.insert(orig_input_node);
    ctx.phys_sm[input].kind = PhysNodeKind::Multiplexer {
        input: orig_input_key,
    };

    todo!()
}

/// Computes the schema that selecting the given expressions on the input node
/// would result in.
fn schema_for_select(
    input: PhysNodeKey,
    exprs: &[ExprIR],
    ctx: &mut LowerExprContext,
) -> PolarsResult<Arc<Schema>> {
    let input_schema = &ctx.phys_sm[input].output_schema;
    let output_schema: Schema = exprs.iter().map(|e| {
        let name = e.output_name();
        let dtype = ctx.expr_arena.get(e.node()).to_dtype(input_schema, Context::Default, &ctx.expr_arena)?;
        PolarsResult::Ok(Field::new(name, dtype))
    }).try_collect()?;
    Ok(Arc::new(output_schema))
}
    

fn build_select_node_with_ctx(
    input: PhysNodeKey,
    exprs: &[ExprIR],
    ctx: &mut LowerExprContext,
) -> PolarsResult<PhysNodeKey> {
    if exprs.iter().all(|e| is_input_independent(e.node(), ctx)) {
        return build_input_independent_node_with_ctx(exprs, ctx);
    }

    // Are we only selecting simple columns?
    let all_simple_columns: Option<Vec<String>> = exprs
        .iter()
        .map(|e| match ctx.expr_arena.get(e.node()) {
            AExpr::Column(name) => Some(name.to_string()),
            _ => None,
        })
        .collect();

    if let Some(columns) = all_simple_columns {
        let input_schema = ctx.phys_sm[input].output_schema.clone();
        if input_schema.len() == columns.len() && input_schema.iter_names().zip(&columns).all(|(l, r)| l == r) {
            // Input node already has the correct schema, just pass through.
            return Ok(input);
        }
        
        let output_schema = Arc::new(input_schema.select(&columns)?);
        let node_kind = PhysNodeKind::SimpleProjection { input, columns };
        return Ok(ctx.phys_sm.insert(PhysNode::new(output_schema, node_kind)));
    }
    
    let (transformed_input, transformed_exprs) = lower_exprs_with_ctx(input, exprs, ctx)?;
    let output_schema = schema_for_select(transformed_input, &transformed_exprs, ctx)?;
    let node_kind = PhysNodeKind::Select { input: transformed_input, selectors: transformed_exprs, extend_original: false };
    Ok(ctx.phys_sm.insert(PhysNode::new(output_schema, node_kind)))
}


/// Lowers an input node plus a set of expressions on that input node to an
/// equivalent (input node, set of expressions) pair, ensuring that the new set
/// of expressions can run on the streaming engine.
pub fn lower_exprs(
    input: PhysNodeKey,
    exprs: &[ExprIR],
    ir_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
    phys_sm: &mut SlotMap<PhysNodeKey, PhysNode>,
) -> PolarsResult<(PhysNodeKey, Vec<ExprIR>)> {
    let mut ctx = LowerExprContext {
        ir_arena,
        expr_arena,
        phys_sm,
        is_elementwise_cache: PlHashMap::new(),
        is_input_independent_cache: PlHashMap::new(),
    };
    lower_exprs_with_ctx(input, exprs, &mut ctx)
}

/// Builds a selection node given an input node and the expressions to select for.
pub fn build_select_node(
    input: PhysNodeKey,
    exprs: &[ExprIR],
    ir_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
    phys_sm: &mut SlotMap<PhysNodeKey, PhysNode>,
) -> PolarsResult<PhysNodeKey> {
    let mut ctx = LowerExprContext {
        ir_arena,
        expr_arena,
        phys_sm,
        is_elementwise_cache: PlHashMap::new(),
        is_input_independent_cache: PlHashMap::new(),
    };
    build_select_node_with_ctx(input, exprs, &mut ctx)
}
