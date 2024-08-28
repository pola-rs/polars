use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use polars_core::frame::DataFrame;
use polars_core::prelude::{Field, InitHashMaps, PlHashMap, PlHashSet};
use polars_core::schema::Schema;
use polars_error::PolarsResult;
use polars_expr::planner::get_expr_depth_limit;
use polars_expr::state::ExecutionState;
use polars_expr::{create_physical_expr, ExpressionConversionState};
use polars_plan::plans::expr_ir::{ExprIR, OutputName};
use polars_plan::plans::{AExpr, LiteralValue};
use polars_plan::prelude::*;
use polars_utils::arena::{Arena, Node};
use polars_utils::itertools::Itertools;
use slotmap::SlotMap;

use super::{PhysNode, PhysNodeKey, PhysNodeKind};

type IRNodeKey = Node;

fn unique_column_name() -> ColumnName {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let idx = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("__POLARS_STMP_{idx}").into()
}

pub(crate) struct ExprCache {
    is_elementwise: PlHashMap<Node, bool>,
    is_input_independent: PlHashMap<Node, bool>,
}

impl ExprCache {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            is_elementwise: PlHashMap::with_capacity(capacity),
            is_input_independent: PlHashMap::with_capacity(capacity),
        }
    }
}

struct LowerExprContext<'a> {
    expr_arena: &'a mut Arena<AExpr>,
    phys_sm: &'a mut SlotMap<PhysNodeKey, PhysNode>,
    cache: &'a mut ExprCache,
}

#[recursive::recursive]
pub(crate) fn is_elementwise(
    expr_key: IRNodeKey,
    arena: &Arena<AExpr>,
    cache: &mut ExprCache,
) -> bool {
    if let Some(ret) = cache.is_elementwise.get(&expr_key) {
        return *ret;
    }

    let ret = match arena.get(expr_key) {
        AExpr::Explode(_) => false,
        AExpr::Alias(inner, _) => is_elementwise(*inner, arena, cache),
        AExpr::Column(_) => true,
        AExpr::Literal(lit) => !matches!(lit, LiteralValue::Series(_) | LiteralValue::Range { .. }),
        AExpr::BinaryExpr { left, op: _, right } => {
            is_elementwise(*left, arena, cache) && is_elementwise(*right, arena, cache)
        },
        AExpr::Cast {
            expr,
            data_type: _,
            options: _,
        } => is_elementwise(*expr, arena, cache),
        AExpr::Sort { .. } | AExpr::SortBy { .. } | AExpr::Gather { .. } => false,
        AExpr::Filter { .. } => false,
        AExpr::Agg(_) => false,
        AExpr::Ternary {
            predicate,
            truthy,
            falsy,
        } => {
            is_elementwise(*predicate, arena, cache)
                && is_elementwise(*truthy, arena, cache)
                && is_elementwise(*falsy, arena, cache)
        },
        AExpr::AnonymousFunction {
            input,
            function: _,
            output_type: _,
            options,
        }
        | AExpr::Function {
            input,
            function: _,
            options,
        } => {
            options.is_elementwise() && input.iter().all(|e| is_elementwise(e.node(), arena, cache))
        },

        AExpr::Window { .. } => false,
        AExpr::Slice { .. } => false,
        AExpr::Len => false,
    };

    cache.is_elementwise.insert(expr_key, ret);
    ret
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
        ctx.expr_arena,
        &mut ctx.cache.is_input_independent,
    )
}

fn build_input_independent_node_with_ctx(
    exprs: &[ExprIR],
    ctx: &mut LowerExprContext,
) -> PolarsResult<PhysNodeKey> {
    let expr_depth_limit = get_expr_depth_limit()?;
    let mut state = ExpressionConversionState::new(false, expr_depth_limit);
    let empty = DataFrame::empty();
    let execution_state = ExecutionState::new();
    let columns = exprs
        .iter()
        .map(|expr| {
            let phys_expr =
                create_physical_expr(expr, Context::Default, ctx.expr_arena, None, &mut state)?;

            phys_expr.evaluate(&empty, &execution_state)
        })
        .try_collect_vec()?;

    let df = Arc::new(DataFrame::new_with_broadcast(columns)?);
    Ok(ctx.phys_sm.insert(PhysNode::new(
        Arc::new(df.schema()),
        PhysNodeKind::InMemorySource { df },
    )))
}

fn simplify_input_nodes(
    orig_input: PhysNodeKey,
    mut input_nodes: PlHashSet<PhysNodeKey>,
    ctx: &mut LowerExprContext,
) -> PolarsResult<PlHashSet<PhysNodeKey>> {
    // Flatten nested zips (ensures the original input columns only occur once).
    if input_nodes.len() > 1 {
        let mut flattened_input_nodes = PlHashSet::with_capacity(input_nodes.len());
        for input_node in input_nodes {
            if let PhysNodeKind::Zip {
                inputs,
                null_extend: false,
            } = &ctx.phys_sm[input_node].kind
            {
                flattened_input_nodes.extend(inputs);
                ctx.phys_sm.remove(input_node);
            } else {
                flattened_input_nodes.insert(input_node);
            }
        }
        input_nodes = flattened_input_nodes;
    }

    // Merge reduce nodes that directly operate on the original input.
    let mut combined_exprs = vec![];
    input_nodes = input_nodes
        .into_iter()
        .filter(|input_node| {
            if let PhysNodeKind::Reduce {
                input: inner,
                exprs,
            } = &ctx.phys_sm[*input_node].kind
            {
                if *inner == orig_input {
                    combined_exprs.extend(exprs.iter().cloned());
                    ctx.phys_sm.remove(*input_node);
                    return false;
                }
            }
            true
        })
        .collect();
    if !combined_exprs.is_empty() {
        let output_schema = schema_for_select(orig_input, &combined_exprs, ctx)?;
        let kind = PhysNodeKind::Reduce {
            input: orig_input,
            exprs: combined_exprs,
        };
        let reduce_node_key = ctx.phys_sm.insert(PhysNode::new(output_schema, kind));
        input_nodes.insert(reduce_node_key);
    }

    Ok(input_nodes)
}

fn build_fallback_node_with_ctx(
    input: PhysNodeKey,
    exprs: &[ExprIR],
    ctx: &mut LowerExprContext,
) -> PolarsResult<PhysNodeKey> {
    // Pre-select only the columns that are needed for this fallback expression.
    let input_schema = &ctx.phys_sm[input].output_schema;
    let select_names: PlHashSet<_> = exprs
        .iter()
        .flat_map(|expr| polars_plan::utils::aexpr_to_leaf_names_iter(expr.node(), ctx.expr_arena))
        .collect();
    let input_node = if input_schema
        .iter_names()
        .any(|name| !select_names.contains(name.as_str()))
    {
        let select_exprs = select_names
            .into_iter()
            .map(|name| {
                ExprIR::new(
                    ctx.expr_arena.add(AExpr::Column(name.clone())),
                    OutputName::ColumnLhs(name),
                )
            })
            .collect_vec();
        build_select_node_with_ctx(input, &select_exprs, ctx)?
    } else {
        input
    };

    let output_schema = schema_for_select(input_node, exprs, ctx)?;
    let expr_depth_limit = get_expr_depth_limit()?;
    let mut conv_state = ExpressionConversionState::new(false, expr_depth_limit);
    let phys_exprs = exprs
        .iter()
        .map(|expr| {
            create_physical_expr(
                expr,
                Context::Default,
                ctx.expr_arena,
                None,
                &mut conv_state,
            )
        })
        .try_collect_vec()?;
    let map = move |df| {
        let exec_state = ExecutionState::new();
        let columns = phys_exprs
            .iter()
            .map(|phys_expr| phys_expr.evaluate(&df, &exec_state))
            .try_collect()?;
        DataFrame::new_with_broadcast(columns)
    };
    let kind = PhysNodeKind::InMemoryMap {
        input: input_node,
        map: Arc::new(map),
    };
    Ok(ctx.phys_sm.insert(PhysNode::new(output_schema, kind)))
}

// In the recursive lowering we don't bother with named expressions at all, so
// we work directly with Nodes.
#[recursive::recursive]
fn lower_exprs_with_ctx(
    input: PhysNodeKey,
    exprs: &[Node],
    ctx: &mut LowerExprContext,
) -> PolarsResult<(PhysNodeKey, Vec<Node>)> {
    // We have to catch this case separately, in case all the input independent expressions are elementwise.
    // TODO: we shouldn't always do this when recursing, e.g. pl.col.a.sum() + 1 will still hit this in the recursion.
    if exprs.iter().all(|e| is_input_independent(*e, ctx)) {
        let expr_irs = exprs
            .iter()
            .map(|e| ExprIR::new(*e, OutputName::Alias(unique_column_name())))
            .collect_vec();
        let node = build_input_independent_node_with_ctx(&expr_irs, ctx)?;
        let out_exprs = expr_irs
            .iter()
            .map(|e| ctx.expr_arena.add(AExpr::Column(e.output_name().into())))
            .collect();
        return Ok((node, out_exprs));
    }

    // Fallback expressions that can directly be applied to the original input.
    let mut fallback_subset = Vec::new();

    // Nodes containing the columns used for executing transformed expressions.
    let mut input_nodes = PlHashSet::new();

    // The final transformed expressions that will be selected from the zipped
    // together transformed nodes.
    let mut transformed_exprs = Vec::with_capacity(exprs.len());

    for expr in exprs.iter().copied() {
        if is_elementwise(expr, ctx.expr_arena, ctx.cache) {
            if !is_input_independent(expr, ctx) {
                input_nodes.insert(input);
            }
            transformed_exprs.push(expr);
            continue;
        }

        match ctx.expr_arena.get(expr).clone() {
            AExpr::Explode(inner) => {
                // While explode is streamable, it is not elementwise, so we
                // have to transform it to a select node.
                let (trans_input, trans_exprs) = lower_exprs_with_ctx(input, &[inner], ctx)?;
                let exploded_name = unique_column_name();
                let trans_inner = ctx.expr_arena.add(AExpr::Explode(trans_exprs[0]));
                let explode_expr =
                    ExprIR::new(trans_inner, OutputName::Alias(exploded_name.clone()));
                let output_schema = schema_for_select(trans_input, &[explode_expr.clone()], ctx)?;
                let node_kind = PhysNodeKind::Select {
                    input: trans_input,
                    selectors: vec![explode_expr.clone()],
                    extend_original: false,
                };
                let node_key = ctx.phys_sm.insert(PhysNode::new(output_schema, node_kind));
                input_nodes.insert(node_key);
                transformed_exprs.push(ctx.expr_arena.add(AExpr::Column(exploded_name)));
            },
            AExpr::Alias(_, _) => unreachable!("alias found in physical plan"),
            AExpr::Column(_) => unreachable!("column should always be streamable"),
            AExpr::Literal(_) => {
                let out_name = unique_column_name();
                let inner_expr = ExprIR::new(expr, OutputName::Alias(out_name.clone()));
                input_nodes.insert(build_input_independent_node_with_ctx(&[inner_expr], ctx)?);
                transformed_exprs.push(ctx.expr_arena.add(AExpr::Column(out_name)));
            },
            AExpr::BinaryExpr { left, op, right } => {
                let (trans_input, trans_exprs) = lower_exprs_with_ctx(input, &[left, right], ctx)?;
                let bin_expr = AExpr::BinaryExpr {
                    left: trans_exprs[0],
                    op,
                    right: trans_exprs[1],
                };
                input_nodes.insert(trans_input);
                transformed_exprs.push(ctx.expr_arena.add(bin_expr));
            },
            AExpr::Ternary {
                predicate,
                truthy,
                falsy,
            } => {
                let (trans_input, trans_exprs) =
                    lower_exprs_with_ctx(input, &[predicate, truthy, falsy], ctx)?;
                let tern_expr = AExpr::Ternary {
                    predicate: trans_exprs[0],
                    truthy: trans_exprs[1],
                    falsy: trans_exprs[2],
                };
                input_nodes.insert(trans_input);
                transformed_exprs.push(ctx.expr_arena.add(tern_expr));
            },
            AExpr::Cast {
                expr: inner,
                data_type,
                options,
            } => {
                let (trans_input, trans_exprs) = lower_exprs_with_ctx(input, &[inner], ctx)?;
                input_nodes.insert(trans_input);
                transformed_exprs.push(ctx.expr_arena.add(AExpr::Cast {
                    expr: trans_exprs[0],
                    data_type,
                    options,
                }));
            },
            AExpr::Sort {
                expr: inner,
                options,
            } => {
                // As we'll refer to the sorted column twice, ensure the inner
                // expr is available as a column by selecting first.
                let sorted_name = unique_column_name();
                let inner_expr_ir = ExprIR::new(inner, OutputName::Alias(sorted_name.clone()));
                let select_node = build_select_node_with_ctx(input, &[inner_expr_ir.clone()], ctx)?;
                let col_expr = ctx.expr_arena.add(AExpr::Column(sorted_name.clone()));
                let kind = PhysNodeKind::Sort {
                    input: select_node,
                    by_column: vec![ExprIR::new(col_expr, OutputName::Alias(sorted_name))],
                    slice: None,
                    sort_options: (&options).into(),
                };
                let output_schema = ctx.phys_sm[select_node].output_schema.clone();
                let node_key = ctx.phys_sm.insert(PhysNode::new(output_schema, kind));
                input_nodes.insert(node_key);
                transformed_exprs.push(col_expr);
            },
            AExpr::SortBy {
                expr: inner,
                by,
                sort_options,
            } => {
                // Select our inputs (if we don't do this we'll waste time sorting irrelevant columns).
                let sorted_name = unique_column_name();
                let by_names = by.iter().map(|_| unique_column_name()).collect_vec();
                let all_inner_expr_irs = [(&sorted_name, inner)]
                    .into_iter()
                    .chain(by_names.iter().zip(by.iter().copied()))
                    .map(|(name, inner)| ExprIR::new(inner, OutputName::Alias(name.clone())))
                    .collect_vec();
                let select_node = build_select_node_with_ctx(input, &all_inner_expr_irs, ctx)?;

                // Sort the inputs.
                let kind = PhysNodeKind::Sort {
                    input: select_node,
                    by_column: by_names
                        .into_iter()
                        .map(|name| {
                            ExprIR::new(
                                ctx.expr_arena.add(AExpr::Column(name.clone())),
                                OutputName::Alias(name),
                            )
                        })
                        .collect(),
                    slice: None,
                    sort_options,
                };
                let output_schema = ctx.phys_sm[select_node].output_schema.clone();
                let sort_node_key = ctx.phys_sm.insert(PhysNode::new(output_schema, kind));

                // Drop the by columns.
                let sorted_col_expr = ctx.expr_arena.add(AExpr::Column(sorted_name.clone()));
                let sorted_col_ir =
                    ExprIR::new(sorted_col_expr, OutputName::Alias(sorted_name.clone()));
                let post_sort_select_node =
                    build_select_node_with_ctx(sort_node_key, &[sorted_col_ir], ctx)?;
                input_nodes.insert(post_sort_select_node);
                transformed_exprs.push(sorted_col_expr);
            },
            AExpr::Gather { .. } => todo!(),
            AExpr::Filter { input: inner, by } => {
                // Select our inputs (if we don't do this we'll waste time filtering irrelevant columns).
                let out_name = unique_column_name();
                let by_name = unique_column_name();
                let inner_expr_ir = ExprIR::new(inner, OutputName::Alias(out_name.clone()));
                let by_expr_ir = ExprIR::new(by, OutputName::Alias(by_name.clone()));
                let select_node =
                    build_select_node_with_ctx(input, &[inner_expr_ir, by_expr_ir], ctx)?;

                // Add a filter node.
                let predicate = ExprIR::new(
                    ctx.expr_arena.add(AExpr::Column(by_name.clone())),
                    OutputName::Alias(by_name),
                );
                let kind = PhysNodeKind::Filter {
                    input: select_node,
                    predicate,
                };
                let output_schema = ctx.phys_sm[select_node].output_schema.clone();
                let filter_node_key = ctx.phys_sm.insert(PhysNode::new(output_schema, kind));
                input_nodes.insert(filter_node_key);
                transformed_exprs.push(ctx.expr_arena.add(AExpr::Column(out_name)));
            },
            AExpr::Agg(mut agg) => match agg {
                // Change agg mutably so we can share the codepath for all of these.
                IRAggExpr::Min {
                    input: ref mut inner,
                    ..
                }
                | IRAggExpr::Max {
                    input: ref mut inner,
                    ..
                }
                | IRAggExpr::Sum(ref mut inner)
                | IRAggExpr::Mean(ref mut inner) => {
                    let (trans_input, trans_exprs) = lower_exprs_with_ctx(input, &[*inner], ctx)?;
                    *inner = trans_exprs[0];

                    let out_name = unique_column_name();
                    let trans_agg_expr = ctx.expr_arena.add(AExpr::Agg(agg));
                    let expr_ir = ExprIR::new(trans_agg_expr, OutputName::Alias(out_name.clone()));
                    let output_schema = schema_for_select(trans_input, &[expr_ir.clone()], ctx)?;
                    let kind = PhysNodeKind::Reduce {
                        input: trans_input,
                        exprs: vec![expr_ir],
                    };
                    let reduce_node_key = ctx.phys_sm.insert(PhysNode::new(output_schema, kind));
                    input_nodes.insert(reduce_node_key);
                    transformed_exprs.push(ctx.expr_arena.add(AExpr::Column(out_name)));
                },
                IRAggExpr::Median(_)
                | IRAggExpr::NUnique(_)
                | IRAggExpr::First(_)
                | IRAggExpr::Last(_)
                | IRAggExpr::Implode(_)
                | IRAggExpr::Quantile { .. }
                | IRAggExpr::Count(_, _)
                | IRAggExpr::Std(_, _)
                | IRAggExpr::Var(_, _)
                | IRAggExpr::AggGroups(_) => {
                    let out_name = unique_column_name();
                    fallback_subset.push(ExprIR::new(expr, OutputName::Alias(out_name.clone())));
                    transformed_exprs.push(ctx.expr_arena.add(AExpr::Column(out_name)));
                },
            },
            AExpr::Len => {
                let out_name = unique_column_name();
                let expr_ir = ExprIR::new(expr, OutputName::Alias(out_name.clone()));
                let output_schema = schema_for_select(input, &[expr_ir.clone()], ctx)?;
                let kind = PhysNodeKind::Reduce {
                    input,
                    exprs: vec![expr_ir],
                };
                let reduce_node_key = ctx.phys_sm.insert(PhysNode::new(output_schema, kind));
                input_nodes.insert(reduce_node_key);
                transformed_exprs.push(ctx.expr_arena.add(AExpr::Column(out_name)));
            },
            AExpr::AnonymousFunction { .. }
            | AExpr::Function { .. }
            | AExpr::Slice { .. }
            | AExpr::Window { .. } => {
                let out_name = unique_column_name();
                fallback_subset.push(ExprIR::new(expr, OutputName::Alias(out_name.clone())));
                transformed_exprs.push(ctx.expr_arena.add(AExpr::Column(out_name)));
            },
        }
    }

    if !fallback_subset.is_empty() {
        input_nodes.insert(build_fallback_node_with_ctx(input, &fallback_subset, ctx)?);
    }

    // Simplify the input nodes (also ensures the original input only occurs
    // once in the zip).
    input_nodes = simplify_input_nodes(input, input_nodes, ctx)?;

    if input_nodes.len() == 1 {
        // No need for any multiplexing/zipping, can directly execute.
        return Ok((input_nodes.into_iter().next().unwrap(), transformed_exprs));
    }

    let zip_inputs = input_nodes.into_iter().collect_vec();
    let output_schema = zip_inputs
        .iter()
        .flat_map(|node| ctx.phys_sm[*node].output_schema.iter_fields())
        .collect();
    let zip_kind = PhysNodeKind::Zip {
        inputs: zip_inputs,
        null_extend: false,
    };
    let zip_node = ctx
        .phys_sm
        .insert(PhysNode::new(Arc::new(output_schema), zip_kind));

    Ok((zip_node, transformed_exprs))
}

/// Computes the schema that selecting the given expressions on the input node
/// would result in.
fn schema_for_select(
    input: PhysNodeKey,
    exprs: &[ExprIR],
    ctx: &mut LowerExprContext,
) -> PolarsResult<Arc<Schema>> {
    let input_schema = &ctx.phys_sm[input].output_schema;
    let output_schema: Schema = exprs
        .iter()
        .map(|e| {
            let name = e.output_name();
            let dtype = ctx.expr_arena.get(e.node()).to_dtype(
                input_schema,
                Context::Default,
                ctx.expr_arena,
            )?;
            PolarsResult::Ok(Field::new(name, dtype))
        })
        .try_collect()?;
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

    // Are we only selecting simple columns, with the same name?
    let all_simple_columns: Option<Vec<String>> = exprs
        .iter()
        .map(|e| match ctx.expr_arena.get(e.node()) {
            AExpr::Column(name) if name.as_ref() == e.output_name() => Some(name.to_string()),
            _ => None,
        })
        .collect();

    if let Some(columns) = all_simple_columns {
        let input_schema = ctx.phys_sm[input].output_schema.clone();
        if input_schema.len() == columns.len()
            && input_schema.iter_names().zip(&columns).all(|(l, r)| l == r)
        {
            // Input node already has the correct schema, just pass through.
            return Ok(input);
        }

        let output_schema = Arc::new(input_schema.select(&columns)?);
        let node_kind = PhysNodeKind::SimpleProjection { input, columns };
        return Ok(ctx.phys_sm.insert(PhysNode::new(output_schema, node_kind)));
    }

    let node_exprs = exprs.iter().map(|e| e.node()).collect_vec();
    let (transformed_input, transformed_exprs) = lower_exprs_with_ctx(input, &node_exprs, ctx)?;
    let trans_expr_irs = exprs
        .iter()
        .zip(transformed_exprs)
        .map(|(e, te)| ExprIR::new(te, OutputName::Alias(e.output_name().into())))
        .collect_vec();
    let output_schema = schema_for_select(transformed_input, &trans_expr_irs, ctx)?;
    let node_kind = PhysNodeKind::Select {
        input: transformed_input,
        selectors: trans_expr_irs,
        extend_original: false,
    };
    Ok(ctx.phys_sm.insert(PhysNode::new(output_schema, node_kind)))
}

/// Lowers an input node plus a set of expressions on that input node to an
/// equivalent (input node, set of expressions) pair, ensuring that the new set
/// of expressions can run on the streaming engine.
///
/// Ensures that if the input node is transformed it has unique column names.
pub fn lower_exprs(
    input: PhysNodeKey,
    exprs: &[ExprIR],
    expr_arena: &mut Arena<AExpr>,
    phys_sm: &mut SlotMap<PhysNodeKey, PhysNode>,
    expr_cache: &mut ExprCache,
) -> PolarsResult<(PhysNodeKey, Vec<ExprIR>)> {
    let mut ctx = LowerExprContext {
        expr_arena,
        phys_sm,
        cache: expr_cache,
    };
    let node_exprs = exprs.iter().map(|e| e.node()).collect_vec();
    let (transformed_input, transformed_exprs) =
        lower_exprs_with_ctx(input, &node_exprs, &mut ctx)?;
    let trans_expr_irs = exprs
        .iter()
        .zip(transformed_exprs)
        .map(|(e, te)| ExprIR::new(te, OutputName::Alias(e.output_name().into())))
        .collect_vec();
    Ok((transformed_input, trans_expr_irs))
}

/// Builds a selection node given an input node and the expressions to select for.
pub fn build_select_node(
    input: PhysNodeKey,
    exprs: &[ExprIR],
    expr_arena: &mut Arena<AExpr>,
    phys_sm: &mut SlotMap<PhysNodeKey, PhysNode>,
    expr_cache: &mut ExprCache,
) -> PolarsResult<PhysNodeKey> {
    let mut ctx = LowerExprContext {
        expr_arena,
        phys_sm,
        cache: expr_cache,
    };
    build_select_node_with_ctx(input, exprs, &mut ctx)
}
