use std::sync::Arc;

use polars_core::frame::DataFrame;
use polars_core::prelude::{DataType, Field, InitHashMaps, PlHashMap, PlHashSet};
use polars_core::schema::{Schema, SchemaExt};
use polars_error::PolarsResult;
use polars_expr::state::ExecutionState;
use polars_expr::{ExpressionConversionState, create_physical_expr};
use polars_ops::frame::{JoinArgs, JoinType};
use polars_plan::plans::AExpr;
use polars_plan::plans::expr_ir::{ExprIR, OutputName};
use polars_plan::prelude::*;
use polars_utils::arena::{Arena, Node};
use polars_utils::itertools::Itertools;
use polars_utils::pl_str::PlSmallStr;
use polars_utils::{unique_column_name, unitvec};
use slotmap::SlotMap;

use super::fmt::fmt_exprs;
use super::{PhysNode, PhysNodeKey, PhysNodeKind, PhysStream, StreamingLowerIRContext};
use crate::physical_plan::lower_group_by::build_group_by_stream;

type ExprNodeKey = Node;

pub(crate) struct ExprCache {
    is_elementwise: PlHashMap<Node, bool>,
    is_input_independent: PlHashMap<Node, bool>,
    is_length_preserving: PlHashMap<Node, bool>,
}

impl ExprCache {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            is_elementwise: PlHashMap::with_capacity(capacity),
            is_input_independent: PlHashMap::with_capacity(capacity),
            is_length_preserving: PlHashMap::with_capacity(capacity),
        }
    }
}

struct LowerExprContext<'a> {
    prepare_visualization: bool,
    expr_arena: &'a mut Arena<AExpr>,
    phys_sm: &'a mut SlotMap<PhysNodeKey, PhysNode>,
    cache: &'a mut ExprCache,
}

impl<'a> From<LowerExprContext<'a>> for StreamingLowerIRContext {
    fn from(value: LowerExprContext<'a>) -> Self {
        Self {
            prepare_visualization: value.prepare_visualization,
        }
    }
}
impl<'a> From<&LowerExprContext<'a>> for StreamingLowerIRContext {
    fn from(value: &LowerExprContext<'a>) -> Self {
        Self {
            prepare_visualization: value.prepare_visualization,
        }
    }
}

pub(crate) fn is_fake_elementwise_function(expr: &AExpr) -> bool {
    // The in-memory engine treats ApplyList as elementwise but this is not actually
    // the case. It doesn't cause any problems for the in-memory engine because of
    // how it does the execution but it causes errors for new-streaming.

    // Some other functions are also marked as elementwise for filter pushdown
    // but aren't actually elementwise (e.g. arguments aren't same length).
    match expr {
        AExpr::AnonymousFunction { options, .. } => {
            options.flags.contains(FunctionFlags::APPLY_LIST)
        },
        AExpr::Function {
            function, options, ..
        } => {
            if options.flags.contains(FunctionFlags::APPLY_LIST) {
                return true;
            }

            use FunctionExpr as F;
            match function {
                #[cfg(feature = "is_in")]
                F::Boolean(BooleanFunction::IsIn { .. }) => true,
                #[cfg(feature = "replace")]
                F::Replace | F::ReplaceStrict { .. } => true,
                _ => false,
            }
        },
        _ => false,
    }
}

pub(crate) fn is_elementwise_rec_cached(
    expr_key: ExprNodeKey,
    arena: &Arena<AExpr>,
    cache: &mut ExprCache,
) -> bool {
    if !cache.is_elementwise.contains_key(&expr_key) {
        cache.is_elementwise.insert(
            expr_key,
            (|| {
                let mut expr_key = expr_key;
                let mut stack = unitvec![];

                loop {
                    let ae = arena.get(expr_key);

                    if is_fake_elementwise_function(ae) {
                        return false;
                    }

                    if !polars_plan::plans::is_elementwise(&mut stack, ae, arena) {
                        return false;
                    }

                    let Some(next_key) = stack.pop() else {
                        break;
                    };

                    expr_key = next_key;
                }

                true
            })(),
        );
    }

    *cache.is_elementwise.get(&expr_key).unwrap()
}

#[recursive::recursive]
pub fn is_input_independent_rec(
    expr_key: ExprNodeKey,
    arena: &Arena<AExpr>,
    cache: &mut PlHashMap<ExprNodeKey, bool>,
) -> bool {
    if let Some(ret) = cache.get(&expr_key) {
        return *ret;
    }

    let ret = match arena.get(expr_key) {
        AExpr::Explode { expr: inner, .. }
        | AExpr::Alias(inner, _)
        | AExpr::Cast {
            expr: inner,
            dtype: _,
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

pub fn is_input_independent(
    expr_key: ExprNodeKey,
    expr_arena: &Arena<AExpr>,
    cache: &mut ExprCache,
) -> bool {
    is_input_independent_rec(expr_key, expr_arena, &mut cache.is_input_independent)
}

fn is_input_independent_ctx(expr_key: ExprNodeKey, ctx: &mut LowerExprContext) -> bool {
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
    let output_schema = compute_output_schema(&Schema::default(), exprs, ctx.expr_arena)?;
    Ok(ctx.phys_sm.insert(PhysNode::new(
        output_schema,
        PhysNodeKind::InputIndependentSelect {
            selectors: exprs.to_vec(),
        },
    )))
}

#[recursive::recursive]
pub fn is_length_preserving_rec(
    expr_key: ExprNodeKey,
    arena: &Arena<AExpr>,
    cache: &mut PlHashMap<ExprNodeKey, bool>,
) -> bool {
    if let Some(ret) = cache.get(&expr_key) {
        return *ret;
    }

    let ret = match arena.get(expr_key) {
        AExpr::Gather { .. }
        | AExpr::Explode { .. }
        | AExpr::Filter { .. }
        | AExpr::Agg(_)
        | AExpr::Slice { .. }
        | AExpr::Len
        | AExpr::Literal(_) => false,

        AExpr::Column(_) => true,

        AExpr::Alias(inner, _)
        | AExpr::Cast {
            expr: inner,
            dtype: _,
            options: _,
        }
        | AExpr::Sort {
            expr: inner,
            options: _,
        }
        | AExpr::SortBy {
            expr: inner,
            by: _,
            sort_options: _,
        } => is_length_preserving_rec(*inner, arena, cache),

        AExpr::BinaryExpr { left, op: _, right } => {
            // As long as at least one input is length-preserving the other side
            // should either broadcast or have the same length.
            is_length_preserving_rec(*left, arena, cache)
                || is_length_preserving_rec(*right, arena, cache)
        },
        AExpr::Ternary {
            predicate,
            truthy,
            falsy,
        } => {
            is_length_preserving_rec(*predicate, arena, cache)
                || is_length_preserving_rec(*truthy, arena, cache)
                || is_length_preserving_rec(*falsy, arena, cache)
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
            // FIXME: actually inspect the functions? This is overly conservative.
            options.is_length_preserving()
                && input
                    .iter()
                    .all(|expr| is_length_preserving_rec(expr.node(), arena, cache))
        },
        AExpr::Window {
            function: _, // Actually shouldn't matter for window functions.
            partition_by: _,
            order_by: _,
            options,
        } => !matches!(options, WindowType::Over(WindowMapping::Explode)),
    };

    cache.insert(expr_key, ret);
    ret
}

#[expect(dead_code)]
pub fn is_length_preserving(
    expr_key: ExprNodeKey,
    expr_arena: &Arena<AExpr>,
    cache: &mut ExprCache,
) -> bool {
    is_length_preserving_rec(expr_key, expr_arena, &mut cache.is_length_preserving)
}

fn is_length_preserving_ctx(expr_key: ExprNodeKey, ctx: &mut LowerExprContext) -> bool {
    is_length_preserving_rec(
        expr_key,
        ctx.expr_arena,
        &mut ctx.cache.is_length_preserving,
    )
}

fn build_fallback_node_with_ctx(
    input: PhysStream,
    exprs: &[ExprIR],
    ctx: &mut LowerExprContext,
) -> PolarsResult<PhysNodeKey> {
    // Pre-select only the columns that are needed for this fallback expression.
    let input_schema = &ctx.phys_sm[input.node].output_schema;
    let mut select_names: PlHashSet<_> = exprs
        .iter()
        .flat_map(|expr| polars_plan::utils::aexpr_to_leaf_names_iter(expr.node(), ctx.expr_arena))
        .collect();
    // To keep the length correct we have to ensure we select at least one
    // column.
    if select_names.is_empty() {
        if let Some(name) = input_schema.iter_names().next() {
            select_names.insert(name.clone());
        }
    }
    let input_stream = if input_schema
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
        build_select_stream_with_ctx(input, &select_exprs, ctx)?
    } else {
        input
    };

    let output_schema = schema_for_select(input_stream, exprs, ctx)?;
    let mut conv_state = ExpressionConversionState::new(false);
    let phys_exprs = exprs
        .iter()
        .map(|expr| {
            create_physical_expr(
                expr,
                Context::Default,
                ctx.expr_arena,
                &ctx.phys_sm[input_stream.node].output_schema,
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

    let format_str = ctx.prepare_visualization.then(|| {
        let mut buffer = String::new();
        buffer.push_str("SELECT [\n");
        fmt_exprs(
            &mut buffer,
            exprs,
            ctx.expr_arena,
            super::fmt::FormatExprStyle::Select,
        );
        buffer.push(']');
        buffer
    });
    let kind = PhysNodeKind::InMemoryMap {
        input: input_stream,
        map: Arc::new(map),
        format_str,
    };
    Ok(ctx.phys_sm.insert(PhysNode::new(output_schema, kind)))
}

fn simplify_input_streams(
    orig_input: PhysStream,
    mut input_streams: PlHashSet<PhysStream>,
    ctx: &mut LowerExprContext,
) -> PolarsResult<PlHashSet<PhysStream>> {
    // Flatten nested zips (ensures the original input columns only occur once).
    if input_streams.len() > 1 {
        let mut flattened_input_streams = PlHashSet::with_capacity(input_streams.len());
        for input_stream in input_streams {
            if let PhysNodeKind::Zip {
                inputs,
                null_extend: false,
            } = &ctx.phys_sm[input_stream.node].kind
            {
                flattened_input_streams.extend(inputs);
                ctx.phys_sm.remove(input_stream.node);
            } else {
                flattened_input_streams.insert(input_stream);
            }
        }
        input_streams = flattened_input_streams;
    }

    // Merge reduce nodes that directly operate on the original input.
    let mut combined_exprs = vec![];
    input_streams = input_streams
        .into_iter()
        .filter(|input_stream| {
            if let PhysNodeKind::Reduce {
                input: inner,
                exprs,
            } = &ctx.phys_sm[input_stream.node].kind
            {
                if *inner == orig_input {
                    combined_exprs.extend(exprs.iter().cloned());
                    ctx.phys_sm.remove(input_stream.node);
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
        input_streams.insert(PhysStream::first(reduce_node_key));
    }

    Ok(input_streams)
}

// In the recursive lowering we don't bother with named expressions at all, so
// we work directly with Nodes.
#[recursive::recursive]
fn lower_exprs_with_ctx(
    input: PhysStream,
    exprs: &[Node],
    ctx: &mut LowerExprContext,
) -> PolarsResult<(PhysStream, Vec<Node>)> {
    // We have to catch this case separately, in case all the input independent expressions are elementwise.
    // TODO: we shouldn't always do this when recursing, e.g. pl.col.a.sum() + 1 will still hit this in the recursion.
    if exprs.iter().all(|e| is_input_independent_ctx(*e, ctx)) {
        let expr_irs = exprs
            .iter()
            .map(|e| ExprIR::new(*e, OutputName::Alias(unique_column_name())))
            .collect_vec();
        let node = build_input_independent_node_with_ctx(&expr_irs, ctx)?;
        let out_exprs = expr_irs
            .iter()
            .map(|e| ctx.expr_arena.add(AExpr::Column(e.output_name().clone())))
            .collect();
        return Ok((PhysStream::first(node), out_exprs));
    }

    // Fallback expressions that can directly be applied to the original input.
    let mut fallback_subset = Vec::new();

    // Streams containing the columns used for executing transformed expressions.
    let mut input_streams = PlHashSet::new();

    // The final transformed expressions that will be selected from the zipped
    // together transformed nodes.
    let mut transformed_exprs = Vec::with_capacity(exprs.len());

    for expr in exprs.iter().copied() {
        if is_elementwise_rec_cached(expr, ctx.expr_arena, ctx.cache) {
            if !is_input_independent_ctx(expr, ctx) {
                input_streams.insert(input);
            }
            transformed_exprs.push(expr);
            continue;
        }

        match ctx.expr_arena.get(expr).clone() {
            AExpr::Explode {
                expr: inner,
                skip_empty,
            } => {
                // While explode is streamable, it is not elementwise, so we
                // have to transform it to a select node.
                let (trans_input, trans_exprs) = lower_exprs_with_ctx(input, &[inner], ctx)?;
                let exploded_name = unique_column_name();
                let trans_inner = ctx.expr_arena.add(AExpr::Explode {
                    expr: trans_exprs[0],
                    skip_empty,
                });
                let explode_expr =
                    ExprIR::new(trans_inner, OutputName::Alias(exploded_name.clone()));
                let output_schema = schema_for_select(trans_input, &[explode_expr.clone()], ctx)?;
                let node_kind = PhysNodeKind::Select {
                    input: trans_input,
                    selectors: vec![explode_expr.clone()],
                    extend_original: false,
                };
                let node_key = ctx.phys_sm.insert(PhysNode::new(output_schema, node_kind));
                input_streams.insert(PhysStream::first(node_key));
                transformed_exprs.push(ctx.expr_arena.add(AExpr::Column(exploded_name)));
            },
            AExpr::Alias(_, _) => unreachable!("alias found in physical plan"),
            AExpr::Column(_) => unreachable!("column should always be streamable"),
            AExpr::Literal(_) => {
                let out_name = unique_column_name();
                let inner_expr = ExprIR::new(expr, OutputName::Alias(out_name.clone()));
                let node_key = build_input_independent_node_with_ctx(&[inner_expr], ctx)?;
                input_streams.insert(PhysStream::first(node_key));
                transformed_exprs.push(ctx.expr_arena.add(AExpr::Column(out_name)));
            },

            AExpr::Function {
                input: ref inner_exprs,
                function: FunctionExpr::ConcatExpr(_rechunk),
                options: _,
            } => {
                // We have to lower each expression separately as they might have different lengths.
                let mut concat_streams = Vec::new();
                let out_name = unique_column_name();
                for inner_expr in inner_exprs {
                    let (trans_input, trans_expr) =
                        lower_exprs_with_ctx(input, &[inner_expr.node()], ctx)?;
                    let select_expr =
                        ExprIR::new(trans_expr[0], OutputName::Alias(out_name.clone()));
                    concat_streams.push(build_select_stream_with_ctx(
                        trans_input,
                        &[select_expr],
                        ctx,
                    )?);
                }

                let output_schema = ctx.phys_sm[concat_streams[0].node].output_schema.clone();
                let node_kind = PhysNodeKind::OrderedUnion {
                    inputs: concat_streams,
                };
                let node_key = ctx.phys_sm.insert(PhysNode::new(output_schema, node_kind));
                input_streams.insert(PhysStream::first(node_key));
                transformed_exprs.push(ctx.expr_arena.add(AExpr::Column(out_name)));
            },

            AExpr::Function {
                input: ref inner_exprs,
                function: FunctionExpr::Unique(maintain_order),
                options: _,
            } => {
                assert!(inner_exprs.len() == 1);
                // Lower to no-aggregate group-by with unique name.
                let tmp_name = unique_column_name();
                let (trans_input, trans_inner_exprs) =
                    lower_exprs_with_ctx(input, &[inner_exprs[0].node()], ctx)?;
                let group_by_key_expr =
                    ExprIR::new(trans_inner_exprs[0], OutputName::Alias(tmp_name.clone()));
                let group_by_output_schema =
                    schema_for_select(trans_input, &[group_by_key_expr.clone()], ctx)?;
                let group_by_stream = build_group_by_stream(
                    trans_input,
                    &[group_by_key_expr],
                    &[],
                    group_by_output_schema,
                    maintain_order,
                    Arc::new(GroupbyOptions::default()),
                    None,
                    ctx.expr_arena,
                    ctx.phys_sm,
                    ctx.cache,
                    StreamingLowerIRContext::from(&*ctx),
                )?;
                input_streams.insert(group_by_stream);
                transformed_exprs.push(ctx.expr_arena.add(AExpr::Column(tmp_name)));
            },

            #[cfg(feature = "is_in")]
            AExpr::Function {
                input: ref inner_exprs,
                function: FunctionExpr::Boolean(BooleanFunction::IsIn { nulls_equal }),
                options: _,
            } if is_scalar_ae(inner_exprs[1].node(), ctx.expr_arena) => {
                // Translate left and right side separately (they could have different lengths).
                let left_on_name = unique_column_name();
                let right_on_name = unique_column_name();
                let (trans_input_left, trans_expr_left) =
                    lower_exprs_with_ctx(input, &[inner_exprs[0].node()], ctx)?;
                let right_expr_exploded_node = match ctx.expr_arena.get(inner_exprs[1].node()) {
                    // expr.implode().explode() ~= expr (and avoids rechunking)
                    AExpr::Agg(IRAggExpr::Implode(n)) => *n,
                    _ => ctx.expr_arena.add(AExpr::Explode {
                        expr: inner_exprs[1].node(),
                        skip_empty: true,
                    }),
                };
                let (trans_input_right, trans_expr_right) =
                    lower_exprs_with_ctx(input, &[right_expr_exploded_node], ctx)?;

                // We have to ensure the left input has the right name for the semi-anti-join to
                // generate the correct output name.
                let left_col_expr = ctx.expr_arena.add(AExpr::Column(left_on_name.clone()));
                let left_select_stream = build_select_stream_with_ctx(
                    trans_input_left,
                    &[ExprIR::new(
                        trans_expr_left[0],
                        OutputName::Alias(left_on_name.clone()),
                    )],
                    ctx,
                )?;

                let node_kind = PhysNodeKind::SemiAntiJoin {
                    input_left: left_select_stream,
                    input_right: trans_input_right,
                    left_on: vec![ExprIR::new(
                        left_col_expr,
                        OutputName::Alias(left_on_name.clone()),
                    )],
                    right_on: vec![ExprIR::new(
                        trans_expr_right[0],
                        OutputName::Alias(right_on_name),
                    )],
                    args: JoinArgs {
                        how: JoinType::Semi,
                        validation: Default::default(),
                        suffix: None,
                        slice: None,
                        nulls_equal,
                        coalesce: Default::default(),
                        maintain_order: Default::default(),
                    },
                    output_bool: true,
                };

                // SemiAntiJoin with output_bool returns a column with the same name as the first
                // input column.
                let output_schema = Schema::from_iter([(left_on_name.clone(), DataType::Boolean)]);
                let node_key = ctx
                    .phys_sm
                    .insert(PhysNode::new(Arc::new(output_schema), node_kind));
                input_streams.insert(PhysStream::first(node_key));
                transformed_exprs.push(left_col_expr);
            },

            ref node @ AExpr::Function {
                input: ref inner_exprs,
                options,
                ..
            }
            | ref node @ AExpr::AnonymousFunction {
                input: ref inner_exprs,
                options,
                ..
            } if options.is_elementwise() && !is_fake_elementwise_function(node) => {
                let inner_nodes = inner_exprs.iter().map(|expr| expr.node()).collect_vec();
                let (trans_input, trans_exprs) = lower_exprs_with_ctx(input, &inner_nodes, ctx)?;

                // The function may be sensitive to names (e.g. pl.struct), so we restore them.
                let new_input = trans_exprs
                    .into_iter()
                    .zip(inner_exprs)
                    .map(|(trans, orig)| {
                        ExprIR::new(trans, OutputName::Alias(orig.output_name().clone()))
                    })
                    .collect_vec();
                let mut new_node = node.clone();
                match &mut new_node {
                    AExpr::Function { input, .. } | AExpr::AnonymousFunction { input, .. } => {
                        *input = new_input;
                    },
                    _ => unreachable!(),
                }
                input_streams.insert(trans_input);
                transformed_exprs.push(ctx.expr_arena.add(new_node));
            },

            AExpr::BinaryExpr { left, op, right } => {
                let (trans_input, trans_exprs) = lower_exprs_with_ctx(input, &[left, right], ctx)?;
                let bin_expr = AExpr::BinaryExpr {
                    left: trans_exprs[0],
                    op,
                    right: trans_exprs[1],
                };
                input_streams.insert(trans_input);
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
                input_streams.insert(trans_input);
                transformed_exprs.push(ctx.expr_arena.add(tern_expr));
            },
            AExpr::Cast {
                expr: inner,
                dtype,
                options,
            } => {
                let (trans_input, trans_exprs) = lower_exprs_with_ctx(input, &[inner], ctx)?;
                input_streams.insert(trans_input);
                transformed_exprs.push(ctx.expr_arena.add(AExpr::Cast {
                    expr: trans_exprs[0],
                    dtype,
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
                let select_stream =
                    build_select_stream_with_ctx(input, &[inner_expr_ir.clone()], ctx)?;
                let col_expr = ctx.expr_arena.add(AExpr::Column(sorted_name.clone()));
                let kind = PhysNodeKind::Sort {
                    input: select_stream,
                    by_column: vec![ExprIR::new(col_expr, OutputName::Alias(sorted_name))],
                    slice: None,
                    sort_options: (&options).into(),
                };
                let output_schema = ctx.phys_sm[select_stream.node].output_schema.clone();
                let node_key = ctx.phys_sm.insert(PhysNode::new(output_schema, kind));
                input_streams.insert(PhysStream::first(node_key));
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
                let select_stream = build_select_stream_with_ctx(input, &all_inner_expr_irs, ctx)?;

                // Sort the inputs.
                let kind = PhysNodeKind::Sort {
                    input: select_stream,
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
                let output_schema = ctx.phys_sm[select_stream.node].output_schema.clone();
                let sort_node_key = ctx.phys_sm.insert(PhysNode::new(output_schema, kind));
                let sort_stream = PhysStream::first(sort_node_key);

                // Drop the by columns.
                let sorted_col_expr = ctx.expr_arena.add(AExpr::Column(sorted_name.clone()));
                let sorted_col_ir =
                    ExprIR::new(sorted_col_expr, OutputName::Alias(sorted_name.clone()));
                let post_sort_select_stream =
                    build_select_stream_with_ctx(sort_stream, &[sorted_col_ir], ctx)?;
                input_streams.insert(post_sort_select_stream);
                transformed_exprs.push(sorted_col_expr);
            },
            AExpr::Filter { input: inner, by } => {
                // Select our inputs (if we don't do this we'll waste time filtering irrelevant columns).
                let out_name = unique_column_name();
                let by_name = unique_column_name();
                let inner_expr_ir = ExprIR::new(inner, OutputName::Alias(out_name.clone()));
                let by_expr_ir = ExprIR::new(by, OutputName::Alias(by_name.clone()));
                let select_stream =
                    build_select_stream_with_ctx(input, &[inner_expr_ir, by_expr_ir], ctx)?;

                // Add a filter node.
                let predicate = ExprIR::new(
                    ctx.expr_arena.add(AExpr::Column(by_name.clone())),
                    OutputName::Alias(by_name),
                );
                let kind = PhysNodeKind::Filter {
                    input: select_stream,
                    predicate,
                };
                let output_schema = ctx.phys_sm[select_stream.node].output_schema.clone();
                let filter_node_key = ctx.phys_sm.insert(PhysNode::new(output_schema, kind));
                input_streams.insert(PhysStream::first(filter_node_key));
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
                | IRAggExpr::First(ref mut inner)
                | IRAggExpr::Last(ref mut inner)
                | IRAggExpr::Sum(ref mut inner)
                | IRAggExpr::Mean(ref mut inner)
                | IRAggExpr::Var(ref mut inner, _ /* ddof */)
                | IRAggExpr::Std(ref mut inner, _ /* ddof */)
                | IRAggExpr::Count(ref mut inner, _ /* count_nulls */) => {
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
                    input_streams.insert(PhysStream::first(reduce_node_key));
                    transformed_exprs.push(ctx.expr_arena.add(AExpr::Column(out_name)));
                },
                IRAggExpr::NUnique(inner) => {
                    // Lower to no-aggregate group-by with unique name feeding into len aggregate.
                    let tmp_name = unique_column_name();
                    let (trans_input, trans_inner_exprs) =
                        lower_exprs_with_ctx(input, &[inner], ctx)?;
                    let group_by_key_expr =
                        ExprIR::new(trans_inner_exprs[0], OutputName::Alias(tmp_name.clone()));
                    let group_by_output_schema =
                        schema_for_select(trans_input, &[group_by_key_expr.clone()], ctx)?;
                    let group_by_stream = build_group_by_stream(
                        trans_input,
                        &[group_by_key_expr],
                        &[],
                        group_by_output_schema,
                        false,
                        Arc::new(GroupbyOptions::default()),
                        None,
                        ctx.expr_arena,
                        ctx.phys_sm,
                        ctx.cache,
                        StreamingLowerIRContext::from(&*ctx),
                    )?;

                    let len_node = ctx.expr_arena.add(AExpr::Len);
                    let len_expr_ir = ExprIR::new(len_node, OutputName::Alias(tmp_name.clone()));
                    let output_schema =
                        schema_for_select(group_by_stream, &[len_expr_ir.clone()], ctx)?;
                    let kind = PhysNodeKind::Reduce {
                        input: group_by_stream,
                        exprs: vec![len_expr_ir],
                    };

                    let reduce_node_key = ctx.phys_sm.insert(PhysNode::new(output_schema, kind));
                    input_streams.insert(PhysStream::first(reduce_node_key));
                    transformed_exprs.push(ctx.expr_arena.add(AExpr::Column(tmp_name)));
                },
                IRAggExpr::Median(_)
                | IRAggExpr::Implode(_)
                | IRAggExpr::Quantile { .. }
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
                input_streams.insert(PhysStream::first(reduce_node_key));
                transformed_exprs.push(ctx.expr_arena.add(AExpr::Column(out_name)));
            },
            AExpr::AnonymousFunction { .. }
            | AExpr::Function { .. }
            | AExpr::Slice { .. }
            | AExpr::Window { .. }
            | AExpr::Gather { .. } => {
                let out_name = unique_column_name();
                fallback_subset.push(ExprIR::new(expr, OutputName::Alias(out_name.clone())));
                transformed_exprs.push(ctx.expr_arena.add(AExpr::Column(out_name)));
            },
        }
    }

    if !fallback_subset.is_empty() {
        let fallback_node = build_fallback_node_with_ctx(input, &fallback_subset, ctx)?;
        input_streams.insert(PhysStream::first(fallback_node));
    }

    // Simplify the input nodes (also ensures the original input only occurs
    // once in the zip).
    input_streams = simplify_input_streams(input, input_streams, ctx)?;

    if input_streams.len() == 1 {
        // No need for any multiplexing/zipping, can directly execute.
        return Ok((input_streams.into_iter().next().unwrap(), transformed_exprs));
    }

    let zip_inputs = input_streams.into_iter().collect_vec();
    let output_schema = zip_inputs
        .iter()
        .flat_map(|stream| ctx.phys_sm[stream.node].output_schema.iter_fields())
        .collect();
    let zip_kind = PhysNodeKind::Zip {
        inputs: zip_inputs,
        null_extend: false,
    };
    let zip_node = ctx
        .phys_sm
        .insert(PhysNode::new(Arc::new(output_schema), zip_kind));

    Ok((PhysStream::first(zip_node), transformed_exprs))
}

/// Computes the schema that selecting the given expressions on the input schema
/// would result in.
pub fn compute_output_schema(
    input_schema: &Schema,
    exprs: &[ExprIR],
    expr_arena: &Arena<AExpr>,
) -> PolarsResult<Arc<Schema>> {
    let output_schema: Schema = exprs
        .iter()
        .map(|e| {
            let name = e.output_name().clone();
            let dtype = e
                .dtype(input_schema, Context::Default, expr_arena)?
                .clone()
                .materialize_unknown(true)
                .unwrap();
            PolarsResult::Ok(Field::new(name, dtype))
        })
        .try_collect()?;
    Ok(Arc::new(output_schema))
}

/// Computes the schema that selecting the given expressions on the input node
/// would result in.
fn schema_for_select(
    input: PhysStream,
    exprs: &[ExprIR],
    ctx: &mut LowerExprContext,
) -> PolarsResult<Arc<Schema>> {
    let input_schema = &ctx.phys_sm[input.node].output_schema;
    compute_output_schema(input_schema, exprs, ctx.expr_arena)
}

fn build_select_stream_with_ctx(
    input: PhysStream,
    exprs: &[ExprIR],
    ctx: &mut LowerExprContext,
) -> PolarsResult<PhysStream> {
    if exprs
        .iter()
        .all(|e| is_input_independent_ctx(e.node(), ctx))
    {
        return Ok(PhysStream::first(build_input_independent_node_with_ctx(
            exprs, ctx,
        )?));
    }

    // Are we only selecting simple columns, with the same name?
    let all_simple_columns: Option<Vec<PlSmallStr>> = exprs
        .iter()
        .map(|e| match ctx.expr_arena.get(e.node()) {
            AExpr::Column(name) if name == e.output_name() => Some(name.clone()),
            _ => None,
        })
        .collect();

    if let Some(columns) = all_simple_columns {
        let input_schema = ctx.phys_sm[input.node].output_schema.clone();
        if input_schema.len() == columns.len()
            && input_schema.iter_names().zip(&columns).all(|(l, r)| l == r)
        {
            // Input node already has the correct schema, just pass through.
            return Ok(input);
        }

        let output_schema = Arc::new(input_schema.try_project(&columns)?);
        let node_kind = PhysNodeKind::SimpleProjection { input, columns };
        let node_key = ctx.phys_sm.insert(PhysNode::new(output_schema, node_kind));
        return Ok(PhysStream::first(node_key));
    }

    // Actual lowering is needed.
    let node_exprs = exprs.iter().map(|e| e.node()).collect_vec();
    let (transformed_input, transformed_exprs) = lower_exprs_with_ctx(input, &node_exprs, ctx)?;
    let trans_expr_irs = exprs
        .iter()
        .zip(transformed_exprs)
        .map(|(e, te)| ExprIR::new(te, OutputName::Alias(e.output_name().clone())))
        .collect_vec();
    let output_schema = schema_for_select(transformed_input, &trans_expr_irs, ctx)?;
    let node_kind = PhysNodeKind::Select {
        input: transformed_input,
        selectors: trans_expr_irs,
        extend_original: false,
    };
    let node_key = ctx.phys_sm.insert(PhysNode::new(output_schema, node_kind));
    Ok(PhysStream::first(node_key))
}

/// Lowers an input node plus a set of expressions on that input node to an
/// equivalent (input node, set of expressions) pair, ensuring that the new set
/// of expressions can run on the streaming engine.
///
/// Ensures that if the input node is transformed it has unique column names.
pub fn lower_exprs(
    input: PhysStream,
    exprs: &[ExprIR],
    expr_arena: &mut Arena<AExpr>,
    phys_sm: &mut SlotMap<PhysNodeKey, PhysNode>,
    expr_cache: &mut ExprCache,
    ctx: StreamingLowerIRContext,
) -> PolarsResult<(PhysStream, Vec<ExprIR>)> {
    let mut ctx = LowerExprContext {
        expr_arena,
        phys_sm,
        cache: expr_cache,
        prepare_visualization: ctx.prepare_visualization,
    };
    let node_exprs = exprs.iter().map(|e| e.node()).collect_vec();
    let (transformed_input, transformed_exprs) =
        lower_exprs_with_ctx(input, &node_exprs, &mut ctx)?;
    let trans_expr_irs = exprs
        .iter()
        .zip(transformed_exprs)
        .map(|(e, te)| ExprIR::new(te, OutputName::Alias(e.output_name().clone())))
        .collect_vec();
    Ok((transformed_input, trans_expr_irs))
}

/// Builds a new selection node given an input stream and the expressions to
/// select for, if needed.
pub fn build_select_stream(
    input: PhysStream,
    exprs: &[ExprIR],
    expr_arena: &mut Arena<AExpr>,
    phys_sm: &mut SlotMap<PhysNodeKey, PhysNode>,
    expr_cache: &mut ExprCache,
    ctx: StreamingLowerIRContext,
) -> PolarsResult<PhysStream> {
    let mut ctx = LowerExprContext {
        expr_arena,
        phys_sm,
        cache: expr_cache,
        prepare_visualization: ctx.prepare_visualization,
    };
    build_select_stream_with_ctx(input, exprs, &mut ctx)
}

/// Builds a new selection node given an input stream and the expressions to
/// select for, if needed. Preserves the length of the input, like in with_columns.
pub fn build_length_preserving_select_stream(
    input: PhysStream,
    exprs: &[ExprIR],
    expr_arena: &mut Arena<AExpr>,
    phys_sm: &mut SlotMap<PhysNodeKey, PhysNode>,
    expr_cache: &mut ExprCache,
    ctx: StreamingLowerIRContext,
) -> PolarsResult<PhysStream> {
    let mut ctx = LowerExprContext {
        expr_arena,
        phys_sm,
        cache: expr_cache,
        prepare_visualization: ctx.prepare_visualization,
    };
    let already_length_preserving = exprs
        .iter()
        .any(|expr| is_length_preserving_ctx(expr.node(), &mut ctx));
    let input_schema = &ctx.phys_sm[input.node].output_schema;
    if exprs.is_empty() || input_schema.is_empty() || already_length_preserving {
        return build_select_stream_with_ctx(input, exprs, &mut ctx);
    }

    // Hacky work-around: append an input column with a temporary name, but
    // remove it from the final selector. This should ensure scalars gets zipped
    // back to the input to broadcast them.
    let tmp_name = unique_column_name();
    let first_col = ctx.expr_arena.add(AExpr::Column(
        input_schema.iter_names_cloned().next().unwrap(),
    ));
    let mut tmp_exprs = Vec::with_capacity(exprs.len() + 1);
    tmp_exprs.extend(exprs.iter().cloned());
    tmp_exprs.push(ExprIR::new(first_col, OutputName::Alias(tmp_name.clone())));

    let out_stream = build_select_stream_with_ctx(input, &tmp_exprs, &mut ctx)?;
    let PhysNodeKind::Select { selectors, .. } = &mut ctx.phys_sm[out_stream.node].kind else {
        unreachable!()
    };
    assert!(selectors.pop().unwrap().output_name() == &tmp_name);
    let out_schema = Arc::make_mut(&mut phys_sm[out_stream.node].output_schema);
    out_schema.shift_remove(tmp_name.as_ref()).unwrap();
    Ok(out_stream)
}
