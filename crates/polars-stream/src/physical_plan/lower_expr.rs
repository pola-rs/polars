use std::sync::Arc;

use polars_core::chunked_array::cast::CastOptions;
use polars_core::frame::DataFrame;
use polars_core::prelude::{
    DataType, Field, IDX_DTYPE, InitHashMaps, PlHashMap, PlHashSet, PlIndexMap, PlIndexSet,
};
use polars_core::scalar::Scalar;
use polars_core::schema::{Schema, SchemaExt};
use polars_error::PolarsResult;
use polars_expr::state::ExecutionState;
use polars_expr::{ExpressionConversionState, create_physical_expr};
use polars_ops::frame::{JoinArgs, JoinType};
use polars_ops::series::{RLE_LENGTH_COLUMN_NAME, RLE_VALUE_COLUMN_NAME};
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
use crate::physical_plan::ZipBehavior;
use crate::physical_plan::lower_group_by::build_group_by_stream;
use crate::physical_plan::lower_ir::{build_filter_stream, build_row_idx_stream};

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
        AExpr::Function { function, .. } => {
            use IRFunctionExpr as F;
            match function {
                #[cfg(feature = "is_in")]
                F::Boolean(IRBooleanFunction::IsIn { .. }) => true,
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
        // Handled separately in `Eval`.
        AExpr::Element => unreachable!(),
        AExpr::StructField(_) => false,
        AExpr::Explode { expr: inner, .. }
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
            null_on_oob: _,
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
            options: _,
            fmt_str: _,
        }
        | AExpr::AnonymousAgg {
            input,
            function: _,
            fmt_str: _,
        }
        | AExpr::Function {
            input,
            function: _,
            options: _,
        } => input
            .iter()
            .all(|expr| is_input_independent_rec(expr.node(), arena, cache)),
        AExpr::Eval {
            expr,
            evaluation: _,
            variant: _,
        } => is_input_independent_rec(*expr, arena, cache),
        AExpr::StructEval { expr, evaluation } => {
            is_input_independent_rec(*expr, arena, cache)
                && evaluation
                    .iter()
                    .all(|expr| is_input_independent_rec(expr.node(), arena, cache))
        },
        #[cfg(feature = "dynamic_group_by")]
        AExpr::Rolling {
            function,
            index_column,
            period: _,
            offset: _,
            closed_window: _,
        } => {
            is_input_independent_rec(*function, arena, cache)
                && is_input_independent_rec(*index_column, arena, cache)
        },
        AExpr::Over {
            function,
            partition_by,
            order_by,
            mapping: _,
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
        // Handled separately in `Eval`.
        AExpr::Element => unreachable!(),
        // Mapped to `Column` in `StructEval`.
        AExpr::StructField(_) => unreachable!(),

        AExpr::Gather { .. }
        | AExpr::Explode { .. }
        | AExpr::Filter { .. }
        | AExpr::Agg(_)
        | AExpr::Slice { .. }
        | AExpr::Len
        | AExpr::Literal(_) => false,

        AExpr::Column(_) => true,

        AExpr::Cast {
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
        AExpr::AnonymousAgg { .. } => false,
        AExpr::AnonymousFunction {
            input,
            function: _,
            options,
            fmt_str: _,
        }
        | AExpr::Function {
            input,
            function: _,
            options,
        } => {
            // TODO: actually inspect the functions? This is overly conservative.
            options.is_length_preserving()
                && input
                    .iter()
                    .all(|expr| is_length_preserving_rec(expr.node(), arena, cache))
        },
        AExpr::Eval {
            expr,
            evaluation: _,
            variant: _,
        } => is_length_preserving_rec(*expr, arena, cache),
        #[cfg(feature = "dynamic_group_by")]
        AExpr::Rolling {
            function: _,
            index_column: _,
            period: _,
            offset: _,
            closed_window: _,
        } => true,
        AExpr::StructEval {
            expr,
            evaluation: _,
        } => is_length_preserving_rec(*expr, arena, cache),
        AExpr::Over {
            function: _, // Actually shouldn't matter for window functions.
            partition_by: _,
            order_by: _,
            mapping,
        } => !matches!(mapping, WindowMapping::Explode),
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
        .flat_map(|expr| {
            polars_plan::utils::aexpr_to_leaf_names_iter(expr.node(), ctx.expr_arena).cloned()
        })
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
        .any(|name| !select_names.contains(name))
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

        DataFrame::new_infer_broadcast(columns)
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
    mut input_streams: PlIndexSet<PhysStream>,
    ctx: &mut LowerExprContext,
) -> PolarsResult<PlIndexSet<PhysStream>> {
    // Flatten nested zips (ensures the original input columns only occur once).
    if input_streams.len() > 1 {
        let mut flattened_input_streams = PlIndexSet::with_capacity(input_streams.len());
        for input_stream in input_streams {
            if let PhysNodeKind::Zip {
                inputs,
                zip_behavior: ZipBehavior::Broadcast,
            } = &ctx.phys_sm[input_stream.node].kind
            {
                flattened_input_streams.extend(inputs);
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

// Assuming that agg_node is a reduction, lowers its input recursively and
// returns a Reduce node as well a node corresponding to the column to select
// from the Reduce node for the aggregate.
fn lower_reduce_node(
    input: PhysStream,
    agg_node: Node,
    ctx: &mut LowerExprContext,
) -> PolarsResult<(PhysStream, Node)> {
    let agg_aexpr = ctx.expr_arena.get(agg_node).clone();
    let mut agg_input = Vec::with_capacity(1);
    agg_aexpr.inputs_rev(&mut agg_input);
    agg_input.reverse();

    let (trans_input, trans_exprs) = lower_exprs_with_ctx(input, &agg_input, ctx)?;
    let trans_agg_node = ctx.expr_arena.add(agg_aexpr.replace_inputs(&trans_exprs));

    let out_name = unique_column_name();
    let expr_ir = ExprIR::new(trans_agg_node, OutputName::Alias(out_name.clone()));
    let output_schema = schema_for_select(trans_input, std::slice::from_ref(&expr_ir), ctx)?;
    let kind = PhysNodeKind::Reduce {
        input: trans_input,
        exprs: vec![expr_ir],
    };

    let reduce_node_key = ctx.phys_sm.insert(PhysNode::new(output_schema, kind));
    let reduce_stream = PhysStream::first(reduce_node_key);
    let out_node = ctx.expr_arena.add(AExpr::Column(out_name));
    Ok((reduce_stream, out_node))
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
    let mut input_streams = PlIndexSet::new();

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
            // Handled separately in `Eval` expressions.
            AExpr::Element => unreachable!(),
            // Mapped to `Column` in `StructEval`.
            AExpr::StructField(_) => unreachable!(),

            AExpr::Explode {
                expr: inner,
                options,
            } => {
                // While explode is streamable, it is not elementwise, so we
                // have to transform it to a select node.
                let (trans_input, trans_exprs) = lower_exprs_with_ctx(input, &[inner], ctx)?;
                let exploded_name = unique_column_name();
                let trans_inner = ctx.expr_arena.add(AExpr::Explode {
                    expr: trans_exprs[0],
                    options,
                });
                let explode_expr =
                    ExprIR::new(trans_inner, OutputName::Alias(exploded_name.clone()));
                let output_schema =
                    schema_for_select(trans_input, std::slice::from_ref(&explode_expr), ctx)?;
                let node_kind = PhysNodeKind::Select {
                    input: trans_input,
                    selectors: vec![explode_expr.clone()],
                    extend_original: false,
                };
                let node_key = ctx.phys_sm.insert(PhysNode::new(output_schema, node_kind));
                input_streams.insert(PhysStream::first(node_key));
                transformed_exprs.push(ctx.expr_arena.add(AExpr::Column(exploded_name)));
            },
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
                function: IRFunctionExpr::Repeat,
                options: _,
            } => {
                assert!(inner_exprs.len() == 2);
                let out_name = unique_column_name();
                let value_expr_ir = inner_exprs[0].with_alias(out_name.clone());
                let repeats_expr_ir = inner_exprs[1].clone();
                let value_stream = build_select_stream_with_ctx(input, &[value_expr_ir], ctx)?;
                let repeats_stream = build_select_stream_with_ctx(input, &[repeats_expr_ir], ctx)?;

                let output_schema = ctx.phys_sm[value_stream.node].output_schema.clone();
                let kind = PhysNodeKind::Repeat {
                    value: value_stream,
                    repeats: repeats_stream,
                };
                let repeat_node_key = ctx.phys_sm.insert(PhysNode::new(output_schema, kind));
                input_streams.insert(PhysStream::first(repeat_node_key));
                transformed_exprs.push(ctx.expr_arena.add(AExpr::Column(out_name)));
            },

            AExpr::Function {
                input: ref inner_exprs,
                function: IRFunctionExpr::ExtendConstant,
                options: _,
            } => {
                assert!(inner_exprs.len() == 3);
                let input_schema = &ctx.phys_sm[input.node].output_schema;
                let out_name = unique_column_name();
                let first_ir = inner_exprs[0].with_alias(out_name.clone());
                let out_dtype = first_ir.dtype(input_schema, ctx.expr_arena)?;
                let mut value_expr_ir = inner_exprs[1].with_alias(out_name.clone());
                let repeats_expr_ir = inner_exprs[2].clone();

                // Cast the value if necessary.
                if value_expr_ir.dtype(input_schema, ctx.expr_arena)? != out_dtype {
                    let cast_expr = AExpr::Cast {
                        expr: value_expr_ir.node(),
                        dtype: out_dtype.clone(),
                        options: CastOptions::NonStrict,
                    };
                    value_expr_ir = ExprIR::new(
                        ctx.expr_arena.add(cast_expr),
                        OutputName::Alias(out_name.clone()),
                    );
                }

                let first_stream = build_select_stream_with_ctx(input, &[first_ir], ctx)?;
                let value_stream = build_select_stream_with_ctx(input, &[value_expr_ir], ctx)?;
                let repeats_stream = build_select_stream_with_ctx(input, &[repeats_expr_ir], ctx)?;

                let output_schema = ctx.phys_sm[first_stream.node].output_schema.clone();
                let repeat_kind = PhysNodeKind::Repeat {
                    value: value_stream,
                    repeats: repeats_stream,
                };
                let repeat_node_key = ctx
                    .phys_sm
                    .insert(PhysNode::new(output_schema.clone(), repeat_kind));

                let concat_kind = PhysNodeKind::OrderedUnion {
                    inputs: vec![first_stream, PhysStream::first(repeat_node_key)],
                };
                let concat_node_key = ctx
                    .phys_sm
                    .insert(PhysNode::new(output_schema, concat_kind));
                input_streams.insert(PhysStream::first(concat_node_key));
                transformed_exprs.push(ctx.expr_arena.add(AExpr::Column(out_name)));
            },

            AExpr::Function {
                input: ref inner_exprs,
                function: IRFunctionExpr::ConcatExpr(_rechunk),
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
                function: IRFunctionExpr::Unique(maintain_order),
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
                    schema_for_select(trans_input, std::slice::from_ref(&group_by_key_expr), ctx)?;
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
                    false,
                )?;
                input_streams.insert(group_by_stream);
                transformed_exprs.push(ctx.expr_arena.add(AExpr::Column(tmp_name)));
            },

            AExpr::Function {
                input: ref inner_exprs,
                function: IRFunctionExpr::UniqueCounts,
                options: _,
            } => {
                // Transform:
                //    expr.unique_counts().alias(name)
                //      ->
                //    .select(expr.alias(name))
                //    .group_by(_ = name, maintain_order=True)
                //      .agg(name = pl.len())
                //    .select(name)

                assert_eq!(inner_exprs.len(), 1);

                let input_schema = &ctx.phys_sm[input.node].output_schema;

                let key_name = unique_column_name();
                let tmp_count_name = unique_column_name();

                let input_expr = &inner_exprs[0];
                let output_dtype = input_expr.dtype(input_schema, ctx.expr_arena)?.clone();
                let group_by_output_schema = Arc::new(Schema::from_iter([
                    (key_name.clone(), output_dtype),
                    (tmp_count_name.clone(), IDX_DTYPE),
                ]));

                let keys = [input_expr.with_alias(key_name)];
                let aggs = [ExprIR::new(
                    ctx.expr_arena.add(AExpr::Len),
                    OutputName::Alias(tmp_count_name.clone()),
                )];

                let stream = build_group_by_stream(
                    input,
                    &keys,
                    &aggs,
                    group_by_output_schema,
                    true,
                    Default::default(),
                    None,
                    ctx.expr_arena,
                    ctx.phys_sm,
                    ctx.cache,
                    StreamingLowerIRContext {
                        prepare_visualization: ctx.prepare_visualization,
                    },
                    false,
                )?;
                transformed_exprs.push(ctx.expr_arena.add(AExpr::Column(tmp_count_name)));
                input_streams.insert(stream);
            },
            AExpr::Function {
                input: ref inner_exprs,
                function:
                    IRFunctionExpr::ValueCounts {
                        sort: false,
                        parallel: _,
                        name: count_name,
                        normalize: false,
                    },
                options: _,
            } => {
                // Transform:
                //    expr.value_counts(
                //      sort=False,
                //      parallel=_,
                //      name=count_name,
                //      normalize=False
                //    ).alias(name)
                //      ->
                //    .select(expr.alias(name))
                //    .group_by(name)
                //      .agg(count_name = pl.len())
                //    .select(pl.struct([name, count_name]))

                assert_eq!(inner_exprs.len(), 1);

                let input_schema = &ctx.phys_sm[input.node].output_schema;

                let tmp_value_name = unique_column_name();
                let tmp_count_name = unique_column_name();

                let input_expr = &inner_exprs[0];
                let output_field = input_expr.field(input_schema, ctx.expr_arena)?;
                let group_by_output_schema = Arc::new(Schema::from_iter([
                    output_field.clone().with_name(tmp_value_name.clone()),
                    Field::new(tmp_count_name.clone(), IDX_DTYPE),
                ]));

                let keys = [input_expr.with_alias(tmp_value_name.clone())];
                let aggs = [ExprIR::new(
                    ctx.expr_arena.add(AExpr::Len),
                    OutputName::Alias(tmp_count_name.clone()),
                )];

                let stream = build_group_by_stream(
                    input,
                    &keys,
                    &aggs,
                    group_by_output_schema,
                    false,
                    Default::default(),
                    None,
                    ctx.expr_arena,
                    ctx.phys_sm,
                    ctx.cache,
                    StreamingLowerIRContext {
                        prepare_visualization: ctx.prepare_visualization,
                    },
                    false,
                )?;

                let value = ExprIR::new(
                    ctx.expr_arena.add(AExpr::Column(tmp_value_name)),
                    OutputName::Alias(output_field.name),
                );
                let count = ExprIR::new(
                    ctx.expr_arena.add(AExpr::Column(tmp_count_name)),
                    OutputName::Alias(count_name.clone()),
                );

                transformed_exprs.push(
                    AExprBuilder::function(
                        vec![value, count],
                        IRFunctionExpr::AsStruct,
                        ctx.expr_arena,
                    )
                    .node(),
                );
                input_streams.insert(stream);
            },

            #[cfg(feature = "mode")]
            AExpr::Function {
                input: ref inner_exprs,
                function: IRFunctionExpr::Mode { maintain_order },
                options: _,
            } => {
                // Transform:
                //    expr.mode()
                //      ->
                //    .select(_t = expr)
                //    .group_by(_t)
                //      .agg(count_name = pl.len())
                //    .select(_t.filter(count_name == count_name.max())

                assert_eq!(inner_exprs.len(), 1);

                let tmp_value_name = unique_column_name();
                let tmp_count_name = unique_column_name();

                let stream = build_select_stream_with_ctx(
                    input,
                    &[inner_exprs[0].with_alias(tmp_value_name.clone())],
                    ctx,
                )?;

                let mut group_by_output_schema =
                    ctx.phys_sm[stream.node].output_schema.as_ref().clone();
                group_by_output_schema.insert(tmp_count_name.clone(), IDX_DTYPE);

                let keys = [AExprBuilder::col(tmp_value_name.clone(), ctx.expr_arena)
                    .expr_ir(tmp_value_name.clone())];
                let aggs = [ExprIR::new(
                    ctx.expr_arena.add(AExpr::Len),
                    OutputName::Alias(tmp_count_name.clone()),
                )];

                let stream = build_group_by_stream(
                    stream,
                    &keys,
                    &aggs,
                    Arc::new(group_by_output_schema),
                    maintain_order,
                    Default::default(),
                    None,
                    ctx.expr_arena,
                    ctx.phys_sm,
                    ctx.cache,
                    StreamingLowerIRContext {
                        prepare_visualization: ctx.prepare_visualization,
                    },
                    false,
                )?;

                let stream = build_select_stream_with_ctx(
                    stream,
                    &[AExprBuilder::col(tmp_value_name.clone(), ctx.expr_arena)
                        .filter(
                            AExprBuilder::col(tmp_count_name.clone(), ctx.expr_arena).eq(
                                AExprBuilder::col(tmp_count_name.clone(), ctx.expr_arena)
                                    .max(ctx.expr_arena),
                                ctx.expr_arena,
                            ),
                            ctx.expr_arena,
                        )
                        .expr_ir(tmp_value_name.clone())],
                    ctx,
                )?;

                transformed_exprs
                    .push(AExprBuilder::col(tmp_value_name.clone(), ctx.expr_arena).node());
                input_streams.insert(stream);
            },

            AExpr::Function {
                input: ref inner_exprs,
                function: IRFunctionExpr::ArgUnique,
                options: _,
            } => {
                // Transform:
                //    expr.arg_unique()
                //      ->
                //    .with_row_index(IDX)
                //    .group_by(expr)
                //    .agg(IDX = IDX.first())
                //    .select(IDX.sort())

                assert_eq!(inner_exprs.len(), 1);

                let expr_name = unique_column_name();
                let idx_name = unique_column_name();

                let stream = build_select_stream_with_ctx(
                    input,
                    &[inner_exprs[0].with_alias(expr_name.clone())],
                    ctx,
                )?;

                let mut group_by_output_schema =
                    ctx.phys_sm[stream.node].output_schema.as_ref().clone();
                group_by_output_schema.insert(idx_name.clone(), IDX_DTYPE);

                let stream = build_row_idx_stream(stream, idx_name.clone(), None, ctx.phys_sm);

                let keys =
                    [AExprBuilder::col(expr_name.clone(), ctx.expr_arena).expr_ir(expr_name)];
                let aggs = [AExprBuilder::col(idx_name.clone(), ctx.expr_arena)
                    .first(ctx.expr_arena)
                    .expr_ir(idx_name.clone())];

                let stream = build_group_by_stream(
                    stream,
                    &keys,
                    &aggs,
                    Arc::new(group_by_output_schema),
                    false,
                    Default::default(),
                    None,
                    ctx.expr_arena,
                    ctx.phys_sm,
                    ctx.cache,
                    StreamingLowerIRContext {
                        prepare_visualization: ctx.prepare_visualization,
                    },
                    false,
                )?;

                let expr = AExprBuilder::col(idx_name.clone(), ctx.expr_arena)
                    .sort(Default::default(), ctx.expr_arena)
                    .expr_ir(idx_name.clone());
                let stream = build_select_stream_with_ctx(stream, &[expr], ctx)?;

                transformed_exprs.push(AExprBuilder::col(idx_name.clone(), ctx.expr_arena).node());
                input_streams.insert(stream);
            },

            #[cfg(feature = "is_in")]
            AExpr::Function {
                input: ref inner_exprs,
                function: IRFunctionExpr::Boolean(IRBooleanFunction::IsIn { nulls_equal }),
                options: _,
            } if is_scalar_ae(inner_exprs[1].node(), ctx.expr_arena) => {
                // Translate left and right side separately (they could have different lengths).

                use polars_core::prelude::ExplodeOptions;
                let left_on_name = unique_column_name();
                let right_on_name = unique_column_name();
                let (trans_input_left, trans_expr_left) =
                    lower_exprs_with_ctx(input, &[inner_exprs[0].node()], ctx)?;
                let right_expr_exploded_node = match ctx.expr_arena.get(inner_exprs[1].node()) {
                    // expr.implode().explode() ~= expr (and avoids rechunking)
                    AExpr::Agg(IRAggExpr::Implode(n)) => *n,
                    _ => AExprBuilder::new_from_node(inner_exprs[1].node())
                        .explode(
                            ctx.expr_arena,
                            ExplodeOptions {
                                empty_as_null: false,
                                keep_nulls: true,
                            },
                        )
                        .node(),
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
                        build_side: None,
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

            #[cfg(feature = "cum_agg")]
            ref agg_expr @ AExpr::Function {
                input: ref inner_exprs,
                function:
                    ref function @ (IRFunctionExpr::CumMin { reverse }
                    | IRFunctionExpr::CumMax { reverse }
                    | IRFunctionExpr::CumSum { reverse }
                    | IRFunctionExpr::CumCount { reverse }
                    | IRFunctionExpr::CumProd { reverse }),
                options: _,
            } if !reverse => {
                use crate::nodes::cum_agg::CumAggKind;

                assert_eq!(inner_exprs.len(), 1);

                let input_schema = &ctx.phys_sm[input.node].output_schema;

                let value_key = unique_column_name();
                let value_dtype =
                    agg_expr.to_dtype(&ToFieldContext::new(ctx.expr_arena, input_schema))?;

                let input = build_select_stream_with_ctx(
                    input,
                    &[inner_exprs[0].with_alias(value_key.clone())],
                    ctx,
                )?;
                let kind = match function {
                    IRFunctionExpr::CumMin { .. } => CumAggKind::Min,
                    IRFunctionExpr::CumMax { .. } => CumAggKind::Max,
                    IRFunctionExpr::CumSum { .. } => CumAggKind::Sum,
                    IRFunctionExpr::CumCount { .. } => CumAggKind::Count,
                    IRFunctionExpr::CumProd { .. } => CumAggKind::Prod,
                    _ => unreachable!(),
                };
                let node_kind = PhysNodeKind::CumAgg { input, kind };

                let output_schema = Schema::from_iter([(value_key.clone(), value_dtype.clone())]);
                let node_key = ctx
                    .phys_sm
                    .insert(PhysNode::new(Arc::new(output_schema), node_kind));
                input_streams.insert(PhysStream::first(node_key));
                transformed_exprs.push(ctx.expr_arena.add(AExpr::Column(value_key)));
            },

            #[cfg(feature = "diff")]
            AExpr::Function {
                input: ref inner_exprs,
                function: IRFunctionExpr::Diff(null_behavior),
                options: _,
            } => {
                use polars_core::scalar::Scalar;
                use polars_core::series::ops::NullBehavior;

                assert_eq!(inner_exprs.len(), 2);

                // Transform:
                //    expr.diff(offset, "ignore")
                //      ->
                //    expr - expr.shift(offset)

                let base_expr_ir = &inner_exprs[0];
                let base_dtype =
                    base_expr_ir.dtype(&ctx.phys_sm[input.node].output_schema, ctx.expr_arena)?;
                let offset_expr_ir = &inner_exprs[1];
                let offset_dtype =
                    offset_expr_ir.dtype(&ctx.phys_sm[input.node].output_schema, ctx.expr_arena)?;

                let mut base = AExprBuilder::new_from_node(base_expr_ir.node());
                let cast_dtype = match base_dtype {
                    DataType::UInt8 => Some(DataType::Int16),
                    DataType::UInt16 => Some(DataType::Int32),
                    DataType::UInt32 | DataType::UInt64 => Some(DataType::Int64),
                    _ => None,
                };
                if let Some(dtype) = cast_dtype {
                    base = base.cast(dtype, ctx.expr_arena);
                }

                let mut offset = AExprBuilder::new_from_node(offset_expr_ir.node());
                if offset_dtype != &DataType::Int64 {
                    offset = offset.cast(DataType::Int64, ctx.expr_arena);
                }

                let shifted = base.shift(offset.node(), ctx.expr_arena);
                let mut output = base.minus(shifted.node(), ctx.expr_arena);

                if null_behavior == NullBehavior::Drop {
                    // Without the column size, slice can only remove leading nulls.
                    // So if the offset was negative, the nulls appeared at the end of the column.
                    // In that case, shift the column forward to move the nulls back to the front.
                    let zero_literal =
                        AExprBuilder::lit(LiteralValue::new_idxsize(0), ctx.expr_arena);
                    let offset_neg = offset.negate(ctx.expr_arena);
                    let offset_if_negative = AExprBuilder::function(
                        vec![offset_neg.expr_ir_unnamed(), zero_literal.expr_ir_unnamed()],
                        IRFunctionExpr::MaxHorizontal,
                        ctx.expr_arena,
                    );
                    output = output.shift(offset_if_negative, ctx.expr_arena);

                    // Remove the nulls that were introduced by the shift
                    let offset_abs = offset.abs(ctx.expr_arena);
                    let null_literal = AExprBuilder::lit(
                        LiteralValue::Scalar(Scalar::null(DataType::Int64)),
                        ctx.expr_arena,
                    );
                    output = output.slice(offset_abs, null_literal, ctx.expr_arena);
                }

                let (stream, nodes) = lower_exprs_with_ctx(input, &[output.node()], ctx)?;
                input_streams.insert(stream);
                transformed_exprs.extend(nodes);
            },

            AExpr::Function {
                input: ref inner_exprs,
                function: IRFunctionExpr::RLE,
                options: _,
            } => {
                assert_eq!(inner_exprs.len(), 1);

                let input_schema = &ctx.phys_sm[input.node].output_schema;

                let value_key = unique_column_name();
                let value_dtype = inner_exprs[0].dtype(input_schema, ctx.expr_arena)?;

                let input = build_select_stream_with_ctx(
                    input,
                    &[inner_exprs[0].with_alias(value_key.clone())],
                    ctx,
                )?;
                let node_kind = PhysNodeKind::Rle(input);

                let output_schema = Schema::from_iter([(
                    value_key.clone(),
                    DataType::Struct(vec![
                        Field::new(
                            PlSmallStr::from_static(RLE_VALUE_COLUMN_NAME),
                            value_dtype.clone(),
                        ),
                        Field::new(PlSmallStr::from_static(RLE_LENGTH_COLUMN_NAME), IDX_DTYPE),
                    ]),
                )]);
                let node_key = ctx
                    .phys_sm
                    .insert(PhysNode::new(Arc::new(output_schema), node_kind));
                input_streams.insert(PhysStream::first(node_key));
                transformed_exprs.push(ctx.expr_arena.add(AExpr::Column(value_key)));
            },

            AExpr::Function {
                input: ref inner_exprs,
                function: IRFunctionExpr::RLEID,
                options: _,
            } => {
                assert_eq!(inner_exprs.len(), 1);

                let value_key = unique_column_name();

                let input = build_select_stream_with_ctx(
                    input,
                    &[inner_exprs[0].with_alias(value_key.clone())],
                    ctx,
                )?;
                let node_kind = PhysNodeKind::RleId(input);

                let output_schema = Schema::from_iter([(value_key.clone(), IDX_DTYPE)]);
                let node_key = ctx
                    .phys_sm
                    .insert(PhysNode::new(Arc::new(output_schema), node_kind));
                input_streams.insert(PhysStream::first(node_key));
                transformed_exprs.push(ctx.expr_arena.add(AExpr::Column(value_key.clone())));
            },

            AExpr::Function {
                input: ref inner_exprs,
                function: IRFunctionExpr::GatherEvery { n, offset },
                options: _,
            } => {
                assert_eq!(inner_exprs.len(), 1);

                let value_key = unique_column_name();

                let input = build_select_stream_with_ctx(
                    input,
                    &[inner_exprs[0].with_alias(value_key.clone())],
                    ctx,
                )?;
                let node_kind = PhysNodeKind::GatherEvery { input, n, offset };

                let output_schema = ctx.phys_sm[input.node].output_schema.clone();
                let node_key = ctx.phys_sm.insert(PhysNode::new(output_schema, node_kind));
                input_streams.insert(PhysStream::first(node_key));
                transformed_exprs.push(ctx.expr_arena.add(AExpr::Column(value_key.clone())));
            },

            AExpr::Function {
                input: ref inner_exprs,
                function: ref function @ (IRFunctionExpr::PeakMin | IRFunctionExpr::PeakMax),
                options: _,
            } => {
                assert_eq!(inner_exprs.len(), 1);

                let value_key = unique_column_name();

                let input = build_select_stream_with_ctx(
                    input,
                    &[inner_exprs[0].with_alias(value_key.clone())],
                    ctx,
                )?;
                let is_peak_max = matches!(function, IRFunctionExpr::PeakMax);
                let node_kind = PhysNodeKind::PeakMinMax { input, is_peak_max };

                let output_schema = Schema::from_iter([(value_key.clone(), DataType::Boolean)]);
                let node_key = ctx
                    .phys_sm
                    .insert(PhysNode::new(Arc::new(output_schema), node_kind));
                input_streams.insert(PhysStream::first(node_key));
                transformed_exprs.push(ctx.expr_arena.add(AExpr::Column(value_key.clone())));
            },

            // pl.row_index() maps to this.
            #[cfg(feature = "range")]
            AExpr::Function {
                input: ref inner_exprs,
                function: IRFunctionExpr::Range(IRRangeFunction::IntRange { step: 1, dtype }),
                options: _,
            } if {
                let start_is_zero = match ctx.expr_arena.get(inner_exprs[0].node()) {
                    AExpr::Literal(lit) => lit.extract_usize().ok() == Some(0),
                    _ => false,
                };
                let stop_is_len = matches!(ctx.expr_arena.get(inner_exprs[1].node()), AExpr::Len);

                dtype == DataType::IDX_DTYPE && start_is_zero && stop_is_len
            } =>
            {
                let out_name = unique_column_name();
                let row_idx_col_aexpr = ctx.expr_arena.add(AExpr::Column(out_name.clone()));
                let row_idx_col_expr_ir =
                    ExprIR::new(row_idx_col_aexpr, OutputName::ColumnLhs(out_name.clone()));
                let row_idx_stream = build_select_stream_with_ctx(
                    build_row_idx_stream(input, out_name, None, ctx.phys_sm),
                    &[row_idx_col_expr_ir],
                    ctx,
                )?;
                input_streams.insert(row_idx_stream);
                transformed_exprs.push(row_idx_col_aexpr);
            },

            #[cfg(feature = "range")]
            AExpr::Function {
                input: ref inner_exprs,
                function: IRFunctionExpr::Range(IRRangeFunction::IntRange { step: 1, dtype }),
                options: _,
            } if {
                let start_is_zero = match ctx.expr_arena.get(inner_exprs[0].node()) {
                    AExpr::Literal(lit) => lit.extract_usize().ok() == Some(0),
                    _ => false,
                };
                let stop_is_count = matches!(
                    ctx.expr_arena.get(inner_exprs[1].node()),
                    AExpr::Agg(IRAggExpr::Count { .. })
                );

                start_is_zero && stop_is_count
            } =>
            {
                let AExpr::Agg(IRAggExpr::Count {
                    input: input_expr,
                    include_nulls,
                }) = ctx.expr_arena.get(inner_exprs[1].node())
                else {
                    unreachable!();
                };
                let (input_expr, include_nulls) = (*input_expr, *include_nulls);

                let out_name = unique_column_name();
                let mut row_idx_col_aexpr = ctx.expr_arena.add(AExpr::Column(out_name.clone()));
                if dtype != IDX_DTYPE {
                    row_idx_col_aexpr = AExprBuilder::new_from_node(row_idx_col_aexpr)
                        .cast(dtype, ctx.expr_arena)
                        .node();
                }
                let row_idx_col_expr_ir =
                    ExprIR::new(row_idx_col_aexpr, OutputName::ColumnLhs(out_name.clone()));

                let mut input_expr = AExprBuilder::new_from_node(input_expr);
                if !include_nulls {
                    input_expr = input_expr.drop_nulls(ctx.expr_arena);
                }
                let input_expr = input_expr.expr_ir_retain_name(ctx.expr_arena);

                let row_idx_stream = build_select_stream_with_ctx(
                    build_row_idx_stream(
                        build_select_stream_with_ctx(input, &[input_expr], ctx)?,
                        out_name,
                        None,
                        ctx.phys_sm,
                    ),
                    &[row_idx_col_expr_ir],
                    ctx,
                )?;
                input_streams.insert(row_idx_stream);
                transformed_exprs.push(row_idx_col_aexpr);
            },

            // Lower arbitrary elementwise functions.
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

            // Lower arbitrary row-separable functions.
            ref node @ AExpr::Function {
                input: ref inner_exprs,
                ref function,
                options,
            } if options.is_row_separable() && !is_fake_elementwise_function(node) => {
                // While these functions are streamable, they are not elementwise, so we
                // have to transform them to a select node.
                let inner_nodes = inner_exprs.iter().map(|x| x.node()).collect_vec();
                let (trans_input, trans_exprs) = lower_exprs_with_ctx(input, &inner_nodes, ctx)?;
                let out_name = unique_column_name();
                let trans_inner = ctx.expr_arena.add(AExpr::Function {
                    input: trans_exprs
                        .iter()
                        .map(|node| ExprIR::from_node(*node, ctx.expr_arena))
                        .collect(),
                    function: function.clone(),
                    options,
                });
                let func_expr = ExprIR::new(trans_inner, OutputName::Alias(out_name.clone()));
                let output_schema =
                    schema_for_select(trans_input, std::slice::from_ref(&func_expr), ctx)?;
                let node_kind = PhysNodeKind::Select {
                    input: trans_input,
                    selectors: vec![func_expr.clone()],
                    extend_original: false,
                };
                let node_key = ctx.phys_sm.insert(PhysNode::new(output_schema, node_kind));
                input_streams.insert(PhysStream::first(node_key));
                transformed_exprs.push(ctx.expr_arena.add(AExpr::Column(out_name)));
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
            AExpr::Eval {
                expr: inner,
                evaluation,
                variant,
            } => match variant {
                EvalVariant::List
                | EvalVariant::ListAgg
                | EvalVariant::Array { as_list: _ }
                | EvalVariant::ArrayAgg => {
                    let (trans_input, trans_expr) = lower_exprs_with_ctx(input, &[inner], ctx)?;
                    let eval_expr = AExpr::Eval {
                        expr: trans_expr[0],
                        evaluation,
                        variant,
                    };
                    input_streams.insert(trans_input);
                    transformed_exprs.push(ctx.expr_arena.add(eval_expr));
                },
                EvalVariant::Cumulative { .. } => {
                    // Cumulative is not elementwise, this would need a special node.
                    let out_name = unique_column_name();
                    fallback_subset.push(ExprIR::new(expr, OutputName::Alias(out_name.clone())));
                    transformed_exprs.push(ctx.expr_arena.add(AExpr::Column(out_name)));
                },
            },
            AExpr::StructEval {
                expr: inner,
                mut evaluation,
            } => {
                // Transform (simplified):
                //    expr.struct.with_fields(evaluation).alias(name)
                //      ->
                //    .select(expr)
                //    .with_columns(validity = expr.is_not_null())
                //    .map(df.struct.unnest()))
                //    .with_columns([evaluation])
                //    .select(pl.when(validity).then(as_struct()).alias(name)
                //
                // Any reference to `StructField(x)` gets remapped to `Column(PREFIX_x)` prior to
                // calling `unnest()`, with PREFIX being unique for each StructEval expression.

                // Evaluate input `expr` and capture `col` references from `evaluation`
                let out_name = unique_column_name();
                let inner_expr_ir = ExprIR::new(inner, OutputName::Alias(out_name.clone()));
                let mut expr_irs = Vec::new();
                expr_irs.push(inner_expr_ir);

                // Any column expression inside evaluation must be added explicitly.
                let eval_col_names: PlHashSet<_> = evaluation
                    .iter()
                    .flat_map(|expr| {
                        polars_plan::utils::aexpr_to_leaf_names_iter(expr.node(), ctx.expr_arena)
                    })
                    .cloned()
                    .collect();
                for name in eval_col_names {
                    expr_irs.push(ExprIR::new(
                        ctx.expr_arena.add(AExpr::Column(name.clone())),
                        OutputName::ColumnLhs(name),
                    ));
                }
                let stream = build_select_stream_with_ctx(input, &expr_irs, ctx)?;

                // Capture validity as an extra column.
                let validity_name = polars_utils::format_pl_smallstr!(
                    "{}{}",
                    out_name,
                    PlSmallStr::from_static("_VLD")
                );
                let validity_input_node = ctx.expr_arena.add(AExpr::Column(out_name.clone()));
                let validity_expr_ir = ExprIR::new(
                    validity_input_node,
                    OutputName::Alias(validity_name.clone()),
                );
                let validity_expr = AExprBuilder::function(
                    vec![validity_expr_ir],
                    IRFunctionExpr::Boolean(IRBooleanFunction::IsNotNull),
                    ctx.expr_arena,
                );
                let validity_node = validity_expr.node();
                let validity_expr_ir =
                    ExprIR::new(validity_node, OutputName::Alias(validity_name.clone()));
                let stream = build_hstack_stream(
                    stream,
                    &[validity_expr_ir],
                    ctx.expr_arena,
                    ctx.phys_sm,
                    ctx.cache,
                    StreamingLowerIRContext {
                        prepare_visualization: ctx.prepare_visualization,
                    },
                )?;

                // Rewrite any `StructField(x)`` expression into a `Col(prefix_x)`` expression.
                let separator = PlSmallStr::from_static("_FLD_");
                let field_prefix = polars_utils::format_pl_smallstr!("{}{}", out_name, separator);
                evaluation.iter_mut().for_each(|e| {
                    e.set_node(structfield_to_column(
                        e.node(),
                        ctx.expr_arena,
                        &field_prefix,
                    ))
                });

                // Unnest.
                let unnest_fn = FunctionIR::Unnest {
                    columns: Arc::new([out_name.clone()]),
                    separator: Some(separator.clone()),
                };
                let input_schema = ctx.phys_sm[stream.node].output_schema.clone();
                let output_schema = unnest_fn.schema(&input_schema)?.into_owned();
                let format_str = ctx.prepare_visualization.then(|| {
                    format!(
                        "UNNEST columns: [{}], separator: \"{}\"",
                        out_name.as_str(),
                        separator.as_str()
                    )
                });
                let map = Arc::new(move |df| unnest_fn.evaluate(df));
                let node_kind = PhysNodeKind::Map {
                    input: stream,
                    map,
                    format_str,
                };
                let node_key = ctx.phys_sm.insert(PhysNode::new(output_schema, node_kind));
                let stream = PhysStream::first(node_key);

                // Evaluate `evaluation`, using `with_columns`.
                // This requires output names to be prefixed, as they refer to the local StructField namespace.
                // Note, native columns are still included in the stream but could be dropped (nice-to-have).
                evaluation.iter_mut().for_each(|e| {
                    *e = e.with_alias(polars_utils::format_pl_smallstr!(
                        "{}{}",
                        &field_prefix,
                        e.output_name()
                    ));
                });
                let stream = build_hstack_stream(
                    stream,
                    &evaluation,
                    ctx.expr_arena,
                    ctx.phys_sm,
                    ctx.cache,
                    StreamingLowerIRContext {
                        prepare_visualization: ctx.prepare_visualization,
                    },
                )?;

                // Nest any column that belongs to the StructField namespace back into a Struct.
                let mut fields_expr_irs = Vec::new();
                let eval_schema = ctx.phys_sm[stream.node].output_schema.clone();
                for (name, _) in eval_schema.iter() {
                    if let Some(stripped_name) = name.strip_prefix(field_prefix.as_str()) {
                        let node = ctx.expr_arena.add(AExpr::Column(name.clone()));
                        fields_expr_irs.push(
                            ExprIR::from_node(node, ctx.expr_arena)
                                .with_alias(PlSmallStr::from_str(stripped_name)),
                        );
                    }
                }
                let as_struct_expr = AExprBuilder::function(
                    fields_expr_irs,
                    IRFunctionExpr::AsStruct,
                    ctx.expr_arena,
                );
                let as_struct_node = as_struct_expr.node();

                // Apply validity.
                let with_validity = AExprBuilder::when_then_otherwise(
                    AExprBuilder::col(validity_name.clone(), ctx.expr_arena),
                    AExprBuilder::new_from_node(as_struct_node),
                    AExprBuilder::lit(
                        LiteralValue::Scalar(Scalar::null(DataType::Null)),
                        ctx.expr_arena,
                    ),
                    ctx.expr_arena,
                );
                let with_validity_node = with_validity.node();
                let validity_expr_ir =
                    ExprIR::new(with_validity_node, OutputName::Alias(out_name.clone()));
                let stream = build_select_stream_with_ctx(stream, &[validity_expr_ir], ctx)?;
                let exit_node = ctx.expr_arena.add(AExpr::Column(out_name.clone()));

                // Finalize.
                input_streams.insert(stream);
                transformed_exprs.push(exit_node);
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
                    build_select_stream_with_ctx(input, std::slice::from_ref(&inner_expr_ir), ctx)?;
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

                let sorted_col_expr = ctx.expr_arena.add(AExpr::Column(sorted_name.clone()));
                input_streams.insert(PhysStream::first(sort_node_key));
                transformed_exprs.push(sorted_col_expr);
            },

            #[cfg(feature = "top_k")]
            AExpr::Function {
                input: inner_exprs,
                function: function @ (IRFunctionExpr::TopK { .. } | IRFunctionExpr::TopKBy { .. }),
                options: _,
            } => {
                // Select our inputs.
                let by = &inner_exprs[2..];
                let out_name = unique_column_name();
                let by_names = by.iter().map(|_| unique_column_name()).collect_vec();
                let data_irs = [(&out_name, &inner_exprs[0])]
                    .into_iter()
                    .chain(by_names.iter().zip(by.iter()))
                    .map(|(name, inner)| ExprIR::new(inner.node(), OutputName::Alias(name.clone())))
                    .collect_vec();
                let data_stream = build_select_stream_with_ctx(input, &data_irs, ctx)?;
                let k_stream = build_select_stream_with_ctx(input, &inner_exprs[1..2], ctx)?;

                // Create 'by' column expressions.
                let out_col_node = ctx.expr_arena.add(AExpr::Column(out_name.clone()));
                let out_col_expr = ExprIR::new(out_col_node, OutputName::Alias(out_name));
                let (by_column, reverse) = match function {
                    IRFunctionExpr::TopK { descending } => {
                        (vec![out_col_expr.clone()], vec![descending])
                    },
                    IRFunctionExpr::TopKBy {
                        descending: reverse,
                    } => {
                        let by_column = by_names
                            .into_iter()
                            .map(|name| {
                                ExprIR::new(
                                    ctx.expr_arena.add(AExpr::Column(name.clone())),
                                    OutputName::Alias(name),
                                )
                            })
                            .collect();
                        (by_column, reverse.clone())
                    },
                    _ => unreachable!(),
                };

                let kind = PhysNodeKind::TopK {
                    input: data_stream,
                    k: k_stream,
                    nulls_last: vec![true; by_column.len()],
                    reverse,
                    by_column,
                    dyn_pred: None,
                };
                let output_schema = ctx.phys_sm[data_stream.node].output_schema.clone();
                let node_key = ctx.phys_sm.insert(PhysNode::new(output_schema, kind));
                input_streams.insert(PhysStream::first(node_key));
                transformed_exprs.push(out_col_node);
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

            AExpr::AnonymousAgg {
                input: _,
                fmt_str: _,
                function: _,
            } => {
                let (trans_stream, trans_expr) = lower_reduce_node(input, expr, ctx)?;
                input_streams.insert(trans_stream);
                transformed_exprs.push(trans_expr);
            },
            // Aggregates.
            AExpr::Agg(agg) => match agg {
                // Change agg mutably so we can share the codepath for all of these.
                IRAggExpr::Min { .. }
                | IRAggExpr::Max { .. }
                | IRAggExpr::First(_)
                | IRAggExpr::FirstNonNull(_)
                | IRAggExpr::Last(_)
                | IRAggExpr::LastNonNull(_)
                | IRAggExpr::Item { .. }
                | IRAggExpr::Sum(_)
                | IRAggExpr::Mean(_)
                | IRAggExpr::Var { .. }
                | IRAggExpr::Std { .. }
                | IRAggExpr::Count { .. } => {
                    let (trans_stream, trans_expr) = lower_reduce_node(input, expr, ctx)?;
                    input_streams.insert(trans_stream);
                    transformed_exprs.push(trans_expr);
                },
                IRAggExpr::NUnique(inner) => {
                    // Lower to no-aggregate group-by with unique name feeding into len aggregate.
                    let tmp_name = unique_column_name();
                    let (trans_input, trans_inner_exprs) =
                        lower_exprs_with_ctx(input, &[inner], ctx)?;
                    let group_by_key_expr =
                        ExprIR::new(trans_inner_exprs[0], OutputName::Alias(tmp_name.clone()));
                    let group_by_output_schema = schema_for_select(
                        trans_input,
                        std::slice::from_ref(&group_by_key_expr),
                        ctx,
                    )?;
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
                        false,
                    )?;

                    let len_node = ctx.expr_arena.add(AExpr::Len);
                    let len_expr_ir = ExprIR::new(len_node, OutputName::Alias(tmp_name.clone()));
                    let output_schema = schema_for_select(
                        group_by_stream,
                        std::slice::from_ref(&len_expr_ir),
                        ctx,
                    )?;
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

            #[cfg(feature = "bitwise")]
            AExpr::Function {
                function:
                    IRFunctionExpr::Bitwise(
                        IRBitwiseFunction::And | IRBitwiseFunction::Or | IRBitwiseFunction::Xor,
                    ),
                ..
            } => {
                let (trans_stream, trans_expr) = lower_reduce_node(input, expr, ctx)?;
                input_streams.insert(trans_stream);
                transformed_exprs.push(trans_expr);
            },

            #[cfg(feature = "approx_unique")]
            AExpr::Function {
                function: IRFunctionExpr::ApproxNUnique,
                ..
            } => {
                let (trans_stream, trans_expr) = lower_reduce_node(input, expr, ctx)?;
                input_streams.insert(trans_stream);
                transformed_exprs.push(trans_expr);
            },

            AExpr::Function {
                function:
                    IRFunctionExpr::Boolean(
                        IRBooleanFunction::Any { .. } | IRBooleanFunction::All { .. },
                    )
                    | IRFunctionExpr::MinBy
                    | IRFunctionExpr::MaxBy
                    | IRFunctionExpr::NullCount,
                ..
            } => {
                let (trans_stream, trans_expr) = lower_reduce_node(input, expr, ctx)?;
                input_streams.insert(trans_stream);
                transformed_exprs.push(trans_expr);
            },

            // Length-based expressions.
            AExpr::Len => {
                let out_name = unique_column_name();
                let expr_ir = ExprIR::new(expr, OutputName::Alias(out_name.clone()));
                let output_schema = schema_for_select(input, std::slice::from_ref(&expr_ir), ctx)?;
                let kind = PhysNodeKind::Reduce {
                    input,
                    exprs: vec![expr_ir],
                };
                let reduce_node_key = ctx.phys_sm.insert(PhysNode::new(output_schema, kind));
                input_streams.insert(PhysStream::first(reduce_node_key));
                transformed_exprs.push(ctx.expr_arena.add(AExpr::Column(out_name)));
            },

            AExpr::Function {
                input: ref inner_exprs,
                function: IRFunctionExpr::ArgWhere,
                options: _,
            } => {
                // pl.arg_where(expr)
                //
                // ->
                // .select(predicate_name = expr)
                // .with_row_index(out_name)
                // .filter(predicate_name)
                // .select(out_name)
                let out_name = unique_column_name();
                let predicate_name = unique_column_name();
                let predicate = build_select_stream_with_ctx(
                    input,
                    &[inner_exprs[0].with_alias(predicate_name.clone())],
                    ctx,
                )?;
                let row_index =
                    build_row_idx_stream(predicate, out_name.clone(), None, ctx.phys_sm);

                let filter_stream = build_filter_stream(
                    row_index,
                    AExprBuilder::col(predicate_name.clone(), ctx.expr_arena)
                        .expr_ir(predicate_name),
                    ctx.expr_arena,
                    ctx.phys_sm,
                    ctx.cache,
                    StreamingLowerIRContext {
                        prepare_visualization: ctx.prepare_visualization,
                    },
                )?;
                input_streams.insert(filter_stream);
                transformed_exprs.push(AExprBuilder::col(out_name.clone(), ctx.expr_arena).node());
            },

            AExpr::Slice {
                input: inner,
                offset,
                length,
            } => {
                let out_name = unique_column_name();
                let inner_expr_ir = ExprIR::new(inner, OutputName::Alias(out_name.clone()));
                let offset_expr_ir = ExprIR::from_node(offset, ctx.expr_arena);
                let length_expr_ir = ExprIR::from_node(length, ctx.expr_arena);
                let input_stream = build_select_stream_with_ctx(input, &[inner_expr_ir], ctx)?;
                let offset_stream = build_select_stream_with_ctx(input, &[offset_expr_ir], ctx)?;
                let length_stream = build_select_stream_with_ctx(input, &[length_expr_ir], ctx)?;

                let output_schema = ctx.phys_sm[input_stream.node].output_schema.clone();
                let kind = PhysNodeKind::DynamicSlice {
                    input: input_stream,
                    offset: offset_stream,
                    length: length_stream,
                };
                let slice_node_key = ctx.phys_sm.insert(PhysNode::new(output_schema, kind));
                input_streams.insert(PhysStream::first(slice_node_key));
                transformed_exprs.push(ctx.expr_arena.add(AExpr::Column(out_name)));
            },

            AExpr::Function {
                input: ref inner_exprs,
                function: func @ (IRFunctionExpr::Shift | IRFunctionExpr::ShiftAndFill),
                options: _,
            } => {
                let out_name = unique_column_name();
                let data_col_expr = inner_exprs[0].with_alias(out_name.clone());
                let trans_data_column = build_select_stream_with_ctx(input, &[data_col_expr], ctx)?;
                let trans_offset =
                    build_select_stream_with_ctx(input, &[inner_exprs[1].clone()], ctx)?;

                let trans_fill = if func == IRFunctionExpr::ShiftAndFill {
                    let fill_expr = inner_exprs[2].with_alias(out_name.clone());
                    Some(build_select_stream_with_ctx(input, &[fill_expr], ctx)?)
                } else {
                    None
                };

                let output_schema = ctx.phys_sm[trans_data_column.node].output_schema.clone();
                let node_key = ctx.phys_sm.insert(PhysNode::new(
                    output_schema,
                    PhysNodeKind::Shift {
                        input: trans_data_column,
                        offset: trans_offset,
                        fill: trans_fill,
                    },
                ));

                input_streams.insert(PhysStream::first(node_key));
                transformed_exprs.push(ctx.expr_arena.add(AExpr::Column(out_name)));
            },

            #[cfg(feature = "ewma")]
            AExpr::Function {
                input: input_exprs,
                function:
                    ewm_variant @ IRFunctionExpr::EwmMean { options }
                    | ewm_variant @ IRFunctionExpr::EwmVar { options }
                    | ewm_variant @ IRFunctionExpr::EwmStd { options },
                options: _,
            } => {
                let out_name = unique_column_name();

                let input = match input_exprs.as_slice() {
                    [input_expr] => build_select_stream_with_ctx(
                        input,
                        &[input_expr.with_alias(out_name.clone())],
                        ctx,
                    )?,
                    _ => panic!("{:?}", input_exprs),
                };

                let input_schema = ctx.phys_sm[input.node].output_schema.clone();
                assert_eq!(input_schema.len(), 1);
                let output_schema = input_schema;

                let kind = match ewm_variant {
                    IRFunctionExpr::EwmMean { .. } => PhysNodeKind::EwmMean { input, options },
                    IRFunctionExpr::EwmVar { .. } => PhysNodeKind::EwmVar { input, options },
                    IRFunctionExpr::EwmStd { .. } => PhysNodeKind::EwmStd { input, options },
                    _ => unreachable!(),
                };
                let node_key = ctx.phys_sm.insert(PhysNode::new(output_schema, kind));
                input_streams.insert(PhysStream::first(node_key));
                transformed_exprs.push(ctx.expr_arena.add(AExpr::Column(out_name)));
            },

            #[cfg(feature = "dynamic_group_by")]
            rolling_function @ AExpr::Rolling {
                function,
                index_column,
                period,
                offset,
                closed_window,
            } => {
                // function.rolling(index_column=index_column)
                //
                // ->
                //
                // .select(*LIVE_COLUMNS(function), _tmp0 = index_column)
                // .rolling(_tmp0)
                // .agg(_tmp1 = function)
                // .select(_tmp1)

                let out_name = unique_column_name();
                let index_column_name = unique_column_name();

                let index_column_expr_ir =
                    AExprBuilder::new_from_node(index_column).expr_ir(index_column_name.clone());

                let input_schema = &ctx.phys_sm[input.node].output_schema;
                let output_dtype = rolling_function
                    .to_dtype(&ToFieldContext::new(ctx.expr_arena, input_schema))?;
                let output_schema = Schema::from_iter([
                    index_column_expr_ir.field(input_schema, ctx.expr_arena)?,
                    Field::new(out_name.clone(), output_dtype),
                ]);

                let input_columns = aexpr_to_leaf_names(function, ctx.expr_arena)
                    .into_iter()
                    .map(|n| AExprBuilder::col(n.clone(), ctx.expr_arena).expr_ir(n))
                    .chain(std::iter::once(index_column_expr_ir.clone()))
                    .collect::<Vec<_>>();
                let input = build_select_stream_with_ctx(input, &input_columns, ctx)?;

                let kind = PhysNodeKind::RollingGroupBy {
                    input,
                    index_column: index_column_name,
                    period,
                    offset,
                    closed: closed_window,
                    slice: None,
                    aggs: vec![AExprBuilder::new_from_node(function).expr_ir(out_name.clone())],
                };
                let node_key = ctx
                    .phys_sm
                    .insert(PhysNode::new(Arc::new(output_schema), kind));
                let input = PhysStream::first(node_key);

                let input = build_select_stream_with_ctx(
                    input,
                    &[AExprBuilder::col(out_name.clone(), ctx.expr_arena)
                        .expr_ir(out_name.clone())],
                    ctx,
                )?;
                input_streams.insert(input);
                transformed_exprs.push(ctx.expr_arena.add(AExpr::Column(out_name)));
            },

            AExpr::AnonymousFunction { .. }
            | AExpr::Function { .. }
            | AExpr::Over { .. }
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
        zip_behavior: ZipBehavior::Broadcast,
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
                .dtype(input_schema, expr_arena)?
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

/// Builds a hstack node given an input stream and the expressions to add.
pub fn build_hstack_stream(
    input: PhysStream,
    exprs: &[ExprIR],
    expr_arena: &mut Arena<AExpr>,
    phys_sm: &mut SlotMap<PhysNodeKey, PhysNode>,
    expr_cache: &mut ExprCache,
    ctx: StreamingLowerIRContext,
) -> PolarsResult<PhysStream> {
    let input_schema = &phys_sm[input.node].output_schema;
    if exprs
        .iter()
        .all(|e| is_elementwise_rec_cached(e.node(), expr_arena, expr_cache))
    {
        let mut output_schema = input_schema.as_ref().clone();
        for expr in exprs {
            output_schema.insert(
                expr.output_name().clone(),
                expr.dtype(input_schema, expr_arena)?
                    .clone()
                    .materialize_unknown(true)?,
            );
        }
        let output_schema = Arc::new(output_schema);

        let selectors = exprs.to_vec();
        let kind = PhysNodeKind::Select {
            input,
            selectors,
            extend_original: true,
        };
        let node_key = phys_sm.insert(PhysNode {
            output_schema,
            kind,
        });

        Ok(PhysStream::first(node_key))
    } else {
        // We already handled the all-streamable case above, so things get more complicated.
        // For simplicity we just do a normal select with all the original columns prepended.
        let mut selectors = PlIndexMap::with_capacity(input_schema.len() + exprs.len());
        for name in input_schema.iter_names() {
            let col_name = name.clone();
            let col_expr = expr_arena.add(AExpr::Column(col_name.clone()));
            selectors.insert(
                name.clone(),
                ExprIR::new(col_expr, OutputName::ColumnLhs(col_name)),
            );
        }
        for expr in exprs {
            selectors.insert(expr.output_name().clone(), expr.clone());
        }
        let selectors = selectors.into_values().collect_vec();
        build_length_preserving_select_stream(
            input, &selectors, expr_arena, phys_sm, expr_cache, ctx,
        )
    }
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
