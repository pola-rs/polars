use std::sync::Arc;

use parking_lot::Mutex;
use polars_core::prelude::{InitHashMaps, PlIndexMap};
use polars_core::schema::Schema;
use polars_error::{PolarsResult, polars_err};
use polars_expr::state::ExecutionState;
use polars_mem_engine::create_physical_plan;
use polars_plan::plans::expr_ir::{ExprIR, OutputName};
use polars_plan::plans::{AExpr, DataFrameUdf, IR, IRAggExpr, NaiveExprMerger, write_group_by};
use polars_plan::prelude::GroupbyOptions;
use polars_utils::arena::{Arena, Node};
use polars_utils::pl_str::PlSmallStr;
use polars_utils::unique_column_name;
use recursive::recursive;
use slotmap::SlotMap;

use super::{ExprCache, PhysNode, PhysNodeKey, PhysNodeKind, PhysStream, StreamingLowerIRContext};
use crate::physical_plan::lower_expr::{
    build_select_stream, compute_output_schema, is_elementwise_rec_cached,
    is_fake_elementwise_function, is_input_independent,
};
use crate::physical_plan::lower_ir::build_slice_stream;
use crate::utils::late_materialized_df::LateMaterializedDataFrame;

#[allow(clippy::too_many_arguments)]
fn build_group_by_fallback(
    input: PhysStream,
    keys: &[ExprIR],
    aggs: &[ExprIR],
    output_schema: Arc<Schema>,
    maintain_order: bool,
    options: Arc<GroupbyOptions>,
    apply: Option<Arc<dyn DataFrameUdf>>,
    expr_arena: &mut Arena<AExpr>,
    phys_sm: &mut SlotMap<PhysNodeKey, PhysNode>,
    format_str: Option<String>,
) -> PolarsResult<PhysStream> {
    let input_schema = phys_sm[input.node].output_schema.clone();
    let lmdf = Arc::new(LateMaterializedDataFrame::default());
    let mut lp_arena = Arena::default();
    let input_lp_node = lp_arena.add(lmdf.clone().as_ir_node(input_schema.clone()));
    let group_by_lp_node = lp_arena.add(IR::GroupBy {
        input: input_lp_node,
        keys: keys.to_vec(),
        aggs: aggs.to_vec(),
        schema: output_schema.clone(),
        maintain_order,
        options,
        apply,
    });
    let executor = Mutex::new(create_physical_plan(
        group_by_lp_node,
        &mut lp_arena,
        expr_arena,
        None,
    )?);

    let group_by_node = PhysNode {
        output_schema,
        kind: PhysNodeKind::InMemoryMap {
            input,
            map: Arc::new(move |df| {
                lmdf.set_materialized_dataframe(df);
                let mut state = ExecutionState::new();
                executor.lock().execute(&mut state)
            }),
            format_str,
        },
    };

    Ok(PhysStream::first(phys_sm.insert(group_by_node)))
}

/// Tries to lower an expression as a 'elementwise scalar agg expression'.
///
/// Such an expression is defined as the elementwise combination of scalar
/// aggregations of elementwise combinations of the input columns / scalar literals.
#[recursive]
fn try_lower_elementwise_scalar_agg_expr(
    expr: Node,
    outer_name: Option<PlSmallStr>,
    expr_merger: &NaiveExprMerger,
    expr_cache: &mut ExprCache,
    expr_arena: &mut Arena<AExpr>,
    agg_exprs: &mut Vec<ExprIR>,
    uniq_input_exprs: &mut PlIndexMap<u32, PlSmallStr>,
) -> Option<Node> {
    // Helper macro to simplify recursive calls.
    macro_rules! lower_rec {
        ($input:expr) => {
            try_lower_elementwise_scalar_agg_expr(
                $input,
                None,
                expr_merger,
                expr_cache,
                expr_arena,
                agg_exprs,
                uniq_input_exprs,
            )
        };
    }

    match expr_arena.get(expr) {
        AExpr::Alias(..) => unreachable!("alias found in physical plan"),

        AExpr::Column(_) => {
            // Implicit implode not yet supported.
            None
        },

        AExpr::Literal(lit) => {
            if lit.is_scalar() {
                Some(expr)
            } else {
                None
            }
        },

        AExpr::Slice { .. }
        | AExpr::Window { .. }
        | AExpr::Sort { .. }
        | AExpr::SortBy { .. }
        | AExpr::Gather { .. } => None,

        // Explode and filter are row-separable and should thus in theory work
        // in a streaming fashion but they change the length of the input which
        // means the same filter/explode should also be applied to the key
        // column, which is not (yet) supported.
        AExpr::Explode { .. } | AExpr::Filter { .. } => None,

        AExpr::BinaryExpr { left, op, right } => {
            let (left, op, right) = (*left, *op, *right);
            let left = lower_rec!(left)?;
            let right = lower_rec!(right)?;
            Some(expr_arena.add(AExpr::BinaryExpr { left, op, right }))
        },

        AExpr::Ternary {
            predicate,
            truthy,
            falsy,
        } => {
            let (predicate, truthy, falsy) = (*predicate, *truthy, *falsy);
            let predicate = lower_rec!(predicate)?;
            let truthy = lower_rec!(truthy)?;
            let falsy = lower_rec!(falsy)?;
            Some(expr_arena.add(AExpr::Ternary {
                predicate,
                truthy,
                falsy,
            }))
        },

        node @ AExpr::Function { input, options, .. }
        | node @ AExpr::AnonymousFunction { input, options, .. }
            if options.is_elementwise() && !is_fake_elementwise_function(node) =>
        {
            let node = node.clone();
            let input = input.clone();
            let new_input = input
                .into_iter()
                .map(|i| {
                    // The function may be sensitive to names (e.g. pl.struct), so we restore them.
                    let new_node = lower_rec!(i.node())?;
                    Some(ExprIR::new(
                        new_node,
                        OutputName::Alias(i.output_name().clone()),
                    ))
                })
                .collect::<Option<Vec<_>>>()?;

            let mut new_node = node.clone();
            match &mut new_node {
                AExpr::Function { input, .. } | AExpr::AnonymousFunction { input, .. } => {
                    *input = new_input;
                },
                _ => unreachable!(),
            }
            Some(expr_arena.add(new_node))
        },

        AExpr::Function { .. } | AExpr::AnonymousFunction { .. } => None,

        AExpr::Cast {
            expr,
            dtype,
            options,
        } => {
            let (expr, dtype, options) = (*expr, dtype.clone(), *options);
            let expr = lower_rec!(expr)?;
            Some(expr_arena.add(AExpr::Cast {
                expr,
                dtype,
                options,
            }))
        },

        AExpr::Agg(agg) => {
            match agg {
                IRAggExpr::Min { input, .. }
                | IRAggExpr::Max { input, .. }
                | IRAggExpr::First(input)
                | IRAggExpr::Last(input)
                | IRAggExpr::Mean(input)
                | IRAggExpr::Sum(input)
                | IRAggExpr::Var(input, ..)
                | IRAggExpr::Std(input, ..)
                | IRAggExpr::Count(input, ..) => {
                    if is_input_independent(*input, expr_arena, expr_cache) {
                        // TODO: we could simply return expr here, but we first need an is_scalar function, because if
                        // it is not a scalar we need to return expr.implode().
                        return None;
                    }

                    if !is_elementwise_rec_cached(*input, expr_arena, expr_cache) {
                        return None;
                    }

                    let mut trans_agg = agg.clone();
                    let input_id = expr_merger.get_uniq_id(*input).unwrap();
                    let input_col = uniq_input_exprs
                        .entry(input_id)
                        .or_insert_with(unique_column_name)
                        .clone();
                    let input_col_node = expr_arena.add(AExpr::Column(input_col.clone()));
                    trans_agg.set_input(input_col_node);
                    let trans_agg_node = expr_arena.add(AExpr::Agg(trans_agg));

                    // Add to aggregation expressions and replace with a reference to its output.
                    let agg_expr = if let Some(name) = outer_name {
                        ExprIR::new(trans_agg_node, OutputName::Alias(name))
                    } else {
                        ExprIR::new(trans_agg_node, OutputName::Alias(unique_column_name()))
                    };
                    let result_node = expr_arena.add(AExpr::Column(agg_expr.output_name().clone()));
                    agg_exprs.push(agg_expr);
                    Some(result_node)
                },
                IRAggExpr::Median(..)
                | IRAggExpr::NUnique(..)
                | IRAggExpr::Implode(..)
                | IRAggExpr::Quantile { .. }
                | IRAggExpr::AggGroups(..) => None, // TODO: allow all aggregates,
            }
        },
        AExpr::Len => {
            let agg_expr = if let Some(name) = outer_name {
                ExprIR::new(expr, OutputName::Alias(name))
            } else {
                ExprIR::new(expr, OutputName::Alias(unique_column_name()))
            };
            let result_node = expr_arena.add(AExpr::Column(agg_expr.output_name().clone()));
            agg_exprs.push(agg_expr);
            Some(result_node)
        },
    }
}

#[allow(clippy::too_many_arguments)]
fn try_build_streaming_group_by(
    input: PhysStream,
    keys: &[ExprIR],
    aggs: &[ExprIR],
    maintain_order: bool,
    options: Arc<GroupbyOptions>,
    apply: Option<Arc<dyn DataFrameUdf>>,
    expr_arena: &mut Arena<AExpr>,
    phys_sm: &mut SlotMap<PhysNodeKey, PhysNode>,
    expr_cache: &mut ExprCache,
    ctx: StreamingLowerIRContext,
) -> Option<PolarsResult<PhysStream>> {
    if apply.is_some() || maintain_order {
        return None; // TODO
    }

    #[cfg(feature = "dynamic_group_by")]
    if options.dynamic.is_some() || options.rolling.is_some() {
        return None; // TODO
    }

    if keys.is_empty() {
        return Some(Err(
            polars_err!(ComputeError: "at least one key is required in a group_by operation"),
        ));
    }

    let all_independent = keys
        .iter()
        .chain(aggs.iter())
        .all(|expr| is_input_independent(expr.node(), expr_arena, expr_cache));
    if all_independent {
        return None;
    }

    // Fill all expressions into the merger, letting us extract common subexpressions later.
    let mut expr_merger = NaiveExprMerger::default();
    for key in keys {
        expr_merger.add_expr(key.node(), expr_arena);
    }
    for agg in aggs {
        expr_merger.add_expr(agg.node(), expr_arena);
    }

    // Extract aggregates, input expressions for those aggregates and replace
    // with agg node output columns.
    let mut uniq_input_exprs = PlIndexMap::new();
    let mut trans_agg_exprs = Vec::new();
    let mut trans_keys = Vec::new();
    let mut trans_output_exprs = Vec::new();
    for key in keys {
        let key_id = expr_merger.get_uniq_id(key.node()).unwrap();
        let uniq_col = uniq_input_exprs
            .entry(key_id)
            .or_insert_with(unique_column_name)
            .clone();

        // Keys might refer to the same column multiple times, we have to give a unique name to it.
        let uniq_name = unique_column_name();
        let trans_key_node = expr_arena.add(AExpr::Column(uniq_col));
        trans_keys.push(ExprIR::new(
            trans_key_node,
            OutputName::Alias(uniq_name.clone()),
        ));
        let output_name = OutputName::Alias(key.output_name().clone());
        let trans_output_node = expr_arena.add(AExpr::Column(uniq_name));
        trans_output_exprs.push(ExprIR::new(trans_output_node, output_name));
    }
    for agg in aggs {
        let trans_node = try_lower_elementwise_scalar_agg_expr(
            agg.node(),
            Some(agg.output_name().clone()),
            &expr_merger,
            expr_cache,
            expr_arena,
            &mut trans_agg_exprs,
            &mut uniq_input_exprs,
        )?;
        let output_name = OutputName::Alias(agg.output_name().clone());
        trans_output_exprs.push(ExprIR::new(trans_node, output_name));
    }

    // We must lower the keys together with the input to the aggregations.
    let mut input_exprs = Vec::new();
    for (uniq_id, name) in uniq_input_exprs.iter() {
        let node = expr_merger.get_node(*uniq_id).unwrap();
        input_exprs.push(ExprIR::new(node, OutputName::Alias(name.clone())));
    }

    let pre_select =
        build_select_stream(input, &input_exprs, expr_arena, phys_sm, expr_cache, ctx).ok()?;

    let input_schema = &phys_sm[pre_select.node].output_schema;
    let group_by_output_schema = compute_output_schema(
        input_schema,
        &[trans_keys.as_slice(), trans_agg_exprs.as_slice()].concat(),
        expr_arena,
    )
    .unwrap();
    let agg_node = phys_sm.insert(PhysNode::new(
        group_by_output_schema,
        PhysNodeKind::GroupBy {
            input: pre_select,
            key: trans_keys,
            aggs: trans_agg_exprs,
        },
    ));

    let post_select = build_select_stream(
        PhysStream::first(agg_node),
        &trans_output_exprs,
        expr_arena,
        phys_sm,
        expr_cache,
        ctx,
    );
    let out = if let Some((offset, len)) = options.slice {
        post_select.map(|s| build_slice_stream(s, offset, len, phys_sm))
    } else {
        post_select
    };
    Some(out)
}

#[allow(clippy::too_many_arguments)]
pub fn build_group_by_stream(
    input: PhysStream,
    keys: &[ExprIR],
    aggs: &[ExprIR],
    output_schema: Arc<Schema>,
    maintain_order: bool,
    options: Arc<GroupbyOptions>,
    apply: Option<Arc<dyn DataFrameUdf>>,
    expr_arena: &mut Arena<AExpr>,
    phys_sm: &mut SlotMap<PhysNodeKey, PhysNode>,
    expr_cache: &mut ExprCache,
    ctx: StreamingLowerIRContext,
) -> PolarsResult<PhysStream> {
    let streaming = try_build_streaming_group_by(
        input,
        keys,
        aggs,
        maintain_order,
        options.clone(),
        apply.clone(),
        expr_arena,
        phys_sm,
        expr_cache,
        ctx,
    );
    if let Some(stream) = streaming {
        stream
    } else {
        let format_str = ctx.prepare_visualization.then(|| {
            let mut buffer = String::new();
            write_group_by(
                &mut buffer,
                0,
                expr_arena,
                keys,
                aggs,
                apply.as_deref(),
                maintain_order,
            )
            .unwrap();
            buffer
        });
        build_group_by_fallback(
            input,
            keys,
            aggs,
            output_schema,
            maintain_order,
            options,
            apply,
            expr_arena,
            phys_sm,
            format_str,
        )
    }
}
