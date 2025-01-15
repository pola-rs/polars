use std::sync::Arc;

use parking_lot::Mutex;
use polars_core::prelude::{InitHashMaps, PlHashMap, PlIndexMap};
use polars_core::schema::Schema;
use polars_error::{polars_err, PolarsResult};
use polars_expr::state::ExecutionState;
use polars_mem_engine::create_physical_plan;
use polars_plan::plans::expr_ir::{ExprIR, OutputName};
use polars_plan::plans::{AExpr, ArenaExprIter, DataFrameUdf, IRAggExpr, IR};
use polars_plan::prelude::GroupbyOptions;
use polars_utils::arena::{Arena, Node};
use polars_utils::itertools::Itertools;
use polars_utils::pl_str::PlSmallStr;
use slotmap::SlotMap;

use super::lower_expr::lower_exprs;
use super::{ExprCache, PhysNode, PhysNodeKey, PhysNodeKind, PhysStream};
use crate::physical_plan::lower_expr::{
    build_select_stream, compute_output_schema, is_input_independent, unique_column_name,
};
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
        },
    };

    Ok(PhysStream::first(phys_sm.insert(group_by_node)))
}

/// Tries to lower an expression as a 'elementwise scalar agg expression'.
///
/// Such an expression is defined as the elementwise combination of scalar
/// aggregations of elementwise combinations of the input columns / scalar literals.
fn try_lower_elementwise_scalar_agg_expr(
    expr: Node,
    inside_agg: bool,
    outer_name: Option<PlSmallStr>,
    expr_arena: &mut Arena<AExpr>,
    agg_exprs: &mut Vec<ExprIR>,
    trans_input_cols: &PlHashMap<PlSmallStr, Node>,
) -> Option<Node> {
    // Helper macro to simplify recursive calls.
    macro_rules! lower_rec {
        ($input:expr, $inside_agg:expr) => {
            try_lower_elementwise_scalar_agg_expr(
                $input,
                $inside_agg,
                None,
                expr_arena,
                agg_exprs,
                trans_input_cols,
            )
        };
    }

    match expr_arena.get(expr) {
        AExpr::Alias(..) => unreachable!("alias found in physical plan"),

        AExpr::Column(c) => {
            if inside_agg {
                Some(trans_input_cols[c])
            } else {
                // Implicit implode not yet supported.
                None
            }
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
        AExpr::Explode(_) | AExpr::Filter { .. } => None,

        AExpr::BinaryExpr { left, op, right } => {
            let (left, op, right) = (*left, *op, *right);
            let left = lower_rec!(left, inside_agg)?;
            let right = lower_rec!(right, inside_agg)?;
            Some(expr_arena.add(AExpr::BinaryExpr { left, op, right }))
        },

        AExpr::Ternary {
            predicate,
            truthy,
            falsy,
        } => {
            let (predicate, truthy, falsy) = (*predicate, *truthy, *falsy);
            let predicate = lower_rec!(predicate, inside_agg)?;
            let truthy = lower_rec!(truthy, inside_agg)?;
            let falsy = lower_rec!(falsy, inside_agg)?;
            Some(expr_arena.add(AExpr::Ternary {
                predicate,
                truthy,
                falsy,
            }))
        },

        node @ AExpr::Function { input, options, .. }
        | node @ AExpr::AnonymousFunction { input, options, .. }
            if options.is_elementwise() =>
        {
            let node = node.clone();
            let input = input.clone();
            let new_inputs = input
                .into_iter()
                .map(|i| lower_rec!(i.node(), inside_agg))
                .collect::<Option<Vec<_>>>()?;
            Some(expr_arena.add(node.replace_inputs(&new_inputs)))
        },

        AExpr::Function { .. } | AExpr::AnonymousFunction { .. } => None,

        AExpr::Cast {
            expr,
            dtype,
            options,
        } => {
            let (expr, dtype, options) = (*expr, dtype.clone(), *options);
            let expr = lower_rec!(expr, inside_agg)?;
            Some(expr_arena.add(AExpr::Cast {
                expr,
                dtype,
                options,
            }))
        },

        AExpr::Agg(agg) => {
            let orig_agg = agg.clone();
            match agg {
                IRAggExpr::Min { input, .. }
                | IRAggExpr::Max { input, .. }
                | IRAggExpr::First(input)
                | IRAggExpr::Last(input)
                | IRAggExpr::Mean(input)
                | IRAggExpr::Sum(input)
                | IRAggExpr::Var(input, ..)
                | IRAggExpr::Std(input, ..) => {
                    // Nested aggregates not supported.
                    if inside_agg {
                        return None;
                    }
                    // Lower and replace input.
                    let trans_input = lower_rec!(*input, true)?;
                    let mut trans_agg = orig_agg;
                    trans_agg.set_input(trans_input);
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
                | IRAggExpr::Count(..)
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

    // We must lower the keys together with the input to the aggregations.
    let mut input_columns = PlIndexMap::new();
    for agg in aggs {
        for (node, expr) in (&*expr_arena).iter(agg.node()) {
            if let AExpr::Column(c) = expr {
                input_columns.insert(c.clone(), node);
            }
        }
    }

    let mut pre_lower_exprs = keys.to_vec();
    for (col, node) in input_columns.iter() {
        pre_lower_exprs.push(ExprIR::new(*node, OutputName::ColumnLhs(col.clone())));
    }
    let Ok((trans_input, trans_exprs)) =
        lower_exprs(input, &pre_lower_exprs, expr_arena, phys_sm, expr_cache)
    else {
        return None;
    };
    let trans_keys = trans_exprs[..keys.len()].to_vec();
    let trans_input_cols: PlHashMap<_, _> = trans_exprs[keys.len()..]
        .iter()
        .zip(input_columns.into_keys())
        .map(|(expr, col)| (col, expr.node()))
        .collect();

    // We must now lower each (presumed) scalar aggregate expression while
    // substituting the translated input columns and extracting the aggregate
    // expressions.
    let mut trans_agg_exprs = Vec::new();
    let mut trans_output_exprs = keys
        .iter()
        .map(|key| {
            let key_node = expr_arena.add(AExpr::Column(key.output_name().clone()));
            ExprIR::from_node(key_node, expr_arena)
        })
        .collect_vec();
    for agg in aggs {
        let trans_node = try_lower_elementwise_scalar_agg_expr(
            agg.node(),
            false,
            Some(agg.output_name().clone()),
            expr_arena,
            &mut trans_agg_exprs,
            &trans_input_cols,
        )?;
        trans_output_exprs.push(ExprIR::new(trans_node, agg.output_name_inner().clone()));
    }

    let input_schema = &phys_sm[trans_input.node].output_schema;
    let group_by_output_schema = compute_output_schema(
        input_schema,
        &[trans_keys.clone(), trans_agg_exprs.clone()].concat(),
        expr_arena,
    )
    .unwrap();
    let agg_node = phys_sm.insert(PhysNode::new(
        group_by_output_schema,
        PhysNodeKind::GroupBy {
            input: trans_input,
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
    );
    Some(post_select)
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
    );
    if let Some(stream) = streaming {
        stream
    } else {
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
        )
    }
}
