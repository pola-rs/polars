use std::sync::Arc;

use parking_lot::Mutex;
use polars_core::frame::DataFrame;
use polars_core::prelude::{Field, InitHashMaps, PlIndexMap, PlIndexSet, SortMultipleOptions};
use polars_core::schema::Schema;
use polars_error::{PolarsResult, polars_err};
use polars_expr::state::ExecutionState;
use polars_mem_engine::create_physical_plan;
use polars_plan::plans::expr_ir::{ExprIR, OutputName};
use polars_plan::plans::{AExpr, IR, IRAggExpr, IRFunctionExpr, NaiveExprMerger, write_group_by};
use polars_plan::prelude::{GroupbyOptions, *};
use polars_utils::arena::{Arena, Node};
use polars_utils::pl_str::PlSmallStr;
use polars_utils::{IdxSize, unique_column_name};
use recursive::recursive;
use slotmap::SlotMap;

use super::{ExprCache, PhysNode, PhysNodeKey, PhysNodeKind, PhysStream, StreamingLowerIRContext};
use crate::physical_plan::lower_expr::{
    build_hstack_stream, build_select_stream, compute_output_schema, is_elementwise_rec_cached,
    is_fake_elementwise_function, is_input_independent,
};
use crate::physical_plan::lower_ir::{build_row_idx_stream, build_slice_stream};
use crate::utils::late_materialized_df::LateMaterializedDataFrame;

#[allow(clippy::too_many_arguments)]
fn build_group_by_fallback(
    input: PhysStream,
    keys: &[ExprIR],
    aggs: &[ExprIR],
    output_schema: Arc<Schema>,
    maintain_order: bool,
    options: Arc<GroupbyOptions>,
    apply: Option<PlanCallback<DataFrame, DataFrame>>,
    expr_arena: &mut Arena<AExpr>,
    phys_sm: &mut SlotMap<PhysNodeKey, PhysNode>,
    format_str: Option<String>,
) -> PolarsResult<PhysStream> {
    let input_schema = phys_sm[input.node].output_schema.clone();
    let lmdf = Arc::new(LateMaterializedDataFrame::default());
    let mut lp_arena = Arena::default();
    let input_lp_node = lp_arena.add(lmdf.clone().as_ir_node(input_schema));
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
        Some(crate::dispatch::build_streaming_query_executor),
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

// Given an aggregate expression returns a column expression which is to
// represent the aggregate result in the post-select.
//
// For each input to this aggregate uniq_input_names is updated to map the
// unique id of the input expressions to an input columns the aggregate
// expression expects.
//
// uniq_agg_exprs is updated with the unique id of the aggregate mapping to
// the aggregate expression and vector of unique input ids for that aggregate.
#[allow(clippy::too_many_arguments)]
fn replace_agg_uniq(
    expr: Node,
    outer_name: Option<PlSmallStr>,
    expr_merger: &NaiveExprMerger,
    _expr_cache: &mut ExprCache,
    expr_arena: &mut Arena<AExpr>,
    agg_exprs: &mut Vec<ExprIR>,
    uniq_input_names: &mut PlIndexMap<u32, PlSmallStr>,
    uniq_agg_exprs: &mut PlIndexMap<u32, (ExprIR, Vec<u32>)>,
) -> Node {
    let aexpr = expr_arena.get(expr).clone();
    let mut inputs = Vec::new();
    aexpr.inputs_rev(&mut inputs);
    inputs.reverse();

    let agg_id = expr_merger.get_uniq_id(expr).unwrap();
    let name = uniq_agg_exprs
        .entry(agg_id)
        .or_insert_with(|| {
            let mut input_ids = Vec::new();
            let input_cols = inputs
                .iter()
                .map(|input| {
                    let input_id = expr_merger.get_uniq_id(*input).unwrap();
                    input_ids.push(input_id);
                    let input_col = uniq_input_names
                        .entry(input_id)
                        .or_insert_with(unique_column_name)
                        .clone();
                    expr_arena.add(AExpr::Column(input_col))
                })
                .collect::<Vec<_>>();
            let trans_agg_node = expr_arena.add(aexpr.replace_inputs(&input_cols));

            // Add to aggregation expressions and replace with a reference to its output.
            let agg_expr = if let Some(name) = outer_name {
                ExprIR::new(trans_agg_node, OutputName::Alias(name))
            } else {
                ExprIR::new(trans_agg_node, OutputName::Alias(unique_column_name()))
            };
            agg_exprs.push(agg_expr.clone());
            (agg_expr.clone(), input_ids)
        })
        .0
        .output_name()
        .clone();
    expr_arena.add(AExpr::Column(name))
}

/// Tries to lower an expression as a 'elementwise scalar agg expression'.
///
/// Such an expression is defined as the elementwise combination of scalar
/// aggregations of elementwise combinations of the input columns / scalar
/// literals.
#[recursive]
#[allow(clippy::too_many_arguments)]
fn try_lower_elementwise_scalar_agg_expr(
    expr: Node,
    outer_name: Option<PlSmallStr>,
    expr_merger: &mut NaiveExprMerger,
    expr_cache: &mut ExprCache,
    expr_arena: &mut Arena<AExpr>,
    agg_exprs: &mut Vec<ExprIR>,
    uniq_input_names: &mut PlIndexMap<u32, PlSmallStr>,
    uniq_agg_exprs: &mut PlIndexMap<u32, (ExprIR, Vec<u32>)>,
) -> Option<Node> {
    // Helper macros to simplify (recursive) calls.
    macro_rules! lower_rec {
        ($input:expr) => {
            try_lower_elementwise_scalar_agg_expr(
                $input,
                None,
                expr_merger,
                expr_cache,
                expr_arena,
                agg_exprs,
                uniq_input_names,
                uniq_agg_exprs,
            )
        };
    }

    macro_rules! replace_agg_uniq {
        ($input:expr) => {
            replace_agg_uniq(
                $input,
                outer_name,
                expr_merger,
                expr_cache,
                expr_arena,
                agg_exprs,
                uniq_input_names,
                uniq_agg_exprs,
            )
        };
    }

    if is_input_independent(expr, expr_arena, expr_cache) {
        if expr_arena.get(expr).is_scalar(expr_arena) {
            return Some(expr);
        } else {
            let agg = IRAggExpr::Implode(expr);
            return Some(expr_arena.add(AExpr::Agg(agg)));
        }
    }

    match expr_arena.get(expr) {
        // Should be handled separately in `Eval`.
        AExpr::Element => unreachable!(),

        AExpr::StructField(_) => {
            // Reflecting StructEval expr state is not yet supported.
            None
        },

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

        #[cfg(feature = "dynamic_group_by")]
        AExpr::Rolling { .. } => None,

        AExpr::Slice { .. }
        | AExpr::Over { .. }
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

        AExpr::Eval {
            expr,
            evaluation,
            variant,
        } => {
            let (expr, evaluation, variant) = (*expr, *evaluation, *variant);
            let expr = lower_rec!(expr)?;
            Some(expr_arena.add(AExpr::Eval {
                expr,
                evaluation,
                variant,
            }))
        },

        AExpr::StructEval { expr, evaluation } => {
            // @TODO: Reflect the lowering result of `expr` into the respective
            // StructField lowering calls.
            let (expr, evaluation) = (*expr, evaluation.clone());
            let expr = lower_rec!(expr)?;

            let new_evaluation = evaluation
                .into_iter()
                .map(|i| {
                    let new_node = lower_rec!(i.node())?;
                    Some(ExprIR::new(
                        new_node,
                        OutputName::Alias(i.output_name().clone()),
                    ))
                })
                .collect::<Option<Vec<_>>>()?;

            Some(expr_arena.add(AExpr::StructEval {
                expr,
                evaluation: new_evaluation,
            }))
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

        #[cfg(feature = "bitwise")]
        AExpr::Function {
            function:
                IRFunctionExpr::Bitwise(
                    IRBitwiseFunction::And | IRBitwiseFunction::Or | IRBitwiseFunction::Xor,
                ),
            ..
        } => Some(replace_agg_uniq!(expr)),

        #[cfg(feature = "approx_unique")]
        AExpr::Function {
            function: IRFunctionExpr::ApproxNUnique,
            ..
        } => Some(replace_agg_uniq!(expr)),

        AExpr::Function {
            function:
                IRFunctionExpr::Boolean(IRBooleanFunction::Any { .. } | IRBooleanFunction::All { .. })
                | IRFunctionExpr::MinBy
                | IRFunctionExpr::MaxBy
                | IRFunctionExpr::NullCount,
            ..
        } => Some(replace_agg_uniq!(expr)),

        AExpr::AnonymousAgg { .. } => Some(replace_agg_uniq!(expr)),

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

            let mut new_node = node;
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
                IRAggExpr::Min { .. }
                | IRAggExpr::Max { .. }
                | IRAggExpr::First(_)
                | IRAggExpr::FirstNonNull(_)
                | IRAggExpr::Last(_)
                | IRAggExpr::LastNonNull(_)
                | IRAggExpr::Item { .. }
                | IRAggExpr::Mean(_)
                | IRAggExpr::Sum(_)
                | IRAggExpr::Var(..)
                | IRAggExpr::Std(..)
                | IRAggExpr::Count { .. } => Some(replace_agg_uniq!(expr)),
                IRAggExpr::NUnique(uniq_input) => {
                    let function = IRFunctionExpr::Unique(false);
                    let uniq_input_expr = ExprIR::from_node(*uniq_input, expr_arena);
                    let uniq_node = expr_arena.add(AExpr::Function {
                        input: vec![uniq_input_expr],
                        options: function.function_options(),
                        function,
                    });

                    let count = IRAggExpr::Count {
                        input: uniq_node,
                        include_nulls: true,
                    };
                    let count_node = expr_arena.add(AExpr::Agg(count));
                    expr_merger.add_expr(count_node, expr_arena);
                    Some(replace_agg_uniq!(count_node))
                },
                IRAggExpr::Median(..)
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
            let agg_id = expr_merger.get_uniq_id(expr).unwrap();
            uniq_agg_exprs.insert(agg_id, (agg_expr.clone(), Vec::new()));
            agg_exprs.push(agg_expr);
            Some(result_node)
        },
    }
}

#[allow(clippy::too_many_arguments)]
fn try_build_streaming_group_by(
    mut input: PhysStream,
    keys: &[ExprIR],
    aggs: &[ExprIR],
    maintain_order: bool,
    options: Arc<GroupbyOptions>,
    apply: Option<PlanCallback<DataFrame, DataFrame>>,
    expr_arena: &mut Arena<AExpr>,
    phys_sm: &mut SlotMap<PhysNodeKey, PhysNode>,
    expr_cache: &mut ExprCache,
    ctx: StreamingLowerIRContext,
) -> PolarsResult<Option<PhysStream>> {
    if apply.is_some() {
        return Ok(None); // TODO
    }

    #[cfg(feature = "dynamic_group_by")]
    if options.dynamic.is_some() || options.rolling.is_some() {
        return Ok(None); // TODO
    }

    if keys.is_empty() {
        return Err(
            polars_err!(ComputeError: "at least one key is required in a group_by operation"),
        );
    }

    // Not supported yet.
    let all_independent = keys
        .iter()
        .chain(aggs.iter())
        .all(|expr| is_input_independent(expr.node(), expr_arena, expr_cache));
    if all_independent {
        return Ok(None);
    }

    // Augment with row index if maintaining order.
    let row_idx_name = unique_column_name();
    let row_idx_node = expr_arena.add(AExpr::Column(row_idx_name.clone()));
    let mut agg_storage;
    let aggs = if maintain_order {
        input = build_row_idx_stream(input, row_idx_name.clone(), None, phys_sm);
        let first_agg_node = expr_arena.add(AExpr::Agg(IRAggExpr::First(row_idx_node)));
        agg_storage = aggs.to_vec();
        agg_storage.push(ExprIR::from_node(first_agg_node, expr_arena));
        &agg_storage
    } else {
        aggs
    };

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
    let mut uniq_input_names = PlIndexMap::new();
    let mut key_ids = PlIndexSet::new();
    let mut trans_agg_exprs = Vec::new();
    let mut trans_keys = Vec::new();
    let mut trans_output_exprs = Vec::new();
    for key in keys {
        let key_id = expr_merger.get_uniq_id(key.node()).unwrap();
        key_ids.insert(key_id);
        let uniq_col = uniq_input_names
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

    let mut uniq_agg_exprs = PlIndexMap::new();

    for agg in aggs {
        let Some(trans_node) = try_lower_elementwise_scalar_agg_expr(
            agg.node(),
            Some(agg.output_name().clone()),
            &mut expr_merger,
            expr_cache,
            expr_arena,
            &mut trans_agg_exprs,
            &mut uniq_input_names,
            &mut uniq_agg_exprs,
        ) else {
            return Ok(None);
        };
        let output_name = OutputName::Alias(agg.output_name().clone());
        trans_output_exprs.push(ExprIR::new(trans_node, output_name));
    }

    // We must lower the keys together with the elementwise inputs to the aggregations.
    let mut elementwise_input_needed = false;
    let mut all_keys_included_in_other_inputs = false;
    let mut elementwise_input_expr_ids = key_ids.clone();
    let mut aggs_with_elementwise_inputs = Vec::new();
    let mut other_agg_input_streams = PlIndexMap::new();
    for (_uniq_agg_id, (agg_expr, input_ids)) in uniq_agg_exprs.iter() {
        if input_ids.iter().all(|i| {
            let node = expr_merger.get_node(*i).unwrap();
            is_elementwise_rec_cached(node, expr_arena, expr_cache)
                || (is_input_independent(node, expr_arena, expr_cache)
                    && is_scalar_ae(node, expr_arena))
        }) {
            aggs_with_elementwise_inputs.push(agg_expr.clone());
            elementwise_input_expr_ids.extend(input_ids.iter().copied());
            elementwise_input_needed = true;
            continue;
        }

        // More than one non-elementwise input to this agg, unsure how to handle this.
        if input_ids.len() != 1 {
            return Ok(None);
        }

        // TODO: fancier input lowering, including filter and elementwise combinations of supported nodes,
        // and move to dedicated function.
        let input_id = input_ids[0];
        let input_node = expr_merger.get_node(input_id).unwrap();
        let input_name = uniq_input_names[&input_id].clone();
        if !other_agg_input_streams.contains_key(&input_id) {
            match expr_arena.get(input_node) {
                AExpr::Function {
                    input: uniq_input,
                    function: IRFunctionExpr::Unique(stable),
                    options: _,
                } => {
                    assert!(uniq_input.len() == 1);
                    if !is_elementwise_rec_cached(uniq_input[0].node(), expr_arena, expr_cache)
                        || maintain_order
                    {
                        // TODO: maintain order is possible but requires including the row index as a first aggregate.
                        return Ok(None);
                    }

                    // We have to uniquify the keys here to prevent name dupes since we uniquified them elsewhere.
                    // TODO: use pre-select as input here.
                    let mut gb_keys = Vec::new();
                    for key_id in &key_ids {
                        gb_keys.push(ExprIR::new(
                            expr_merger.get_node(*key_id).unwrap(),
                            OutputName::Alias(uniq_input_names[key_id].clone()),
                        ));
                    }
                    gb_keys.push(uniq_input[0].with_alias(input_name));
                    let aggs = &[];
                    let options = Arc::new(GroupbyOptions::default());
                    let maintain_order = *stable;
                    let Some(input_stream) = try_build_streaming_group_by(
                        input,
                        &gb_keys,
                        aggs,
                        maintain_order,
                        options,
                        None,
                        expr_arena,
                        phys_sm,
                        expr_cache,
                        ctx,
                    )?
                    else {
                        return Ok(None);
                    };
                    other_agg_input_streams.insert(input_id, (input_stream, Vec::new()));
                    all_keys_included_in_other_inputs = true;
                },
                _ => return Ok(None),
            }
        }

        other_agg_input_streams[&input_id].1.push(agg_expr.clone());
    }

    let mut elementwise_input_exprs = Vec::new();
    for uniq_id in elementwise_input_expr_ids {
        let name = &uniq_input_names[&uniq_id];
        let node = expr_merger.get_node(uniq_id).unwrap();
        elementwise_input_exprs.push(ExprIR::new(node, OutputName::Alias(name.clone())));
    }

    // If all inputs are input independent add a dummy column so the group sizes are correct. See #23868.
    if elementwise_input_exprs
        .iter()
        .all(|e| is_input_independent(e.node(), expr_arena, expr_cache))
    {
        elementwise_input_needed = true;
        let dummy_col_name = phys_sm[input.node].output_schema.get_at_index(0).unwrap().0;
        let dummy_col = expr_arena.add(AExpr::Column(dummy_col_name.clone()));
        elementwise_input_exprs.push(ExprIR::new(
            dummy_col,
            OutputName::ColumnLhs(dummy_col_name.clone()),
        ));
    }

    let pre_select = build_select_stream(
        input,
        &elementwise_input_exprs,
        expr_arena,
        phys_sm,
        expr_cache,
        ctx,
    )?;

    // Reconstruct the output schema of this node.
    let mut group_by_output_schema = Schema::default();
    let mut inputs = Vec::new();
    let mut key_per_input = Vec::new();
    let mut aggs_per_input = Vec::new();
    if elementwise_input_needed || !all_keys_included_in_other_inputs {
        let this_input_schema = &phys_sm[pre_select.node].output_schema;
        let exprs = [
            trans_keys.as_slice(),
            aggs_with_elementwise_inputs.as_slice(),
        ]
        .concat();
        let elementwise_out_schema =
            compute_output_schema(this_input_schema, &exprs, expr_arena).unwrap();
        group_by_output_schema.merge((*elementwise_out_schema).clone());
        inputs.push(pre_select);
        key_per_input.push(trans_keys.clone());
        aggs_per_input.push(aggs_with_elementwise_inputs);
    }
    for (_input_id, (stream, aggs)) in other_agg_input_streams {
        let this_input_schema = &phys_sm[stream.node].output_schema;
        let exprs = [trans_keys.as_slice(), aggs.as_slice()].concat();
        let this_out_schema = compute_output_schema(this_input_schema, &exprs, expr_arena).unwrap();
        group_by_output_schema.merge((*this_out_schema).clone());
        inputs.push(stream);
        key_per_input.push(trans_keys.clone());
        aggs_per_input.push(aggs);
    }
    let group_by_output_schema = Arc::new(group_by_output_schema);

    let agg_node = phys_sm.insert(PhysNode::new(
        group_by_output_schema.clone(),
        PhysNodeKind::GroupBy {
            inputs,
            key_per_input,
            aggs_per_input,
        },
    ));

    // Sort the input based on the first row index if maintaining order.
    let post_select_input = if maintain_order {
        let sort_node = phys_sm.insert(PhysNode::new(
            group_by_output_schema,
            PhysNodeKind::Sort {
                input: PhysStream::first(agg_node),
                by_column: vec![ExprIR::from_node(row_idx_node, expr_arena)],
                slice: None,
                sort_options: SortMultipleOptions::new(),
            },
        ));
        trans_output_exprs.pop(); // Remove row idx from post-select.
        PhysStream::first(sort_node)
    } else {
        PhysStream::first(agg_node)
    };

    let post_select = build_select_stream(
        post_select_input,
        &trans_output_exprs,
        expr_arena,
        phys_sm,
        expr_cache,
        ctx,
    )?;

    let out = if let Some((offset, len)) = options.slice {
        build_slice_stream(post_select, offset, len, phys_sm)
    } else {
        post_select
    };
    Ok(Some(out))
}

#[expect(clippy::too_many_arguments)]
pub fn try_build_sorted_group_by(
    input: PhysStream,
    keys: &[ExprIR],
    aggs: &[ExprIR],
    output_schema: Arc<Schema>,
    maintain_order: bool,
    options: Arc<GroupbyOptions>,
    apply: Option<PlanCallback<DataFrame, DataFrame>>,
    expr_arena: &mut Arena<AExpr>,
    phys_sm: &mut SlotMap<PhysNodeKey, PhysNode>,
    expr_cache: &mut ExprCache,
    ctx: StreamingLowerIRContext,
    are_keys_sorted: bool,
) -> PolarsResult<Option<PhysStream>> {
    let input_schema = phys_sm[input.node].output_schema.as_ref();

    if keys.is_empty()
        || apply.is_some()
        || options.is_rolling()
        || options.is_dynamic()
        || (!are_keys_sorted && maintain_order)
        || keys.iter().any(|k| {
            k.dtype(input_schema, expr_arena)
                .is_ok_and(|dtype| dtype.contains_unknown())
        })
    {
        return Ok(None);
    }

    let mut input = input;
    let mut input_column = unique_column_name();
    let mut projected = false;
    let mut row_encoded: Option<Vec<Field>> = None;

    if keys.len() > 1 || keys[0].dtype(input_schema, expr_arena)?.is_nested() {
        let key_fields = keys
            .iter()
            .map(|k| k.field(input_schema, expr_arena))
            .collect::<PolarsResult<Vec<_>>>()?;
        let expr = AExprBuilder::function(
            keys.to_vec(),
            IRFunctionExpr::RowEncode(
                key_fields.iter().map(|k| k.dtype().clone()).collect(),
                RowEncodingVariant::Ordered {
                    descending: None,
                    nulls_last: None,
                    broadcast_nulls: None,
                },
            ),
            expr_arena,
        )
        .expr_ir(input_column.clone());
        input = build_hstack_stream(input, &[expr], expr_arena, phys_sm, expr_cache, ctx)?;
        projected = true;
        row_encoded = Some(key_fields);
    } else if !matches!(expr_arena.get(keys[0].node()), AExpr::Column(c) if c == keys[0].output_name())
    {
        input = build_hstack_stream(
            input,
            &[keys[0].with_alias(input_column.clone())],
            expr_arena,
            phys_sm,
            expr_cache,
            ctx,
        )?;
        projected = true;
    } else {
        input_column = keys[0].output_name().clone();
    }

    let key = AExprBuilder::col(input_column.clone(), expr_arena).expr_ir(input_column.clone());

    let schema = phys_sm[input.node].output_schema.clone();
    if !are_keys_sorted {
        let row_idx_name = unique_column_name();
        input = build_row_idx_stream(input, row_idx_name.clone(), None, phys_sm);

        let row_idx_expr =
            AExprBuilder::col(row_idx_name.clone(), expr_arena).expr_ir(row_idx_name.clone());

        input = PhysStream::first(phys_sm.insert(PhysNode {
            output_schema: phys_sm[input.node].output_schema.clone(),
            kind: PhysNodeKind::Sort {
                input,
                by_column: vec![key, row_idx_expr],
                slice: None,
                sort_options: SortMultipleOptions::default(),
            },
        }));
    }

    let mut gb_output_schema = Schema::with_capacity(aggs.len() + 1);
    gb_output_schema.insert(
        input_column.clone(),
        schema.get(input_column.as_str()).unwrap().clone(),
    );
    for agg in aggs {
        let field = agg.field(schema.as_ref(), expr_arena)?;
        let dtype = if agg.is_scalar(expr_arena) {
            field.dtype
        } else {
            field.dtype.implode()
        };
        gb_output_schema.insert(field.name, dtype);
    }
    input = PhysStream::first(
        phys_sm.insert(PhysNode {
            output_schema: Arc::new(gb_output_schema.clone()),
            kind: PhysNodeKind::SortedGroupBy {
                input,
                key: input_column.clone(),
                aggs: aggs.to_vec(),
                slice: options
                    .slice
                    .filter(|(o, _)| *o >= 0)
                    .map(|(o, l)| (o as IdxSize, l as IdxSize)),
            },
        }),
    );
    if let Some((offset, length)) = options.slice.as_ref().filter(|(o, _)| *o < 0) {
        input = build_slice_stream(input, *offset, *length, phys_sm);
    }

    if projected {
        if let Some(key_fields) = row_encoded {
            let expr =
                AExprBuilder::col(input_column.clone(), expr_arena).expr_ir(input_column.clone());
            let expr = AExprBuilder::function(
                vec![expr],
                IRFunctionExpr::RowDecode(
                    key_fields,
                    RowEncodingVariant::Ordered {
                        descending: None,
                        nulls_last: None,
                        broadcast_nulls: None,
                    },
                ),
                expr_arena,
            )
            .expr_ir(input_column.clone());
            input = build_hstack_stream(input, &[expr], expr_arena, phys_sm, expr_cache, ctx)?;

            // Unnest the row encoded columns.
            input = PhysStream::first(phys_sm.insert(PhysNode {
                output_schema: output_schema.clone(),
                kind: PhysNodeKind::Map {
                    input,
                    map: Arc::new(move |df: DataFrame| df.unnest([input_column.clone()], None))
                        as _,
                    format_str: ctx.prepare_visualization.then(|| "UNNEST".to_string()),
                },
            }));

            let exprs = output_schema
                .iter_names()
                .map(|name| AExprBuilder::col(name.clone(), expr_arena).expr_ir(name.clone()))
                .collect::<Vec<_>>();
            input = build_select_stream(input, &exprs, expr_arena, phys_sm, expr_cache, ctx)?;
        } else {
            let exprs = std::iter::once(input_column)
                .map(|name| (name, output_schema.get_at_index(0).unwrap().0.clone()))
                .chain(
                    output_schema
                        .iter_names_cloned()
                        .skip(1)
                        .map(|name| (name.clone(), name.clone())),
                )
                .map(|(col_name, out_name)| {
                    AExprBuilder::col(col_name, expr_arena).expr_ir(out_name)
                })
                .collect::<Vec<_>>();
            input = build_select_stream(input, &exprs, expr_arena, phys_sm, expr_cache, ctx)?;
        }
    }

    Ok(Some(input))
}

#[allow(clippy::too_many_arguments)]
pub fn build_group_by_stream(
    input: PhysStream,
    keys: &[ExprIR],
    aggs: &[ExprIR],
    output_schema: Arc<Schema>,
    maintain_order: bool,
    options: Arc<GroupbyOptions>,
    apply: Option<PlanCallback<DataFrame, DataFrame>>,
    expr_arena: &mut Arena<AExpr>,
    phys_sm: &mut SlotMap<PhysNodeKey, PhysNode>,
    expr_cache: &mut ExprCache,
    ctx: StreamingLowerIRContext,
    are_keys_sorted: bool,
) -> PolarsResult<PhysStream> {
    #[cfg(feature = "dynamic_group_by")]
    if let Some(rolling_options) = options.as_ref().rolling.as_ref()
        && keys.is_empty()
        && apply.is_none()
    {
        let mut input = PhysStream::first(
            phys_sm.insert(PhysNode::new(
                output_schema.clone(),
                PhysNodeKind::RollingGroupBy {
                    input,
                    index_column: rolling_options.index_column.clone(),
                    period: rolling_options.period,
                    offset: rolling_options.offset,
                    closed: rolling_options.closed_window,
                    slice: options
                        .slice
                        .filter(|(o, _)| *o >= 0)
                        .map(|(o, l)| (o as IdxSize, l as IdxSize)),
                    aggs: aggs.to_vec(),
                },
            )),
        );
        if let Some((offset, length)) = options.slice.as_ref().filter(|(o, _)| *o < 0) {
            input = build_slice_stream(input, *offset, *length, phys_sm);
        }
        return Ok(input);
    } else if let Some(dynamic_options) = options.as_ref().dynamic.as_ref()
        && keys.is_empty()
        && apply.is_none()
    {
        let mut input = PhysStream::first(
            phys_sm.insert(PhysNode::new(
                output_schema.clone(),
                PhysNodeKind::DynamicGroupBy {
                    input,
                    options: dynamic_options.clone(),
                    aggs: aggs.to_vec(),
                    slice: options
                        .slice
                        .filter(|(o, _)| *o >= 0)
                        .map(|(o, l)| (o as IdxSize, l as IdxSize)),
                },
            )),
        );
        if let Some((offset, length)) = options.slice.as_ref().filter(|(o, _)| *o < 0) {
            input = build_slice_stream(input, *offset, *length, phys_sm);
        }
        return Ok(input);
    }

    if (are_keys_sorted || std::env::var("POLARS_FORCE_SORTED_GROUP_BY").is_ok_and(|v| v == "1"))
        && let Some(stream) = try_build_sorted_group_by(
            input,
            keys,
            aggs,
            output_schema.clone(),
            maintain_order,
            options.clone(),
            apply.clone(),
            expr_arena,
            phys_sm,
            expr_cache,
            ctx,
            are_keys_sorted,
        )?
    {
        Ok(stream)
    } else if let Some(stream) = try_build_streaming_group_by(
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
    )? {
        Ok(stream)
    } else {
        let format_str = ctx.prepare_visualization.then(|| {
            let mut buffer = String::new();
            write_group_by(
                &mut buffer,
                0,
                expr_arena,
                keys,
                aggs,
                apply.as_ref(),
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
