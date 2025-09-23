use std::sync::Arc;

use polars_core::frame::UniqueKeepStrategy;
use polars_core::prelude::PlHashMap;
#[cfg(feature = "asof_join")]
use polars_ops::frame::JoinType;
use polars_ops::frame::MaintainOrderJoin;
use polars_utils::arena::{Arena, Node};
use polars_utils::idx_vec::UnitVec;
use polars_utils::unique_id::UniqueId;

use super::expr_pushdown::{adjust_for_with_columns_context, get_frame_observing, zip};
use crate::dsl::{PartitionVariantIR, SinkTypeIR, UnionOptions};
use crate::plans::set_order::expr_pushdown::FrameOrderObserved;
use crate::plans::{AExpr, IR, is_scalar_ae};

pub(super) fn pushdown_orders(
    roots: &[Node],
    ir_arena: &mut Arena<IR>,
    expr_arena: &Arena<AExpr>,
    outputs: &mut PlHashMap<Node, Vec<(Node, usize)>>,
    cache_proxy: &PlHashMap<UniqueId, Vec<Node>>,
) -> PlHashMap<Node, UnitVec<bool>> {
    let mut orders: PlHashMap<Node, UnitVec<bool>> = PlHashMap::default();
    let mut node_hits: PlHashMap<Node, usize> = PlHashMap::default();
    let mut stack = Vec::new();

    stack.extend(roots.iter().copied());

    while let Some(node) = stack.pop() {
        // @Hack. The IR creates caches for every path at the moment. That is super hacky. So is
        // this, but we need to work around it.
        let node = match ir_arena.get(node) {
            IR::Cache { id, .. } => cache_proxy.get(id).unwrap()[0],
            _ => node,
        };

        debug_assert!(!orders.contains_key(&node));

        let node_outputs = &outputs[&node];
        let hits = node_hits.entry(node).or_default();
        *hits += 1;
        if *hits < node_outputs.len() {
            continue;
        }

        let all_outputs_unordered = !node_outputs
            .iter()
            .any(|(to_node, to_input_idx)| orders[to_node][*to_input_idx]);

        // Pushdown simplification rules.
        let mut ir = ir_arena.get_mut(node);
        use MaintainOrderJoin as MOJ;
        let node_ordering: UnitVec<bool> = match ir {
            IR::Cache { .. } if all_outputs_unordered => [false].into(),
            IR::Cache { .. } => [true].into(),
            IR::Sort {
                input,
                slice,
                sort_options: _,
                ..
            } if slice.is_none() && all_outputs_unordered => {
                // _ -> Unordered
                //
                // Remove sort.
                let input = *input;
                _ = ir_arena.take(node);

                let node_outputs = outputs.remove(&node).unwrap();
                for (to_node, to_input_idx) in node_outputs {
                    *ir_arena
                        .get_mut(to_node)
                        .inputs_mut()
                        .nth(to_input_idx)
                        .unwrap() = input;
                    outputs
                        .get_mut(&input)
                        .unwrap()
                        .push((to_node, to_input_idx));
                }
                outputs.get_mut(&input).unwrap().retain(|(n, _)| *n != node);

                stack.push(input);
                continue;
            },
            IR::Sort {
                by_column,
                sort_options,
                ..
            } => {
                let is_order_observing = sort_options.maintain_order || {
                    adjust_for_with_columns_context(zip(by_column
                        .iter()
                        .map(|e| get_frame_observing(expr_arena.get(e.node()), expr_arena))))
                    .is_err()
                };
                [is_order_observing].into()
            },
            IR::GroupBy {
                keys,
                aggs,
                maintain_order,
                apply,
                options,
                ..
            } => {
                *maintain_order &= !all_outputs_unordered;

                let is_order_observing = apply.is_some()
                    || options.is_dynamic()
                    || options.is_rolling()
                    || *maintain_order
                    || {
                        // _ -> Unordered
                        //   to
                        // maintain_order = false
                        // and
                        // Unordered -> Unordered (if no order sensitive expressions)

                        let expr_observing = adjust_for_with_columns_context(zip(keys
                            .iter()
                            .chain(aggs.iter())
                            .map(|e| get_frame_observing(expr_arena.get(e.node()), expr_arena))))
                        .is_err();

                        expr_observing
                            // The auto-implode is also other sensitive.
                            || aggs.iter().any(|agg| !is_scalar_ae(agg.node(), expr_arena))
                    };
                [is_order_observing].into()
            },
            #[cfg(feature = "merge_sorted")]
            IR::MergeSorted {
                input_left,
                input_right,
                ..
            } => {
                if all_outputs_unordered {
                    // MergeSorted
                    // (_, _) -> Unordered
                    //   to
                    // UnorderedUnion([left, right])

                    *ir = IR::Union {
                        inputs: vec![*input_left, *input_right],
                        options: UnionOptions {
                            maintain_order: false,
                            ..Default::default()
                        },
                    };
                    [false; 2].into()
                } else {
                    [true; 2].into()
                }
            },
            #[cfg(feature = "asof_join")]
            IR::Join { options, .. } if matches!(options.args.how, JoinType::AsOf(_)) => {
                [true; 2].into()
            },
            IR::Join {
                input_left: _,
                input_right: _,
                schema: _,
                left_on: _,
                right_on: _,
                options,
            } if all_outputs_unordered => {
                // If the join maintains order, but the output has undefined order. Remove the
                // ordering.
                if !matches!(options.args.maintain_order, MOJ::None) {
                    let mut new_options = options.as_ref().clone();
                    new_options.args.maintain_order = MOJ::None;
                    *options = Arc::new(new_options);
                }

                // Join `on` expressions are elementwise so we don't have to inspect the order
                // sensitivity.
                [false, false].into()
            },
            IR::Join {
                input_left: _,
                input_right: _,
                schema: _,
                left_on: _,
                right_on: _,
                options,
            } => {
                use MaintainOrderJoin as M;
                let left_input = matches!(
                    options.args.maintain_order,
                    M::Left | M::LeftRight | M::RightLeft
                );
                let right_input = matches!(
                    options.args.maintain_order,
                    M::Right | M::RightLeft | M::LeftRight
                );

                [left_input, right_input].into()
            },
            IR::Distinct { input: _, options } => {
                options.maintain_order &= !all_outputs_unordered;

                let is_order_observing = options.maintain_order
                    || matches!(
                        options.keep_strategy,
                        UniqueKeepStrategy::First | UniqueKeepStrategy::Last
                    );
                [is_order_observing].into()
            },
            IR::MapFunction { input: _, function } => {
                let is_order_observing = (function.has_equal_order() && !all_outputs_unordered)
                    || !function.is_input_order_agnostic();
                [is_order_observing].into()
            },
            IR::SimpleProjection { .. } => [!all_outputs_unordered].into(),
            IR::Slice { .. } => [true].into(),
            IR::HStack { input, exprs, .. } => {
                let input = *input;
                let mut observing = zip(exprs
                    .iter()
                    .map(|e| get_frame_observing(expr_arena.get(e.node()), expr_arena)));

                let input_schema = ir_arena.get(input).schema(ir_arena).as_ref().clone();
                ir = ir_arena.get_mut(node);
                let IR::HStack { exprs, .. } = ir else {
                    unreachable!()
                };

                let mut hits = 0;
                for expr in exprs {
                    hits += usize::from(input_schema.contains(expr.output_name()));
                }

                if hits < input_schema.len() {
                    observing = adjust_for_with_columns_context(observing);
                }

                let is_order_observing = match observing {
                    Ok(o) => o.has_frame_ordering() && !all_outputs_unordered,
                    Err(FrameOrderObserved) => true,
                };
                [is_order_observing].into()
            },
            IR::Select { expr: exprs, .. } => {
                let observing = zip(exprs
                    .iter()
                    .map(|e| get_frame_observing(expr_arena.get(e.node()), expr_arena)));
                let is_order_observing = match observing {
                    Ok(o) => o.has_frame_ordering() && !all_outputs_unordered,
                    Err(FrameOrderObserved) => true,
                };
                [is_order_observing].into()
            },

            IR::Filter {
                input: _,
                predicate,
            } => {
                let observing = adjust_for_with_columns_context(get_frame_observing(
                    expr_arena.get(predicate.node()),
                    expr_arena,
                ));
                let is_order_observing = match observing {
                    Ok(o) => o.has_frame_ordering() && !all_outputs_unordered,
                    Err(FrameOrderObserved) => true,
                };
                [is_order_observing].into()
            },

            IR::Union { inputs, options } => {
                if options.slice.is_none() && all_outputs_unordered {
                    options.maintain_order = false;
                }
                std::iter::repeat_n(
                    options.slice.is_some() || options.maintain_order,
                    inputs.len(),
                )
                .collect()
            },

            IR::HConcat { inputs, .. } => std::iter::repeat_n(true, inputs.len()).collect(),

            #[cfg(feature = "python")]
            IR::PythonScan { .. } => UnitVec::new(),

            IR::Sink { payload, .. } => {
                let is_order_observing = payload.maintain_order()
                    || match payload {
                        SinkTypeIR::Memory => false,
                        SinkTypeIR::File(_) => false,
                        SinkTypeIR::Callback(_) => false,
                        SinkTypeIR::Partition(p) => match &p.variant {
                            PartitionVariantIR::MaxSize(_) => false,
                            PartitionVariantIR::Parted { .. } => true,
                            PartitionVariantIR::ByKey { key_exprs, .. } => {
                                adjust_for_with_columns_context(zip(key_exprs.iter().map(|e| {
                                    get_frame_observing(expr_arena.get(e.node()), expr_arena)
                                })))
                                .is_err()
                            },
                        },
                    };

                [is_order_observing].into()
            },
            IR::Scan { .. } | IR::DataFrameScan { .. } => UnitVec::new(),

            IR::ExtContext { contexts, .. } => {
                // This node is nonsense. Just do the most conservative thing you can.
                std::iter::repeat_n(true, contexts.len() + 1).collect()
            },

            IR::SinkMultiple { .. } | IR::Invalid => unreachable!(),
        };

        let prev_value = orders.insert(node, node_ordering);
        assert!(prev_value.is_none());

        stack.extend(ir.inputs());
    }

    orders
}
