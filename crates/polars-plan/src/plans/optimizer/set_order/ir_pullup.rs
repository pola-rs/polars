use std::sync::Arc;

use polars_core::frame::UniqueKeepStrategy;
use polars_core::prelude::PlHashMap;
use polars_ops::frame::MaintainOrderJoin;
use polars_utils::arena::{Arena, Node};
use polars_utils::idx_vec::UnitVec;
use polars_utils::unique_id::UniqueId;

use super::expr_pullup::is_output_ordered;
use crate::dsl::SinkTypeIR;
use crate::plans::{AExpr, IR};

pub(super) fn pullup_orders(
    leaves: &[Node],
    ir_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
    outputs: &mut PlHashMap<Node, Vec<(Node, usize)>>,
    orders: &mut PlHashMap<Node, UnitVec<bool>>,
    cache_proxy: &PlHashMap<UniqueId, Vec<Node>>,
) {
    let mut hits: PlHashMap<Node, usize> = PlHashMap::default();
    let mut stack = Vec::new();

    for leaf in leaves {
        stack.extend(outputs[leaf].iter().map(|v| v.0));
    }

    while let Some(node) = stack.pop() {
        // @Hack. The IR creates caches for every path at the moment. That is super hacky. So is
        // this, but we need to work around it.
        let node = match ir_arena.get(node) {
            IR::Cache { id, .. } => cache_proxy.get(id).unwrap()[0],
            _ => node,
        };

        let hits = hits.entry(node).or_default();
        *hits += 1;
        if *hits < orders[&node].len() {
            continue;
        }

        let node_outputs = &outputs[&node];
        let mut ir = ir_arena.get_mut(node);

        let inputs_ordered = orders.get_mut(&node).unwrap();

        macro_rules! set_unordered_output {
            () => {
                for (output, edge) in node_outputs {
                    orders.get_mut(output).unwrap()[*edge] = false;
                }
            };
        }

        // Pullup simplification rules.
        use MaintainOrderJoin as MOJ;
        match ir {
            IR::Sort { sort_options, .. } => {
                // Unordered -> _     ==>    maintain_order=false
                sort_options.maintain_order &= inputs_ordered[0];
            },
            IR::GroupBy {
                keys,
                maintain_order,
                ..
            } => {
                if !inputs_ordered[0] {
                    // Unordered -> _
                    //   to
                    // maintain_order = false
                    // and
                    // Unordered -> Unordered

                    let keys_produce_order = keys
                        .iter()
                        .any(|k| is_output_ordered(expr_arena.get(k.node()), expr_arena, false));
                    if !keys_produce_order {
                        *maintain_order = false;
                        set_unordered_output!();
                    }
                }
            },
            IR::Sink { input: _, payload } => {
                if !inputs_ordered[0] {
                    // Set maintain order to false if input is unordered
                    match payload {
                        SinkTypeIR::Memory => {},
                        SinkTypeIR::File(s) => s.sink_options.maintain_order = false,
                        SinkTypeIR::Partition(s) => s.sink_options.maintain_order = false,
                        SinkTypeIR::Callback(s) => s.maintain_order = false,
                    }
                }
            },
            IR::Join { options, .. } => {
                let left_unordered = !inputs_ordered[0];
                let right_unordered = !inputs_ordered[1];

                let maintain_order = options.args.maintain_order;

                if (left_unordered && matches!(maintain_order, MOJ::Left | MOJ::RightLeft))
                    || (right_unordered && matches!(maintain_order, MOJ::Right | MOJ::LeftRight))
                {
                    // If we are maintaining order of a side, but that input has no guaranteed order,
                    // remove the maintain ordering from that side.

                    let mut new_options = options.as_ref().clone();
                    new_options.args.maintain_order = match maintain_order {
                        _ if left_unordered && right_unordered => MOJ::None,
                        MOJ::Left if left_unordered => MOJ::None,
                        MOJ::RightLeft if left_unordered => MOJ::Right,
                        MOJ::Right if right_unordered => MOJ::None,
                        MOJ::LeftRight if right_unordered => MOJ::Left,
                        _ => unreachable!(),
                    };

                    if matches!(new_options.args.maintain_order, MOJ::None) {
                        set_unordered_output!();
                    }
                    *options = Arc::new(new_options);
                }
            },
            IR::Distinct { input: _, options } => {
                if !inputs_ordered[0] {
                    options.maintain_order = false;
                    if options.keep_strategy != UniqueKeepStrategy::None {
                        options.keep_strategy = UniqueKeepStrategy::Any;
                    }
                    set_unordered_output!();
                }
            },

            #[cfg(feature = "python")]
            IR::PythonScan { .. } => {},
            IR::Scan { .. } | IR::DataFrameScan { .. } => {},
            #[cfg(feature = "merge_sorted")]
            IR::MergeSorted { .. } => {
                // An input being unordered is technically valid as it is possible for all values
                // to be the same in which case the rows are sorted.
            },
            IR::Union { options, .. } => {
                // Even if the inputs are unordered. The output still has an order given by the
                // order of the inputs.

                if !options.maintain_order && !inputs_ordered.iter().any(|i| *i) {
                    set_unordered_output!();
                }
            },
            IR::MapFunction { input: _, function } => {
                if !function.is_order_producing(inputs_ordered[0]) {
                    set_unordered_output!();
                }
            },

            IR::Select { expr, .. } => {
                if !expr.iter().any(|e| {
                    is_output_ordered(expr_arena.get(e.node()), expr_arena, inputs_ordered[0])
                }) {
                    set_unordered_output!();
                }
            },

            IR::HStack { input, .. } => {
                let input = *input;
                let input_schema = ir_arena.get(input).schema(ir_arena).as_ref().clone();
                ir = ir_arena.get_mut(node);
                let IR::HStack { exprs, .. } = ir else {
                    unreachable!()
                };

                let has_any_ordered_expression = exprs.iter().any(|e| {
                    is_output_ordered(expr_arena.get(e.node()), expr_arena, inputs_ordered[0])
                });
                let only_overwrites_existing_columns = exprs
                    .iter()
                    .filter(|e| input_schema.contains(e.output_name()))
                    .count()
                    == input_schema.len();
                let is_output_unordered =
                    !has_any_ordered_expression && only_overwrites_existing_columns;

                if is_output_unordered {
                    set_unordered_output!();
                }
            },

            IR::Filter {
                input: _,
                predicate: _,
            } => {
                if !inputs_ordered[0] {
                    // @Performance:
                    // This can be optimized to IR::Slice {
                    //     input,
                    //     offset: 0,
                    //     length: predicate.sum()
                    // }
                    set_unordered_output!();
                }
            },

            IR::Cache { .. }
            | IR::SimpleProjection { .. }
            | IR::Slice { .. }
            | IR::HConcat { .. }
            | IR::ExtContext { .. } => {
                if !inputs_ordered.iter().any(|i| *i) {
                    set_unordered_output!();
                }
            },

            IR::SinkMultiple { .. } | IR::Invalid => unreachable!(),
        }

        stack.extend(node_outputs.iter().map(|v| v.0));
    }
}
