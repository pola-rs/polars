//! Pass to obtain and optimize using exhaustive row-order information.
//!
//! This pass attaches order information to all the IR node input and output ports.
//!
//! The pass performs two passes over the IR graph. First, it assigns and pushes ordering down from
//! the sinks to the leaves. Second, it pulls those orderings back up from the leaves to the sinks.
//! The two passes weaken order guarantees and simplify IR nodes where possible.
//!
//! When the two passes are done, we are left with a map from all the nodes to `PortOrder` which
//! contains the input and output port ordering information.

use std::sync::Arc;

use polars_core::frame::UniqueKeepStrategy;
use polars_core::prelude::PlHashMap;
use polars_ops::frame::{JoinType, MaintainOrderJoin};
use polars_utils::arena::{Arena, Node};
use polars_utils::idx_vec::UnitVec;
use polars_utils::unique_id::UniqueId;
use polars_utils::unitvec;

use super::IR;
use crate::dsl::{SinkTypeIR, UnionOptions};
use crate::plans::ir::inputs::Inputs;
use crate::plans::{AExpr, is_order_sensitive_amortized, is_scalar_ae};

#[derive(Debug, Clone, Copy)]
pub enum OutputOrder {
    Ordered,
    Random,
    /// Output ordering preserves that of the input.
    PreservesInput,
}

/// The ordering of the input and output ports of an IR node.
///
/// This gives information about how row ordering may be received, observed and passed an IR node.
#[derive(Debug, Clone)]
pub struct PortOrder {
    /// Indicates whether the ordering of each input is observed.
    pub input_order_observe: UnitVec<bool>,
    pub output_orders: UnitVec<OutputOrder>,
}

impl PortOrder {
    pub fn new(
        input_order_observe: impl IntoIterator<Item = bool>,
        output_orders: impl IntoIterator<Item = OutputOrder>,
    ) -> Self {
        Self {
            input_order_observe: input_order_observe.into_iter().collect(),
            output_orders: output_orders.into_iter().collect(),
        }
    }

    fn set_all_outputs_unordered(&mut self) {
        self.output_orders.iter_mut().for_each(|o| {
            use OutputOrder as O;
            match o {
                O::Ordered => {
                    // Should not be setting ordered output to unordered
                    if cfg!(debug_assertions) {
                        panic!()
                    }
                },
                O::Random | O::PreservesInput => *o = O::Random,
            }
        });
    }

    /// # Panics
    /// Panics if `self.input_order_observe.len() != N`.
    fn input_order_observe_arr<const N: usize>(&self) -> [bool; N] {
        <[_; N]>::try_from(self.input_order_observe.as_slice()).unwrap_or_else(|_| {
            panic!(
                "have {} inputs but expected {} inputs",
                self.input_order_observe.len(),
                N
            )
        })
    }
}

fn pushdown_orders(
    roots: &[Node],
    ir_arena: &mut Arena<IR>,
    expr_arena: &Arena<AExpr>,
    outputs: &mut PlHashMap<Node, UnitVec<Node>>,
    cache_proxy: &PlHashMap<UniqueId, Vec<Node>>,
) -> PlHashMap<Node, PortOrder> {
    let mut orders: PlHashMap<Node, PortOrder> = PlHashMap::default();
    let mut node_hits: PlHashMap<Node, Vec<(usize, Node)>> = PlHashMap::default();
    let mut aexpr_stack = Vec::new();
    let mut stack = Vec::new();

    stack.extend(roots.iter().map(|n| (*n, None)));

    while let Some((node, outgoing)) = stack.pop() {
        // @Hack. The IR creates caches for every path at the moment. That is super hacky. So is
        // this, but we need to work around it.
        let node = match ir_arena.get(node) {
            IR::Cache { id, .. } => cache_proxy.get(id).unwrap()[0],
            _ => node,
        };

        debug_assert!(!orders.contains_key(&node));

        let node_outputs = &outputs[&node];
        let hits = node_hits.entry(node).or_default();
        if let Some(outgoing) = outgoing {
            hits.push(outgoing);
            if hits.len() < node_outputs.len() {
                continue;
            }
        }

        let any_receiver_observes_order = hits.iter().any(|(to_input_idx, to_node)| {
            orders.get_mut(to_node).unwrap().input_order_observe[*to_input_idx]
        });

        // Pushdown simplification rules.
        let ir = ir_arena.get_mut(node);

        use PortOrder as P;
        let mut node_ordering: PortOrder = match ir {
            IR::Cache { .. } => {
                P::new([any_receiver_observes_order], [OutputOrder::PreservesInput])
            },
            IR::Sort { input, slice, .. } if slice.is_none() && !any_receiver_observes_order => {
                // _ -> Unordered
                //
                // Remove sort.
                let input = *input;
                let (to_input_idx, to_node) = outgoing.unwrap();
                *ir_arena
                    .get_mut(to_node)
                    .inputs_mut()
                    .nth(to_input_idx)
                    .unwrap() = input;

                // @Performance: Linear search
                *outputs
                    .get_mut(&input)
                    .unwrap()
                    .iter_mut()
                    .find(|o| **o == node)
                    .unwrap() = to_node;

                stack.push((input, outgoing));
                continue;
            },
            IR::Sort {
                by_column,
                sort_options,
                slice,
                ..
            } => {
                if !any_receiver_observes_order {
                    sort_options.maintain_order = false;
                }

                let observes_input_order = sort_options.maintain_order || slice.is_some() || {
                    let mut has_order_sensitive = false;
                    for e in by_column {
                        let aexpr = expr_arena.get(e.node());
                        has_order_sensitive |=
                            is_order_sensitive_amortized(aexpr, expr_arena, &mut aexpr_stack);
                    }

                    has_order_sensitive
                };

                P::new([observes_input_order], [OutputOrder::Ordered])
            },
            IR::GroupBy {
                keys,
                aggs,
                maintain_order,
                apply,
                options,
                ..
            } => {
                if !any_receiver_observes_order {
                    *maintain_order = false;
                }

                let (observes_input_order, output_order) = if *maintain_order
                    || apply.is_some()
                    || options.is_dynamic()
                    || options.is_rolling()
                {
                    (true, OutputOrder::PreservesInput)
                } else {
                    let mut has_order_sensitive = false;
                    for e in keys.iter().chain(aggs.iter()) {
                        let aexpr = expr_arena.get(e.node());
                        has_order_sensitive |=
                            is_order_sensitive_amortized(aexpr, expr_arena, &mut aexpr_stack);
                    }

                    // The auto-implode is also other sensitive.
                    has_order_sensitive |=
                        aggs.iter().any(|agg| !is_scalar_ae(agg.node(), expr_arena));

                    (has_order_sensitive, OutputOrder::Random)
                };

                P::new([observes_input_order], [output_order])
            },
            #[cfg(feature = "merge_sorted")]
            IR::MergeSorted {
                input_left,
                input_right,
                ..
            } => {
                if !any_receiver_observes_order {
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

                    P::new([false, false], [OutputOrder::Random])
                } else {
                    // Output order is derived
                    P::new([true, true], [OutputOrder::PreservesInput])
                }
            },
            #[cfg(feature = "asof_join")]
            IR::Join { options, .. } if matches!(options.args.how, JoinType::AsOf(_)) => {
                // Output order is derived
                P::new([true, true], [OutputOrder::PreservesInput])
            },
            IR::Join {
                input_left: _,
                input_right: _,
                schema: _,
                left_on,
                right_on,
                options,
            } if !any_receiver_observes_order => {
                // If the join maintains order, but the output has undefined order. Remove the
                // ordering.
                use {MaintainOrderJoin as MOJ, PortOrder as P};
                if !matches!(options.args.maintain_order, MOJ::None) {
                    Arc::make_mut(options).args.maintain_order = MOJ::None;
                }
                let mut input_order_observe = [false, false];

                // If either side does not need to maintain order, don't maintain the old on that
                // side.
                for (i, on) in [left_on, right_on].iter().enumerate() {
                    let mut has_order_sensitive = false;
                    for e in on.iter() {
                        let aexpr = expr_arena.get(e.node());
                        has_order_sensitive |=
                            is_order_sensitive_amortized(aexpr, expr_arena, &mut aexpr_stack);
                    }

                    if has_order_sensitive {
                        input_order_observe[i] = true
                    }
                }

                P::new(input_order_observe, [OutputOrder::Random])
            },
            IR::Join {
                input_left: _,
                input_right: _,
                schema: _,
                left_on,
                right_on,
                options,
            } => {
                assert!(any_receiver_observes_order);
                let mut left_has_order_sensitive = false;
                let mut right_has_order_sensitive = false;

                for e in left_on {
                    let aexpr = expr_arena.get(e.node());
                    left_has_order_sensitive |=
                        is_order_sensitive_amortized(aexpr, expr_arena, &mut aexpr_stack);
                }
                for e in right_on {
                    let aexpr = expr_arena.get(e.node());
                    right_has_order_sensitive |=
                        is_order_sensitive_amortized(aexpr, expr_arena, &mut aexpr_stack);
                }

                use MaintainOrderJoin as M;

                let output_order = if matches!(options.args.maintain_order, M::None) {
                    OutputOrder::Random
                } else {
                    OutputOrder::PreservesInput
                };

                let observe_input_order = !matches!(options.args.maintain_order, M::None)
                    || options.args.slice.is_some()
                    || left_has_order_sensitive
                    || right_has_order_sensitive;

                P::new([observe_input_order, observe_input_order], [output_order])
            },
            IR::Distinct { input: _, options } => {
                use UniqueKeepStrategy as UKS;

                if !any_receiver_observes_order
                    && matches!(options.keep_strategy, UKS::Any | UKS::None)
                {
                    options.maintain_order = false;
                }

                let observe_input_order = options.maintain_order
                    || match options.keep_strategy {
                        UKS::First | UKS::Last => true,
                        UKS::Any | UKS::None => false,
                    };

                let output_order = if options.maintain_order {
                    OutputOrder::PreservesInput
                } else {
                    OutputOrder::Random
                };

                P::new([observe_input_order], [output_order])
            },
            IR::MapFunction { input: _, function } => {
                let observe_input_order =
                    any_receiver_observes_order || function.observes_input_order();

                let output_order = function.output_order();

                P::new([observe_input_order], [output_order])
            },
            IR::SimpleProjection { .. } => {
                P::new([any_receiver_observes_order], [OutputOrder::PreservesInput])
            },
            IR::Slice { .. } => P::new([true], [OutputOrder::PreservesInput]),
            IR::HStack { exprs, .. } => {
                let mut has_order_sensitive = false;
                for e in exprs {
                    let aexpr = expr_arena.get(e.node());
                    has_order_sensitive |=
                        is_order_sensitive_amortized(aexpr, expr_arena, &mut aexpr_stack);
                }

                let observe_input_order = any_receiver_observes_order || has_order_sensitive;

                // FIXME: overly strict; output order dependent on projection expression output order
                let output_order = if has_order_sensitive {
                    OutputOrder::Ordered
                } else {
                    OutputOrder::PreservesInput
                };

                P::new([observe_input_order], [output_order])
            },
            IR::Select { expr: exprs, .. } => {
                let mut has_order_sensitive = false;

                for e in exprs {
                    let aexpr = expr_arena.get(e.node());
                    has_order_sensitive |=
                        is_order_sensitive_amortized(aexpr, expr_arena, &mut aexpr_stack);
                }

                let observe_input_order = any_receiver_observes_order || has_order_sensitive;
                // FIXME: overly strict; output order dependent on projection expression output order
                let output_order = if has_order_sensitive {
                    OutputOrder::Ordered
                } else {
                    OutputOrder::PreservesInput
                };

                P::new([observe_input_order], [output_order])
            },

            IR::Filter {
                input: _,
                predicate,
            } => {
                let is_order_sensitive = is_order_sensitive_amortized(
                    expr_arena.get(predicate.node()),
                    expr_arena,
                    &mut aexpr_stack,
                );

                let observe_input_order = any_receiver_observes_order || is_order_sensitive;

                P::new([observe_input_order], [OutputOrder::PreservesInput])
            },

            IR::Union { inputs, options } => {
                if !any_receiver_observes_order {
                    options.maintain_order = false;
                }

                let observe_input_order = options.slice.is_some() || options.maintain_order;

                let output_order = if options.maintain_order {
                    // Output always has ordering regardless of input orderings - you will always see
                    // all of the rows from the first input before seeing rows from the second.
                    OutputOrder::Ordered
                } else {
                    OutputOrder::Random
                };

                P::new(unitvec![observe_input_order; inputs.len()], [output_order])
            },

            IR::HConcat { inputs, .. } => P::new(
                unitvec![any_receiver_observes_order; inputs.len()],
                [OutputOrder::PreservesInput],
            ),

            #[cfg(feature = "python")]
            IR::PythonScan { .. } => P::new([], [OutputOrder::Ordered]),

            IR::Sink { payload, .. } => {
                let observe_input_order = payload.maintain_order();
                P::new([observe_input_order], [OutputOrder::PreservesInput])
            },
            IR::Scan { .. } | IR::DataFrameScan { .. } => P::new([], [OutputOrder::Ordered]),

            IR::ExtContext { contexts, .. } => {
                // This node is nonsense. Just do the most conservative thing you can.
                P::new(unitvec![true; contexts.len() + 1], [OutputOrder::Ordered])
            },

            IR::SinkMultiple { .. } | IR::Invalid => unreachable!(),
        };

        // We make the code above simpler by pretending every node except caches always only has
        // one output. We correct for that here.
        if hits.len() > 1 && node_ordering.output_orders.len() == 1 {
            node_ordering.output_orders = unitvec![node_ordering.output_orders[0]; hits.len()]
        }

        let prev_value = orders.insert(node, node_ordering);
        assert!(prev_value.is_none());

        stack.extend(
            ir.inputs()
                .enumerate()
                .map(|(to_input_idx, input)| (input, Some((to_input_idx, node)))),
        );
    }

    orders
}

fn pullup_orders(
    leaves: &[Node],
    ir_arena: &mut Arena<IR>,
    outputs: &mut PlHashMap<Node, UnitVec<Node>>,
    orders: &mut PlHashMap<Node, PortOrder>,
    cache_proxy: &PlHashMap<UniqueId, Vec<Node>>,
) {
    let mut hits: PlHashMap<Node, Vec<(usize, Node)>> = PlHashMap::default();
    let mut stack = Vec::new();

    // Orderings of the inputs to this node.
    let mut orderings_to_this_node_scratch: Vec<OutputOrder> = Vec::new();

    for leaf in leaves {
        stack.extend(
            outputs[leaf]
                .iter()
                .enumerate()
                .map(|(i, v)| (*v, (i, *leaf))),
        );
    }

    while let Some((node, outgoing)) = stack.pop() {
        // @Hack. The IR creates caches for every path at the moment. That is super hacky. So is
        // this, but we need to work around it.
        let node = match ir_arena.get(node) {
            IR::Cache { id, .. } => cache_proxy.get(id).unwrap()[0],
            _ => node,
        };

        let hits = hits.entry(node).or_default();
        hits.push(outgoing);
        if hits.len() < orders[&node].input_order_observe.len() {
            continue;
        }

        let node_outputs = &outputs[&node];
        let ir = ir_arena.get_mut(node);

        orderings_to_this_node_scratch.clear();
        orderings_to_this_node_scratch.extend(
            hits.iter()
                .map(|(to_output_idx, to_node)| orders[to_node].output_orders[*to_output_idx]),
        );

        let node_ordering = orders.get_mut(&node).unwrap();

        use OutputOrder as O;

        let mut default_materialized_output_order = None;

        assert_eq!(
            orderings_to_this_node_scratch.len(),
            node_ordering.input_order_observe.len()
        );

        for (output_order_to_this_node, input_order_observe) in orderings_to_this_node_scratch
            .iter()
            .zip(node_ordering.input_order_observe.iter_mut())
        {
            if let O::Random = output_order_to_this_node {
                *input_order_observe = false;
            }

            match output_order_to_this_node {
                O::Random => default_materialized_output_order.get_or_insert(O::Random),
                O::Ordered => default_materialized_output_order.insert(O::Ordered),
                O::PreservesInput => unreachable!(),
            };
        }

        // Pullup simplification rules.
        use MaintainOrderJoin as MOJ;
        match ir {
            IR::Sort { sort_options, .. } => {
                let [observe_order] = node_ordering.input_order_observe_arr();

                // Unordered -> _     ==>    maintain_order=false
                if !observe_order {
                    sort_options.maintain_order = false;
                }
            },
            IR::GroupBy { maintain_order, .. } => {
                let [observe_order] = node_ordering.input_order_observe_arr();

                if !observe_order {
                    *maintain_order = false;
                    node_ordering.set_all_outputs_unordered();
                }
            },
            IR::Sink { input: _, payload } => {
                let [observe_order] = node_ordering.input_order_observe_arr();

                if !observe_order {
                    // Set maintain order to false if input is unordered
                    match payload {
                        SinkTypeIR::Memory => {},
                        SinkTypeIR::File(s) => s.sink_options.maintain_order = false,
                        SinkTypeIR::Partition(s) => s.sink_options.maintain_order = false,
                    }
                    node_ordering.set_all_outputs_unordered();
                }
            },
            IR::Join { options, .. } => {
                let [observe_left_order, observe_right_order] =
                    node_ordering.input_order_observe_arr();

                let left_unordered = !observe_left_order;
                let right_unordered = !observe_right_order;

                let maintain_order = options.args.maintain_order;

                if (left_unordered
                    && matches!(maintain_order, MOJ::Left | MOJ::RightLeft | MOJ::LeftRight))
                    || (right_unordered
                        && matches!(maintain_order, MOJ::Right | MOJ::RightLeft | MOJ::LeftRight))
                {
                    // If we are maintaining order of a side, but that input has no guaranteed order,
                    // remove the maintain ordering from that side.

                    Arc::make_mut(options).args.maintain_order = match maintain_order {
                        _ if left_unordered && right_unordered => MOJ::None,
                        MOJ::Left | MOJ::LeftRight if left_unordered => MOJ::None,
                        MOJ::RightLeft if left_unordered => MOJ::Right,
                        MOJ::Right | MOJ::RightLeft if right_unordered => MOJ::None,
                        MOJ::LeftRight if right_unordered => MOJ::Left,
                        _ => unreachable!(),
                    };

                    if matches!(options.args.maintain_order, MOJ::None) {
                        node_ordering.set_all_outputs_unordered();
                    }
                }
            },
            IR::Distinct { input: _, options } => {
                let [observe_order] = node_ordering.input_order_observe_arr();

                if !observe_order {
                    options.maintain_order = false;
                    node_ordering.set_all_outputs_unordered();
                }
            },

            #[cfg(feature = "python")]
            IR::PythonScan { .. } => {},
            IR::Scan { .. } | IR::DataFrameScan { .. } => {},
            #[cfg(feature = "merge_sorted")]
            IR::MergeSorted { .. } => {},

            IR::Cache { .. }
            | IR::SimpleProjection { .. }
            | IR::Slice { .. }
            | IR::HStack { .. }
            | IR::Filter { .. }
            | IR::Select { .. }
            | IR::HConcat { .. }
            | IR::Union { .. }
            | IR::MapFunction { .. } => {},
            IR::ExtContext { .. } => {},

            IR::SinkMultiple { .. } | IR::Invalid => unreachable!(),
        }

        for o in node_ordering.output_orders.iter_mut() {
            if let O::PreservesInput = o {
                // Note: This is conservatively initialized to `Ordered` if any input is `Ordered`.
                *o = default_materialized_output_order.unwrap();
            }
        }

        stack.extend(
            node_outputs
                .iter()
                .enumerate()
                .map(|(i, v)| (*v, (i, node))),
        );
    }
}

/// Optimize the orderings used in the IR plan and get the relative orderings of all edges.
///
/// All roots should be `Sink` nodes and no `SinkMultiple` or `Invalid` are allowed to be part of
/// the graph.
pub fn simplify_and_fetch_orderings(
    roots: &[Node],
    ir_arena: &mut Arena<IR>,
    expr_arena: &Arena<AExpr>,
) -> PlHashMap<Node, PortOrder> {
    let mut leaves = Vec::new();
    let mut outputs = PlHashMap::default();
    let mut cache_proxy = PlHashMap::<UniqueId, Vec<Node>>::default();

    // Get the per-node outputs and leaves
    {
        let mut stack = Vec::new();

        for root in roots {
            assert!(matches!(ir_arena.get(*root), IR::Sink { .. }));
            outputs.insert(*root, UnitVec::new());
            stack.extend(ir_arena.get(*root).inputs().map(|node| (*root, node)));
        }

        while let Some((parent, node)) = stack.pop() {
            let ir = ir_arena.get(node);
            let node = match ir {
                IR::Cache { id, .. } => {
                    let nodes = cache_proxy.entry(*id).or_default();
                    nodes.push(node);
                    nodes[0]
                },
                _ => node,
            };

            let outputs = outputs.entry(node).or_default();
            let has_been_visisited_before = !outputs.is_empty();
            outputs.push(parent);

            if has_been_visisited_before {
                continue;
            }

            let inputs = ir.inputs();
            if matches!(inputs, Inputs::Empty) {
                leaves.push(node);
            }
            stack.extend(inputs.map(|input| (node, input)));
        }
    }

    // Pushdown and optimize orders from the roots to the leaves.
    let mut orders = pushdown_orders(roots, ir_arena, expr_arena, &mut outputs, &cache_proxy);
    // Pullup orders from the leaves to the roots.
    pullup_orders(&leaves, ir_arena, &mut outputs, &mut orders, &cache_proxy);

    // @Hack. Since not all caches might share the same node and the input of caches might have
    // been updated, we need to ensure that all caches again have the same input.
    //
    // This can be removed when all caches with the same id share the same IR node.
    for nodes in cache_proxy.into_values() {
        let updated_node = nodes[0];
        let order = orders[&updated_node].clone();
        let IR::Cache {
            input: updated_input,
            id: _,
        } = ir_arena.get(updated_node)
        else {
            unreachable!();
        };
        let updated_input = *updated_input;
        for n in &nodes[1..] {
            let IR::Cache { input, id: _ } = ir_arena.get_mut(*n) else {
                unreachable!();
            };

            orders.insert(*n, order.clone());
            *input = updated_input;
        }
    }

    orders
}
