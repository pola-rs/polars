use std::sync::Arc;

use polars_core::frame::UniqueKeepStrategy;
use polars_core::prelude::PlHashMap;
use polars_ops::frame::{JoinType, MaintainOrderJoin};
use polars_utils::arena::{Arena, Node};
use polars_utils::idx_vec::UnitVec;

use super::IR;
use crate::dsl::{SinkTypeIR, UnionOptions};
use crate::plans::ir::inputs::Inputs;
use crate::plans::{AExpr, is_order_sensitive_amortized, is_scalar_ae};

#[derive(Debug, Clone, Copy)]
pub enum InputOrder {
    /// The input may receive data in an undefined order.
    Unordered,
    /// The input propagates ordering into one or more of its outputs.
    Preserving,
    /// The input observes and propagates ordering into one or more of its outputs.
    Observing,
    /// The input observes and terminates the ordering.
    Consuming,
}

#[derive(Debug)]
pub struct NodeEdgeOrdering {
    pub inputs: Box<[InputOrder]>,
    pub output_ordered: Box<[bool]>,
}

impl NodeEdgeOrdering {
    fn new(inputs: impl Into<Box<[InputOrder]>>, output_ordered: impl Into<Box<[bool]>>) -> Self {
        Self {
            inputs: inputs.into(),
            output_ordered: output_ordered.into(),
        }
    }

    fn set_unordered_output(&mut self) {
        self.output_ordered.iter_mut().for_each(|o| *o = false);
    }
}

/// Remove ordering from both sides if either side has an undefined order.
fn simplify_edge(tx: bool, rx: InputOrder) -> (bool, InputOrder) {
    use InputOrder as I;
    match (tx, rx) {
        (false, _) | (_, I::Unordered) => (false, I::Unordered),
        (o, i) => (o, i),
    }
}

fn pushdown_orders(
    roots: &[Node],
    ir_arena: &mut Arena<IR>,
    expr_arena: &Arena<AExpr>,
    outputs: &mut PlHashMap<Node, UnitVec<Node>>,
) -> PlHashMap<Node, NodeEdgeOrdering> {
    let mut orders: PlHashMap<Node, NodeEdgeOrdering> = PlHashMap::default();
    let mut node_hits: PlHashMap<Node, Vec<(usize, Node)>> = PlHashMap::default();
    let mut aexpr_stack = Vec::new();
    let mut stack = Vec::new();
    let mut output_port_orderings = Vec::new();

    stack.extend(roots.iter().map(|n| (*n, None)));

    while let Some((node, outgoing)) = stack.pop() {
        debug_assert!(!orders.contains_key(&node));

        let node_outputs = &outputs[&node];
        let hits = node_hits.entry(node).or_default();
        if let Some(outgoing) = outgoing {
            hits.push(outgoing);
            if hits.len() < node_outputs.len() {
                continue;
            }
        }

        output_port_orderings.clear();
        output_port_orderings.extend(
            hits.iter().map(|(to_input_idx, to_node)| {
                orders.get_mut(to_node).unwrap().inputs[*to_input_idx]
            }),
        );

        let all_outputs_unordered = output_port_orderings
            .iter()
            .all(|i| matches!(i, I::Unordered));

        // Pushdown simplification rules.
        let ir = ir_arena.get_mut(node);
        use {InputOrder as I, MaintainOrderJoin as MOJ, NodeEdgeOrdering as NEO};
        let mut node_ordering = match ir {
            IR::Cache { .. } if all_outputs_unordered => NEO::new([I::Unordered], [false]),
            IR::Cache { .. } => NEO::new(
                [I::Preserving],
                output_port_orderings
                    .iter()
                    .map(|i| !matches!(i, I::Unordered))
                    .collect::<Box<[_]>>(),
            ),
            IR::Sort { input, slice, .. } if slice.is_none() && all_outputs_unordered => {
                // Remove sort if output is unordered, and we are not slicing.
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
                ..
            } => {
                let input = if sort_options.maintain_order {
                    I::Consuming
                } else {
                    let mut has_order_sensitive = false;
                    for e in by_column {
                        let aexpr = expr_arena.get(e.node());
                        has_order_sensitive |=
                            is_order_sensitive_amortized(aexpr, expr_arena, &mut aexpr_stack);
                    }

                    if has_order_sensitive {
                        I::Consuming
                    } else {
                        I::Unordered
                    }
                };

                NEO::new([input], [true])
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

                let (input, output) = if apply.is_some()
                    || options.dynamic.is_some()
                    || options.rolling.is_some()
                    || *maintain_order
                {
                    (I::Consuming, true)
                } else {
                    // _ -> Unordered
                    //   to
                    // maintain_order = false
                    // and
                    // Unordered -> Unordered (if no order sensitive expressions)

                    *maintain_order = false;
                    let mut has_order_sensitive = false;
                    for e in keys.iter().chain(aggs.iter()) {
                        let aexpr = expr_arena.get(e.node());
                        has_order_sensitive |=
                            is_order_sensitive_amortized(aexpr, expr_arena, &mut aexpr_stack);
                    }

                    // The auto-implode is also other sensitive.
                    has_order_sensitive |=
                        aggs.iter().any(|agg| !is_scalar_ae(agg.node(), expr_arena));

                    (
                        if has_order_sensitive {
                            I::Consuming
                        } else {
                            I::Unordered
                        },
                        false,
                    )
                };

                NEO::new([input], [output])
            },
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
                    NEO::new([I::Unordered, I::Unordered], [false])
                } else {
                    NEO::new([I::Observing, I::Observing], [true])
                }
            },
            IR::Join { options, .. } if matches!(options.args.how, JoinType::AsOf(_)) => {
                NEO::new([I::Observing, I::Observing], [!all_outputs_unordered])
            },
            IR::Join {
                input_left: _,
                input_right: _,
                schema: _,
                left_on,
                right_on,
                options,
            } if all_outputs_unordered => {
                // If the join maintains order, but the output has underfined order. Remove the
                // ordering.
                if !matches!(options.args.maintain_order, MOJ::None) {
                    let mut new_options = options.as_ref().clone();
                    new_options.args.maintain_order = MOJ::None;
                    *options = Arc::new(new_options);
                }

                let mut inputs = [I::Consuming, I::Consuming];

                // If either side does not need to maintain order, don't maintain the old on that
                // side.
                for (i, on) in [left_on, right_on].iter().enumerate() {
                    let mut has_order_sensitive = false;
                    for e in on.iter() {
                        let aexpr = expr_arena.get(e.node());
                        has_order_sensitive |=
                            is_order_sensitive_amortized(aexpr, expr_arena, &mut aexpr_stack);
                    }

                    if !has_order_sensitive {
                        inputs[i] = I::Unordered;
                    }
                }

                NEO::new(inputs, [false])
            },
            IR::Join {
                input_left: _,
                input_right: _,
                schema: _,
                left_on,
                right_on,
                options,
            } => {
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
                let left_input = match (
                    options.args.maintain_order,
                    left_has_order_sensitive,
                    options.args.slice.is_some(),
                ) {
                    (M::Left | M::LeftRight, true, _)
                    | (M::Left | M::LeftRight | M::RightLeft, _, true) => I::Observing,
                    (M::Left | M::LeftRight, false, _) => I::Preserving,
                    (M::RightLeft, _, _) | (_, true, _) => I::Consuming,
                    _ => I::Unordered,
                };
                let right_input = match (
                    options.args.maintain_order,
                    right_has_order_sensitive,
                    options.args.slice.is_some(),
                ) {
                    (M::Right | M::RightLeft, true, _)
                    | (M::Right | M::LeftRight | M::RightLeft, _, true) => I::Observing,
                    (M::Right | M::RightLeft, false, _) => I::Preserving,
                    (M::LeftRight, _, _) | (_, true, _) => I::Consuming,
                    _ => I::Unordered,
                };
                let output = !matches!(options.args.maintain_order, M::None);

                NEO::new([left_input, right_input], [output])
            },
            IR::Distinct { input: _, options } => {
                options.maintain_order &= !all_outputs_unordered;

                let input = if options.maintain_order
                    || matches!(
                        options.keep_strategy,
                        UniqueKeepStrategy::First | UniqueKeepStrategy::Last
                    ) {
                    I::Observing
                } else {
                    I::Unordered
                };
                NEO::new([input], [options.maintain_order])
            },
            IR::MapFunction { input: _, function } => {
                let input = if function.is_streamable() {
                    if all_outputs_unordered {
                        I::Unordered
                    } else {
                        I::Preserving
                    }
                } else {
                    I::Consuming
                };

                NEO::new([input], [!all_outputs_unordered])
            },
            IR::SimpleProjection { .. } => {
                let input = if all_outputs_unordered {
                    I::Unordered
                } else {
                    I::Preserving
                };
                NEO::new([input], [!all_outputs_unordered])
            },
            IR::Slice { .. } => NEO::new([I::Observing], [!all_outputs_unordered]),
            IR::HStack { exprs, .. } => {
                let mut has_order_sensitive = false;
                for e in exprs {
                    let aexpr = expr_arena.get(e.node());
                    has_order_sensitive |=
                        is_order_sensitive_amortized(aexpr, expr_arena, &mut aexpr_stack);
                }

                let input = if has_order_sensitive {
                    I::Observing
                } else if all_outputs_unordered {
                    I::Unordered
                } else {
                    I::Preserving
                };

                NEO::new([input], [!all_outputs_unordered])
            },
            IR::Select { expr: exprs, .. } => {
                let mut has_order_sensitive = false;
                let mut all_scalar = true;

                for e in exprs {
                    let aexpr = expr_arena.get(e.node());
                    has_order_sensitive |=
                        is_order_sensitive_amortized(aexpr, expr_arena, &mut aexpr_stack);
                    all_scalar &= is_scalar_ae(e.node(), expr_arena);
                }

                let input = if has_order_sensitive {
                    I::Observing
                } else if all_outputs_unordered {
                    I::Unordered
                } else {
                    I::Preserving
                };
                let output = !all_outputs_unordered && !all_scalar;

                NEO::new([input], [output])
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

                let input = if is_order_sensitive {
                    I::Observing
                } else if all_outputs_unordered {
                    I::Unordered
                } else {
                    I::Preserving
                };

                NEO::new([input], [!all_outputs_unordered])
            },

            IR::Union { inputs, options } => {
                // @NOTE: It seems we cannot trust the `maintain_order` for now.
                // if !options.maintain_order {
                //     return NEO::new(vec![I::Unordered; inputs.len()], [false]);
                // }

                let input = if options.slice.is_some() {
                    I::Observing
                } else {
                    I::Preserving
                };

                NEO::new(vec![input; inputs.len()], [true])
            },

            IR::HConcat { inputs, .. } => {
                NEO::new(vec![I::Observing; inputs.len()], [!all_outputs_unordered])
            },

            #[cfg(feature = "python")]
            IR::PythonScan { .. } => NEO::new([], [!all_outputs_unordered]),

            IR::Sink { payload, .. } => {
                let input = if payload.maintain_order() {
                    I::Consuming
                } else {
                    I::Unordered
                };
                NEO::new([input], [])
            },
            IR::Scan { .. } | IR::DataFrameScan { .. } => NEO::new([], [!all_outputs_unordered]),

            IR::ExtContext { contexts, .. } => {
                // This node is nonsense. Just do the most conservative thing you can.
                NEO::new(
                    vec![I::Consuming; contexts.len() + 1],
                    [!all_outputs_unordered],
                )
            },

            IR::SinkMultiple { .. } | IR::Invalid => unreachable!(),
        };

        // We make the code above simpler by pretending every node except caches always only has
        // one output. We correct for that here.
        if output_port_orderings.len() > 1 && node_ordering.output_ordered.len() == 1 {
            node_ordering.output_ordered =
                vec![node_ordering.output_ordered[0]; output_port_orderings.len()]
                    .into_boxed_slice();
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
    leafs: &[Node],
    ir_arena: &mut Arena<IR>,
    outputs: &mut PlHashMap<Node, UnitVec<Node>>,
    orders: &mut PlHashMap<Node, NodeEdgeOrdering>,
) {
    let mut hits: PlHashMap<Node, Vec<(usize, Node)>> = PlHashMap::default();
    let mut stack = Vec::new();

    let mut txs = Vec::new();

    for leaf in leafs {
        stack.extend(
            outputs[leaf]
                .iter()
                .enumerate()
                .map(|(i, v)| (*v, (i, *leaf))),
        );
    }

    while let Some((node, outgoing)) = stack.pop() {
        let hits = hits.entry(node).or_default();
        hits.push(outgoing);
        if hits.len() < orders[&node].inputs.len() {
            continue;
        }

        let node_outputs = &outputs[&node];
        let ir = ir_arena.get_mut(node);

        txs.clear();
        txs.extend(
            hits.iter()
                .map(|(to_output_idx, to_node)| orders[to_node].output_ordered[*to_output_idx]),
        );

        let node_ordering = orders.get_mut(&node).unwrap();
        assert_eq!(txs.len(), node_ordering.inputs.len());
        for (tx, rx) in txs.iter().zip(node_ordering.inputs.iter_mut()) {
            // @NOTE: We don't assign tx back here since it would be redundant.
            (_, *rx) = simplify_edge(*tx, *rx);
        }

        // Pullup simplification rules.
        use {InputOrder as I, MaintainOrderJoin as MOJ};
        match ir {
            IR::Cache { .. } => {
                // Cache:
                // Unordered -> [_]
                //   to
                // Unordered -> [Unordered]
                if matches!(node_ordering.inputs[0], I::Unordered) {
                    node_ordering
                        .output_ordered
                        .iter_mut()
                        .for_each(|o| *o = false);
                }
            },
            IR::Sort { sort_options, .. } => {
                if matches!(node_ordering.inputs[0], I::Unordered) && sort_options.maintain_order {
                    // Unordered -> _   ==>   maintain_order=false
                    sort_options.maintain_order = false;
                }
            },
            IR::GroupBy { maintain_order, .. } => {
                if matches!(node_ordering.inputs[0], I::Unordered)
                    && (*maintain_order || !node_ordering.output_ordered[0])
                {
                    // Unordered -> _
                    //   to
                    // maintain_order = false
                    // and
                    // Unordered -> Unordered

                    *maintain_order = false;
                    node_ordering.set_unordered_output();
                }
            },
            IR::Sink { input: _, payload } => {
                if matches!(node_ordering.inputs[0], I::Unordered) {
                    // Set maintain order to false if input is unordered
                    match payload {
                        SinkTypeIR::Memory => {},
                        SinkTypeIR::File(s) => s.sink_options.maintain_order = false,
                        SinkTypeIR::Partition(s) => s.sink_options.maintain_order = false,
                    }
                }
            },
            IR::Join {
                input_left: _,
                input_right: _,
                schema: _,
                left_on: _,
                right_on: _,
                options,
            } => {
                let left_unordered = matches!(node_ordering.inputs[0], I::Unordered);
                let right_unordered = matches!(node_ordering.inputs[1], I::Unordered);

                let maintain_order = options.args.maintain_order;

                if (left_unordered
                    && matches!(maintain_order, MOJ::Left | MOJ::RightLeft | MOJ::LeftRight))
                    || (right_unordered
                        && matches!(maintain_order, MOJ::Right | MOJ::RightLeft | MOJ::LeftRight))
                {
                    // If we are maintaining order of a side, but that input has no guaranteed order,
                    // remove the maintain ordering from that side.

                    let mut new_options = options.as_ref().clone();
                    new_options.args.maintain_order = match maintain_order {
                        _ if left_unordered && right_unordered => MOJ::None,
                        MOJ::Left | MOJ::LeftRight if left_unordered => MOJ::None,
                        MOJ::RightLeft if left_unordered => MOJ::Right,
                        MOJ::Right | MOJ::RightLeft if right_unordered => MOJ::None,
                        MOJ::LeftRight if right_unordered => MOJ::Left,
                        _ => unreachable!(),
                    };

                    if matches!(new_options.args.maintain_order, MOJ::None) {
                        node_ordering.set_unordered_output();
                    }
                    *options = Arc::new(new_options);
                }
            },
            IR::Distinct { input: _, options } => {
                if matches!(node_ordering.inputs[0], I::Unordered) {
                    options.maintain_order = false;
                    options.keep_strategy = UniqueKeepStrategy::Any;
                    node_ordering.set_unordered_output();
                }
            },
            IR::MapFunction { input: _, function } => {
                if function.is_streamable() && matches!(node_ordering.inputs[0], I::Unordered) {
                    node_ordering.output_ordered[0] = false;
                }
            },
            IR::SimpleProjection { .. }
            | IR::Slice { .. }
            | IR::HStack { .. }
            | IR::Filter { .. }
            | IR::Select { .. } => {
                if matches!(node_ordering.inputs[0], I::Unordered) {
                    node_ordering.set_unordered_output();
                }
            },

            #[cfg(feature = "python")]
            IR::PythonScan { .. } => {},
            IR::Scan { .. } | IR::DataFrameScan { .. } => {},

            IR::MergeSorted { .. } => {
                // An input being unordered is technically valid as it is possible for all values
                // to be the same in which case the rows are sorted.
            },

            IR::Union { .. } => {
                // Even if the inputs are unordered. The output still has an order given by the
                // order of the inputs.
            },
            IR::HConcat { .. } | IR::ExtContext { .. } => {
                if node_ordering
                    .inputs
                    .iter()
                    .all(|i| matches!(i, I::Unordered))
                {
                    node_ordering.set_unordered_output();
                }
            },

            IR::SinkMultiple { .. } | IR::Invalid => unreachable!(),
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
/// All roots should be `Sink` nodes and no `SinkMultiple`, `Invalid` or `ExtContext` are allowed
/// to be part of the graph.
pub fn simplify_and_fetch_orderings(
    roots: &[Node],
    ir_arena: &mut Arena<IR>,
    expr_arena: &Arena<AExpr>,
) -> PlHashMap<Node, NodeEdgeOrdering> {
    let mut leafs = Vec::new();
    let mut outputs = PlHashMap::default();

    // Get the per-node outputs and leafs
    {
        let mut stack = Vec::new();

        for root in roots {
            assert!(matches!(ir_arena.get(*root), IR::Sink { .. }));
            outputs.insert(*root, UnitVec::new());
            stack.extend(ir_arena.get(*root).inputs().map(|node| (*root, node)));
        }

        while let Some((input, node)) = stack.pop() {
            let outputs = outputs.entry(node).or_default();
            let has_been_visisited_before = !outputs.is_empty();
            outputs.push(input);

            let ir = ir_arena.get(node);
            if has_been_visisited_before {
                continue;
            }

            let inputs = ir.inputs();
            if matches!(inputs, Inputs::Empty) {
                leafs.push(node);
            }
            stack.extend(inputs.map(|input| (node, input)));
        }
    }

    // Pushdown and optimize orders from the roots to the leafs.
    let mut orders = pushdown_orders(roots, ir_arena, expr_arena, &mut outputs);
    // Pullup orders from the leafs to the roots.
    pullup_orders(&leafs, ir_arena, &mut outputs, &mut orders);

    orders
}
