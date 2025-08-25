use std::sync::Arc;

use polars_core::frame::UniqueKeepStrategy;
use polars_core::prelude::PlHashMap;
use polars_ops::frame::{JoinType, MaintainOrderJoin};
use polars_utils::IdxSize;
use polars_utils::arena::{Arena, Node};
use polars_utils::idx_vec::UnitVec;

use super::IR;
use crate::dsl::{SinkTypeIR, UnionOptions};
use crate::plans::ir::inputs::Inputs;
use crate::plans::{AExpr, is_order_sensitive_amortized};

#[derive(Debug, Clone, Copy)]
pub enum InputOrder {
    /// The input may receive data in an undefined order.
    Unordered,
    /// The input propagates ordering into on of its outputs.
    Preserving,
    /// The input observes and propagates ordering into on of its outputs.
    Observing,
    /// The input observes and terminates the ordering.
    Consuming,
}

#[derive(Debug, Clone, Copy)]
pub enum OutputOrder {
    /// The output has an undefined output order.
    Unordered,
    /// The output propagates order from all of the inputs with `InputOrder::Preserving` or
    /// `InputOrder::Observing`.
    Preserving,
    /// The output produces a new ordering.
    Producing,
}

#[derive(Debug)]
pub struct NodeEdgeOrdering {
    pub inputs: Box<[InputOrder]>,
    pub outputs: Box<[OutputOrder]>,
}

impl NodeEdgeOrdering {
    fn new(inputs: impl Into<Box<[InputOrder]>>, outputs: impl Into<Box<[OutputOrder]>>) -> Self {
        Self {
            inputs: inputs.into(),
            outputs: outputs.into(),
        }
    }
}

struct IROrderContext<'a> {
    aexpr_stack: &'a mut Vec<Node>,
}

fn ir_top_level_to_order(
    node: Node,
    ir_arena: &Arena<IR>,
    expr_arena: &Arena<AExpr>,
    ctx: &mut IROrderContext,
    outputs: &PlHashMap<Node, UnitVec<Node>>,
) -> NodeEdgeOrdering {
    use {InputOrder as I, NodeEdgeOrdering as NEO, OutputOrder as O};

    match ir_arena.get(node) {
        #[cfg(feature = "python")]
        IR::PythonScan { .. } => NEO::new([], [O::Producing]),
        IR::Scan { .. } | IR::DataFrameScan { .. } => NEO::new([], [O::Producing]),

        IR::Slice { .. } => NEO::new([I::Observing], [O::Preserving]),
        IR::Filter { predicate, .. } => {
            let aexpr = expr_arena.get(predicate.node());
            let input = if is_order_sensitive_amortized(aexpr, expr_arena, ctx.aexpr_stack) {
                I::Observing
            } else {
                I::Preserving
            };
            NEO::new([input], [O::Preserving])
        },
        IR::SimpleProjection { .. } => NEO::new([I::Preserving], [O::Preserving]),
        IR::Select { expr, .. } => {
            let mut all_scalar = true;
            let mut has_order_sensitive = false;

            for e in expr {
                let aexpr = expr_arena.get(e.node());
                has_order_sensitive |=
                    is_order_sensitive_amortized(aexpr, expr_arena, ctx.aexpr_stack);
                all_scalar &= aexpr.is_scalar(expr_arena);
            }

            // @NOTE: If a select has all scalars, the output is a DF of height 1.
            let (input, output) = match (has_order_sensitive, all_scalar) {
                (true, false) => (I::Consuming, O::Producing),
                (true, true) => (I::Consuming, O::Unordered),
                (false, true) => (I::Unordered, O::Unordered),
                (false, false) => (I::Preserving, O::Preserving),
            };

            NEO::new([input], [output])
        },
        IR::Sort {
            by_column,
            sort_options,
            ..
        } => {
            if sort_options.maintain_order {
                return NEO::new([I::Consuming], [O::Producing]);
            }

            let mut has_order_sensitive = false;

            for e in by_column {
                let aexpr = expr_arena.get(e.node());
                has_order_sensitive |=
                    is_order_sensitive_amortized(aexpr, expr_arena, ctx.aexpr_stack);
            }

            let input = if has_order_sensitive {
                I::Consuming
            } else {
                I::Unordered
            };

            NEO::new([input], [O::Producing])
        },
        IR::Cache { .. } => NEO::new([I::Preserving], vec![O::Preserving; outputs[&node].len()]),
        IR::GroupBy {
            input: _,
            keys,
            aggs,
            schema: _,
            maintain_order,
            options,
            apply,
        } => {
            if *maintain_order {
                return NEO::new([I::Observing], [O::Preserving]);
            }

            if options.dynamic.is_some() || options.rolling.is_some() {
                return NEO::new([I::Observing], [O::Preserving]);
            }

            if apply.is_some() {
                return NEO::new([I::Consuming], [O::Unordered]);
            }

            let mut has_order_sensitive = false;

            for e in keys.iter().chain(aggs) {
                let aexpr = expr_arena.get(e.node());
                has_order_sensitive |=
                    is_order_sensitive_amortized(aexpr, expr_arena, ctx.aexpr_stack);
            }

            let input = if has_order_sensitive {
                I::Consuming
            } else {
                I::Unordered
            };

            NEO::new([input], [O::Unordered])
        },
        IR::Join {
            input_left: _,
            input_right: _,
            schema: _,
            left_on,
            right_on,
            options,
        } => {
            if matches!(options.args.how, JoinType::AsOf(_)) {
                return NEO::new([I::Observing, I::Observing], [O::Preserving]);
            }

            let mut left_has_order_sensitive = false;
            let mut right_has_order_sensitive = false;

            for e in left_on {
                let aexpr = expr_arena.get(e.node());
                left_has_order_sensitive |=
                    is_order_sensitive_amortized(aexpr, expr_arena, ctx.aexpr_stack);
            }
            for e in right_on {
                let aexpr = expr_arena.get(e.node());
                right_has_order_sensitive |=
                    is_order_sensitive_amortized(aexpr, expr_arena, ctx.aexpr_stack);
            }

            use MaintainOrderJoin as M;
            NEO::new(
                [
                    match (
                        options.args.maintain_order,
                        left_has_order_sensitive,
                        options.args.slice.is_some(),
                    ) {
                        (M::Left | M::LeftRight, true, _)
                        | (M::Left | M::LeftRight | M::RightLeft, _, true) => I::Observing,
                        (M::Left | M::LeftRight, false, _) => I::Preserving,
                        (M::RightLeft, _, _) | (_, true, _) => I::Consuming,
                        _ => I::Unordered,
                    },
                    match (
                        options.args.maintain_order,
                        right_has_order_sensitive,
                        options.args.slice.is_some(),
                    ) {
                        (M::Right | M::RightLeft, true, _)
                        | (M::Right | M::LeftRight | M::RightLeft, _, true) => I::Observing,
                        (M::Right | M::RightLeft, false, _) => I::Preserving,
                        (M::LeftRight, _, _) | (_, true, _) => I::Consuming,
                        _ => I::Unordered,
                    },
                ],
                [match options.args.maintain_order {
                    M::None => O::Unordered,
                    _ => O::Preserving,
                }],
            )
        },
        IR::HStack { exprs, .. } => {
            let mut has_order_sensitive = false;

            for e in exprs {
                let aexpr = expr_arena.get(e.node());
                has_order_sensitive |=
                    is_order_sensitive_amortized(aexpr, expr_arena, ctx.aexpr_stack);
            }

            let (input, output) = if has_order_sensitive {
                (I::Consuming, O::Producing)
            } else {
                (I::Preserving, O::Preserving)
            };

            NEO::new([input], [output])
        },
        IR::Distinct { input: _, options } => {
            if options.maintain_order {
                let input = if options.slice.is_some() {
                    I::Observing
                } else {
                    I::Preserving
                };
                return NEO::new([input], [O::Preserving]);
            }

            if matches!(
                options.keep_strategy,
                UniqueKeepStrategy::First | UniqueKeepStrategy::Last
            ) {
                return NEO::new([I::Consuming], [O::Unordered]);
            }

            NEO::new([I::Unordered], [O::Unordered])
        },
        IR::MapFunction { input: _, function } => {
            if function.is_streamable() {
                return NEO::new([I::Preserving], [O::Preserving]);
            }

            NEO::new([I::Consuming], [O::Producing])
        },
        IR::Union { inputs, options } => {
            let input = if options.slice.is_some() {
                I::Observing
            } else {
                I::Preserving
            };

            NEO::new(vec![input; inputs.len()], [O::Preserving])
        },
        IR::HConcat { inputs, .. } => NEO::new(vec![I::Observing; inputs.len()], [O::Preserving]),

        IR::Sink { input, payload } => {
            let input = if payload.maintain_order() {
                I::Consuming
            } else {
                I::Unordered
            };
            NEO::new([input], [])
        },
        IR::MergeSorted {
            input_left: _,
            input_right: _,
            key: _,
        } => NEO::new([I::Observing, I::Observing], [O::Preserving]),

        IR::ExtContext { .. } | IR::SinkMultiple { .. } | IR::Invalid => unreachable!(),
    }
}

/// Remove ordering from both sides if either side has an undefined order.
fn simplify_edge(tx: OutputOrder, rx: InputOrder) -> (OutputOrder, InputOrder) {
    use {InputOrder as I, OutputOrder as O};
    match (tx, rx) {
        (O::Unordered, _) | (_, I::Unordered) => (O::Unordered, I::Unordered),
        (o, i) => (o, i),
    }
}

/// Remove preserving and observing where no-one is observing anymore.
fn simplify_1x1_node(n: &mut NodeEdgeOrdering) {
    use {InputOrder as I, OutputOrder as O};
    (n.inputs[0], n.outputs[0]) = match (n.inputs[0], n.outputs[0]) {
        (I::Observing, o @ (O::Unordered | O::Producing)) => (I::Consuming, o),
        (I::Preserving, o @ (O::Unordered | O::Producing)) => (I::Unordered, o),
        (i @ (I::Consuming | I::Unordered), O::Preserving) => (i, O::Unordered),
        (i, o) => (i, o),
    }
}

fn pushdown_orders(
    roots: &[Node],
    ir_arena: &mut Arena<IR>,
    expr_arena: &Arena<AExpr>,
    outputs: &mut PlHashMap<Node, UnitVec<Node>>,
) -> PlHashMap<Node, NodeEdgeOrdering> {
    let mut orders: PlHashMap<Node, NodeEdgeOrdering> = PlHashMap::default();
    let mut cache_hits: PlHashMap<Node, Vec<(usize, Node)>> = PlHashMap::default();
    let mut ctx = IROrderContext {
        aexpr_stack: &mut Vec::new(),
    };
    let mut stack = Vec::new();

    stack.extend(roots.iter().map(|n| (*n, None)));

    while let Some((node, outgoing)) = stack.pop() {
        let node_outputs = &outputs[&node];
        let ir = ir_arena.get(node);

        debug_assert!(!orders.contains_key(&node));

        let mut node_ordering = match ir {
            IR::Cache { .. } => {
                let cache_hits = cache_hits.entry(node).or_default();
                cache_hits.push(outgoing.unwrap());
                if cache_hits.len() < node_outputs.len() {
                    continue;
                }

                let mut node_ordering =
                    ir_top_level_to_order(node, ir_arena, expr_arena, &mut ctx, outputs);

                // Simplify the edges ordering' that connect the cache to all its outputs.
                for (i, (to_input_idx, to_node)) in cache_hits.iter().enumerate() {
                    // Edge: Node Out -> To In
                    let tx = &mut node_ordering.outputs[i];
                    let rx = &mut orders.get_mut(to_node).unwrap().inputs[*to_input_idx];
                    (*tx, *rx) = simplify_edge(*tx, *rx);
                }

                node_ordering
            },
            _ => {
                let mut node_ordering =
                    ir_top_level_to_order(node, ir_arena, expr_arena, &mut ctx, outputs);

                // Simplify the edge ordering that connects this node to the output.
                if let Some((to_input_idx, to_node)) = outgoing {
                    // Edge: Node Out -> To In
                    let tx = &mut node_ordering.outputs[0];
                    let rx = &mut orders.get_mut(&to_node).unwrap().inputs[to_input_idx];
                    (*tx, *rx) = simplify_edge(*tx, *rx);
                }

                node_ordering
            },
        };

        // Pushdown simplification rules.
        let mut ir = ir_arena.get_mut(node);
        use {InputOrder as I, MaintainOrderJoin as MOJ, OutputOrder as O};
        match ir {
            IR::Cache { .. } => {
                // _ -> [Unordered]
                //   to
                // Unordered -> [Unordered]
                if node_ordering
                    .outputs
                    .iter()
                    .all(|o| matches!(o, O::Unordered))
                {
                    node_ordering.inputs[0] = I::Unordered;
                }
            },
            IR::Sort { input, slice, .. } => {
                if matches!(node_ordering.outputs[0], O::Unordered) {
                    match slice {
                        None => {
                            // Remove sort if output is Unordered
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
                        Some((offset, len)) => {
                            // Sort with unordered output and slice. Just remove the sort and keep the
                            // slice.
                            *ir = IR::Slice {
                                input: *input,
                                offset: *offset,
                                len: *len as IdxSize,
                            }
                        },
                    }
                }
            },
            IR::GroupBy {
                keys,
                aggs,
                maintain_order,
                apply,
                options,
                ..
            } => {
                if matches!(node_ordering.outputs[0], O::Unordered)
                    && apply.is_none()
                    && *maintain_order
                {
                    if options.dynamic.is_none() && options.rolling.is_none() {
                        node_ordering.inputs[0] = I::Consuming;
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
                                is_order_sensitive_amortized(aexpr, expr_arena, ctx.aexpr_stack);
                        }

                        if !has_order_sensitive {
                            node_ordering.inputs[0] = I::Unordered;
                        }
                    }
                }
            },
            IR::MergeSorted {
                input_left,
                input_right,
                ..
            } => {
                if matches!(node_ordering.outputs[0], O::Unordered) {
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

                    node_ordering.inputs[0] = I::Unordered;
                    node_ordering.inputs[1] = I::Unordered;
                }
            },
            IR::Join {
                input_left: _,
                input_right: _,
                schema: _,
                left_on,
                right_on,
                options,
            } => {
                if matches!(node_ordering.outputs[0], O::Unordered) {
                    // If the join maintains order, but the output has underfined order. Remove the
                    // ordering.
                    if !matches!(options.args.maintain_order, MOJ::None) {
                        let mut new_options = options.as_ref().clone();
                        new_options.args.maintain_order = MOJ::None;
                        *options = Arc::new(new_options);
                    }

                    match options.args.how {
                        JoinType::AsOf(_) => {
                            node_ordering.inputs[0] = I::Consuming;
                            node_ordering.inputs[1] = I::Consuming;
                        },
                        _ => {
                            // If either side does not need to maintain order, don't maintain the old on that
                            // side.
                            for (i, on) in [left_on, right_on].iter().enumerate() {
                                if matches!(node_ordering.inputs[i], I::Unordered) {
                                    continue;
                                }

                                let mut has_order_sensitive = false;
                                for e in on.iter() {
                                    let aexpr = expr_arena.get(e.node());
                                    has_order_sensitive |= is_order_sensitive_amortized(
                                        aexpr,
                                        expr_arena,
                                        ctx.aexpr_stack,
                                    );
                                }
                                node_ordering.inputs[i] = I::Unordered;
                            }
                        },
                    }
                }
            },
            IR::Distinct { input: _, options } => {
                if matches!(node_ordering.outputs[0], O::Unordered) && options.maintain_order {
                    options.maintain_order = false;
                    if matches!(
                        options.keep_strategy,
                        UniqueKeepStrategy::Any | UniqueKeepStrategy::None
                    ) {
                        node_ordering.inputs[0] = I::Unordered;
                    }
                }
            },
            IR::MapFunction { input: _, function } => {
                if function.is_streamable() && matches!(node_ordering.outputs[0], O::Unordered) {
                    node_ordering.inputs[0] = I::Unordered;
                }
            },
            IR::SimpleProjection { .. } => {
                if matches!(node_ordering.outputs[0], O::Unordered) {
                    node_ordering.inputs[0] = I::Unordered;
                }
            },
            IR::Slice { .. } => {
                if matches!(node_ordering.outputs[0], O::Unordered) {
                    node_ordering.inputs[0] = I::Consuming;
                }
            },
            IR::HStack { exprs, .. } | IR::Select { expr: exprs, .. } => {
                if matches!(node_ordering.outputs[0], O::Unordered) {
                    let mut has_order_sensitive = false;
                    for e in exprs {
                        let aexpr = expr_arena.get(e.node());
                        has_order_sensitive |=
                            is_order_sensitive_amortized(aexpr, expr_arena, ctx.aexpr_stack);
                    }
                    if !has_order_sensitive {
                        node_ordering.inputs[0] = I::Unordered;
                    }
                }
            },

            IR::Filter {
                input: _,
                predicate,
            } => {
                if matches!(node_ordering.outputs[0], O::Unordered)
                    && !is_order_sensitive_amortized(
                        expr_arena.get(predicate.node()),
                        expr_arena,
                        ctx.aexpr_stack,
                    )
                {
                    node_ordering.inputs[0] = I::Unordered;
                }
            },

            IR::Union { options, .. } => {
                if matches!(node_ordering.outputs[0], O::Unordered) && options.maintain_order {
                    options.maintain_order = false;
                    node_ordering
                        .inputs
                        .iter_mut()
                        .for_each(|i| *i = I::Unordered);
                }
            },

            IR::HConcat { .. } => {
                if matches!(node_ordering.outputs[0], O::Unordered) {
                    node_ordering
                        .inputs
                        .iter_mut()
                        .for_each(|i| *i = I::Consuming);
                }
            },

            #[cfg(feature = "python")]
            IR::PythonScan { .. } => {},

            IR::Scan { .. } | IR::DataFrameScan { .. } | IR::Sink { .. } => {},

            IR::ExtContext { .. } | IR::SinkMultiple { .. } | IR::Invalid => unreachable!(),
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
    expr_arena: &Arena<AExpr>,
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
                .map(|(to_output_idx, to_node)| orders[to_node].outputs[*to_output_idx]),
        );

        let node_ordering = orders.get_mut(&node).unwrap();
        assert_eq!(txs.len(), node_ordering.inputs.len());
        for (tx, rx) in txs.iter().zip(node_ordering.inputs.iter_mut()) {
            // @NOTE: We don't assign tx back here since it would be redundant.
            (_, *rx) = simplify_edge(*tx, *rx);
        }

        // Pullup simplification rules.
        use {InputOrder as I, MaintainOrderJoin as MOJ, OutputOrder as O};
        match ir {
            IR::Cache { .. } => {
                // Cache:
                // Unordered -> [_]
                //   to
                // Unordered -> [Unordered]
                if matches!(node_ordering.inputs[0], I::Unordered) {
                    node_ordering
                        .outputs
                        .iter_mut()
                        .for_each(|o| *o = O::Unordered);
                }
            },
            IR::Sort {
                input,
                sort_options,
                ..
            } => {
                if matches!(node_ordering.inputs[0], I::Unordered) && sort_options.maintain_order {
                    // Unordered -> _   ==>   maintain_order=false
                    sort_options.maintain_order = false;
                }
            },
            IR::GroupBy {
                keys,
                aggs,
                maintain_order,
                apply,
                ..
            } => {
                if matches!(node_ordering.inputs[0], I::Unordered)
                    && apply.is_none()
                    && (*maintain_order || !matches!(node_ordering.outputs[0], O::Unordered))
                {
                    // Unordered -> _
                    //   to
                    // maintain_order = false
                    // and
                    // Unordered -> Unordered

                    *maintain_order = false;
                    node_ordering.outputs[0] = O::Unordered;
                }
            },
            IR::Sink { input, payload } => {
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
                left_on,
                right_on,
                options,
            } => {
                let left_unordered = matches!(node_ordering.inputs[0], I::Unordered);
                let right_unordered = matches!(node_ordering.inputs[1], I::Unordered);

                let maintain_order = options.args.maintain_order;

                if (left_unordered
                    && !matches!(maintain_order, MOJ::Left | MOJ::RightLeft | MOJ::LeftRight))
                    || (right_unordered
                        && !matches!(maintain_order, MOJ::Right | MOJ::RightLeft | MOJ::LeftRight))
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
                        node_ordering.outputs[0] = O::Unordered;
                    }
                    *options = Arc::new(new_options);
                }
            },
            IR::Distinct { input: _, options } => {
                if matches!(node_ordering.inputs[0], I::Unordered)
                    && (options.maintain_order
                        || matches!(
                            options.keep_strategy,
                            UniqueKeepStrategy::First | UniqueKeepStrategy::Last
                        ))
                {
                    options.maintain_order = false;
                    options.keep_strategy = UniqueKeepStrategy::Any;

                    node_ordering.outputs[0] = O::Unordered;
                }
            },
            IR::MapFunction { input: _, function } => {
                if function.is_streamable() && matches!(node_ordering.inputs[0], I::Unordered) {
                    node_ordering.outputs[0] = O::Unordered;
                }
            },
            IR::SimpleProjection { .. }
            | IR::Slice { .. }
            | IR::HStack { .. }
            | IR::Filter { .. }
            | IR::Select { .. } => {
                if matches!(node_ordering.inputs[0], I::Unordered) {
                    node_ordering.outputs[0] = O::Unordered;
                }
            },

            #[cfg(feature = "python")]
            IR::PythonScan { .. } => {},
            IR::Scan { .. } | IR::DataFrameScan { .. } => {},

            IR::MergeSorted {
                input_left,
                input_right,
                ..
            } => {
                // An input being unordered is technically valid as it is possible for all values
                // to be the same in which case the rows are sorted.

                if node_ordering
                    .inputs
                    .iter()
                    .all(|i| matches!(i, I::Unordered))
                {
                    node_ordering.outputs[0] = O::Unordered;
                }
            },

            IR::Union { options, .. } => {
                // @WARN: If you update the maintain_order flag to actually do something, make sure
                // to adjust the code below as well and remove the assert.
                assert!(!options.maintain_order);

                if node_ordering
                    .inputs
                    .iter()
                    .all(|i| matches!(i, I::Unordered))
                {
                    node_ordering.outputs[0] = O::Producing;
                }
            },
            IR::HConcat { .. } => {
                if node_ordering
                    .inputs
                    .iter()
                    .all(|i| matches!(i, I::Unordered))
                {
                    node_ordering.outputs[0] = O::Unordered;
                }
            },

            IR::ExtContext { .. } | IR::SinkMultiple { .. } | IR::Invalid => unreachable!(),
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
            outputs.push(input);

            let ir = ir_arena.get(node);
            if matches!(ir, IR::Cache { .. }) && outputs.len() > 1 {
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
    pullup_orders(&leafs, ir_arena, expr_arena, &mut outputs, &mut orders);

    orders
}
