pub mod expr;

use std::sync::Arc;

use polars_core::frame::UniqueKeepStrategy;
use polars_core::prelude::{PlHashMap, ScratchMap};
use polars_utils::arena::{Arena, Node};
use polars_utils::scratch_vec::ScratchVec;
use slotmap::{SlotMap, new_key_type};

use crate::dsl::{SinkTypeIR, UnionOptions};
use crate::plans::ir_traversal::edge_provider::{EdgesProvider, IRTraversalGraphEdgeProvider};
use crate::plans::ir_traversal::ir_graph::{IRNodeEdgeKeys, build_ir_traversal_graph};
use crate::plans::ir_traversal::ir_node_key::IRNodeKey;
use crate::plans::optimizer::sortedness::pullup_sorted_single;
use crate::plans::partitioning::frame::FramePartitioning;
use crate::plans::simplify_ordering::expr::{ExprOrderSimplifier, ObservableOrders};
use crate::plans::{IRAggExpr, Sorted, is_scalar_ae};
use crate::prelude::{AExpr, IR};

#[derive(Debug, Clone)]
pub enum Edge {
    Ordered(FramePartitioning),
    Unordered(FramePartitioning),
}

impl Default for Edge {
    fn default() -> Self {
        Self::Ordered(FramePartitioning::new())
    }
}

impl AsMut<Edge> for Edge {
    fn as_mut(&mut self) -> &mut Edge {
        self
    }
}

impl AsMut<FramePartitioning> for Edge {
    fn as_mut(&mut self) -> &mut FramePartitioning {
        match self {
            Self::Ordered(v) => v,
            Self::Unordered(v) => v,
        }
    }
}

impl Edge {
    pub fn is_unordered(&self) -> bool {
        matches!(self, Self::Unordered(_))
    }

    pub fn partitioning(&self) -> &FramePartitioning {
        match self {
            Self::Ordered(v) => v,
            Self::Unordered(v) => v,
        }
    }

    fn set_unordered(&mut self) {
        match self {
            Self::Ordered(p) => {
                *self = Self::Unordered(std::mem::take(p));
                self.unordered_partitioning_correction();
            },
            Self::Unordered(_) => {},
        }
    }

    fn unordered_partitioning_correction(&mut self) {
        if !self.is_unordered() {
            return;
        }

        let partitioning: &mut FramePartitioning = self.as_mut();

        for i in 0..partitioning.len() {
            if !matches!(
                partitioning.get_index(i).unwrap().1,
                Sorted {
                    column: _,
                    descending: None,
                    nulls_last: None,
                }
            ) {
                let sorted = partitioning.make_mut().get_index_mut(i).unwrap().1;
                sorted.descending = None;
                sorted.nulls_last = None;
            }
        }
    }
}

new_key_type! {
    pub struct EdgeKey;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VisitState {
    Pre,
    Post,
}

pub fn simplify_and_fetch_orderings(
    roots: &[Node],
    ir_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
) -> (
    PlHashMap<IRNodeKey, IRNodeEdgeKeys<EdgeKey>>,
    SlotMap<EdgeKey, Edge>,
) {
    let (mut ir_nodes_stack, mut ir_node_to_edge_keys_map, mut edges_map, cache_updater) =
        build_ir_traversal_graph(roots, ir_arena);

    let eos_revisit_cache = &mut PlHashMap::default();
    let ae_nodes_scratch = &mut ScratchVec::default();
    let mut deleted_idxs = vec![];

    let mut simplifier = SimplifyIRNodeOrder {
        expr_arena,
        eos_revisit_cache,
        ae_nodes_scratch,
    };

    for (i, node) in ir_nodes_stack.iter().copied().enumerate() {
        if simplifier.simplify_ir_node_orders(
            node,
            ir_arena,
            IRTraversalGraphEdgeProvider {
                ir_node_edge_keys: ir_node_to_edge_keys_map
                    .get(&IRNodeKey::new(node, ir_arena))
                    .unwrap(),
                edges_map: &mut edges_map,
            },
            VisitState::Pre,
        ) {
            deleted_idxs.push(i);
            unlink_node(node, ir_arena, &mut ir_node_to_edge_keys_map);
        }
    }

    let mut column_names_scratch = ScratchMap::default();

    for (i, node) in ir_nodes_stack.drain(..).enumerate().rev() {
        if deleted_idxs.last() == Some(&i) {
            deleted_idxs.pop();
            continue;
        }

        if simplifier.simplify_ir_node_orders(
            node,
            ir_arena,
            IRTraversalGraphEdgeProvider {
                ir_node_edge_keys: ir_node_to_edge_keys_map
                    .get(&IRNodeKey::new(node, ir_arena))
                    .unwrap(),
                edges_map: &mut edges_map,
            },
            VisitState::Post,
        ) {
            unlink_node(node, ir_arena, &mut ir_node_to_edge_keys_map);
        }

        let mut edge_provider = IRTraversalGraphEdgeProvider {
            ir_node_edge_keys: ir_node_to_edge_keys_map
                .get(&IRNodeKey::new(node, ir_arena))
                .unwrap(),
            edges_map: &mut edges_map,
        };

        pullup_sorted_single(
            node,
            ir_arena,
            simplifier.expr_arena,
            &mut edge_provider,
            &mut column_names_scratch,
        );

        edge_provider
            .map_out_edges_mut(|e: &mut Edge| e.unordered_partitioning_correction())
            .for_each(|_| ());
    }

    cache_updater.update_cache_nodes(ir_arena);

    (ir_node_to_edge_keys_map, edges_map)
}

struct SimplifyIRNodeOrder<'a> {
    expr_arena: &'a mut Arena<AExpr>,
    eos_revisit_cache: &'a mut PlHashMap<Node, ObservableOrders>,
    ae_nodes_scratch: &'a mut ScratchVec<Node>,
}

impl SimplifyIRNodeOrder<'_> {
    /// Returns if the node was deleted.
    fn simplify_ir_node_orders(
        &mut self,
        current_ir_node: Node,
        ir_arena: &mut Arena<IR>,
        mut edges_provider: impl EdgesProvider<Edge>,
        visit_state: VisitState,
    ) -> bool {
        use ObservableOrders as O;

        macro_rules! unpack_edges {
            ($total:literal) => {
                edges_provider.unpack_edges_mut::<_, _, $total>().unwrap()
            };
        }

        macro_rules! expr_order_simplifier {
            () => {{
                self.eos_revisit_cache.clear();
                ExprOrderSimplifier::new(self.expr_arena, self.eos_revisit_cache)
            }};
        }

        match ir_arena.get_mut(current_ir_node) {
            IR::Select { .. } | IR::HStack { .. } => {
                let (exprs, is_hstack) = match ir_arena.get_mut(current_ir_node) {
                    IR::Select { expr, .. } => (expr, false),
                    IR::HStack { exprs, schema, .. } => {
                        let v = schema.len() != exprs.len();
                        (exprs, v)
                    },
                    _ => unreachable!(),
                };

                let ([in_edge], [out_edge]) = unpack_edges!(2);

                let mut eos = expr_order_simplifier!();
                let ae_nodes_scratch = self.ae_nodes_scratch.get();

                ae_nodes_scratch.extend(exprs.iter().map(|eir| eir.node()));

                let exprs_observable_orders = eos.simplify_projected_exprs(
                    ae_nodes_scratch,
                    out_edge.is_unordered() && (in_edge.is_unordered() || !is_hstack),
                );

                let input_order_observe = ((exprs_observable_orders.contains(O::COLUMN)
                    || is_hstack)
                    && !out_edge.is_unordered())
                    || (is_hstack && exprs_observable_orders.contains(O::INDEPENDENT))
                    || eos.internally_observed_orders().contains(O::COLUMN);

                if !input_order_observe {
                    in_edge.set_unordered();
                }

                if !exprs_observable_orders.contains(O::INDEPENDENT)
                    && (in_edge.is_unordered()
                        || !(is_hstack || exprs_observable_orders.contains(O::COLUMN)))
                {
                    out_edge.set_unordered();
                }
            },

            IR::Sort {
                input: _,
                by_column,
                slice,
                sort_options,
            } => {
                let ([in_edge], [out_edge]) = unpack_edges!(2);

                if out_edge.is_unordered() && slice.is_none() {
                    debug_assert_eq!(visit_state, VisitState::Pre);
                    *in_edge = out_edge.clone();
                    return true;
                }

                let mut eos = expr_order_simplifier!();
                let ae_nodes_scratch = self.ae_nodes_scratch.get();

                ae_nodes_scratch.extend(by_column.iter().map(|eir| eir.node()));

                let key_exprs_observable_orders =
                    eos.simplify_projected_exprs(ae_nodes_scratch, false);

                if in_edge.is_unordered()
                    || !(sort_options.maintain_order
                        || eos.internally_observed_orders().contains(O::COLUMN)
                        || key_exprs_observable_orders.contains(O::INDEPENDENT))
                {
                    in_edge.set_unordered();
                    sort_options.maintain_order = false;
                }
            },

            IR::Filter {
                input: _,
                predicate,
            } => {
                let ([in_edge], [out_edge]) = unpack_edges!(2);

                let mut eos = expr_order_simplifier!();
                let predicate_observable_orders =
                    eos.simplify_projected_exprs(&[predicate.node()], false);

                if out_edge.is_unordered()
                    && !(eos.internally_observed_orders().contains(O::COLUMN)
                        || predicate_observable_orders.contains(O::INDEPENDENT))
                {
                    in_edge.set_unordered();
                }

                if in_edge.is_unordered() {
                    out_edge.set_unordered();
                }
            },

            IR::GroupBy {
                input: _,
                keys,
                aggs,
                schema: _,
                maintain_order,
                options,
                apply,
            } => {
                let ([in_edge], [out_edge]) = unpack_edges!(2);

                // Put the implode in for the expr order optimizer.
                for agg in aggs.iter_mut() {
                    if !is_scalar_ae(agg.node(), self.expr_arena) {
                        agg.set_node(self.expr_arena.add(AExpr::Agg(IRAggExpr::Implode {
                            input: agg.node(),
                            maintain_order: true,
                        })));
                    }
                }

                let mut eos = expr_order_simplifier!();
                let ae_nodes_scratch = self.ae_nodes_scratch.get();

                ae_nodes_scratch.extend(keys.iter().map(|eir| eir.node()));
                let keys_observable = eos.simplify_projected_exprs(
                    ae_nodes_scratch,
                    in_edge.is_unordered() && !*maintain_order,
                );

                ae_nodes_scratch.clear();
                ae_nodes_scratch.extend(aggs.iter().map(|eir| eir.node()));
                eos.simplify_projected_exprs(ae_nodes_scratch, false);

                let order_observing_options =
                    apply.is_some() || options.is_dynamic() || options.is_rolling();

                if !(order_observing_options
                    || keys_observable.contains(O::INDEPENDENT)
                    || eos.internally_observed_orders().contains(O::COLUMN)
                    || (*maintain_order
                        && keys_observable.contains(O::COLUMN)
                        && !out_edge.is_unordered()))
                {
                    in_edge.set_unordered();
                }

                if out_edge.is_unordered()
                    || !*maintain_order
                    || (in_edge.is_unordered() && !keys_observable.contains(O::INDEPENDENT))
                {
                    out_edge.set_unordered();
                    *maintain_order = false;
                }
            },

            IR::Distinct { input: _, options } => {
                use UniqueKeepStrategy as K;

                let ([in_edge], [out_edge]) = unpack_edges!(2);

                if !options.maintain_order || out_edge.is_unordered() {
                    options.maintain_order = false;
                    out_edge.set_unordered();
                }

                if in_edge.is_unordered()
                    || (!options.maintain_order
                        && match options.keep_strategy {
                            K::First | K::Last => false,
                            K::Any | K::None => true,
                        })
                {
                    options.maintain_order = false;

                    match options.keep_strategy {
                        K::First | K::Last => options.keep_strategy = K::Any,
                        K::Any | K::None => {},
                    };

                    in_edge.set_unordered();
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
                use polars_ops::prelude::JoinType;

                let ([in_edge_lhs, in_edge_rhs], [out_edge]) = unpack_edges!(3);

                let mut eos = expr_order_simplifier!();

                let ae_nodes_scratch = self.ae_nodes_scratch.get();
                ae_nodes_scratch.extend(left_on.iter().map(|eir| eir.node()));
                let left_keys_observable = eos.simplify_projected_exprs(ae_nodes_scratch, false);

                ae_nodes_scratch.clear();
                ae_nodes_scratch.extend(right_on.iter().map(|eir| eir.node()));
                let right_keys_observable = eos.simplify_projected_exprs(ae_nodes_scratch, false);

                // Join keys should be elementwise.
                assert!(!(left_keys_observable | right_keys_observable).contains(O::INDEPENDENT));
                assert!(!eos.internally_observed_orders().contains(O::COLUMN));

                #[cfg(feature = "asof_join")]
                if let JoinType::AsOf(_) = &options.args.how {
                    if in_edge_lhs.is_unordered()
                        || (out_edge.is_unordered() && in_edge_rhs.is_unordered())
                    {
                        in_edge_lhs.set_unordered();
                        in_edge_rhs.set_unordered();
                        out_edge.set_unordered();
                    }

                    return false;
                }

                use polars_ops::prelude::MaintainOrderJoin as JO;

                if out_edge.is_unordered() || options.args.maintain_order == JO::None {
                    out_edge.set_unordered();
                    in_edge_lhs.set_unordered();
                    in_edge_rhs.set_unordered();
                    Arc::make_mut(options).args.maintain_order = JO::None;
                }

                if in_edge_lhs.is_unordered() || options.args.maintain_order == JO::Right {
                    in_edge_lhs.set_unordered();

                    match options.args.maintain_order {
                        JO::Left => Arc::make_mut(options).args.maintain_order = JO::None,
                        JO::LeftRight | JO::RightLeft => {
                            Arc::make_mut(options).args.maintain_order = JO::Right
                        },
                        JO::None | JO::Right => {},
                    }
                }

                if in_edge_rhs.is_unordered()
                    || options.args.maintain_order == JO::Left
                    || match &options.args.how {
                        #[cfg(feature = "semi_anti_join")]
                        JoinType::Semi | JoinType::Anti => true,
                        _ => false,
                    }
                {
                    in_edge_rhs.set_unordered();

                    match options.args.maintain_order {
                        JO::Right => Arc::make_mut(options).args.maintain_order = JO::None,
                        JO::RightLeft | JO::LeftRight => {
                            Arc::make_mut(options).args.maintain_order = JO::Left
                        },
                        JO::None | JO::Left => {},
                    }
                }
            },

            IR::Union { inputs: _, options } => {
                assert_eq!(edges_provider.num_out_edges(), 1);

                let out_edge = edges_provider.get_out_edge_mut(0);

                if !options.maintain_order || out_edge.is_unordered() {
                    options.maintain_order = false;
                    out_edge.set_unordered();

                    edges_provider
                        .map_in_edges_mut(|e| e.set_unordered())
                        .for_each(|_| ());
                }

                // Note, having no ordered inputs still cannot de-order the out edge, since the rows
                // of each input are still ordered to fully appear before the next input.
            },

            #[cfg(feature = "merge_sorted")]
            IR::MergeSorted {
                input_left,
                input_right,
                key: _,
            } => {
                let ([in_edge_lhs, in_edge_rhs], [out_edge]) = unpack_edges!(3);

                if out_edge.is_unordered()
                    || (in_edge_lhs.is_unordered() && in_edge_rhs.is_unordered())
                {
                    out_edge.set_unordered();
                    in_edge_lhs.set_unordered();
                    in_edge_rhs.set_unordered();

                    let input_left = *input_left;
                    let input_right = *input_right;

                    ir_arena.replace(
                        current_ir_node,
                        IR::Union {
                            inputs: vec![input_left, input_right],
                            options: UnionOptions {
                                maintain_order: false,
                                ..Default::default()
                            },
                        },
                    );
                }
            },

            IR::MapFunction { input: _, function } => {
                let ([in_edge], [out_edge]) = unpack_edges!(2);

                if !function.observes_input_order()
                    && (!function.has_equal_order() || out_edge.is_unordered())
                {
                    in_edge.set_unordered();
                }

                if !function.is_order_producing(!in_edge.is_unordered())
                    && (in_edge.is_unordered() || !function.has_equal_order())
                {
                    out_edge.set_unordered();
                }
            },

            IR::HConcat { .. } | IR::Slice { .. } | IR::ExtContext { .. } => {
                if edges_provider
                    .map_in_edges_mut(|e| e.is_unordered())
                    .all(|x| x)
                {
                    edges_provider
                        .map_out_edges_mut(|e| e.set_unordered())
                        .for_each(|_| ())
                }
            },

            IR::SimpleProjection { .. } => {
                let ([in_edge], [out_edge]) = unpack_edges!(2);

                if in_edge.is_unordered() || out_edge.is_unordered() {
                    in_edge.set_unordered();
                    out_edge.set_unordered();
                }
            },

            IR::Cache { .. } => {
                assert_eq!(edges_provider.num_in_edges(), 1);

                if edges_provider.get_in_edge_mut(0).is_unordered() {
                    edges_provider
                        .map_out_edges_mut(|e| e.set_unordered())
                        .for_each(|_| ())
                } else if edges_provider
                    .map_out_edges_mut(|e| e.is_unordered())
                    .all(|x| x)
                {
                    edges_provider
                        .map_in_edges_mut(|e| e.set_unordered())
                        .for_each(|_| ())
                }
            },

            IR::Sink { input: _, payload } => {
                let ([in_edge], []) = unpack_edges!(1);

                if let SinkTypeIR::Partitioned(options) = payload {
                    let mut eos = expr_order_simplifier!();
                    let ae_nodes_scratch = self.ae_nodes_scratch.get();

                    ae_nodes_scratch.extend(options.expr_irs_iter().map(|eir| eir.node()));
                    let observable = eos.simplify_projected_exprs(ae_nodes_scratch, false);

                    // Partition key exprs should be elementwise
                    assert!(!observable.contains(O::INDEPENDENT));
                    assert!(!eos.internally_observed_orders().contains(O::COLUMN));
                }

                dbg!((&in_edge, visit_state));

                if !payload.maintain_order() || in_edge.is_unordered() {
                    in_edge.set_unordered();
                    payload.set_maintain_order(false);
                }
            },

            #[cfg(feature = "python")]
            IR::PythonScan { .. } => {},

            IR::Scan { .. } | IR::DataFrameScan { .. } => {},

            IR::SinkMultiple { .. } | IR::Invalid => unreachable!(),
        };

        false
    }
}

fn unlink_node(
    current_ir_node: Node,
    ir_arena: &mut Arena<IR>,
    ir_node_to_edge_keys_map: &mut PlHashMap<IRNodeKey, IRNodeEdgeKeys<EdgeKey>>,
) {
    let input_to_current_ir_node = {
        let mut inputs = ir_arena.get(current_ir_node).inputs();
        let node = inputs.next().unwrap();
        assert!(inputs.next().is_none());
        node
    };

    let current_ir_node_edges = ir_node_to_edge_keys_map
        .get(&IRNodeKey::new(current_ir_node, ir_arena))
        .unwrap();

    let IRNodeEdgeKeys {
        out_nodes,
        in_edges,
        ..
    } = current_ir_node_edges;

    assert_eq!(out_nodes.len(), 1);
    assert_eq!(in_edges.len(), 1);

    let current_in_edge_key = in_edges[0];

    let consumer_node = out_nodes[0];

    let mut iter = ir_arena
        .get_mut(consumer_node)
        .inputs_mut()
        .enumerate()
        .filter(|(_, node)| **node == current_ir_node);

    let (consumer_node_input_idx, node) = iter.next().unwrap();
    *node = input_to_current_ir_node;
    assert!(iter.next().is_none());
    drop(iter);

    let [
        Some(IRNodeEdgeKeys {
            in_edges: consumer_node_in_edges,
            ..
        }),
        Some(IRNodeEdgeKeys {
            out_edges: out_edges_of_new_input_node,
            out_nodes: out_nodes_of_new_input_node,
            ..
        }),
    ] = ir_node_to_edge_keys_map.get_disjoint_mut([
        &IRNodeKey::new(consumer_node, ir_arena),
        &IRNodeKey::new(input_to_current_ir_node, ir_arena),
    ])
    else {
        unreachable!()
    };

    let out_edge_idx_in_new_input_node = out_edges_of_new_input_node
        .iter()
        .position(|k| *k == current_in_edge_key)
        .unwrap();

    out_edges_of_new_input_node[out_edge_idx_in_new_input_node] =
        consumer_node_in_edges[consumer_node_input_idx];
    out_nodes_of_new_input_node[out_edge_idx_in_new_input_node] = consumer_node;
}
