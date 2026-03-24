use std::ops::{BitOr, BitOrAssign};
use std::sync::atomic::{AtomicU8, Ordering};

use bitflags::bitflags;
use polars_core::prelude::{InitHashMaps, PlHashMap, PlHashSet};
use polars_error::PolarsResult;
use polars_utils::UnitVec;
use polars_utils::arena::{Arena, Node};
use polars_utils::relaxed_cell::RelaxedCell;
use slotmap::{SlotMap, new_key_type};

use crate::dsl::EvalVariant;
use crate::plans::{AExpr, IRAggExpr, IRFunctionExpr, is_scalar_ae};

/// Tracks orders that can be observed in the output of an expression.
///
/// This also allows distinguishing if an output is strictly column ordered (i.e. contains no other
/// observable ordering).
///
/// This currently does not support distinguishing the origin(s) of independent orders.
bitflags! {
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub(crate) struct ObservableOrders: u8 {
        /// Ordering of a column can be observed. Note that this does not capture information on whether
        /// the column itself is ordered (e.g. this is not the case after an unstable unique).
        const COLUMN = 1 << 0;
        /// Order originating from a non-column node can be observed.
        /// E.g.: sort()
        const INDEPENDENT = 1 << 1;
    }
}

#[derive(Default)]
pub(crate) struct ZipState {
    pub(crate) saw_mixed_inputs: bool,
    num_ordered_inputs: usize,

    #[cfg(debug_assertions)]
    called: bool,
}

impl ZipState {
    /// Note: Must not be called with a non-empty initial accumulator, as that will cause `num_ordered_inputs` tracking
    /// to be incorrect.
    pub(crate) fn reduce(
        &mut self,
        left: ObservableOrders,
        right: ObservableOrders,
    ) -> ObservableOrders {
        use ObservableOrders as O;

        #[cfg(debug_assertions)]
        if !self.called {
            assert!(left.is_empty());
            self.called = true;
        }

        // Mainly want to catch Column<>Independent.
        // In the general case we also catch Independent<>Independent since we don't have ordering
        // provenance information to check ordering equality.
        self.saw_mixed_inputs |= (left.contains(O::INDEPENDENT) && right != O::empty())
            || (right.contains(O::INDEPENDENT) && left != O::empty());

        self.num_ordered_inputs += (right != O::empty()) as usize;

        left | right
    }
}

pub(crate) struct ExprOrderSimplifier<'a> {
    column_ordering: ObservableOrders,
    struct_field_ordering: Option<ObservableOrders>,
    list_element_ordering: ObservableOrders,

    /// Entries for nodes whose subtrees will not change on revisits with different recursion states.
    /// Guide: Any aexpr node that recurses with `RecursionState::new(false)`.
    revisit_cache: PlHashMap<Node, ObservableOrders>,
    internal_observer: OrderObserver,

    expr_arena: &'a mut Arena<AExpr>,
}

struct OrderObserver(
    // Atomic for interior mutability
    AtomicU8,
);

impl OrderObserver {
    fn new() -> Self {
        Self(AtomicU8::new(0))
    }

    fn observed_orders(&self) -> ObservableOrders {
        ObservableOrders::from_bits(self.0.load(Ordering::Relaxed)).unwrap()
    }

    fn observe_orders(&self, observable_orders: ObservableOrders) {
        self.0
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |x| {
                Some((ObservableOrders::from_bits(x).unwrap() | observable_orders).bits())
            })
            .unwrap();
    }
}

impl<'a> ExprOrderSimplifier<'a> {
    pub(crate) fn new(column_ordered: Option<bool>, expr_arena: &'a mut Arena<AExpr>) -> Self {
        Self {
            column_ordering: match column_ordered {
                Some(true) | None => ObservableOrders::COLUMN,
                Some(false) => ObservableOrders::empty(),
            },
            struct_field_ordering: None,
            list_element_ordering: ObservableOrders::INDEPENDENT,

            revisit_cache: PlHashMap::new(),
            internal_observer: OrderObserver::new(),

            expr_arena,
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum RecursionState {
    /// Inverted from "observe", since it could be unknown whether a consumer observes order.
    /// But it is always known whether deordering rewrites should be allowed.
    Recurse {
        allow_deordering_rewrites: bool,
    },
    DeorderAndReturn,
}

impl RecursionState {
    const fn new(allow_deordering_rewrites: bool) -> Self {
        Self::Recurse {
            allow_deordering_rewrites,
        }
    }

    fn to_zip_previsit_state(&self) -> Self {
        use RecursionState::*;

        match self {
            Recurse {
                allow_deordering_rewrites: _,
            } => Recurse {
                allow_deordering_rewrites: false,
            },
            DeorderAndReturn => DeorderAndReturn,
        }
    }

    fn to_zip_revisit_deorder_state(&self) -> Option<Self> {
        use RecursionState::*;

        match self {
            Recurse {
                allow_deordering_rewrites,
            } => allow_deordering_rewrites.then_some(Self::DeorderAndReturn),
            DeorderAndReturn => None,
        }
    }

    fn allows_deordering_rewrites(&self) -> bool {
        match self {
            Self::Recurse {
                allow_deordering_rewrites,
            } => *allow_deordering_rewrites,
            Self::DeorderAndReturn => true,
        }
    }
}

impl ExprOrderSimplifier<'_> {
    /// Returns all orderings that were observed by this expression tree.
    pub(crate) fn simplify_and_resolve_observable_orderings(
        &mut self,
        current_ae_node: Node,
        observe_top_order: Option<bool>,
    ) -> ObservableOrders {
        self.list_element_ordering = ObservableOrders::INDEPENDENT;

        self.rec(
            current_ae_node,
            RecursionState::Recurse {
                allow_deordering_rewrites: observe_top_order == Some(false),
            },
        )
    }

    pub(crate) fn internally_observed_orders(&self) -> ObservableOrders {
        self.internal_observer.observed_orders()
    }

    fn internal_observe(&self, observable_orders: ObservableOrders) {
        self.internal_observer.observe_orders(observable_orders);
    }

    #[recursive::recursive]
    fn rec<'os>(&mut self, current_ae_node: Node, recursion: RecursionState) -> ObservableOrders {
        const NO_DEORDER: RecursionState = RecursionState::new(false);
        const ALLOW_DEORDER: RecursionState = RecursionState::new(true);

        use ObservableOrders as O;

        match self.expr_arena.get(current_ae_node) {
            AExpr::Column(_) => self.column_ordering,

            AExpr::Literal(lv) => {
                if lv.is_scalar() {
                    O::empty()
                } else {
                    O::INDEPENDENT
                }
            },

            AExpr::Eval {
                expr,
                evaluation,
                variant,
            } => {
                if let Some(o) = self.revisit_cache.get(&current_ae_node) {
                    return *o;
                }

                let expr = *expr;
                let evaluation = *evaluation;

                let out = match variant {
                    EvalVariant::Array { as_list: _ }
                    | EvalVariant::ArrayAgg
                    | EvalVariant::List
                    | EvalVariant::ListAgg => self.rec(expr, NO_DEORDER),
                    EvalVariant::Cumulative { min_samples: _ } => {
                        let expr_ordering = self.rec(expr, NO_DEORDER);

                        expr_ordering | O::INDEPENDENT
                    },
                };

                let sublist_observable_orderings = self.rec(evaluation, NO_DEORDER);
                // Register this ordering to the global list item ordering.
                // TODO: This would cause effects between exprs from unrelated subtrees; should
                // be stored in `ObservableOrders` instead.
                self.list_element_ordering |= sublist_observable_orderings;

                self.revisit_cache.insert(current_ae_node, out);

                out
            },
            AExpr::Element => self.list_element_ordering,

            #[cfg(feature = "dtype-struct")]
            AExpr::StructEval { expr, evaluation } => {
                if let Some(o) = self.revisit_cache.get(&current_ae_node) {
                    return *o;
                }

                let evaluation_len = evaluation.len();

                let struct_expr = *expr;
                let subtree_recursion = recursion.to_zip_previsit_state();
                let struct_field_ordering = self.rec(struct_expr, subtree_recursion);

                let prev_struct_field_ordering =
                    self.struct_field_ordering.replace(struct_field_ordering);

                let mut zs = ZipState::default();

                let evaluation_observable = (0..evaluation_len).fold(O::empty(), |acc, i| {
                    let AExpr::StructEval { evaluation, .. } = self.expr_arena.get(current_ae_node)
                    else {
                        unreachable!()
                    };

                    zs.reduce(acc, self.rec(evaluation[i].node(), subtree_recursion))
                });

                let output_observable = zs.reduce(struct_field_ordering, evaluation_observable);

                if zs.saw_mixed_inputs {
                    self.internal_observe(output_observable);
                    self.revisit_cache
                        .insert(current_ae_node, output_observable);
                } else if zs.num_ordered_inputs <= 1 {
                    self.struct_field_ordering = prev_struct_field_ordering;

                    let output_observable =
                        if let Some(recursion) = recursion.to_zip_revisit_deorder_state() {
                            self.rec(current_ae_node, recursion)
                        } else {
                            output_observable
                        };

                    self.revisit_cache
                        .insert(current_ae_node, output_observable);

                    return output_observable;
                }

                self.struct_field_ordering = prev_struct_field_ordering;

                output_observable
            },

            AExpr::BinaryExpr { .. } | AExpr::Ternary { .. } => {
                if let Some(o) = self.revisit_cache.get(&current_ae_node) {
                    return *o;
                }

                let (nodes, ternary_predicate_node) = match self.expr_arena.get(current_ae_node) {
                    AExpr::BinaryExpr { left, op: _, right } => ([*left, *right], None),
                    AExpr::Ternary {
                        predicate,
                        truthy,
                        falsy,
                    } => ([*truthy, *falsy], Some(*predicate)),
                    _ => unreachable!(),
                };

                let mut zs = ZipState::default();
                let subtree_recursion = recursion.to_zip_previsit_state();

                let output_observable = nodes.iter().fold(O::empty(), |acc, node| {
                    zs.reduce(acc, self.rec(*node, subtree_recursion))
                });

                if let Some(ternary_predicate_node) = ternary_predicate_node {
                    zs.reduce(
                        output_observable,
                        self.rec(ternary_predicate_node, subtree_recursion),
                    );
                }

                if zs.saw_mixed_inputs {
                    self.internal_observe(output_observable);
                    self.revisit_cache
                        .insert(current_ae_node, output_observable);
                } else if zs.num_ordered_inputs <= 1 {
                    let output_observable =
                        if let Some(recursion) = recursion.to_zip_revisit_deorder_state() {
                            self.rec(current_ae_node, recursion)
                        } else {
                            output_observable
                        };

                    self.revisit_cache
                        .insert(current_ae_node, output_observable);

                    return output_observable;
                }

                output_observable
            },

            #[cfg(feature = "dtype-struct")]
            AExpr::StructField(_) => self.struct_field_ordering.unwrap(),

            AExpr::Cast { expr, .. } => self.rec(*expr, recursion),
            AExpr::Explode { expr, .. } => {
                let observable_in_input = self.rec(*expr, recursion);

                observable_in_input | O::INDEPENDENT
            },
            AExpr::Len => O::empty(),
            AExpr::Sort { expr, options } => {
                let expr = *expr;
                let maintain_order = options.maintain_order;

                if recursion.allows_deordering_rewrites() {
                    self.expr_arena
                        .replace(current_ae_node, self.expr_arena.get(expr).clone());
                    return self.rec(current_ae_node, ALLOW_DEORDER);
                } else {
                    if maintain_order {
                        self.rec(expr, NO_DEORDER) | O::INDEPENDENT
                    } else {
                        self.rec(expr, ALLOW_DEORDER);
                        O::INDEPENDENT
                    }
                }
            },

            AExpr::Filter { input, by } => {
                if let Some(o) = self.revisit_cache.get(&current_ae_node) {
                    return *o;
                }

                let input = *input;
                let by = *by;

                let observable_in_input = self.rec(input, NO_DEORDER);
                let observable_in_by = self.rec(by, NO_DEORDER);

                let mut zs = ZipState::default();
                let observable_in_input = zs.reduce(O::empty(), observable_in_input);
                let zipped_observable = zs.reduce(observable_in_input, observable_in_by);

                if zs.saw_mixed_inputs {
                    self.internal_observe(zipped_observable);
                }

                self.revisit_cache
                    .insert(current_ae_node, observable_in_input);

                observable_in_input
            },

            AExpr::Gather {
                expr,
                idx,
                returns_scalar,
                null_on_oob: _,
            } => {
                let expr = *expr;
                let idx = *idx;
                let returns_scalar = *returns_scalar;

                if let Some(o) = self.revisit_cache.get(&current_ae_node) {
                    return *o;
                }

                let observable_in_expr = self.rec(expr, NO_DEORDER);
                let observable_in_idx = self.rec(idx, NO_DEORDER);

                self.internal_observe(observable_in_expr);

                let out = if returns_scalar {
                    O::empty()
                } else {
                    observable_in_idx
                };

                self.revisit_cache.insert(current_ae_node, out);

                out
            },

            AExpr::Over {
                function,
                partition_by,
                order_by,
                mapping: _,
            } => {
                if let Some(o) = self.revisit_cache.get(&current_ae_node) {
                    return *o;
                }

                let function = *function;
                let partition_by_len = partition_by.len();
                let order_by = order_by.as_ref().map(|(node, _)| *node);

                let observable_in_function = self.rec(function, NO_DEORDER);
                let observable_in_partition_by =
                    (0..partition_by_len).fold(O::empty(), |acc, i| {
                        let AExpr::Over { partition_by, .. } = self.expr_arena.get(current_ae_node)
                        else {
                            unreachable!()
                        };
                        acc | self.rec(partition_by[i], NO_DEORDER)
                    });
                let observable_in_order_by =
                    order_by.map_or(O::empty(), |node| self.rec(node, NO_DEORDER));

                let zipped_observable =
                    observable_in_function | observable_in_partition_by | observable_in_order_by;
                self.internal_observe(zipped_observable);

                let output_observable = zipped_observable | O::INDEPENDENT;

                self.revisit_cache
                    .insert(current_ae_node, output_observable);

                output_observable
            },

            #[cfg(feature = "dynamic_group_by")]
            AExpr::Rolling {
                function,
                index_column,
                period: _,
                offset: _,
                closed_window: _,
            } => {
                if let Some(o) = self.revisit_cache.get(&current_ae_node) {
                    return *o;
                }

                let function = *function;
                let index_column = *index_column;

                let observable_in_function = self.rec(function, NO_DEORDER);
                let observable_in_index_column = self.rec(index_column, NO_DEORDER);

                self.internal_observe(observable_in_function);
                self.internal_observe(observable_in_index_column);

                let output_observable =
                    observable_in_function | observable_in_index_column | O::INDEPENDENT;
                self.revisit_cache
                    .insert(current_ae_node, output_observable);

                output_observable
            },

            AExpr::SortBy {
                expr,
                by,
                sort_options,
            } => {
                let expr = *expr;
                let maintain_order = sort_options.maintain_order;
                let by_len = by.len();

                if recursion.allows_deordering_rewrites() {
                    self.expr_arena
                        .replace(current_ae_node, self.expr_arena.get(expr).clone());
                    return self.rec(current_ae_node, ALLOW_DEORDER);
                }

                let mut zs = ZipState::default();
                let observable_in_input = self.rec(expr, NO_DEORDER);

                let observable_in_by = (0..by_len).fold(O::empty(), |acc, i| {
                    let AExpr::SortBy { by, .. } = self.expr_arena.get(current_ae_node) else {
                        unreachable!()
                    };
                    zs.reduce(acc, self.rec(by[i], NO_DEORDER))
                });

                let zipped_observable = zs.reduce(observable_in_input, observable_in_by);

                if zs.saw_mixed_inputs {
                    self.internal_observe(zipped_observable);
                } else if zs.num_ordered_inputs == 0 {
                    return self.rec(current_ae_node, RecursionState::DeorderAndReturn);
                }

                if maintain_order {
                    zipped_observable | O::INDEPENDENT
                } else {
                    O::INDEPENDENT
                }
            },

            AExpr::Slice {
                input,
                offset,
                length,
            } => {
                let input = *input;
                let offset = *offset;
                let length = *length;

                debug_assert!(is_scalar_ae(offset, self.expr_arena));
                debug_assert!(is_scalar_ae(length, self.expr_arena));

                let observable_in_offset = self.rec(offset, NO_DEORDER);
                let observable_in_length = self.rec(length, NO_DEORDER);

                debug_assert_eq!(observable_in_offset, O::empty());
                debug_assert_eq!(observable_in_length, O::empty());

                let output_observable = self.rec(input, recursion);

                self.internal_observe(output_observable);

                output_observable
            },

            AExpr::Function {
                input,
                function: IRFunctionExpr::MinBy | IRFunctionExpr::MaxBy,
                ..
            } => {
                if let Some(o) = self.revisit_cache.get(&current_ae_node) {
                    return *o;
                }

                assert_eq!(input.len(), 2);
                let of = input[0].node();
                let by = input[1].node();

                let observable_in_of = self.rec(of, NO_DEORDER);
                let observable_in_by = self.rec(by, NO_DEORDER);

                self.internal_observe(observable_in_of);
                self.internal_observe(observable_in_by);

                let output_observable = O::empty();
                self.revisit_cache
                    .insert(current_ae_node, output_observable);

                output_observable
            },

            AExpr::AnonymousFunction { input, options, .. }
            | AExpr::Function { input, options, .. } => {
                if let Some(o) = self.revisit_cache.get(&current_ae_node) {
                    return *o;
                }

                let input_len = input.len();
                let observes_input_order = options.flags.observes_input_order();
                let terminates_input_order = options.flags.terminates_input_order();
                let non_order_producing = options.flags.non_order_producing();

                let subtree_recursion = if observes_input_order {
                    NO_DEORDER
                } else {
                    recursion.to_zip_previsit_state()
                };

                let mut zs = ZipState::default();

                let zipped_observable = (0..input_len).fold(O::empty(), |acc, i| {
                    let (AExpr::AnonymousFunction { input, .. } | AExpr::Function { input, .. }) =
                        self.expr_arena.get(current_ae_node)
                    else {
                        unreachable!()
                    };

                    zs.reduce(acc, self.rec(input[i].node(), subtree_recursion))
                });

                let output_observable = match (terminates_input_order, non_order_producing) {
                    (false, false) => zipped_observable | O::INDEPENDENT,
                    (false, true) => zipped_observable,
                    (true, false) => O::INDEPENDENT,
                    (true, true) => O::empty(),
                };

                if zs.saw_mixed_inputs {
                    self.internal_observe(zipped_observable);
                    self.revisit_cache
                        .insert(current_ae_node, output_observable);
                } else if !observes_input_order {
                    let output_observable =
                        if let Some(recursion) = recursion.to_zip_revisit_deorder_state() {
                            self.rec(current_ae_node, recursion)
                        } else if recursion != RecursionState::DeorderAndReturn
                            && terminates_input_order
                        {
                            self.rec(current_ae_node, RecursionState::DeorderAndReturn)
                        } else {
                            output_observable
                        };

                    self.revisit_cache
                        .insert(current_ae_node, output_observable);

                    return output_observable;
                };

                output_observable
            },

            AExpr::AnonymousAgg {
                input,
                fmt_str: _,
                function: _,
            } => {
                if let Some(o) = self.revisit_cache.get(&current_ae_node) {
                    return *o;
                }

                let input_len = input.len();

                let zipped_observable = (0..input_len).fold(O::empty(), |acc, i| {
                    let AExpr::AnonymousAgg { input, .. } = self.expr_arena.get(current_ae_node)
                    else {
                        unreachable!()
                    };

                    acc | self.rec(input[i].node(), NO_DEORDER)
                });

                self.internal_observe(zipped_observable);

                let output_observable = zipped_observable | O::INDEPENDENT;

                self.revisit_cache
                    .insert(current_ae_node, output_observable);

                output_observable
            },

            AExpr::Agg(agg) => {
                if let Some(o) = self.revisit_cache.get(&current_ae_node) {
                    return *o;
                }

                let output_observable = match agg {
                    // Input order agnostic aggregations.
                    IRAggExpr::Min { input: node, .. }
                    | IRAggExpr::Max { input: node, .. }
                    | IRAggExpr::Median(node)
                    | IRAggExpr::NUnique(node)
                    | IRAggExpr::Mean(node)
                    | IRAggExpr::Sum(node)
                    | IRAggExpr::Count { input: node, .. }
                    | IRAggExpr::Std(node, _)
                    | IRAggExpr::Var(node, _)
                    | IRAggExpr::Item { input: node, .. } => {
                        self.rec(*node, ALLOW_DEORDER);
                        O::empty()
                    },
                    IRAggExpr::Quantile { expr, quantile, .. } => {
                        let expr = *expr;
                        let quantile = *quantile;
                        self.rec(expr, ALLOW_DEORDER);
                        self.rec(quantile, ALLOW_DEORDER);

                        O::empty()
                    },

                    // Input order observing aggregations.
                    IRAggExpr::First(node)
                    | IRAggExpr::FirstNonNull(node)
                    | IRAggExpr::Last(node)
                    | IRAggExpr::LastNonNull(node) => {
                        let observable = self.rec(*node, NO_DEORDER);

                        self.internal_observe(observable);

                        O::empty()
                    },

                    IRAggExpr::Implode {
                        input,
                        maintain_order,
                    } => {
                        let input = *input;
                        let maintain_order = *maintain_order;

                        let input_observable =
                            self.rec(input, RecursionState::new(!maintain_order));

                        if input_observable.is_empty() && maintain_order {
                            self.expr_arena.replace(
                                current_ae_node,
                                AExpr::Agg(IRAggExpr::Implode {
                                    input,
                                    maintain_order: false,
                                }),
                            );
                        }

                        O::empty()
                    },

                    IRAggExpr::AggGroups(node) => {
                        let observable = self.rec(*node, NO_DEORDER);

                        self.internal_observe(observable);

                        observable | O::INDEPENDENT
                    },
                };

                self.revisit_cache
                    .insert(current_ae_node, output_observable);
                output_observable
            },
        }
    }
}
