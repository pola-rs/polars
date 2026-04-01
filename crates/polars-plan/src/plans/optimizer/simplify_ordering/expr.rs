use bitflags::bitflags;
use polars_core::prelude::PlHashMap;
use polars_utils::arena::{Arena, Node};

use crate::dsl::EvalVariant;
use crate::plans::{AExpr, IRAggExpr, IRFunctionExpr, is_length_preserving_ae};

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

use _order_acc::ExprOrderAcc;

mod _order_acc {
    use polars_utils::arena::Node;

    use super::ObservableOrders;

    /// Order accumulator, tracks additional properties used to reason on projecting multiple exprs.
    #[derive(Default)]
    pub(crate) struct ExprOrderAcc {
        acc: ObservableOrders,
        /// Used to detect order observation triggered by projecting exprs with different ordering
        /// alongside each other.
        saw_mixed_inputs: bool,
        /// In the case of multiple projections de-ordering can only take place iff only a single
        /// one of those projections has ordering (and there were no mixed inputs). We cannot
        /// otherwise de-order multiple exprs as that would destroy horizontal ordering relations.
        num_ordered_inputs: usize,
        last_ordered_node: Option<Node>,
    }

    impl ExprOrderAcc {
        pub(crate) fn add(&mut self, right: ObservableOrders, right_node: Node) {
            use ObservableOrders as O;

            self.saw_mixed_inputs |= (self.acc.contains(O::INDEPENDENT) && !right.is_empty())
                || (right.contains(O::INDEPENDENT) && !self.acc.is_empty());

            if !right.is_empty() {
                self.num_ordered_inputs += 1;
                self.last_ordered_node = Some(right_node);
            }

            self.acc |= right;
        }

        pub(crate) fn accumulated_orders(&self) -> ObservableOrders {
            self.acc
        }

        pub(crate) fn saw_mixed_inputs(&self) -> bool {
            self.saw_mixed_inputs
        }

        pub(super) fn single_ordered_node(&self) -> Option<Node> {
            (self.num_ordered_inputs == 1).then(|| self.last_ordered_node.unwrap())
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
struct RecursionState {
    allow_deorder: bool,
}

impl RecursionState {
    const NO_DEORDER: RecursionState = RecursionState {
        allow_deorder: false,
    };
    const ALLOW_DEORDER: RecursionState = RecursionState {
        allow_deorder: true,
    };

    fn allows_deorder(&self) -> bool {
        self.allow_deorder
    }
}

pub(crate) struct ExprOrderSimplifier<'a> {
    struct_field_ordering: Option<ObservableOrders>,

    /// Entries for nodes whose subtrees will no longer change when revisited with a de-ordering
    /// recursion state.
    revisit_cache: &'a mut PlHashMap<Node, ObservableOrders>,
    internally_observed: ObservableOrders,

    expr_arena: &'a mut Arena<AExpr>,
}

impl<'a> ExprOrderSimplifier<'a> {
    pub fn new(
        expr_arena: &'a mut Arena<AExpr>,
        revisit_cache: &'a mut PlHashMap<Node, ObservableOrders>,
    ) -> Self {
        Self {
            struct_field_ordering: None,

            revisit_cache,
            internally_observed: ObservableOrders::empty(),

            expr_arena,
        }
    }
}

impl ExprOrderSimplifier<'_> {
    pub fn simplify_projected_exprs(
        &mut self,
        ae_nodes: &[Node],
        allow_deordering_top: bool,
    ) -> ObservableOrders {
        let mut acc = ExprOrderAcc::default();

        for node in ae_nodes.iter().copied() {
            acc.add(self.rec(node, RecursionState::NO_DEORDER), node)
        }

        let acc_observable = acc.accumulated_orders();

        if acc.saw_mixed_inputs() {
            self.internal_observe(acc_observable);
        }

        if let Some(node) = acc.single_ordered_node()
            && allow_deordering_top
        {
            self.rec(node, RecursionState::ALLOW_DEORDER)
        } else {
            acc_observable
        }
    }

    pub fn internally_observed_orders(&self) -> ObservableOrders {
        self.internally_observed
    }

    fn internal_observe(&mut self, observable_orders: ObservableOrders) {
        self.internally_observed |= observable_orders;
    }

    #[recursive::recursive]
    fn rec(&mut self, current_ae_node: Node, recursion: RecursionState) -> ObservableOrders {
        use ObservableOrders as O;
        use RecursionState as RS;

        macro_rules! check_return_cached {
            () => {
                if let Some(o) = self.revisit_cache.get(&current_ae_node) {
                    return *o;
                }
            };
        }

        macro_rules! cache_output {
            ($o:expr) => {
                let existing = self.revisit_cache.insert(current_ae_node, $o);
                debug_assert!(existing.is_none());
            };
        }

        match self.expr_arena.get_mut(current_ae_node) {
            AExpr::Column(_) => O::COLUMN,

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
                check_return_cached!();

                let expr = *expr;
                let evaluation = *evaluation;
                let variant = *variant;

                let mut expr_ordering = self.rec(expr, RS::NO_DEORDER);

                match variant {
                    EvalVariant::Array { as_list: _ }
                    | EvalVariant::ArrayAgg
                    | EvalVariant::List
                    | EvalVariant::ListAgg => {},
                    EvalVariant::Cumulative { min_samples: _ } => {
                        self.internal_observe(expr_ordering);
                        expr_ordering |= O::INDEPENDENT;
                    },
                };

                self.rec(evaluation, RS::NO_DEORDER);

                cache_output!(expr_ordering);

                expr_ordering
            },
            AExpr::Element => O::INDEPENDENT,

            #[cfg(feature = "dtype-struct")]
            AExpr::StructEval { expr, evaluation } => {
                check_return_cached!();

                let evaluation_len = evaluation.len();

                let struct_expr = *expr;
                let struct_field_ordering = self.rec(struct_expr, RS::NO_DEORDER);

                let prev_struct_field_ordering =
                    self.struct_field_ordering.replace(struct_field_ordering);

                let mut acc = ExprOrderAcc::default();
                acc.add(struct_field_ordering, struct_expr);

                for i in 0..evaluation_len {
                    let AExpr::StructEval { evaluation, .. } = self.expr_arena.get(current_ae_node)
                    else {
                        unreachable!()
                    };

                    let node = evaluation[i].node();
                    acc.add(self.rec(node, RS::NO_DEORDER), node);
                }

                let mut output_observable = acc.accumulated_orders();
                let mut should_cache = false;

                if acc.saw_mixed_inputs() {
                    self.internal_observe(output_observable);
                    should_cache = true;
                } else if let Some(node) = acc.single_ordered_node()
                    && recursion.allows_deorder()
                {
                    output_observable = self.rec(node, RS::ALLOW_DEORDER);
                    should_cache = true;
                }

                self.struct_field_ordering = prev_struct_field_ordering;

                if should_cache {
                    cache_output!(output_observable);
                }

                output_observable
            },

            #[cfg(feature = "dtype-struct")]
            AExpr::StructField(_) => self.struct_field_ordering.unwrap(),

            AExpr::BinaryExpr { .. } | AExpr::Ternary { .. } => {
                check_return_cached!();

                let (nodes, ternary_mask_node) = match self.expr_arena.get(current_ae_node) {
                    AExpr::BinaryExpr { left, op: _, right } => ([*left, *right], None),
                    AExpr::Ternary {
                        predicate,
                        truthy,
                        falsy,
                    } => ([*truthy, *falsy], Some(*predicate)),
                    _ => unreachable!(),
                };

                let mut acc = ExprOrderAcc::default();

                for node in nodes {
                    acc.add(self.rec(node, RS::NO_DEORDER), node);
                }

                let mut output_observable = acc.accumulated_orders();

                if let Some(ternary_mask_node) = ternary_mask_node {
                    acc.add(
                        self.rec(ternary_mask_node, RS::NO_DEORDER),
                        ternary_mask_node,
                    );
                }

                let mut should_cache = false;

                if acc.saw_mixed_inputs() {
                    self.internal_observe(output_observable);
                    should_cache = true;
                } else if let Some(node) = acc.single_ordered_node()
                    && recursion.allows_deorder()
                {
                    output_observable = self.rec(node, RS::ALLOW_DEORDER);

                    if Some(node) == ternary_mask_node {
                        output_observable = O::empty();
                    }

                    should_cache = true;
                }

                if should_cache {
                    cache_output!(output_observable);
                }

                output_observable
            },

            AExpr::Cast { expr, .. } => {
                let expr = *expr;
                self.rec(expr, recursion)
            },
            AExpr::Explode { expr, .. } => {
                let expr = *expr;
                let observable_in_input = self.rec(expr, recursion);

                observable_in_input | O::INDEPENDENT
            },
            AExpr::Len => O::empty(),
            AExpr::Sort { expr, options } => {
                let expr = *expr;
                debug_assert!(!options.maintain_order);
                let maintain_order = false;

                if recursion.allows_deorder() {
                    self.expr_arena
                        .replace(current_ae_node, self.expr_arena.get(expr).clone());

                    return self.rec(current_ae_node, recursion);
                }

                let mut out = self.rec(
                    expr,
                    RecursionState {
                        allow_deorder: !maintain_order,
                    },
                );

                if maintain_order {
                    out |= O::INDEPENDENT;
                } else {
                    out = O::INDEPENDENT;
                }

                out
            },

            AExpr::Filter { input, by } => {
                check_return_cached!();

                let input = *input;
                let by = *by;

                let observable_in_input = self.rec(input, RS::NO_DEORDER);
                let observable_in_by = self.rec(by, RS::NO_DEORDER);

                let mut acc = ExprOrderAcc::default();
                acc.add(observable_in_input, input);
                acc.add(observable_in_by, by);

                if acc.saw_mixed_inputs() {
                    self.internal_observe(acc.accumulated_orders());
                } else if observable_in_input.is_empty() && !observable_in_by.is_empty() {
                    self.rec(by, RS::ALLOW_DEORDER);
                }

                cache_output!(observable_in_input);

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

                check_return_cached!();

                let observable_in_expr = self.rec(expr, RS::NO_DEORDER);
                let observable_in_idx = self.rec(idx, RS::NO_DEORDER);

                self.internal_observe(observable_in_expr);

                let output_observable = if returns_scalar || observable_in_expr.is_empty() {
                    O::empty()
                } else {
                    observable_in_idx
                };

                cache_output!(output_observable);

                output_observable
            },

            AExpr::Over {
                function,
                partition_by,
                order_by,
                mapping: _,
            } => {
                check_return_cached!();

                let function = *function;
                let partition_by_len = partition_by.len();
                let order_by = order_by.as_ref().map(|(node, _)| *node);

                let observable_in_function = self.rec(function, RS::NO_DEORDER);
                let observable_in_partition_by = (0..partition_by_len)
                    .map(|i| {
                        let AExpr::Over { partition_by, .. } = self.expr_arena.get(current_ae_node)
                        else {
                            unreachable!()
                        };

                        self.rec(partition_by[i], RS::NO_DEORDER)
                    })
                    .fold(O::empty(), |acc, v| acc | v);
                let observable_in_order_by =
                    order_by.map_or(O::empty(), |node| self.rec(node, RS::NO_DEORDER));

                let acc_observable =
                    observable_in_function | observable_in_partition_by | observable_in_order_by;
                self.internal_observe(acc_observable);

                let output_observable = acc_observable | O::INDEPENDENT;

                cache_output!(output_observable);

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
                check_return_cached!();

                let function = *function;
                let index_column = *index_column;

                let observable_in_function = self.rec(function, RS::NO_DEORDER);
                let observable_in_index_column = self.rec(index_column, RS::NO_DEORDER);

                self.internal_observe(observable_in_function);
                self.internal_observe(observable_in_index_column);

                let output_observable =
                    observable_in_function | observable_in_index_column | O::INDEPENDENT;

                cache_output!(output_observable);

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

                if recursion.allows_deorder()
                    && is_length_preserving_ae(expr, self.expr_arena)
                    && (0..by_len).all(|i| {
                        let AExpr::SortBy { by, .. } = self.expr_arena.get(current_ae_node) else {
                            unreachable!()
                        };

                        let node = by[i];
                        is_length_preserving_ae(node, self.expr_arena)
                    })
                {
                    self.expr_arena
                        .replace(current_ae_node, self.expr_arena.get(expr).clone());

                    return self.rec(current_ae_node, recursion);
                }

                let mut acc = ExprOrderAcc::default();
                let observable_in_input = self.rec(expr, recursion);
                acc.add(observable_in_input, expr);

                for i in 0..by_len {
                    let AExpr::SortBy { by, .. } = self.expr_arena.get(current_ae_node) else {
                        unreachable!()
                    };

                    let node = by[i];
                    acc.add(self.rec(node, RS::NO_DEORDER), node);
                }

                if acc.saw_mixed_inputs() {
                    self.internal_observe(acc.accumulated_orders());
                }

                if maintain_order {
                    observable_in_input | O::INDEPENDENT
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

                let observable_in_offset = self.rec(offset, RS::NO_DEORDER);
                let observable_in_length = self.rec(length, RS::NO_DEORDER);
                let observable_in_input = self.rec(input, recursion);

                let mut acc = ExprOrderAcc::default();
                acc.add(observable_in_offset, offset);
                acc.add(observable_in_length, length);
                acc.add(observable_in_input, input);

                self.internal_observe(observable_in_input);

                if acc.saw_mixed_inputs() {
                    self.internal_observe(acc.accumulated_orders());
                }

                observable_in_input
            },

            AExpr::Function {
                input,
                function: IRFunctionExpr::MinBy | IRFunctionExpr::MaxBy,
                ..
            } => {
                check_return_cached!();

                assert_eq!(input.len(), 2);
                let of = input[0].node();
                let by = input[1].node();

                let observable_in_of = self.rec(of, RS::NO_DEORDER);
                let observable_in_by = self.rec(by, RS::NO_DEORDER);

                self.internal_observe(observable_in_of);
                self.internal_observe(observable_in_by);

                let output_observable = O::empty();

                cache_output!(output_observable);

                output_observable
            },

            AExpr::AnonymousFunction { input, options, .. }
            | AExpr::Function { input, options, .. } => {
                check_return_cached!();

                let input_len = input.len();
                let observes_input_order = options.flags.observes_input_order();
                let terminates_input_order = options.flags.terminates_input_order();
                let non_order_producing = options.flags.non_order_producing();

                let mut acc = ExprOrderAcc::default();

                for i in 0..input_len {
                    let (AExpr::AnonymousFunction { input, .. } | AExpr::Function { input, .. }) =
                        self.expr_arena.get(current_ae_node)
                    else {
                        unreachable!()
                    };

                    let node = input[i].node();
                    acc.add(self.rec(node, RS::NO_DEORDER), node);
                }

                if observes_input_order {
                    self.internal_observe(acc.accumulated_orders());
                }

                let mut should_cache = false;

                if acc.saw_mixed_inputs() {
                    should_cache = true;
                    self.internal_observe(acc.accumulated_orders());
                };

                let input_order = if let Some(node) = acc.single_ordered_node()
                    && !observes_input_order
                    && (recursion.allows_deorder() || terminates_input_order)
                {
                    should_cache = true;
                    self.rec(node, RS::ALLOW_DEORDER)
                } else {
                    acc.accumulated_orders()
                };

                let output_observable = match (terminates_input_order, non_order_producing) {
                    (false, false) => input_order | O::INDEPENDENT,
                    (false, true) => input_order,
                    (true, false) => O::INDEPENDENT,
                    (true, true) => O::empty(),
                };

                if should_cache {
                    cache_output!(output_observable);
                }

                output_observable
            },

            AExpr::AnonymousAgg {
                input,
                fmt_str: _,
                function: _,
            } => {
                check_return_cached!();

                let input_len = input.len();

                let acc_observable = (0..input_len)
                    .map(|i| {
                        let AExpr::AnonymousAgg { input, .. } =
                            self.expr_arena.get(current_ae_node)
                        else {
                            unreachable!()
                        };

                        self.rec(input[i].node(), RS::NO_DEORDER)
                    })
                    .fold(O::empty(), |acc, v| acc | v);

                self.internal_observe(acc_observable);

                let output_observable = acc_observable | O::INDEPENDENT;

                cache_output!(output_observable);

                output_observable
            },

            AExpr::Agg(agg) => {
                check_return_cached!();

                let output_observable = match agg {
                    IRAggExpr::First(node)
                    | IRAggExpr::FirstNonNull(node)
                    | IRAggExpr::Last(node)
                    | IRAggExpr::LastNonNull(node) => {
                        let node = *node;
                        let input_observable = self.rec(node, RS::NO_DEORDER);
                        self.internal_observe(input_observable);

                        O::empty()
                    },

                    IRAggExpr::Min { input: node, .. }
                    | IRAggExpr::Max { input: node, .. }
                    | IRAggExpr::Mean(node)
                    | IRAggExpr::Median(node)
                    | IRAggExpr::Sum(node)
                    | IRAggExpr::Item { input: node, .. } => {
                        let node = *node;
                        self.rec(node, RS::ALLOW_DEORDER);
                        O::empty()
                    },

                    IRAggExpr::NUnique(node)
                    | IRAggExpr::Count { input: node, .. }
                    | IRAggExpr::Std(node, _)
                    | IRAggExpr::Var(node, _) => {
                        let node = *node;
                        self.rec(node, RS::ALLOW_DEORDER);
                        O::empty()
                    },
                    IRAggExpr::Quantile { expr, quantile, .. } => {
                        let expr = *expr;
                        let quantile = *quantile;

                        self.rec(expr, RS::ALLOW_DEORDER);
                        let sublist_observable = self.rec(quantile, RS::NO_DEORDER);
                        self.internal_observe(sublist_observable);

                        O::empty()
                    },

                    IRAggExpr::Implode {
                        input,
                        maintain_order,
                    } => {
                        let input = *input;
                        let maintain_order = *maintain_order;

                        let sublist_observable = self.rec(
                            input,
                            RecursionState {
                                allow_deorder: !maintain_order,
                            },
                        );

                        let mut should_cache = !maintain_order;

                        if maintain_order {
                            self.internal_observe(sublist_observable);

                            // Note: De-ordering of implodes requires tracking orders at nesting
                            // levels.

                            if sublist_observable.is_empty() {
                                should_cache = true;

                                self.expr_arena.replace(
                                    current_ae_node,
                                    AExpr::Agg(IRAggExpr::Implode {
                                        input,
                                        maintain_order: false,
                                    }),
                                );
                            }
                        }

                        if !should_cache {
                            return O::empty();
                        }

                        O::empty()
                    },

                    IRAggExpr::AggGroups(node) => {
                        let node = *node;
                        let input_observable = self.rec(node, RS::NO_DEORDER);
                        self.internal_observe(input_observable);

                        input_observable | O::INDEPENDENT
                    },
                };

                cache_output!(output_observable);

                output_observable
            },
        }
    }
}
