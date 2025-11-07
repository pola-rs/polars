use std::ops::{BitOr, BitOrAssign};

use polars_utils::arena::Arena;

use crate::dsl::EvalVariant;
use crate::plans::{AExpr, IRAggExpr};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ColumnOrderObserved;

/// Tracks orders that can be observed in the output of an expression.
///
/// This also allows distinguishing if an output is strictly column ordered (i.e. contains no other
/// observable ordering).
///
/// This currently does not support distinguishing the origin(s) of independent orders.
#[repr(u8)]
#[derive(Debug, Clone, Copy)]
pub enum ObservableOrders {
    /// No ordering can be observed.
    None = 0b00,

    /// Ordering of a column can be observed. Note that this does not capture information on whether
    /// the column itself is ordered (e.g. this is not the case after an unstable unique).
    Column = 0b01,

    /// Order originating from a non-column node can be observed.
    /// E.g.: sort()
    Independent = 0b10,

    /// Both the ordering of a column, as well as independent ordering can be observed.
    /// E.g.: explode()
    Both = 0b11,
}

impl BitOr for ObservableOrders {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self::from_u8((self as u8) | (rhs as u8)).unwrap()
    }
}

impl BitOrAssign for ObservableOrders {
    fn bitor_assign(&mut self, rhs: Self) {
        *self = Self::from_u8((*self as u8) | (rhs as u8)).unwrap();
    }
}

impl ObservableOrders {
    pub const fn from_u8(v: u8) -> Option<Self> {
        Some(match v {
            0b00 => Self::None,
            0b01 => Self::Column,
            0b10 => Self::Independent,
            0b11 => Self::Both,

            _ => return None,
        })
    }

    /// Combines output ordering for expressions being projected alongside each other.
    ///
    /// Returns `Err(ColumnOrderObserved)` if a side contains column ordering and the other side
    /// contains a non-column ordering.
    pub fn zip_with(self, other: Self) -> Result<Self, ColumnOrderObserved> {
        use ObservableOrders as O;

        match (self, other) {
            (v, O::None)
            | (O::None, v)
            | (v @ O::Independent, O::Independent)
            | (v @ O::Column, O::Column) => Ok(v),

            // Otherwise, one side contains column ordering, and the other side
            // contains independent ordering, which observes the column ordering.
            _ => Err(ColumnOrderObserved),
        }
    }

    pub fn column_ordering_observable(self) -> bool {
        matches!(self, Self::Column | Self::Both)
    }
}

pub fn zip(
    orders: impl IntoIterator<Item = Result<ObservableOrders, ColumnOrderObserved>>,
) -> Result<ObservableOrders, ColumnOrderObserved> {
    let mut output_order = ObservableOrders::None;
    for order in orders {
        output_order = output_order.zip_with(order?)?;
    }
    Ok(output_order)
}

pub fn adjust_for_with_columns_context(
    order: Result<ObservableOrders, ColumnOrderObserved>,
) -> Result<ObservableOrders, ColumnOrderObserved> {
    order?.zip_with(ObservableOrders::Column)
}

/// Returns the observable orderings in the output of this `AExpr`.
///
/// If within the expression tree an expression observes a `Column` ordering, this instead returns
/// `Err(ColumnOrderObserved)`.
pub fn resolve_observable_orders(
    aexpr: &AExpr,
    expr_arena: &Arena<AExpr>,
) -> Result<ObservableOrders, ColumnOrderObserved> {
    ObservableOrdersResolver::new(ObservableOrders::Column, expr_arena)
        .resolve_observable_orders(aexpr)
}

pub(super) struct ObservableOrdersResolver<'a> {
    column_ordering: ObservableOrders,
    expr_arena: &'a Arena<AExpr>,
}

impl<'a> ObservableOrdersResolver<'a> {
    pub(super) fn new(column_ordering: ObservableOrders, expr_arena: &'a Arena<AExpr>) -> Self {
        Self {
            column_ordering,
            expr_arena,
        }
    }

    #[recursive::recursive]
    pub(super) fn resolve_observable_orders(
        &self,
        aexpr: &AExpr,
    ) -> Result<ObservableOrders, ColumnOrderObserved> {
        macro_rules! rec {
            ($expr:expr) => {{ self.resolve_observable_orders(self.expr_arena.get($expr))? }};
        }

        macro_rules! zip {
            ($($expr:expr),*) => {{ zip([$(Ok(rec!($expr))),*])? }};
        }

        use ObservableOrders as O;
        Ok(match aexpr {
            // This should never reached as we don't recurse on the Eval evaluation expression.
            AExpr::Element => unreachable!(),

            // Explode creates local orders.
            //
            // The following observes order:
            //
            // a: [[1, 2], [3]]
            // b: [[3], [4, 5]]
            //
            // col(a).explode() * col(b).explode()
            AExpr::Explode { expr, .. } => rec!(*expr) | O::Independent,

            AExpr::Column(_) => self.column_ordering,
            AExpr::Literal(lv) if lv.is_scalar() => O::None,
            AExpr::Literal(_) => O::Independent,

            AExpr::Cast { expr, .. } => rec!(*expr),

            // Elementwise can be seen as a `zip + op`.
            AExpr::BinaryExpr { left, op: _, right } => zip!(*left, *right),
            AExpr::Ternary {
                predicate,
                truthy,
                falsy,
            } => zip!(*predicate, *truthy, *falsy),

            // Filter has to check whether zipping observes order, otherwise it propagates expr order.
            AExpr::Filter { input, by } => {
                let input = rec!(*input);
                input.zip_with(rec!(*by))?;
                input
            },

            AExpr::Sort { expr, options } => {
                if options.maintain_order {
                    rec!(*expr) | O::Independent
                } else {
                    _ = rec!(*expr);
                    O::Independent
                }
            },
            AExpr::SortBy {
                expr,
                by,
                sort_options,
            } => {
                let mut zipped = rec!(*expr);
                for e in by {
                    zipped = zipped.zip_with(rec!(*e))?;
                }

                if sort_options.maintain_order {
                    zipped | O::Independent
                } else {
                    O::Independent
                }
            },
            // Fow now only non-observing aggregations
            AExpr::AnonymousStreamingAgg {
                input: _,
                fmt_str: _,
                function: _,
            } => O::None,
            AExpr::Agg(agg) => match agg {
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
                    // Input order is deregarded, but must not observe order.
                    _ = rec!(*node);
                    O::None
                },
                IRAggExpr::Quantile { expr, quantile, .. } => {
                    // Input and quantile order is deregarded, but must not observe order.
                    _ = rec!(*expr);
                    _ = rec!(*quantile);
                    O::None
                },

                // Input order observing aggregations.
                IRAggExpr::Implode(node) | IRAggExpr::First(node) | IRAggExpr::Last(node) => {
                    if rec!(*node).column_ordering_observable() {
                        return Err(ColumnOrderObserved);
                    }
                    O::None
                },

                // @NOTE: This aggregation makes very little sense. We do the most pessimistic thing
                // possible here.
                IRAggExpr::AggGroups(node) => {
                    if rec!(*node).column_ordering_observable() {
                        return Err(ColumnOrderObserved);
                    }

                    O::Independent
                },
            },

            AExpr::Gather {
                expr,
                idx,
                returns_scalar,
            } => {
                let expr = rec!(*expr);
                let idx = rec!(*idx);

                // We need to ensure that the values come in column order. The order of the idxes is
                // propagated.
                if expr.column_ordering_observable() {
                    return Err(ColumnOrderObserved);
                }

                if *returns_scalar { O::None } else { idx }
            },
            AExpr::AnonymousFunction { input, options, .. }
            | AExpr::Function { input, options, .. } => {
                let input_ordering = if input.is_empty() {
                    O::None
                } else {
                    zip(input.iter().map(|e| Ok(rec!(e.node()))))?
                };

                if input_ordering.column_ordering_observable()
                    && options.flags.observes_input_order()
                {
                    return Err(ColumnOrderObserved);
                }

                match (
                    options.flags.terminates_input_order(),
                    options.flags.non_order_producing(),
                ) {
                    (false, false) => input_ordering | O::Independent,
                    (false, true) => input_ordering,
                    (true, false) => O::Independent,
                    (true, true) => O::None,
                }
            },

            AExpr::Eval {
                expr,
                evaluation: _,
                variant,
            } => match variant {
                EvalVariant::Array { as_list: _ }
                | EvalVariant::ArrayAgg
                | EvalVariant::List
                | EvalVariant::ListAgg => rec!(*expr),
                EvalVariant::Cumulative { min_samples: _ } => {
                    let expr = rec!(*expr);
                    if expr.column_ordering_observable() {
                        return Err(ColumnOrderObserved);
                    }
                    expr
                },
            },

            AExpr::Window {
                function,
                partition_by,
                order_by,
                options: _,
            } => {
                let input = rec!(*function);

                // @Performance.
                // All of the code below might be a bit pessimistic, several window function variants
                // are length preserving and/or propagate order in specific ways.
                if input.column_ordering_observable() {
                    return Err(ColumnOrderObserved);
                }
                for e in partition_by {
                    if rec!(*e).column_ordering_observable() {
                        return Err(ColumnOrderObserved);
                    }
                }
                if let Some((e, _)) = &order_by
                    && rec!(*e).column_ordering_observable()
                {
                    return Err(ColumnOrderObserved);
                }
                O::Independent
            },
            AExpr::Slice {
                input,
                offset,
                length,
            } => {
                // @NOTE
                // `offset` and `length` are supposed to be scalars, they have to resolved as they
                // might be order observing, but are not important for the output order.
                _ = rec!(*offset);
                _ = rec!(*length);

                let input = rec!(*input);
                if input.column_ordering_observable() {
                    return Err(ColumnOrderObserved);
                }
                input
            },
            AExpr::Len => O::None,
        })
    }
}
