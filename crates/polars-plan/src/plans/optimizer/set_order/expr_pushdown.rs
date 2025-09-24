use std::ops::{BitOr, BitOrAssign};

use polars_utils::arena::Arena;

use crate::dsl::EvalVariant;
use crate::plans::{AExpr, IRAggExpr};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameOrderObserved;

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExprOutputOrder {
    /// The expression has no defined output order.
    None = 0b00,

    /// The expression's output order is reliant on the input dataframe's order.
    Frame = 0b01,

    /// The expression's output order is completely independent from the frame order.
    Independent = 0b10,

    /// The expression's output order is both observing the frame order and some other independent
    /// order.
    Both = 0b11,
}

impl BitOr for ExprOutputOrder {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self::from_u8((self as u8) | (rhs as u8)).unwrap()
    }
}

impl BitOrAssign for ExprOutputOrder {
    fn bitor_assign(&mut self, rhs: Self) {
        *self = Self::from_u8((*self as u8) | (rhs as u8)).unwrap();
    }
}

impl ExprOutputOrder {
    pub const fn from_u8(v: u8) -> Option<Self> {
        Some(match v {
            0b00 => Self::None,
            0b01 => Self::Frame,
            0b10 => Self::Independent,
            0b11 => Self::Both,

            _ => return None,
        })
    }

    /// Do a elementwise zip between two output orderings.
    pub fn zip_with(self, other: Self) -> Result<Self, FrameOrderObserved> {
        match (self, other) {
            (Self::None, v) | (v, Self::None) => Ok(v),
            (Self::Both, _) | (_, Self::Both) => Err(FrameOrderObserved),
            (v, w) if v == w => Ok(v),
            _ => Err(FrameOrderObserved),
        }
    }

    /// Does the output order observe an ordering (in)directly derived from the frame ordering.
    pub fn has_frame_ordering(self) -> bool {
        matches!(self, Self::Frame | Self::Both)
    }
}

pub fn zip(
    orders: impl IntoIterator<Item = Result<ExprOutputOrder, FrameOrderObserved>>,
) -> Result<ExprOutputOrder, FrameOrderObserved> {
    use ExprOutputOrder as O;
    let mut orders = orders.into_iter();
    let Some(output_order) = orders.next() else {
        return Ok(O::Independent);
    };
    let mut output_order = output_order?;
    for order in orders {
        output_order = output_order.zip_with(order?)?;
    }
    Ok(output_order)
}

pub fn adjust_for_with_columns_context(
    order: Result<ExprOutputOrder, FrameOrderObserved>,
) -> Result<ExprOutputOrder, FrameOrderObserved> {
    order?.zip_with(ExprOutputOrder::Frame)
}

/// Determine whether the output observes the order of the expressions input frame.
///
/// This answers the question:
/// > Given that my output is (un)ordered, can my input be unordered?
#[recursive::recursive]
pub fn get_frame_observing(
    aexpr: &AExpr,
    expr_arena: &Arena<AExpr>,
) -> Result<ExprOutputOrder, FrameOrderObserved> {
    macro_rules! rec {
        ($expr:expr) => {{ get_frame_observing(expr_arena.get($expr), expr_arena)? }};
    }

    macro_rules! zip {
        ($($expr:expr),*) => {{ zip([$(Ok(rec!($expr))),*])? }};
    }

    use ExprOutputOrder as O;
    Ok(match aexpr {
        // Explode creates local orders.
        //
        // The following observes order:
        //
        // a: [[1, 2], [3]]
        // b: [[3], [4, 5]]
        //
        // col(a).explode() * col(b).explode()
        AExpr::Explode { expr, .. } => rec!(*expr) | O::Independent,

        AExpr::Column(_) => O::Frame,
        AExpr::Literal(lv) if lv.is_scalar() => O::None,
        AExpr::Literal(_) => O::Independent,

        AExpr::Cast { expr, .. } => rec!(*expr),

        // Elementwise can be seen as a `zip + op`.
        AExpr::BinaryExpr { left, op: _, right } => zip!(*left, *right),
        AExpr::Filter { input, by } => zip!(*input, *by),
        AExpr::Ternary {
            predicate,
            truthy,
            falsy,
        } => zip!(*predicate, *truthy, *falsy),

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
            | IRAggExpr::Var(node, _) => {
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
                if rec!(*node).has_frame_ordering() {
                    return Err(FrameOrderObserved);
                }
                O::None
            },

            // @NOTE: This aggregation makes very little sense. We do the most pessimistic thing
            // possible here.
            IRAggExpr::AggGroups(_) => return Err(FrameOrderObserved),
        },

        AExpr::Gather {
            expr,
            idx,
            returns_scalar,
        } => {
            let expr = rec!(*expr);
            let idx = rec!(*idx);

            // We need to ensure that the values come in frame order. The order of the idxes is
            // propagated.
            if expr.has_frame_ordering() {
                return Err(FrameOrderObserved);
            }

            if *returns_scalar { O::None } else { idx }
        },
        AExpr::AnonymousFunction { input, options, .. }
        | AExpr::Function { input, options, .. } => {
            // By definition, does not observe any input so cannot observe any ordering.
            if input.is_empty() {
                return Ok(
                    if options.flags.returns_scalar() || options.flags.is_output_unordered() {
                        O::None
                    } else {
                        O::Independent
                    },
                );
            }

            // Elementwise are regarded as a `zip + function`.
            if options.flags.is_elementwise() {
                return zip(input.iter().map(|e| Ok(rec!(e.node()))));
            }

            if options.flags.propagates_order() {
                // Propagate the order of the singular input, this is for expressions like
                // `drop_nulls` and `rechunk`.
                assert_eq!(input.len(), 1);
                match rec!(input[0].node()) {
                    v if !(options.flags.is_row_separable()
                        || options.flags.is_input_order_agnostic())
                        && v.has_frame_ordering() =>
                    {
                        return Err(FrameOrderObserved);
                    },
                    v => v,
                }
            } else if options.flags.is_input_order_agnostic() {
                // There are also expressions that are entirely input order agnostic like
                // `null_count` and `unique(maintain_order=False)`
                for e in input {
                    _ = rec!(e.node());
                }

                if options.returns_scalar() || options.flags.is_output_unordered() {
                    O::None
                } else {
                    O::Independent
                }
            } else {
                // For other expressions, we are observing frame order i.f.f. any of the inputs are
                // observing frame order.
                for e in input {
                    if rec!(e.node()).has_frame_ordering() {
                        return Err(FrameOrderObserved);
                    }
                }

                if options.returns_scalar() || options.flags.is_output_unordered() {
                    O::None
                } else {
                    O::Independent
                }
            }
        },

        AExpr::Eval {
            expr,
            evaluation: _,
            variant,
        } => match variant {
            EvalVariant::List => rec!(*expr),
            EvalVariant::Cumulative { min_samples: _ } => {
                let expr = rec!(*expr);
                if expr.has_frame_ordering() {
                    return Err(FrameOrderObserved);
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
            if input.has_frame_ordering() {
                return Err(FrameOrderObserved);
            }
            for e in partition_by {
                if rec!(*e).has_frame_ordering() {
                    return Err(FrameOrderObserved);
                }
            }
            if let Some((e, _)) = &order_by
                && rec!(*e).has_frame_ordering()
            {
                return Err(FrameOrderObserved);
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
            if input.has_frame_ordering() {
                return Err(FrameOrderObserved);
            }
            input
        },
        AExpr::Len => O::None,
    })
}
