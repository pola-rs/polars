use polars_utils::arena::Arena;

use crate::dsl::EvalVariant;
use crate::plans::{AExpr, IRAggExpr};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameOrdering;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExprOutputOrder {
    /// The expression's output order is solely reliant on whether the input dataframe has a
    /// defined order.
    Frame,

    /// The expression's output order is derived from the input dataframe order, but not
    /// necessarily equal to the frame order.
    ///
    /// Changing the order of the input dataframe might change the output order, but it cannot be
    /// regarded as the same ordering as the frame ordering or any order derived ordering.
    Derived,

    /// The expression's output order is completely independent from the frame order. This may mean
    /// that the output is unordered, has a constant order or is derived from another independent order.
    Independent,

    /// The expression is a scalar and thus has no order.
    ///
    /// This is a special case of the independent output order in that it can be elementwise zipped
    /// with any order ordering to keep that ordering.
    Scalar,
}

impl ExprOutputOrder {
    /// Derive from the current output order while not necessarily preserving length.
    pub fn length_altering_derive(self) -> Self {
        match self {
            Self::Frame | Self::Derived => Self::Derived,
            Self::Independent | Self::Scalar => Self::Independent,
        }
    }

    /// Derive from the current output order while preserving length.
    pub fn length_preserving_derive(self) -> Self {
        match self {
            Self::Frame | Self::Derived => Self::Derived,
            Self::Independent => Self::Independent,
            Self::Scalar => Self::Scalar,
        }
    }

    /// Do a elementwise zip between two output orderings.
    pub fn zip_with(self, other: Self) -> Result<Self, FrameOrdering> {
        match (self, other) {
            (Self::Scalar, v) | (v, Self::Scalar) => Ok(v),
            (Self::Derived, _) | (_, Self::Derived) => Err(FrameOrdering),
            (v, w) if v == w => Ok(v),
            _ => Err(FrameOrdering),
        }
    }

    /// Does the output order observe an ordering (in)directly derived from the frame ordering.
    pub fn observes_frame(self) -> bool {
        matches!(self, Self::Frame | Self::Derived)
    }

    fn independent(is_scalar: bool) -> Self {
        if is_scalar {
            Self::Scalar
        } else {
            Self::Independent
        }
    }
}

pub fn zip(
    orders: impl IntoIterator<Item = Result<ExprOutputOrder, FrameOrdering>>,
) -> Result<ExprOutputOrder, FrameOrdering> {
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
    order: Result<ExprOutputOrder, FrameOrdering>,
) -> Result<ExprOutputOrder, FrameOrdering> {
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
) -> Result<ExprOutputOrder, FrameOrdering> {
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
        AExpr::Explode { expr, .. } => rec!(*expr).length_altering_derive(),

        AExpr::Column(_) => O::Frame,
        AExpr::Literal(lv) => O::independent(lv.is_scalar()),

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
                rec!(*expr).length_preserving_derive()
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
                zipped.length_preserving_derive()
            } else {
                O::independent(matches!(zipped, O::Scalar))
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
                O::Scalar
            },
            IRAggExpr::Quantile { expr, quantile, .. } => {
                // Input and quantile order is deregarded, but must not observe order.
                _ = rec!(*expr);
                _ = rec!(*quantile);
                O::Scalar
            },

            // Input order observing aggregations.
            IRAggExpr::Implode(node) | IRAggExpr::First(node) | IRAggExpr::Last(node) => {
                if rec!(*node).observes_frame() {
                    return Err(FrameOrdering);
                }
                O::Scalar
            },

            // @NOTE: This aggregation makes very little sense. We do the most pessimistic thing
            // possible here.
            IRAggExpr::AggGroups(_) => return Err(FrameOrdering),
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
            if expr.observes_frame() {
                return Err(FrameOrdering);
            }

            if *returns_scalar { O::Scalar } else { idx }
        },
        AExpr::AnonymousFunction { input, options, .. }
        | AExpr::Function { input, options, .. } => {
            // By definition, does not observe any input so cannot observe any ordering.
            if input.is_empty() {
                O::independent(options.returns_scalar())
            } else if options.flags.is_elementwise() {
                // Elementwise are regarded as a `zip + function`.
                zip(input.iter().map(|e| Ok(rec!(e.node()))))?
            } else if options.flags.propagates_order() {
                // Propagate the order of the singular input, this is for expressions like
                // `drop_nulls` and `rechunk`.
                assert_eq!(input.len(), 1);
                match rec!(input[0].node()) {
                    v if !options.flags.is_row_separable() && v.observes_frame() => {
                        return Err(FrameOrdering);
                    },
                    O::Scalar if !options.flags.is_length_preserving() => O::Independent,
                    v => v,
                }
            } else if options.flags.is_input_order_agnostic() {
                // There are also expressions that are entirely input order agnostic like
                // `null_count` and `unique(maintain_order=False)`
                for e in input {
                    _ = rec!(e.node());
                }
                O::independent(options.flags.returns_scalar())
            } else {
                // For other expressions, we are observing frame order i.f.f. any of the inputs are
                // observing frame order.
                for e in input {
                    if rec!(e.node()).observes_frame() {
                        return Err(FrameOrdering);
                    }
                }
                O::independent(options.flags.returns_scalar())
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
                if expr.observes_frame() {
                    return Err(FrameOrdering);
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
            if input.observes_frame() {
                return Err(FrameOrdering);
            }
            for e in partition_by {
                if rec!(*e).observes_frame() {
                    return Err(FrameOrdering);
                }
            }
            if let Some((e, _)) = &order_by
                && rec!(*e).observes_frame()
            {
                return Err(FrameOrdering);
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

            if rec!(*input).observes_frame() {
                return Err(FrameOrdering);
            }
            O::Independent
        },
        AExpr::Len => O::Scalar,
    })
}
