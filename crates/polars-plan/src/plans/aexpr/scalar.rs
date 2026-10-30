use super::*;
use crate::plans::projection_height::{ExprProjectionHeight, aexpr_projection_height_rec};

pub fn is_scalar_ae(node: Node, arena: &Arena<AExpr>) -> bool {
    use ExprProjectionHeight as H;

    match aexpr_projection_height_rec(
        node,
        arena,
        &mut Default::default(),
        &mut Default::default(),
    ) {
        H::Scalar => true,
        H::Column | H::Unknown | H::Range => false,
    }
}

/// A literal with exactly one value, including a single row `Series`.
pub fn is_single_literal_ae(node: Node, arena: &Arena<AExpr>) -> bool {
    matches!(arena.get(node), AExpr::Literal(lv) if lv.is_single_value())
}

pub fn is_length_preserving_ae(node: Node, arena: &Arena<AExpr>) -> bool {
    use ExprProjectionHeight as H;

    match aexpr_projection_height_rec(
        node,
        arena,
        &mut Default::default(),
        &mut Default::default(),
    ) {
        H::Column => true,
        H::Scalar | H::Unknown | H::Range => false,
    }
}

pub fn is_known_length_ae(node: Node, arena: &Arena<AExpr>) -> bool {
    use ExprProjectionHeight as H;

    match aexpr_projection_height_rec(
        node,
        arena,
        &mut Default::default(),
        &mut Default::default(),
    ) {
        H::Column | H::Scalar | H::Range => true,
        H::Unknown => false,
    }
}

/// Matches `int_range(0, length)` with step 1, where `length` is `len()` or `count()`, and returns
/// `length` and the dtype of the range.
///
/// `length` may be cast to a dtype that holds any length, as a strict cast to a narrower one fails
/// on one more row than a cast of the range would.
#[cfg(feature = "range")]
pub fn int_range_over_length_ae(node: Node, arena: &Arena<AExpr>) -> Option<(Node, DataType)> {
    let AExpr::Function {
        input,
        function: IRFunctionExpr::Range(IRRangeFunction::IntRange { step: 1, dtype }),
        ..
    } = arena.get(node)
    else {
        return None;
    };
    let [start, end] = input.as_slice() else {
        return None;
    };
    if !matches!(arena.get(start.node()), AExpr::Literal(lit) if lit.extract_usize().ok() == Some(0))
    {
        return None;
    }
    let mut length = end.node();
    if let AExpr::Cast {
        expr,
        dtype: cast_dtype,
        ..
    } = arena.get(length)
        && (*cast_dtype == IDX_DTYPE
            || matches!(
                cast_dtype,
                DataType::Int64 | DataType::UInt64 | DataType::Int128 | DataType::UInt128
            ))
    {
        length = *expr;
    }
    matches!(
        arena.get(length),
        AExpr::Len | AExpr::Agg(IRAggExpr::Count { .. })
    )
    .then(|| (length, dtype.clone()))
}

/// Whether every output value only depends on the values in its own row and on the set of rows
/// in the group, not on the order of those rows.
#[recursive::recursive]
pub fn is_order_insensitive_ae(node: Node, arena: &Arena<AExpr>) -> bool {
    let inputs_insensitive = |input: &[ExprIR]| {
        input
            .iter()
            .all(|e| is_order_insensitive_ae(e.node(), arena))
    };

    match arena.get(node) {
        AExpr::Column(_) | AExpr::Len => true,
        AExpr::Literal(lv) => lv.is_scalar(),
        AExpr::Cast { expr, .. } => is_order_insensitive_ae(*expr, arena),
        AExpr::BinaryExpr { left, op: _, right } => {
            is_order_insensitive_ae(*left, arena) && is_order_insensitive_ae(*right, arena)
        },
        AExpr::Ternary {
            predicate,
            truthy,
            falsy,
        } => [*predicate, *truthy, *falsy]
            .into_iter()
            .all(|n| is_order_insensitive_ae(n, arena)),
        AExpr::Agg(agg) => match agg {
            IRAggExpr::Min { input, .. }
            | IRAggExpr::Max { input, .. }
            | IRAggExpr::Median(input)
            | IRAggExpr::NUnique(input)
            | IRAggExpr::Item { input, .. }
            | IRAggExpr::Mean(input)
            | IRAggExpr::Sum { input, .. }
            | IRAggExpr::Count { input, .. }
            | IRAggExpr::Std(input, _)
            | IRAggExpr::Var(input, _) => is_order_insensitive_ae(*input, arena),
            IRAggExpr::First(_)
            | IRAggExpr::FirstNonNull(_)
            | IRAggExpr::Last(_)
            | IRAggExpr::LastNonNull(_)
            | IRAggExpr::Implode { .. } => false,
        },
        AExpr::Function {
            function: IRFunctionExpr::SetSortedFlag(_),
            ..
        } => false,
        AExpr::Function { input, options, .. } => {
            let flags = options.flags;
            !flags.observes_input_order()
                && flags.non_order_producing()
                && (flags.is_length_preserving() || flags.returns_scalar())
                && inputs_insensitive(input)
        },
        _ => false,
    }
}
