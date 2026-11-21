use polars_utils::idx_vec::UnitVec;
use polars_utils::unitvec;

use super::*;
use crate::plans::projection_height::{
    ExprHeightOptions, ExprProjectionHeight, aexpr_projection_height_rec,
    aexpr_projection_height_rec_with,
};

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

/// Raises if an argument that must be a scalar is not, in the expression at `node`.
///
/// This runs after type coercion, which can turn a flat column into one list of values.
pub fn ensure_scalar_arguments(node: Node, arena: &Arena<AExpr>) -> PolarsResult<()> {
    // The second value is the height of a `StructField` in the expression.
    let mut stack: UnitVec<(Node, ExprProjectionHeight)> =
        unitvec![(node, ExprProjectionHeight::Unknown)];
    let mut inputs: UnitVec<Node> = unitvec![];
    while let Some((node, struct_field)) = stack.pop() {
        let aexpr = arena.get(node);
        ensure_node_scalar_arguments(aexpr, arena, struct_field).map_err(|e| {
            e.with_expr_context(ExprIRDisplay::display_node(node, arena).to_string().into())
        })?;

        match aexpr {
            AExpr::Eval { evaluation, .. } => {
                stack.push((*evaluation, ExprProjectionHeight::Unknown));
            },
            #[cfg(feature = "dtype-struct")]
            AExpr::StructEval {
                expr, evaluation, ..
            } => {
                let options = ExprHeightOptions {
                    struct_field,
                    ..Default::default()
                };
                let struct_height = aexpr_projection_height_rec_with(
                    *expr,
                    arena,
                    &mut Default::default(),
                    &mut Default::default(),
                    options,
                );
                stack.push((*expr, struct_field));
                stack.extend(evaluation.iter().map(|e| (e.node(), struct_height)));
                continue;
            },
            _ => {},
        }
        inputs.clear();
        aexpr.inputs(&mut inputs);
        stack.extend(inputs.iter().map(|&input| (input, struct_field)));
    }
    Ok(())
}

fn ensure_node_scalar_arguments(
    aexpr: &AExpr,
    arena: &Arena<AExpr>,
    struct_field: ExprProjectionHeight,
) -> PolarsResult<()> {
    match aexpr {
        AExpr::Gather {
            idx,
            returns_scalar: true,
            ..
        } => ensure_scalar_ae(*idx, arena, struct_field, "index", "get"),
        AExpr::Slice { offset, length, .. } => {
            ensure_scalar_ae(*offset, arena, struct_field, "offset", "slice")?;
            ensure_scalar_ae(*length, arena, struct_field, "length", "slice")
        },
        AExpr::Function {
            function, input, ..
        } => {
            let name: &dyn std::fmt::Display = match function {
                #[cfg(feature = "approx_quantile")]
                IRFunctionExpr::ApproxQuantileEstimate { .. } => &"approx_quantile",
                function => function,
            };
            for &(idx, arg) in function.scalar_inputs() {
                let input = input.get(idx).ok_or_else(
                    || polars_err!(InvalidOperation: "missing argument '{arg}' in `{name}`"),
                )?;
                ensure_scalar_ae(input.node(), arena, struct_field, arg, name)?;
            }
            Ok(())
        },
        _ => Ok(()),
    }
}

/// Raises unless the argument `arg` of `function` at `node` is a scalar, where a `StructField`
/// has the height `struct_field`.
fn ensure_scalar_ae(
    node: Node,
    arena: &Arena<AExpr>,
    struct_field: ExprProjectionHeight,
    arg: &str,
    function: impl std::fmt::Display,
) -> PolarsResult<()> {
    let options = ExprHeightOptions {
        struct_field,
        scalar_argument: true,
        ..Default::default()
    };
    let height = aexpr_projection_height_rec_with(
        node,
        arena,
        &mut Default::default(),
        &mut Default::default(),
        options,
    );
    polars_ensure!(
        height == ExprProjectionHeight::Scalar,
        ShapeMismatch: "'{arg}' must be a scalar value in `{function}`; use a literal or an \
        aggregation such as `.item()`, `.first()` or, for a list, `.implode()`"
    );
    Ok(())
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
