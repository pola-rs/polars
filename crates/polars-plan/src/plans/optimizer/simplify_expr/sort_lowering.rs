use crate::plans::optimizer::repeated_keys::{first_expr_occurrences, retain_sort_keys};
use crate::plans::*;

/// Rewrites expressions that only read the first or last rows of a sort:
/// - `x.sort().first()` and `x.sort().last()` -> `x.min()` or `x.max()` in the order of the sort
/// - `x.sort_by(a).first()` and `x.sort_by(a).last()` -> `x.min_by(a)` or `x.max_by(a)` in the
///   order of the sort
/// - `x.sort_by(by).first()` -> `x.min_by(row_encode(by))`
/// - `x.sort_by(by).last()` -> `x.max_by(row_encode(by))`
/// - `x.sort().first()` and `x.sort().slice(offset, len)` -> a sort that only has to output its
///   first `1` or `offset + len` rows.
pub(super) fn lower_sort_reads(
    node: Node,
    arena: &mut Arena<AExpr>,
    schema: &Schema,
) -> Option<AExpr> {
    match arena.get(node) {
        AExpr::Agg(IRAggExpr::First(input)) => {
            let input = *input;
            sort_first_last_to_min_max(input, false, arena, schema)
                .or_else(|| sort_by_first_last_to_min_max_by(input, false, arena, schema))
                .or_else(|| {
                    let input = sort_with_limit(input, 1, arena)?;
                    Some(AExpr::Agg(IRAggExpr::First(input)))
                })
        },
        AExpr::Agg(IRAggExpr::Last(input)) => {
            let input = *input;
            sort_first_last_to_min_max(input, true, arena, schema)
                .or_else(|| sort_by_first_last_to_min_max_by(input, true, arena, schema))
        },
        AExpr::Slice {
            input,
            offset,
            length,
        } => {
            let (input, offset, length) = (*input, *offset, *length);
            let limit = slice_head_len(offset, length, arena)?;
            let input = sort_with_limit(input, limit, arena)?;
            Some(AExpr::Slice {
                input,
                offset,
                length,
            })
        },
        _ => None,
    }
}

/// The number of leading rows a slice reads, if `offset` and `length` are literals, `offset` is not
/// negative and the slice is not empty.
fn slice_head_len(offset: Node, length: Node, arena: &Arena<AExpr>) -> Option<IdxSize> {
    let (AExpr::Literal(offset), AExpr::Literal(length)) = (arena.get(offset), arena.get(length))
    else {
        return None;
    };
    let end = offset
        .extract_usize()
        .ok()?
        .checked_add(length.extract_usize().ok()?)?;
    if end == 0 {
        return None;
    }
    IdxSize::try_from(end).ok()
}

/// Adds a copy of the `Sort` or `SortBy` at `node` that only has to output its first `limit` rows.
/// Returns `None` if the sort already has the same or a smaller limit.
///
/// The copy may output more rows, so its users must still slice it.
fn sort_with_limit(node: Node, limit: IdxSize, arena: &mut Arena<AExpr>) -> Option<Node> {
    let mut sort = arena.get(node).clone();
    let current_limit = match &mut sort {
        AExpr::Sort { options, .. } => &mut options.limit,
        AExpr::SortBy { sort_options, .. } => &mut sort_options.limit,
        _ => return None,
    };
    if current_limit.is_some_and(|l| l <= limit) {
        return None;
    }
    *current_limit = Some(limit);
    Some(arena.add(sort))
}

/// Whether `min`, `max`, `min_by` and `max_by` support `dtype` in the order `sort` uses.
fn has_total_min_max(dtype: &DataType) -> bool {
    #[cfg(feature = "dtype-f16")]
    if matches!(dtype, DataType::Float16) {
        return false;
    }
    dtype.is_primitive_numeric()
        || dtype.is_temporal()
        || dtype.is_decimal()
        || dtype.is_categorical()
        || dtype.is_enum()
        || matches!(
            dtype,
            DataType::Boolean | DataType::String | DataType::Binary
        )
}

/// A float `max` that propagates NaN needs the `propagate_nans` feature.
fn can_propagate_nans(dtype: &DataType) -> bool {
    !dtype.is_float() || cfg!(feature = "propagate_nans")
}

fn sort_first_last_to_min_max(
    input: Node,
    is_last: bool,
    arena: &mut Arena<AExpr>,
    schema: &Schema,
) -> Option<AExpr> {
    let AExpr::Sort { expr, options } = arena.get(input) else {
        return None;
    };
    let dtype = arena
        .get(*expr)
        .to_dtype(&ToFieldContext::new(arena, schema))
        .ok()?;
    let is_max = is_last != options.descending;
    if !has_total_min_max(&dtype) || (is_max && !can_propagate_nans(&dtype)) {
        return None;
    }

    let input = *expr;
    let propagate_nans = is_max && dtype.is_float();
    let propagate_nulls = options.nulls_last == is_last;
    Some(AExpr::Agg(if is_max {
        IRAggExpr::Max {
            input,
            propagate_nans,
            propagate_nulls,
        }
    } else {
        IRAggExpr::Min {
            input,
            propagate_nans,
            propagate_nulls,
        }
    }))
}

/// The ordered row encoding sorts like `by`, nulls included. Only done if the sort does not keep
/// the order of equal rows, as `min_by` and `max_by` may pick any of them.
fn sort_by_first_last_to_min_max_by(
    input: Node,
    is_last: bool,
    arena: &mut Arena<AExpr>,
    schema: &Schema,
) -> Option<AExpr> {
    let AExpr::SortBy {
        expr,
        by,
        sort_options,
    } = arena.get(input)
    else {
        return None;
    };
    if sort_options.maintain_order
        || by.is_empty()
        || !is_length_preserving_ae(*expr, arena)
        || !by.iter().all(|e| is_length_preserving_ae(*e, arena))
    {
        return None;
    }

    let expr = *expr;
    let mut by = by.clone();
    let mut sort_options = sort_options.clone();
    // This runs during conversion, before the optimizer removes repeated keys.
    if let Some(keep) = first_expr_occurrences(by.iter().copied(), arena) {
        retain_sort_keys(&mut by, &mut sort_options, &keep);
    }
    let SortMultipleOptions {
        descending,
        nulls_last,
        ..
    } = sort_options;

    let ctx = ToFieldContext::new(arena, schema);
    let dtypes = by
        .iter()
        .map(|e| arena.get(*e).to_dtype(&ctx).ok())
        .collect::<Option<Vec<_>>>()?;
    if dtypes.iter().any(|dt| dt.is_object()) {
        return None;
    }

    let expr = AExprBuilder::new_from_node(expr);
    if let ([by], [dtype]) = (by.as_slice(), dtypes.as_slice())
        && (has_total_min_max(dtype) || dtype.is_null())
    {
        let is_max = is_last != descending[0];
        let order = MinMaxByOrder::Total {
            nulls_largest: nulls_last[0] != descending[0],
        };
        let out = if is_max {
            expr.max_by(*by, order, arena)
        } else {
            expr.min_by(*by, order, arena)
        };
        return Some(arena.get(out.node()).clone());
    }

    let by = by
        .into_iter()
        .map(|e| ExprIR::from_node(e, arena))
        .collect();
    let encoded = AExprBuilder::row_encode(
        by,
        dtypes,
        RowEncodingVariant::Ordered {
            descending: Some(descending),
            nulls_last: Some(nulls_last),
            broadcast_nulls: None,
        },
        arena,
    );
    let out = if is_last {
        expr.max_by(encoded, MinMaxByOrder::SkipNulls, arena)
    } else {
        expr.min_by(encoded, MinMaxByOrder::SkipNulls, arena)
    };
    Some(arena.get(out.node()).clone())
}
