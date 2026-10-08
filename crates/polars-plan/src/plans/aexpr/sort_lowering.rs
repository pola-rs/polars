use polars_core::chunked_array::ops::sort::_broadcast_bools;

use super::*;

/// The number of leading rows a slice reads, if `offset` and `length` are literals, `offset` is not
/// negative and the slice is not empty.
pub fn slice_head_len(offset: Node, length: Node, arena: &Arena<AExpr>) -> Option<IdxSize> {
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
///
/// The copy may output more rows, so its users must still slice it.
pub fn sort_with_limit(node: Node, limit: IdxSize, arena: &mut Arena<AExpr>) -> Option<Node> {
    let mut sort = arena.get(node).clone();
    let current_limit = match &mut sort {
        AExpr::Sort { options, .. } => &mut options.limit,
        AExpr::SortBy { sort_options, .. } => &mut sort_options.limit,
        _ => return None,
    };
    *current_limit = Some(current_limit.map_or(limit, |l| l.min(limit)));
    Some(arena.add(sort))
}

/// Lowers `x.sort_by(by).first()` at `node` to `x.min_by(row_encode(by))`, and
/// `x.sort_by(by).last()` to `x.max_by(row_encode(by))`.
///
/// The ordered row encoding sorts like `by`, nulls included. Only done if the sort does not keep
/// the order of equal rows, as `min_by` and `max_by` may pick any of them.
pub fn sort_by_first_last_to_min_max_by(
    node: Node,
    arena: &mut Arena<AExpr>,
    schema: &Schema,
) -> Option<Node> {
    let (input, is_last) = match arena.get(node) {
        AExpr::Agg(IRAggExpr::First(input)) => (*input, false),
        AExpr::Agg(IRAggExpr::Last(input)) => (*input, true),
        _ => return None,
    };
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
    let by = by.clone();
    let mut descending = sort_options.descending.clone();
    let mut nulls_last = sort_options.nulls_last.clone();
    _broadcast_bools(by.len(), &mut descending);
    _broadcast_bools(by.len(), &mut nulls_last);

    let ctx = ToFieldContext::new(arena, schema);
    let dtypes = by
        .iter()
        .map(|e| arena.get(*e).to_dtype(&ctx).ok())
        .collect::<Option<Vec<_>>>()?;
    if dtypes.iter().any(|dt| dt.is_object()) {
        return None;
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
    let expr = AExprBuilder::new_from_node(expr);
    let out = if is_last {
        expr.max_by(encoded, arena)
    } else {
        expr.min_by(encoded, arena)
    };
    Some(out.node())
}
