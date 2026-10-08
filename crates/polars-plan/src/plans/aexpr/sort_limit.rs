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
