use IR::*;
use polars_core::error::PolarsResult;
use polars_core::prelude::*;
use polars_utils::arena::{Arena, Node};

use super::OptimizationRule;
use crate::plans::{AExpr, ExprIR, is_sorted};
use crate::prelude::IR;

pub struct CoalesceSort {}

impl OptimizationRule for CoalesceSort {
    fn optimize_plan(
        &mut self,
        lp_arena: &mut Arena<IR>,
        expr_arena: &mut Arena<AExpr>,
        node: Node,
    ) -> PolarsResult<Option<IR>> {
        if let Some(result) = try_coalesce_sorts(node, lp_arena, expr_arena) {
            return Ok(Some(result));
        }
        if let Some(result) = try_prune_sort_with_sortedness(node, lp_arena, expr_arena) {
            return Ok(Some(result));
        }
        Ok(None)
    }
}

/// If two consecutive sort nodes share a prefix of sort columns, replace them with
/// the sort node that covers the most columns.
fn try_coalesce_sorts(node: Node, lp_arena: &Arena<IR>, expr_arena: &Arena<AExpr>) -> Option<IR> {
    let IR::Sort {
        input,
        by_column,
        slice,
        sort_options,
    } = lp_arena.get(node)
    else {
        return None;
    };
    let IR::Sort {
        input: in_input,
        by_column: in_by_column,
        slice: in_slice,
        sort_options: in_sort_options,
    } = lp_arena.get(*input)
    else {
        return None;
    };

    let slice = match (slice, in_slice) {
        (s @ Some((offset, len, None)), None) => s,
        (None, s @ Some((offset, len, None))) => s,
        _ => return None,
    };

    let expr_eq = |e1: &&ExprIR, e2: &&ExprIR| {
        AExpr::is_expr_equal_to(
            expr_arena.get(e1.node()),
            expr_arena.get(e2.node()),
            expr_arena,
        )
    };

    let main_has_most_cols = prefix_dominance(by_column.iter(), in_by_column.iter(), expr_eq)?;
    let (merged_options, main_has_most_cols2) =
        coalesce_sort_multiple_options(sort_options, in_sort_options)?;
    debug_assert_eq!(main_has_most_cols, main_has_most_cols2);

    Some(if main_has_most_cols {
        Sort {
            input: *in_input,
            by_column: by_column.clone(),
            slice: slice.clone(),
            sort_options: merged_options,
        }
    } else {
        Sort {
            input: *in_input,
            by_column: in_by_column.clone(),
            slice: slice.clone(),
            sort_options: merged_options,
        }
    })
}

fn try_prune_sort_with_sortedness(
    node: Node,
    lp_arena: &Arena<IR>,
    expr_arena: &Arena<AExpr>,
) -> Option<IR> {
    let IR::Sort {
        input,
        by_column,
        slice,
        sort_options,
    } = lp_arena.get(node)
    else {
        return None;
    };
    if by_column.iter().any(|e| !expr_arena.get(e.node()).is_col()) {
        return None;
    }
    let by = by_column
        .iter()
        .map(|e| expr_arena.get(e.node()).to_name(expr_arena));
    let sort_props = Iterator::zip(
        sort_options.descending.iter(),
        sort_options.nulls_last.iter(),
    );
    let node_sort_cols = by.zip(sort_props);

    let input_sortedness = is_sorted(*input, lp_arena, expr_arena)?;
    let input_sort_cols = input_sortedness
        .0
        .iter()
        .map(|s| (&s.column, s.descending, s.nulls_last));
    if !prefix_dominance(input_sort_cols, node_sort_cols, |n1, n2| {
        *n1.0 == n2.0 && n1.1 == Some(*n2.1.0) && n1.2 == Some(*n2.1.1)
    })? {
        return None;
    }

    // We can safely prune this sort node
    if let Some((offset, len, None)) = slice {
        Some(IR::Slice {
            input: *input,
            offset: *offset,
            len: *len as IdxSize,
        })
    } else {
        Some(lp_arena.get(*input).clone())
    }
}

/// Checks whether one iterator is a prefix of the other (or they are equal).
///
/// Returns `Some(true)` if the left iterator has at least as many elements as the right,
/// `Some(false)` if the right iterator is strictly longer, and `None` if the iterators
/// diverge before either is exhausted.
fn prefix_dominance<T, U, I1, I2, EQ>(iter1: I1, iter2: I2, eq: EQ) -> Option<bool>
where
    I1: IntoIterator<Item = T>,
    I2: IntoIterator<Item = U>,
    EQ: Fn(&T, &U) -> bool,
{
    let mut iter1 = iter1.into_iter();
    let mut iter2 = iter2.into_iter();
    loop {
        match (iter1.next(), iter2.next()) {
            (Some(a), Some(b)) if eq(&a, &b) => {},
            (Some(_), Some(_)) => return None,
            (_, None) => return Some(true),
            (None, Some(_)) => return Some(false),
        }
    }
}

fn coalesce_sort_multiple_options(
    node: &SortMultipleOptions,
    input: &SortMultipleOptions,
) -> Option<(SortMultipleOptions, bool)> {
    let node_sort_props = Iterator::zip(node.descending.iter(), node.nulls_last.iter());
    let input_sort_propes: std::iter::Zip<std::slice::Iter<'_, bool>, std::slice::Iter<'_, bool>> =
        Iterator::zip(input.descending.iter(), input.nulls_last.iter());
    let node_sorts_most_cols = prefix_dominance(node_sort_props, input_sort_propes, |n, i| n == i)?;

    // We don't care about the multithreaded criterion

    let maintain_order = node.maintain_order && input.maintain_order;
    let limit = match (node.limit, input.limit) {
        (Some(l1), Some(l2)) => Some(IdxSize::min(l1, l2)),
        (Some(l), None) | (None, Some(l)) => Some(l),
        (None, None) => None,
    };
    let result = SortMultipleOptions {
        maintain_order,
        limit,
        ..if node_sorts_most_cols {
            node.clone()
        } else {
            input.clone()
        }
    };
    Some((result, node_sorts_most_cols))
}
