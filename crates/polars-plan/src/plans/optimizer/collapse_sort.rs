use polars_core::error::PolarsResult;
use polars_core::prelude::*;
use polars_utils::arena::{Arena, Node};

use super::OptimizationRule;
use crate::plans::{AExpr, is_sorted};
use crate::prelude::*;

pub struct CollapseSort {}

impl OptimizationRule for CollapseSort {
    fn optimize_plan(
        &mut self,
        lp_arena: &mut Arena<IR>,
        expr_arena: &mut Arena<AExpr>,
        node: Node,
    ) -> PolarsResult<Option<IR>> {
        if let Some(result) = try_collapse_sorts(node, lp_arena, expr_arena) {
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
fn try_collapse_sorts(node: Node, lp_arena: &Arena<IR>, expr_arena: &Arena<AExpr>) -> Option<IR> {
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

    if !sort_options.maintain_order {
        return Some(IR::Sort {
            input: *in_input,
            by_column: by_column.clone(),
            slice: slice.clone(),
            sort_options: sort_options.clone(),
        });
    }

    let mut by_column = by_column.clone();
    let mut descending = sort_options.descending.clone();
    let mut nulls_last = sort_options.nulls_last.clone();
    let in_so_iter = Iterator::zip(
        in_sort_options.descending.iter(),
        in_sort_options.nulls_last.iter(),
    );
    let mut l_stack = Default::default();
    let mut r_stack = Default::default();
    for (by, (d, nl)) in in_by_column.iter().zip(in_so_iter) {
        let by_node = expr_arena.get(by.node());
        let expr_is_eq = |x: &ExprIR| {
            by_node.is_expr_equal_to_amortized(
                expr_arena.get(x.node()),
                expr_arena,
                &mut l_stack,
                &mut r_stack,
            )
        };
        if !by_column.iter().any(expr_is_eq) {
            by_column.push(by.clone());
            descending.push(*d);
            nulls_last.push(*nl);
        }
    }

    let slice = match (slice, in_slice) {
        (Some((o1, l1, None)), Some((o2, l2, None))) => Some((o1 + o2, usize::min(*l1, *l2), None)),
        (s @ Some((_, _, None)), None) | (None, s @ Some((_, _, None))) => s.to_owned(),
        (None, None) => None,
        _ => return None, // TODO: Implement dynamic slices too
    };
    let maintain_order = in_sort_options.maintain_order;
    let limit = match (sort_options.limit, in_sort_options.limit) {
        (Some(l1), Some(l2)) => Some(IdxSize::min(l1, l2)),
        (Some(l), None) | (None, Some(l)) => Some(l),
        (None, None) => None,
    };
    let sort_options = SortMultipleOptions {
        descending,
        nulls_last,
        maintain_order,
        limit,
        ..sort_options.clone()
    };
    Some(IR::Sort {
        input: *in_input,
        by_column,
        slice,
        sort_options,
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
    let node_sortedness_props = by.zip(sort_props);

    let input_sortedness = is_sorted(*input, lp_arena, expr_arena)?;
    let input_sortedness_props = input_sortedness
        .0
        .iter()
        .map(|s| (&s.column, s.descending, s.nulls_last));
    let node_sorts_most_columns =
        prefix_dominance(input_sortedness_props, node_sortedness_props, |n1, n2| {
            *n1.0 == n2.0 && n1.1 == Some(*n2.1.0) && n1.2 == Some(*n2.1.1)
        })?;
    if !node_sorts_most_columns {
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
